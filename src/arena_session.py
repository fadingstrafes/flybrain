"""Fixed-step sensory -> CNS -> motor -> body loop, independent of rendering."""

import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd

from src.arena_model import Arena, DEFAULT_ARENA_SIZE
from src.arena_learning import ExplorationLearner
from src.live_brain import LiveBrain, SensoryMotorBridge
from src.arena_vision import BinocularVision, VisualNavigator
from src.visual_input import VisualInput
from src.stimulus_recording import StimulusRecording


class ArenaSession:
    def __init__(self, graph, watched_ids=(), device="auto", neural_hz=60, exploration=True,
                 max_events=2_000_000, cache=Path("data/cache"), learning=True,
                 policy_path=None, seed=42, training=True, arena_size=DEFAULT_ARENA_SIZE,
                 vision=True, avoidance=True, max_art_events=2_000_000):
        self.art_recording = StimulusRecording(max_art_events)
        self.graph = graph
        self.cache = Path(cache)
        self.brain = LiveBrain(graph, device=device)
        self.bridge = SensoryMotorBridge(graph, cache=cache)
        self.arena = Arena(half_width=arena_size[0]/2, half_height=arena_size[1]/2)
        self.arena.obstacles = tuple(o for o in self.arena.obstacles
            if abs(o[0])+o[2] < self.arena.half_width and abs(o[1])+o[2] < self.arena.half_height)
        self.vision = BinocularVision() if vision else None
        self.visual_input = VisualInput(graph, cache) if vision else None
        self.avoidance_enabled = avoidance
        self.dt = 1.0 / neural_hz
        self.neural_hz = neural_hz
        self.exploration = exploration
        self.max_events = max_events
        self.learner = ExplorationLearner(policy_path, seed=seed, training=training) if learning else None
        lookup = {int(body): i for i, body in enumerate(graph.ids)}
        self.watched = np.array([lookup.get(int(body), -1) for body in watched_ids], dtype=np.int64)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.paused = False
        self.manual_touch = np.zeros(6, dtype=np.float32)
        self.art_enabled = False
        self.art_strength = 1.1
        self.art_limit = 2048
        self.reset()

    def reset(self):
        with self.lock:
            self.brain.reset()
            self.bridge.reset()
            self.arena.reset()
            self.counts = np.zeros(len(self.graph.ids), dtype=np.int32)
            self.activity = np.zeros(len(self.graph.ids), dtype=np.float32)
            self.recorded_indices, self.recorded_times = [], []
            self.recorded_count = 0
            self.recording_stopped_step = None
            self.trajectory = []
            self.manual_touch.fill(0)
            self.art_indices = np.empty(0, dtype=np.int64)
            self.art_levels = np.empty(0, dtype=np.float32)
            self.art_cursor = 0
            self.art_target_counts = np.zeros(len(self.graph.ids), dtype=np.int64)
            self.art_injected_steps = 0
            self.art_last_targets = 0
            self.art_recording.reset()
            self.touch = np.zeros(6, dtype=np.float32)
            self.proprio = np.zeros(6, dtype=np.float32)
            self.last_ms = 0.0
            self.error = None
            self.pattern = None
            self.flight_requested = False
            self.flight_override = None
            self.navigator = VisualNavigator()
            self.visual_frame = None
            self.visual_spikes = 0
            self.visual_mask = np.zeros(len(self.graph.ids),dtype=bool)
            if self.visual_input is not None:
                self.visual_mask[self.visual_input.indices] = True
            if self.vision is not None:
                self.vision.reset()
                self.visual_frame = self.vision.sample(self.arena, self.dt)
            if self.learner is not None:
                self.learner.reset_episode()

    def set_art_stimulus(self, indices, levels):
        indices = np.asarray(indices, dtype=np.int64)
        levels = np.asarray(levels, dtype=np.float32)
        if (indices.ndim != 1 or levels.shape != indices.shape or not np.isfinite(levels).all()
                or np.any(indices < 0) or np.any(indices >= len(self.graph.ids))):
            raise ValueError("Invalid mapped artwork input")
        with self.lock:
            self.art_indices = indices.copy()
            self.art_levels = np.clip(levels, 0, 1)

    def art_stimulus(self):
        """Bounded round-robin current injection; synaptic propagation is unchanged."""
        self.art_last_targets = 0
        if not self.art_enabled or not len(self.art_indices) or self.art_strength <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        count = min(self.art_limit, len(self.art_indices))
        slots = (np.arange(count)+self.art_cursor)%len(self.art_indices)
        self.art_cursor = (self.art_cursor+count)%len(self.art_indices)
        indices = self.art_indices[slots]
        currents = self.art_levels[slots]*self.art_strength
        self.art_last_targets = count
        return indices, currents

    def tick(self):
        with self.lock:
            if self.paused or self.brain.halted:
                return
            start = time.perf_counter()
            self.touch, self.proprio = self.arena.sense()
            vision_interval = max(1, round(self.neural_hz/20))
            if self.vision is not None and self.brain.step_count % vision_interval == 0:
                self.visual_frame = self.vision.sample(self.arena, self.dt*vision_interval)
            stimulus_touch = np.maximum(self.touch, self.manual_touch)
            if self.learner is not None and self.exploration:
                self.learner.visit(self.arena.position)
                interval = max(1, round(self.neural_hz * .5))
                if self.brain.step_count % interval == 0:
                    perception = self.arena.pose()
                    perception["clearances"] = self.visual_frame.clearances if self.visual_frame is not None else None
                    self.pattern = self.learner.decide(stimulus_touch, perception)
                    if self.flight_override is None and self.learner.names[self.learner.action] == "takeoff":
                        self.flight_requested = True
                    elif self.flight_override is None and self.learner.names[self.learner.action] == "land":
                        self.flight_requested = False
            elif self.learner is not None:
                # Do not assign outcomes of manually disabled drive to a learned action.
                self.learner.last_state = None
                self.learner.new_tiles = 0
            indices, currents = self.bridge.stimulus(stimulus_touch, self.proprio, self.arena.elapsed,
                                                    exploration=self.exploration, pattern=self.pattern,
                                                    flight_requested=self.flight_requested)
            if self.visual_frame is not None:
                vi, vc = self.visual_input.stimulus(self.visual_frame.signals)
                indices = np.concatenate([indices,vi])
                currents = np.concatenate([currents,vc])
            ai, ac = self.art_stimulus()
            if len(ai):
                indices = np.concatenate([indices, ai])
                currents = np.concatenate([currents, ac])
            fired = self.brain.step(indices, currents)
            if len(ai):
                # Record the exact float32 currents consumed by a completed
                # neural step, including the step that triggers the guard.
                np.add.at(self.art_target_counts, ai, 1)
                self.art_injected_steps += 1
                self.art_recording.append(self.brain.step_count - 1, self.graph.ids[ai], ac)
            self.visual_spikes += int(np.count_nonzero(self.visual_mask[fired]))
            self.counts[fired] += 1
            self.activity *= 0.85
            self.activity[self.activity < .05] = 0
            self.activity[fired] += 1
            step = self.brain.step_count - 1
            if self.recording_stopped_step is None:
                if self.recorded_count + len(fired) <= self.max_events:
                    if len(fired):
                        self.recorded_indices.append(fired.astype(np.int32))
                        self.recorded_times.append(np.full(len(fired), step, dtype=np.int32))
                        self.recorded_count += len(fired)
                else:
                    self.recording_stopped_step = step
            self.bridge.observe(fired, self.dt)
            navigation = None
            if self.visual_frame is not None and self.avoidance_enabled:
                navigation = self.navigator.command(self.visual_frame.clearances, self.arena.speed)
            else:
                self.navigator.active = False
            self.arena.step(self.bridge.activation, self.dt, self.bridge.wing_activation, self.flight_requested, navigation)
            self.manual_touch *= np.exp(-self.dt / 0.3)
            # A bounded path for drawing; full telemetry goes to the session CSV.
            self.trajectory.append([step, self.arena.elapsed, *self.arena.position,
                                    self.arena.yaw, self.arena.speed, self.brain.active_count,
                                    self.bridge.motor_spikes, *stimulus_touch, *self.bridge.activation,
                                    self.learner.action if self.learner else -1,
                                    self.learner.episode_reward if self.learner else 0,
                                    len(self.learner.visited) if self.learner else 0,
                                    self.arena.altitude, self.arena.vertical_speed,
                                    *self.bridge.wing_activation, int(self.flight_requested),
                                    *(self.visual_frame.clearances if self.visual_frame is not None else [np.nan]*3),
                                    int(self.navigator.active), self.arena.contacts])
            self.last_ms = (time.perf_counter() - start) * 1000

    def snapshot(self, whole_brain=False):
        with self.lock:
            watched = np.zeros(len(self.watched), dtype=np.float32)
            valid = self.watched >= 0
            watched[valid] = self.activity[self.watched[valid]]
            pose = self.arena.pose()
            pose.update({"activation": self.bridge.activation.copy(), "extension": self.bridge.extension.copy(),
                         "touch": np.maximum(self.touch, self.manual_touch).copy(), "proprio": self.proprio.copy(),
                         "watched_activity": watched, "step": self.brain.step_count,
                         "active": self.brain.active_count, "motor_spikes": self.bridge.motor_spikes,
                         "brain_ms": self.last_ms, "paused": self.paused, "halted": self.brain.halted,
                         "exploration": self.exploration, "error": self.error,
                         "path": np.array([row[2:4] for row in self.trajectory[-2400::6]]),
                         "recording_complete": self.recording_stopped_step is None})
            pose["wing_activation"] = self.bridge.wing_activation.copy()
            pose["wing_spikes"] = self.bridge.wing_spikes
            pose["flight_requested"] = self.flight_requested
            pose["learning"] = self.learner.summary() if self.learner is not None else None
            pose["vision"] = self.visual_frame
            pose["avoidance_enabled"] = self.avoidance_enabled and self.vision is not None
            pose["avoidance_active"] = self.navigator.active
            pose["visual_spikes"] = self.visual_spikes
            pose["arena_size"] = (self.arena.half_width*2,self.arena.half_height*2)
            pose["obstacles"] = self.arena.obstacles
            pose["art_input"] = {"enabled": self.art_enabled, "strength": self.art_strength,
                                 "mapped": len(self.art_indices), "targets": self.art_last_targets,
                                 "limit": self.art_limit, "injected_steps": self.art_injected_steps,
                                 "recording": self.art_recording.summary(self.brain.step_count)}
            stride = max(1, len(self.trajectory)//1200)
            pose["map_path"] = np.asarray([row[2:4] for row in self.trajectory[::stride][-1200:]])
            if whole_brain:
                pose["brain_activity"] = self.activity.copy()
                pose["brain_counts"] = self.counts.copy()
            return pose

    def start(self):
        def work():
            try:
                while not self.stop_event.is_set():
                    start = time.perf_counter()
                    self.tick()
                    self.stop_event.wait(max(0.001, self.dt - (time.perf_counter() - start)))
            except Exception as error:
                with self.lock:
                    self.error = f"{type(error).__name__}: {error}"
                    self.paused = True
        self.thread = threading.Thread(target=work, name="FlyBrain simulation", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def export(self, output):
        output = Path(output)
        with self.lock:
            output.mkdir(parents=True, exist_ok=True)
            indices = np.concatenate(self.recorded_indices) if self.recorded_indices else np.empty(0, dtype=np.int32)
            times = np.concatenate(self.recorded_times) if self.recorded_times else np.empty(0, dtype=np.int32)
            np.savez_compressed(output / "spike_events.npz", time=times, bodyId=self.graph.ids[indices])
            np.savez_compressed(output / "spike_counts.npz", neuron_ids=self.graph.ids, spike_counts=self.counts)
            self.art_recording.save(output / "art_input_events.npz", self.brain.step_count, self.neural_hz)
            metadata = pd.read_parquet(self.cache / "metadata.parquet", columns=["bodyId", "instance", "superclass"])
            active = np.flatnonzero(self.counts)
            ranked = active[np.argsort(self.counts[active])[::-1]][:50]
            pd.DataFrame({"bodyId": self.graph.ids[ranked], "spikes": self.counts[ranked]}).merge(
                metadata.drop_duplicates("bodyId"), on="bodyId", how="left"
            ).to_csv(output / "top_active.csv", index=False)
            motor = pd.read_csv(self.cache / "motor_map.csv")
            counts = pd.Series(self.counts, index=self.graph.ids)
            motor["spikes"] = motor.bodyId.map(counts).fillna(0).astype(int)
            motor.loc[motor.spikes > 0, ["bodyId", "instance", "spikes"]].sort_values(
                "spikes", ascending=False).to_csv(output / "active_motor_neurons.csv", index=False)
            columns = ["step", "seconds", "x", "y", "yaw", "speed", "active_neurons", "motor_spikes"]
            columns += ["touch_" + name.replace(" ", "_") for name in self.bridge.leg_names]
            columns += ["motor_" + name.replace(" ", "_") for name in self.bridge.leg_names]
            columns += ["sensory_action", "learning_reward", "visited_tiles"]
            columns += ["altitude", "vertical_speed", "wing_L", "wing_R", "flight_requested"]
            columns += ["vision_left", "vision_front", "vision_right", "avoidance_active", "contact_steps"]
            pd.DataFrame(self.trajectory, columns=columns).to_csv(output / "trajectory.csv", index=False)
            params = {
                "dataset": "male-cns:v1.0", "backend": "live-arena-" + str(self.brain.device),
                "steps": self.brain.step_count, "completed_steps": self.brain.step_count,
                "recording_complete": self.recording_stopped_step is None,
                "recording_stopped_step": self.recording_stopped_step,
                "recorded_events": self.recorded_count, "total_spikes": int(self.counts.sum()),
                "aborted": self.brain.halted, "abort_reason": "live retained-population activity guard" if self.brain.halted else None,
                "neural_hz": self.neural_hz, "leak": 0.92, "threshold": 1.0, "min_weight": 5,
                "synaptic_gain": 0.35, "source_body_count": self.graph.source_size,
                "retained_body_count": len(self.graph.ids),
                "model": "Exploratory sensory drive and body kinematics are imposed modeling assumptions",
                "distance": self.arena.distance, "obstacles": self.arena.obstacles,
                "arena_half_size": [self.arena.half_width, self.arena.half_height],
                "learning": self.learner.summary() if self.learner is not None else None,
                "flight_model": "Wing motor limited lift toward requested altitude; toy gravity and drag",
                "vision": self.vision is not None,
                "visual_avoidance_assist": self.avoidance_enabled and self.vision is not None,
                "visual_relay_spikes": self.visual_spikes, "contact_steps": self.arena.contacts,
                "avoidance_interventions": self.navigator.interventions,
                "visual_mapping": "Synthetic retinal sectors to annotated T4/T5 relays; not measured retinotopy",
                "art_input": {"enabled_at_export": self.art_enabled, "strength_at_export": self.art_strength,
                              "max_targets_per_step": self.art_limit, "injected_steps": self.art_injected_steps,
                              "target_deliveries": int(self.art_target_counts.sum()),
                              "recording": self.art_recording.summary(self.brain.step_count),
                              "mapping": "View-projected soma/root positions sample media/ink brightness; bounded round-robin external current, not retinal physiology"},
            }
            if self.visual_input is not None:
                pd.DataFrame({"bodyId": self.visual_input.body_ids}).to_csv(output/"visual_input_neurons.csv",index=False)
            (output / "params.json").write_text(json.dumps(params, indent=2) + "\n")
            targeted = self.art_target_counts > 0
            pd.DataFrame({"bodyId": self.graph.ids[targeted], "input_steps": self.art_target_counts[targeted]}).to_csv(
                output / "art_input_neurons.csv", index=False)
            if self.learner is not None:
                self.learner.save(output / "policy.json")
                if self.learner.training:
                    self.learner.save()
        return output

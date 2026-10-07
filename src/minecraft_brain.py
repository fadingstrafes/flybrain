"""Local Minecraft protocol and fixed sparse connectome adapter (protocol version 1)."""
import json
from collections import deque
import threading
import time
from pathlib import Path
import numpy as np
from src.minecraft_learning import MinecraftLearner, ACTIONS, FEATURES
from src.minecraft_telemetry import BodyTelemetry
from src.minecraft_drives import SurvivalDrives, validate_world, state_features, estimated_drive
from src.minecraft_vision import CompoundEyes, decode_vision


def validate_observation(data):
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('Expected protocol version 1')
    if not isinstance(data.get('session'), str) or not 1 <= len(data['session']) <= 80:
        raise ValueError('Invalid session')
    for name in ('life', 'seq'):
        if type(data.get(name)) is not int or not 0 <= data[name] < 2**53:
            raise ValueError('Invalid sequence/life')
    if type(data.get('dead')) is not bool:
        raise ValueError('Expected dead boolean')
    for name in ('health', 'food'):
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1024:
            raise ValueError('Invalid health/food')
    senses = np.asarray(data.get('senses'), dtype=float)
    if senses.shape != (32,) or not np.isfinite(senses).all() or np.any((senses < 0) | (senses > 1)):
        raise ValueError('Expected 32 finite sensory levels in [0, 1]')
    validate_world(data.get('world'))
    decode_vision(data.get('vision'))
    return data


class NeuralAdapter:
    """Synthetic channel mapping onto annotated sensory cells; no physiology claim.

    Each channel stimulates 16 cells. Readout pools actual CNS spikes into 64
    deterministic bins. Both maps are explicit engineering assumptions.
    """
    def __init__(self, graph, cache=Path('data/cache'), device='auto'):
        import pandas as pd
        from src.live_brain import LiveBrain
        self.brain = LiveBrain(graph, device=device)
        self.ids = graph.ids
        sensory = pd.read_csv(Path(cache) / 'sensory_map.csv')
        candidates = np.intersect1d(self.ids, sensory.bodyId.to_numpy(dtype=np.int64))
        if len(candidates) < 512:
            raise ValueError('Need at least 512 annotated sensory cells')
        # Seeded selection avoids assigning adjacent body IDs as a sensory organ.
        chosen = np.random.default_rng(1701).choice(candidates, 512, replace=False)
        self.inputs = self.brain.indices(chosen).reshape(32, 16)
        self.eyes = CompoundEyes(self.ids,cache)
        self.bins = np.arange(len(self.ids)) % 64
        source_ids = np.load(Path(cache)/'neuron_ids.npy', mmap_mode='r')
        source_labels = np.load(Path(cache)/'nt_label.npy', mmap_mode='r')
        indices = np.searchsorted(source_ids, self.ids)
        if np.any(indices >= len(source_ids)) or not np.array_equal(source_ids[indices], self.ids):
            raise ValueError('Transmitter labels do not align with live neuron IDs')
        labels = np.char.lower(np.asarray(source_labels[indices]))
        self.nt_groups = {name: np.flatnonzero(labels == name) for name in
                          ('acetylcholine', 'gaba', 'glutamate', 'dopamine', 'serotonin', 'octopamine')}
        self.reset()

    def reset(self):
        self.brain.reset()
        self.activity = np.zeros(len(self.ids), dtype=np.float32)
        self.counts = np.zeros(len(self.ids), dtype=np.int64)
        self.filtered = np.zeros(64)
        self.recent_counts = np.zeros(len(self.ids), dtype=np.int32)
        self.eyes.reset()
        self.optic_features=np.zeros(32)

    def set_vision(self,vision):
        self.eyes.update(decode_vision(vision),None if vision is None else vision['frame'])

    def encode(self, senses):
        histogram = np.zeros(64)
        self.recent_counts.fill(0)
        currents = np.repeat(np.asarray(senses, dtype=np.float32) * 1.1, 16)
        indices=self.inputs.ravel()
        if self.eyes.rgb is not None:
            indices=np.r_[indices,self.eyes.indices]
            currents=np.r_[currents,self.eyes.currents]
        for _ in range(6):
            fired = self.brain.step(indices, currents)
            if self.brain.halted:
                raise RuntimeError('Neural activity guard triggered; restart the brain service')
            histogram += np.bincount(self.bins[fired], minlength=64)
            self.activity *= .9
            self.activity[fired] = 1
            self.counts[fired] += 1
            self.recent_counts[fired] += 1
        self.filtered = .6 * self.filtered + .4 * (1 - np.exp(-histogram / 20))
        self.optic_features=self.eyes.readout(self.recent_counts)
        return np.r_[1., self.filtered]

    def snapshot(self):
        return dict(self.eyes.snapshot(self.recent_counts),brain_activity=self.activity.copy(), brain_counts=self.counts.copy(),
                    brain_recent=self.recent_counts.copy(),
                    transmitters={name: dict(total=len(indices),
                        active=int(np.count_nonzero(self.recent_counts[indices])),
                        spikes=int(self.recent_counts[indices].sum()))
                        for name, indices in self.nt_groups.items()},
                    active=int(np.count_nonzero(self.recent_counts)), step=self.brain.step_count)


class MinecraftSession:
    def __init__(self, adapter, learner, log_path=None):
        self.adapter, self.learner = adapter, learner
        self.lock = threading.RLock()
        self.identity = None
        self.retired_sessions = deque(maxlen=32)
        self.last_seq = -1
        self.last_response = None
        self.previous = None
        self.terminal = False
        self.status = dict(action='idle', health=0, food=0, reward=0)
        self.body = BodyTelemetry()
        self.drives = SurvivalDrives()
        self.received_at = None
        self.log_path = Path(log_path) if log_path else None

    def step(self, observation):
        o = validate_observation(observation)
        with self.lock:
            identity = (o['session'], o['life'])
            if self.identity is not None:
                if o['session'] in self.retired_sessions:
                    raise ValueError('Retired client session')
                if o['session'] == self.identity[0]:
                    if o['life'] < self.identity[1] or (o['life'] > self.identity[1] and o['seq'] <= self.last_seq):
                        raise ValueError('Out-of-order life')
                elif identity != self.identity:
                    self.retired_sessions.append(self.identity[0])
            if identity != self.identity:
                self.adapter.reset()
                self.body.reset()
                self.drives.reset()
                self.drives.baseline(o)
                self.learner.reset_episode()
                self.previous = None
                self.last_seq = -1
                self.terminal = False
                self.identity = identity
            if o['seq'] == self.last_seq:
                return self.last_response
            if o['seq'] < self.last_seq:
                raise ValueError('Out-of-order observation')
            if self.terminal and not o['dead']:
                raise ValueError('Respawn must increment life')
            if hasattr(self.adapter,'set_vision'): self.adapter.set_vision(o.get('vision'))
            features = np.zeros(FEATURES) if o['dead'] else np.r_[state_features(self.adapter.encode(o['senses']),o),getattr(self.adapter,'optic_features',np.zeros(32))]
            reward = 0.
            components = {}
            if self.previous is not None and not self.terminal:
                old, old_features, action = self.previous
                components = self.drives.transition(old,o)
                reward = sum(components.values())
                self.learner.learn(old_features, action, reward, features, o['dead'])
            action = 0 if o['dead'] else self.learner.choose(features)
            response = dict(version=1, session=o['session'], life=o['life'], seq=o['seq'],
                            action=ACTIONS[action], respawn=o['dead'])
            self.previous = None if o['dead'] else (dict(o), features.copy(), action)
            self.terminal = o['dead']
            self.last_seq, self.last_response = o['seq'], response
            self.status = dict(action=ACTIONS[action], health=o['health'], food=o['food'], reward=reward,
                               deaths=self.learner.deaths, updates=self.learner.updates,
                               life=o['life'], seq=o['seq'], training=self.learner.training)
            self.status.update(self.body.update(o, ACTIONS[action]))
            self.status.update(reward_components=components, reward_pulse=max(0.,reward),
                               drive_state=estimated_drive(o,components,self.drives.idle_ticks),
                               idle_ticks=self.drives.idle_ticks, world=o.get('world',{}))
            self.received_at = time.monotonic()
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open('a') as stream:
                    recorded=dict(o)
                    if 'vision' in recorded:
                        recorded['vision']={k:v for k,v in o['vision'].items() if k!='rgb'}
                    stream.write(json.dumps(dict(observation=recorded, response=response, **self.status))+'\n')
            return response

    def snapshot(self):
        with self.lock:
            return dict(self.adapter.snapshot(), **self.status,
                        observation_age=None if self.received_at is None else time.monotonic()-self.received_at)

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.arena_session import ArenaSession
from src.live_brain import LiveBrain, project_graph


class StimulusRecordingTests(unittest.TestCase):
    def make_session(self, **kwargs):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        cache = Path(folder.name)
        pd.DataFrame(columns=["bodyId", "side", "system", "subclass"]).to_csv(
            cache / "sensory_map.csv", index=False)
        pd.DataFrame([dict(bodyId=815344, instance="Ti extensor MN_L", side="L",
                           leg_pair="front", limb="leg", action="extend")]).to_csv(
            cache / "motor_map.csv", index=False)
        pd.DataFrame(dict(bodyId=[513052, 815344], instance=["DNg15_R", "Ti extensor MN_L"],
                          superclass=["descending", "vnc_motor"])).to_parquet(cache / "metadata.parquet")
        graph = project_graph(csr_matrix([[0, 8], [0, 0]]), np.array([513052, 815344]),
                              np.array([1, 0], dtype="f4"), [815344])
        session = ArenaSession(graph, device="cpu", cache=cache, learning=False,
                               vision=False, exploration=False, **kwargs)
        session.brain.abort_fraction = 0
        session.art_enabled = True
        return session, cache / "output"

    def test_exported_currents_reproduce_spikes_with_round_robin_and_silent_tail(self):
        session, output = self.make_session()
        session.tick()  # Silent leading step.
        session.set_art_stimulus([0], [1])
        for _ in range(8):
            session.tick()
        self.assertGreater(session.counts[1], 0)  # Motor fired through the connection.
        session.set_art_stimulus([0, 1], [.75, .25])
        session.art_limit = 1
        session.art_strength = 2
        for _ in range(4):
            session.tick()
        session.paused = True
        session.tick()
        session.paused = False
        session.art_enabled = False
        for _ in range(3):
            session.tick()
        session.export(output)
        with np.load(output / "art_input_events.npz", allow_pickle=False) as saved:
            self.assertEqual(int(saved["completed_steps"]), 16)
            self.assertEqual(int(saved["recorded_prefix_steps"]), 16)
            self.assertTrue(bool(saved["recording_complete"]))
            self.assertEqual(int(saved["recording_stopped_step"]), -1)
            np.testing.assert_array_equal(saved["time"], np.arange(1, 13))
            np.testing.assert_array_equal(saved["bodyId"][-4:], [513052, 815344, 513052, 815344])
            np.testing.assert_array_equal(saved["current"][-4:], [1.5, .5, 1.5, .5])
            replay = LiveBrain(session.graph, device="cpu", abort_fraction=0)
            replay_times, replay_ids = [], []
            for step in range(int(saved["completed_steps"])):
                selected = saved["time"] == step
                fired = replay.step(replay.indices(saved["bodyId"][selected]), saved["current"][selected])
                replay_times.extend([step] * len(fired))
                replay_ids.extend(session.graph.ids[fired])
        with np.load(output / "spike_events.npz") as spikes:
            np.testing.assert_array_equal(replay_times, spikes["time"])
            np.testing.assert_array_equal(replay_ids, spikes["bodyId"])
        params = json.loads((output / "params.json").read_text())
        self.assertEqual(params["art_input"]["target_deliveries"], 12)
        self.assertEqual(params["art_input"]["recording"]["recorded_events"], 12)

    def test_cap_keeps_whole_step_prefix_and_totals_continue(self):
        session, output = self.make_session(max_art_events=3, max_events=0)
        session.set_art_stimulus([0, 1], [1, .5])
        session.tick()  # Two input deliveries fit; spike log overflows separately.
        session.tick()  # Whole step omitted, including the otherwise fitting event.
        session.art_limit = 1
        session.tick()  # Must not resume recording in the remaining slot.
        session.export(output)
        with np.load(output / "art_input_events.npz") as saved:
            np.testing.assert_array_equal(saved["time"], [0, 0])
            self.assertFalse(bool(saved["recording_complete"]))
            self.assertEqual(int(saved["recording_stopped_step"]), 1)
            self.assertEqual(int(saved["recorded_prefix_steps"]), 1)
            self.assertEqual(int(saved["completed_steps"]), 3)
        params = json.loads((output / "params.json").read_text())
        self.assertEqual(params["recording_stopped_step"], 0)
        self.assertEqual(params["art_input"]["target_deliveries"], 5)
        self.assertEqual(params["art_input"]["injected_steps"], 3)
        self.assertEqual(session.snapshot()["art_input"]["recording"]["recording_stopped_step"], 1)

    def test_exact_cap_is_complete_until_another_delivery_and_reset_replaces_export(self):
        session, output = self.make_session(max_art_events=1)
        session.set_art_stimulus([0], [1])
        session.tick()
        session.set_art_stimulus([], [])
        session.tick()
        session.export(output)
        with np.load(output / "art_input_events.npz") as saved:
            self.assertTrue(bool(saved["recording_complete"]))
            self.assertEqual(int(saved["recorded_prefix_steps"]), 2)
        session.set_art_stimulus([0], [1])
        session.tick()
        self.assertEqual(session.art_recording.stopped_step, 2)
        session.reset()
        session.export(output)
        with np.load(output / "art_input_events.npz") as saved:
            self.assertEqual(len(saved["time"]), 0)
            self.assertEqual(len(saved["bodyId"]), 0)
            self.assertEqual(len(saved["current"]), 0)
            self.assertTrue(bool(saved["recording_complete"]))
            self.assertEqual(int(saved["completed_steps"]), 0)
        self.assertEqual(len(pd.read_csv(output / "art_input_neurons.csv")), 0)

    def test_zero_cap_disables_recording_without_disabling_input(self):
        session, output = self.make_session(max_art_events=0)
        session.tick()
        session.set_art_stimulus([0], [1])
        session.tick()
        session.export(output)
        self.assertEqual(session.counts[0], 1)
        self.assertEqual(session.art_target_counts[0], 1)
        with np.load(output / "art_input_events.npz") as saved:
            self.assertEqual(len(saved["time"]), 0)
            self.assertFalse(bool(saved["recording_complete"]))
            self.assertEqual(int(saved["recording_stopped_step"]), 1)

    def test_guard_step_recorded_but_halted_and_failed_steps_are_not(self):
        session, output = self.make_session()
        session.brain.abort_fraction = .2
        session.set_art_stimulus([0], [1])
        session.tick()
        self.assertTrue(session.brain.halted)
        session.tick()
        session.export(output)
        with np.load(output / "art_input_events.npz") as saved:
            np.testing.assert_array_equal(saved["time"], [0])
            self.assertEqual(int(saved["completed_steps"]), 1)
        session.reset()
        session.set_art_stimulus([0], [1])
        from unittest.mock import patch
        with patch.object(session.brain, "step", side_effect=RuntimeError("failed step")):
            with self.assertRaisesRegex(RuntimeError, "failed step"):
                session.tick()
        self.assertEqual(session.art_recording.recorded_events, 0)
        self.assertEqual(session.art_injected_steps, 0)
        self.assertEqual(int(session.art_target_counts.sum()), 0)


if __name__ == "__main__":
    unittest.main()

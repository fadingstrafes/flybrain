import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.run_data import load_run, sum_saved_spikes


class RunDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)

    def save(self, times, counts, **params):
        times = np.asarray(times, dtype=np.int32)
        np.savez_compressed(self.run / "spike_events.npz", time=times,
                            bodyId=np.full(times.size, 513052, dtype=np.int64))
        np.savez_compressed(self.run / "spike_counts.npz",
                            spike_counts=np.asarray(counts, dtype=np.int32))
        (self.run / "params.json").write_text(json.dumps({"steps": 20, **params}))

    def test_complete_legacy_run_keeps_silent_tail(self):
        self.save([5, 2], [2])
        run = load_run(self.run)
        self.assertEqual(run.playback_steps, 20)
        self.assertEqual(run.times.tolist(), [2, 5])
        self.assertTrue(run.recording_complete)
        self.assertIsNone(run.completed_steps)

    def test_legacy_truncation_is_not_silence(self):
        self.save([2, 5], [200])
        run = load_run(self.run)
        self.assertFalse(run.recording_complete)
        self.assertEqual(run.playback_steps, 6)
        self.assertIn("INCOMPLETE", run.notices[0])

    def test_recording_cutoff_keeps_known_silent_interval(self):
        self.save([2, 5], [200], completed_steps=20, recording_complete=False,
                  recording_stopped_step=10)
        self.assertEqual(load_run(self.run).playback_steps, 10)

    def test_aborted_run_uses_completed_duration(self):
        self.save([2, 5], [2], completed_steps=8, aborted=True)
        run = load_run(self.run)
        self.assertEqual(run.playback_steps, 8)
        self.assertTrue(any("aborted" in notice for notice in run.notices))

    def test_zero_spikes_and_legacy_missing_event_file(self):
        self.save([], [0], completed_steps=20)
        self.assertEqual(load_run(self.run).playback_steps, 20)
        (self.run / "spike_events.npz").unlink()
        self.assertTrue(load_run(self.run).recording_complete)

    def test_missing_events_with_spikes_is_an_error(self):
        self.save([2], [1])
        (self.run / "spike_events.npz").unlink()
        with self.assertRaises(RuntimeError):
            load_run(self.run)

    def test_inconsistent_counts_and_timestamps_are_rejected(self):
        self.save([2, 3], [1])
        with self.assertRaises(ValueError):
            load_run(self.run)
        self.save([21], [1])
        with self.assertRaises(ValueError):
            load_run(self.run)

    def test_streamed_sum_crosses_chunk_boundary_without_int32_overflow(self):
        counts = np.full(1_000_005, 3000, dtype=np.int32)
        np.savez_compressed(self.run / "spike_counts.npz", spike_counts=counts)
        self.assertEqual(sum_saved_spikes(self.run / "spike_counts.npz"), 3_000_015_000)


if __name__ == "__main__":
    unittest.main()

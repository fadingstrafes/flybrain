"""Small asymmetric circuits test the real CLI without modifying production caches."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import torch

from src.run_data import load_run


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), "ROCm GPU is unavailable")
class GPUSimulationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cache = self.root / "data" / "cache"
        cache.mkdir(parents=True)
        ids = np.array([513052, 800002, 800206, 800373, 815344], dtype=np.int64)
        np.save(cache / "neuron_ids.npy", ids)
        np.save(cache / "nt_sign.npy", np.array([1, -1, 0, 1, 1], dtype=np.float32))
        np.save(cache / "nt_label.npy", np.array(["acetylcholine", "gaba", "", "acetylcholine", "acetylcholine"]))
        # Direction, inhibition, neutral transmitter and runtime edge filtering.
        W = csr_matrix((np.array([8, 4, 2, 8, 8, 8], dtype=np.int32),
                        ([0, 0, 0, 1, 2, 3], [1, 2, 3, 4, 4, 4])), shape=(5, 5))
        save_npz(cache / "connectome_raw.npz", W)
        pd.DataFrame({
            "bodyId": ids, "instance": ["DNg15_R", "IN1", "IN2", "IN3", "Ti extensor MN_L"],
            "superclass": ["descending_neuron", "vnc_intrinsic", "vnc_intrinsic", "vnc_intrinsic", "vnc_motor"],
            "consensus_nt": ["acetylcholine", "gaba", "", "acetylcholine", "acetylcholine"],
        }).to_parquet(cache / "metadata.parquet")

    def run_sim(self, backend, name, *extra):
        output = self.root / name
        command = [sys.executable, "-B", str(ROOT / "src" / f"sim_full{backend}.py"),
                   "--stimulate", "513052", "--steps", "20", "--stim-start", "2",
                   "--stim-end", "8", "--synaptic-gain", "1", "--min-weight", "3",
                   "--abort-fraction", "0", "--output", str(output), *extra]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return output

    def test_cpu_gpu_events_agree_for_asymmetric_signed_graph(self):
        cpu = self.run_sim("", "cpu")
        gpu = self.run_sim("_gpu", "gpu")
        for filename, keys in [("spike_events.npz", ["time", "bodyId"]),
                               ("spike_counts.npz", ["spike_counts", "neuron_ids"])]:
            with np.load(cpu / filename) as a, np.load(gpu / filename) as b:
                for key in keys:
                    np.testing.assert_array_equal(a[key], b[key])
        run = load_run(gpu)
        self.assertTrue(run.recording_complete)
        self.assertEqual(run.completed_steps, 20)
        # This fixture must actually propagate forward, not just agree on stimulus.
        self.assertIn(800002, run.body_ids)
        self.assertNotIn(815344, run.body_ids)

    def test_recording_limit_and_abort_metadata(self):
        output = self.run_sim("_gpu", "limited", "--max-recorded-events", "1")
        run = load_run(output)
        self.assertFalse(run.recording_complete)
        self.assertEqual(run.playback_steps, 3)
        self.assertGreater(run.total_spikes, len(run.times))
        output = self.run_sim("_gpu", "aborted", "--abort-fraction", "0.1")
        params = json.loads((output / "params.json").read_text())
        self.assertTrue(params["aborted"])
        self.assertEqual(params["completed_steps"], 3)

    def test_reusing_output_with_no_spikes_clears_old_events(self):
        self.run_sim("_gpu", "reuse")
        output = self.run_sim("_gpu", "reuse", "--stim-current", "0")
        run = load_run(output)
        self.assertEqual(run.total_spikes, 0)
        self.assertEqual(run.times.size, 0)
        self.assertTrue(pd.read_csv(output / "active_motor_neurons.csv").empty)
        self.assertTrue(pd.read_csv(output / "top_active.csv").empty)


if __name__ == "__main__":
    unittest.main()

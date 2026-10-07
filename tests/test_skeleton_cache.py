from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import cache_run_skeletons


class SkeletonCacheTests(unittest.TestCase):
    def test_cached_selection_works_without_credentials_or_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "run"
            skeletons = run / "skeleton_cache"
            skeletons.mkdir(parents=True)
            np.savez_compressed(run / "spike_events.npz", time=[0, 1], bodyId=[513052, 513052])
            np.savez_compressed(skeletons / "513052.npz", segments=np.zeros((1, 2, 3)))
            pd.DataFrame({"bodyId": [513052], "instance": ["DNg15_R"],
                          "superclass": ["descending_neuron"]}).to_parquet(root / "metadata.parquet")
            with patch.object(cache_run_skeletons, "CACHE", root), \
                    patch("sys.argv", ["cache_run_skeletons", "--run", str(run)]), \
                    patch.dict("os.environ", {}, clear=True):
                cache_run_skeletons.main()
            self.assertEqual(pd.read_csv(skeletons / "selected.csv").bodyId.tolist(), [513052])

    def test_empty_selection_preserves_existing_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "run"
            skeletons = run / "skeleton_cache"
            skeletons.mkdir(parents=True)
            selected = skeletons / "selected.csv"
            selected.write_text("bodyId\n815344\n")
            np.savez_compressed(run / "spike_events.npz", time=[0], bodyId=[513052])
            pd.DataFrame({"bodyId": [513052], "instance": ["DNg15_R"],
                          "superclass": ["descending_neuron"]}).to_parquet(root / "metadata.parquet")
            with patch.object(cache_run_skeletons, "CACHE", root), \
                    patch("sys.argv", ["cache_run_skeletons", "--run", str(run)]):
                cache_run_skeletons.main()
            self.assertEqual(selected.read_text(), "bodyId\n815344\n")


if __name__ == "__main__":
    unittest.main()

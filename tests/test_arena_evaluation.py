from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.arena_learning import ExplorationLearner
from src.arena_model import Arena
from src.arena_session import ArenaSession
from src.evaluate_arena import METRICS, episode_start, run_episode, summarize
from src.live_brain import project_graph


class EvaluationTests(unittest.TestCase):
    def test_starts_are_repeatable_clear_and_separate_from_training(self):
        arena = Arena()
        for seed in (11, 22, 33):
            for episode in range(4):
                position, yaw = episode_start(arena, seed, episode, 1)
                repeated, repeated_yaw = episode_start(arena, seed, episode, 1)
                np.testing.assert_array_equal(position, repeated)
                self.assertEqual(yaw, repeated_yaw)
                self.assertFalse(np.array_equal(position, episode_start(arena, seed, episode, 0)[0]))
                self.assertTrue(all(np.linalg.norm(position - [x, y]) > r + 2
                                    for x, y, r in arena.obstacles))

    def test_episode_reset_removes_previous_controller_and_freezes_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            pd.DataFrame([dict(bodyId=513052, side="L", system="front_leg", subclass="leg bristle")]).to_csv(cache / "sensory_map.csv", index=False)
            pd.DataFrame([dict(bodyId=815344, side="L", leg_pair="front", limb="leg", action="extend")]).to_csv(cache / "motor_map.csv", index=False)
            graph = project_graph(csr_matrix([[0, 8], [0, 0]]), np.array([513052, 815344]),
                                  np.array([1, 0], dtype=np.float32), [815344])
            session = ArenaSession(graph, cache=cache, device="cpu", learning=False, vision=False)
            session.brain.abort_fraction = 0
            start = (np.array([0., -3.5]), .25)
            baseline = run_episode(session, start, 61, 5)
            learner = ExplorationLearner(training=False)
            learner.q[:, 6] = 1  # Select takeoff and leave a persistent flight request.
            session.learner = learner
            original = learner.q.copy()
            run_episode(session, start, 61, 5)
            self.assertTrue(session.flight_requested)
            np.testing.assert_array_equal(learner.q, original)
            self.assertEqual(learner.updates, 0)
            session.learner = None
            repeated = run_episode(session, start, 61, 5)
            self.assertEqual(baseline, repeated)
            self.assertFalse(session.flight_requested)

    def test_paired_summary_excludes_incomplete_pairs_without_hiding_aborts(self):
        rows = []
        for episode in (0, 1):
            for controller, value in (("fixed", 2), ("untrained", 3), ("trained", 5)):
                rows.append(dict(seed=11, episode=episode, assist=False, phase="evaluation",
                                 controller=controller, halted=episode == 1 and controller == "trained",
                                 completed_steps=60, requested_steps=60,
                                 **{key: value for key in METRICS}))
        report = summarize(rows)
        self.assertEqual(report["groups"][2]["episodes"], 2)
        self.assertEqual(report["groups"][2]["complete_episodes"], 1)
        self.assertEqual(report["paired"][0]["complete_pairs"], 1)
        self.assertEqual(report["paired"][0]["mean_deltas"]["distance"], 3)
        self.assertEqual(report["paired"][1]["mean_deltas"]["distance"], 2)


if __name__ == "__main__":
    unittest.main()

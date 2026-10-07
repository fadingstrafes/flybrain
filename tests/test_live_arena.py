import tempfile
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from src.arena_learning import ExplorationLearner
from src.arena_model import Arena, LegGait, solve_knee
from src.live_brain import LiveBrain, project_graph


class ProjectionTests(unittest.TestCase):
    def test_projection_retains_all_effective_sources_and_original_normalization(self):
        ids = np.array([10, 20, 30, 40, 513052, 815344])
        signs = np.array([1, -1, 0, 0, 1, 0], dtype=np.float32)
        # Largest edge goes to an omitted neutral body; it must still normalize row 0.
        W = csr_matrix((np.array([8, 2, 16, 8, 4, 8], dtype=np.int32),
                        ([0, 0, 1, 2, 4, 4], [2, 5, 5, 5, 0, 5])), shape=(6, 6))
        graph = project_graph(W, ids, signs, [513052, 815344, 999999])
        self.assertEqual(graph.ids.tolist(), [10, 20, 513052, 815344])
        self.assertEqual(graph.row_max.tolist(), [8, 16, 8, 0])
        self.assertEqual(int(graph.weights[0, 3]), 2)
        brain = LiveBrain(graph, device="cpu", gain=.5, min_weight=1, abort_fraction=0)
        full_v = np.zeros(6, dtype=np.float32)
        previous = np.zeros(6, dtype=np.float32)
        row_max = np.asarray(W.max(axis=1).toarray()).ravel()
        scale = np.divide(signs * .5, row_max, out=np.zeros(6, dtype=np.float32), where=row_max > 0)
        keep = np.searchsorted(ids, graph.ids)
        for step in range(24):
            full_v *= .92
            full_v += W.T @ previous
            stimulus = 1.25 if step < 12 else 0
            full_v[4] += stimulus
            fired = full_v >= 1
            full_v[fired] = 0
            previous = fired * scale
            live_fired = brain.step(brain.indices([513052]), [stimulus])
            np.testing.assert_allclose(brain.voltage.numpy(), full_v[keep], atol=1e-6)
            np.testing.assert_array_equal(live_fired, np.flatnonzero(fired[keep]))

    def test_live_reset_removes_neural_history(self):
        graph = project_graph(csr_matrix([[0, 4], [0, 0]]), np.array([513052, 815344]),
                              np.array([1, 0], dtype=np.float32), [815344])
        brain = LiveBrain(graph, device="cpu", min_weight=1)
        brain.step([0], [2])
        brain.reset()
        self.assertEqual(brain.step_count, 0)
        self.assertEqual(len(brain.step()), 0)


class BodyTests(unittest.TestCase):
    def test_zero_motor_output_cannot_generate_motion(self):
        arena = Arena()
        start = arena.position.copy()
        for _ in range(120): arena.step(np.zeros(6), 1/60)
        np.testing.assert_array_equal(arena.position, start)

    def test_motor_asymmetry_steers_and_bounds_prevent_escape(self):
        arena = Arena(obstacles=())
        for _ in range(60): arena.step([.05, .05, .05, .8, .8, .8], 1/60)
        self.assertGreater(arena.yaw, 0)
        arena.position = np.array([0.0, arena.half_height - arena.radius - .01])
        arena.yaw = 0
        arena.turn_rate = 0
        for _ in range(120): arena.step(np.ones(6), 1/60)
        self.assertLessEqual(arena.position[1], arena.half_height - arena.radius)
        self.assertGreater(arena.contacts, 0)
        touch, _ = arena.sense()
        self.assertGreater(touch.max(), 0)

    def test_obstacle_collision_keeps_body_outside(self):
        arena = Arena(obstacles=((0, -2, .8),))
        for _ in range(180): arena.step(np.ones(6), 1/60)
        self.assertGreaterEqual(np.linalg.norm(arena.position - [0, -2]), .8 + arena.radius - 1e-9)

    def test_takeoff_requires_wing_output_and_landing_reaches_floor(self):
        arena = Arena(obstacles=())
        for _ in range(120): arena.step(np.zeros(6), 1/60, (0,0), True)
        self.assertEqual(arena.altitude, 0)
        for _ in range(180): arena.step(np.zeros(6), 1/60, (1,1), True)
        self.assertGreater(arena.altitude, 1.5)
        for _ in range(240): arena.step(np.zeros(6), 1/60, (1,1), False)
        self.assertAlmostEqual(arena.altitude, 0, places=5)

    def test_stance_feet_remain_in_world_space_and_swing_clears_ground(self):
        gait = LegGait()
        gait.reset([0,0], 0)
        before = gait.feet.copy()
        gait.update([0,.01], 0, 0, .5, .1, 1/60)
        np.testing.assert_array_equal(gait.feet, before)
        gait.update([0,.01], 0, 0, .5, 2*np.pi*.65, 1/60)
        self.assertGreater(gait.feet[0, 2], .02)

    def test_ik_preserves_segment_lengths_for_reachable_foot(self):
        hip=np.array([0,0,.5]);foot=np.array([.7,.2,.02])
        knee=solve_knee(hip,foot,[1,0,1])
        self.assertAlmostEqual(np.linalg.norm(knee-hip), .55)
        self.assertAlmostEqual(np.linalg.norm(foot-knee), .60)


class LearningTests(unittest.TestCase):
    @staticmethod
    def pose(x=0, contacts=0):
        return {"position": np.array([float(x), 0]), "speed": .5, "distance": abs(x), "contacts": contacts}

    def test_rewards_update_the_policy_and_survive_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            learner = ExplorationLearner(path, seed=5)
            learner.visit([0, 0])
            learner.decide(np.zeros(6), self.pose())
            first = learner.action
            learner.visit([1, 0])
            learner.decide(np.zeros(6), self.pose(1))
            self.assertGreater(learner.q[0, first], 0)
            self.assertEqual(learner.updates, 1)
            learner.save()
            loaded = ExplorationLearner(path)
            np.testing.assert_array_equal(loaded.q, learner.q)
            self.assertEqual(loaded.updates, 1)

    def test_stuck_contact_is_penalized_and_evaluation_is_frozen(self):
        learner = ExplorationLearner(seed=1)
        learner.decide(np.ones(6), self.pose())
        action, state = learner.action, learner.last_state
        learner.decide(np.ones(6), self.pose(0, contacts=30))
        self.assertLess(learner.q[state, action], 0)
        learner.training = False
        original = learner.q.copy()
        learner.decide(np.ones(6), self.pose(1, contacts=30))
        np.testing.assert_array_equal(original, learner.q)
        self.assertEqual(learner.epsilon, 0)

    def test_new_episode_preserves_learned_values(self):
        learner = ExplorationLearner()
        learner.q[0, 0] = 2
        learner.visit([0, 0])
        learner.reset_episode()
        self.assertEqual(learner.q[0, 0], 2)
        self.assertFalse(learner.visited)
        self.assertIsNone(learner.last_state)

    def test_walking_policy_migrates_without_forgetting(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'walking.json'
            old=np.arange(96,dtype=float).reshape(16,6)
            path.write_text(json.dumps({'version':1,'dataset':'male-cns:v1.0','q':old.tolist(),
                                        'updates':138,'lifetime_reward':12.5}))
            learner=ExplorationLearner(path)
            np.testing.assert_array_equal(learner.q[:16,:6],old)
            self.assertEqual(learner.q.shape,(256,8))
            self.assertEqual(learner.updates,138)
            self.assertTrue(np.all(learner.q[16:]==0))


if __name__ == "__main__":
    unittest.main()

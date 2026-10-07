import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import moderngl
import numpy as np
import pandas as pd

from src.arena_learning import ExplorationLearner
from src.arena_model import Arena
from src.arena_vision import BinocularVision, VisualNavigator
from src.brain_view import BrainMap, BrainRenderer
from src.visual_input import VisualInput


class VisionTests(unittest.TestCase):
    def test_eye_depth_tracks_wall_distance_and_rotation_changes_image(self):
        arena = Arena(obstacles=())
        arena.position = np.array([0., arena.half_height-3])
        vision = BinocularVision()
        near = vision.sample(arena, .05)
        self.assertEqual(near.images.shape, (2, 30, 90, 3))
        self.assertAlmostEqual(near.clearances[1], 2.4, places=2)
        arena.position[1] -= 2
        far = vision.sample(arena, .05)
        self.assertGreater(far.clearances[1], near.clearances[1]+1.9)
        arena.yaw = np.pi
        away = vision.sample(arena, .05)
        self.assertEqual(away.clearances[1], vision.max_range)
        self.assertFalse(np.array_equal(far.images, away.images))

    def test_obstacle_is_seen_on_ground_but_can_be_flown_over(self):
        arena = Arena(obstacles=((0, 0, 1),))
        arena.position = np.array([0., -4.])
        vision = BinocularVision()
        ground = vision.sample(arena, .05)
        self.assertLess(ground.clearances[1], 3)
        arena.altitude = 1.8
        air = vision.sample(arena, .05)
        self.assertEqual(air.clearances[1], vision.max_range)

    def test_visual_assist_reduces_wall_contacts_with_identical_motor_drive(self):
        results = []
        for enabled in (False, True):
            arena = Arena(half_width=8, half_height=6, obstacles=())
            vision, navigator = BinocularVision(), VisualNavigator()
            for step in range(900):
                if step % 3 == 0:
                    frame = vision.sample(arena, .05)
                command = navigator.command(frame.clearances, arena.speed) if enabled else None
                arena.step(np.full(6, .5), 1/60, navigation=command)
            results.append((arena.contacts, arena.distance))
        self.assertGreater(results[0][0], 400)
        self.assertEqual(results[1][0], 0)
        self.assertGreater(results[1][1], results[0][1]*2)

    def test_assist_cannot_power_a_silent_body(self):
        arena = Arena(obstacles=())
        before = arena.position.copy()
        for _ in range(120):
            arena.step(np.zeros(6), 1/60, navigation=(2.8, .5))
        np.testing.assert_array_equal(arena.position, before)
        self.assertEqual(arena.yaw, 0)

    def test_visual_inputs_only_target_matching_annotated_relays(self):
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame({"bodyId": [10, 20, 30, 40, 50],
                          "type": ["T4a", "T5d", "T4a", "T5a", "other"],
                          "somaSide": ["L", "R", "L", "R", "L"]}).to_parquet(Path(directory)/"metadata.parquet")
            graph = SimpleNamespace(ids=np.array([10, 20, 30, 50]), signs=np.array([1, 1, 0, 1]))
            relay = VisualInput(graph, directory)
            np.testing.assert_array_equal(relay.body_ids, [10, 20])
            indices, currents = relay.stimulus(np.ones((2, 9)))
            np.testing.assert_array_equal(indices, [0, 1])
            self.assertTrue(np.all(currents > 1))
            self.assertTrue(np.all(relay.stimulus(np.zeros((2, 9)))[1] == 0))

    def test_visual_policy_migration_preserves_values_and_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"memory.json"
            old = np.arange(256).reshape(32, 8)
            path.write_text(json.dumps({"version": 2, "dataset": "male-cns:v1.0", "q": old.tolist(),
                                        "updates": 1234, "lifetime_reward": 42.5}))
            learner = ExplorationLearner(path)
            np.testing.assert_array_equal(learner.q[:32], old)
            self.assertTrue(np.all(learner.q[32:] == 0))
            state = learner.state(np.zeros(6), .5, clearances=[12, 2, 12])
            self.assertEqual(state, 64)
            learner.save()
            loaded = ExplorationLearner(path)
            np.testing.assert_array_equal(loaded.q, learner.q)
            self.assertEqual(loaded.updates, 1234)
            self.assertEqual(loaded.lifetime_reward, 42.5)


class BrainViewTests(unittest.TestCase):
    def make_map(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        pd.DataFrame({"bodyId": [513052, 815344, 123],
                      "instance": ["DNg15_R", "Ti extensor MN_L", "unknown"],
                      "type": ["DNg15", "Ti extensor", None],
                      "superclass": ["descending_neuron", "vnc_motor", None],
                      "somaLocation": [[0., 0., 0.], None, None],
                      "tosomaLocation": [None, [1., 1., 1.], None]}).to_parquet(Path(directory.name)/"metadata.parquet")
        return BrainMap(np.array([513052, 815344, 123, 999]), directory.name)

    def test_missing_coordinates_are_omitted_and_picking_respects_active_filter(self):
        brain = self.make_map()
        np.testing.assert_array_equal(brain.indices, [0, 1])
        brain.points = np.array([[0, 0, 0], [.1, 0, 0]], dtype="f4")
        self.assertEqual(brain.pick(50, 50, np.eye(4), (0, 0, 100, 100)), 0)
        self.assertEqual(brain.details([7, 0, 0, 0])["bodyId"], 513052)
        self.assertEqual(brain.details([7, 0, 0, 0])["spikes"], 7)
        brain.selected = None
        self.assertEqual(brain.pick(50, 50, np.eye(4), (0, 0, 100, 100), np.array([0, 1, 0, 0])), 1)
        self.assertIsNone(brain.pick(95, 95, np.eye(4), (0, 0, 100, 100)))

    def test_point_renderer_displays_activity_and_hides_silent_cells(self):
        try:
            ctx = moderngl.create_standalone_context(require=330, backend="egl")
        except Exception as error:
            self.skipTest(f"EGL context unavailable: {error}")
        self.addCleanup(ctx.release)
        fbo = ctx.simple_framebuffer((128, 128))
        self.addCleanup(fbo.release)
        fbo.use()
        brain = self.make_map()
        brain.points = np.array([[-.5, 0, 0], [.5, 0, 0]], dtype="f4")
        renderer = BrainRenderer(ctx, brain)
        self.addCleanup(renderer.release)
        ctx.enable(moderngl.BLEND)
        camera = {"yaw": 0, "pitch": .3, "distance": 4, "active_only": True}
        for active in (False, True):
            fbo.clear(0, 0, 0, 0)
            renderer.draw({"brain_activity": np.array([0, int(active), 0, 0])}, (0, 0, 128, 128), camera)
            pixels = np.frombuffer(fbo.read(), dtype=np.uint8)
            self.assertEqual(bool(pixels.any()), active)
            self.assertEqual(ctx.error, "GL_NO_ERROR")


if __name__ == "__main__":
    unittest.main()

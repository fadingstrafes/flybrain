import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import numpy as np
from src.minecraft_brain import MinecraftSession, validate_observation
from src.minecraft_learning import MinecraftLearner, FEATURES
from src.minecraft_service import make_server


class FakeBrain:
    def __init__(self): self.resets = 0
    def reset(self): self.resets += 1
    def encode(self, senses): return np.r_[1., np.tile(senses,2)]
    def snapshot(self): return {}


def observation(seq=0, life=0, dead=False, health=20, food=10):
    return dict(version=1, session='test', seq=seq, life=life, dead=dead,
                health=health, food=food, senses=[.5]*32)


class MinecraftTests(unittest.TestCase):
    def test_body_estimates_track_injury_and_reset(self):
        from src.minecraft_telemetry import BodyTelemetry
        body = BodyTelemetry()
        body.update(observation(health=20), 'idle')
        injured = body.update(observation(health=14), 'idle')
        self.assertAlmostEqual(injured['pain_proxy'], .6)
        self.assertEqual(injured['body_state'], 'Recent injury')
        rested = body.update(observation(health=14), 'idle')
        self.assertLess(rested['pain_proxy'], injured['pain_proxy'])
        body.reset()
        self.assertEqual(body.update(observation(health=20), 'idle')['pain_proxy'],0)

    def test_death_checkpoint_reload_and_respawn(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'policy.json'
            learner = MinecraftLearner(path)
            brain = FakeBrain()
            session = MinecraftSession(brain, learner)
            session.step(observation())
            session.step(observation(1, food=16))
            # Carpet restores hunger during its death handler; this must not
            # award food reward to the action that killed the agent.
            death = observation(2, dead=True, health=0, food=20)
            response = session.step(death)
            self.assertTrue(response['respawn'])
            self.assertEqual(session.status['reward'], -5.)
            updates = learner.updates
            self.assertEqual(session.step(death), response)
            session.step(observation(3, dead=True, health=0))
            self.assertEqual(learner.updates, updates)
            self.assertEqual(learner.deaths, 1)
            restored = MinecraftLearner(path)
            np.testing.assert_array_equal(restored.weights, learner.weights)
            self.assertEqual(len(restored.replay), 2)
            self.assertTrue(restored.replay[-1][-1])
            session.step(observation(4, life=1))
            self.assertEqual(learner.updates, updates)
            self.assertEqual(brain.resets, 2)
            self.assertFalse(learner.traces.any())

    def test_terminal_target_and_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'policy.json'
            learner = MinecraftLearner(path)
            learner.weights.fill(1.)
            x = np.zeros(FEATURES); x[0] = 1
            learner.learn(x, 0, -2, x*100, True)
            self.assertLess(learner.weights[0,0], 1.)
            original = path.read_bytes()
            frozen = MinecraftLearner(path, training=False)
            frozen.learn(x, 0, 10, x, True)
            frozen.save()
            self.assertEqual(original, path.read_bytes())

    def test_order_validation_and_no_cross_session_credit(self):
        with tempfile.TemporaryDirectory() as d:
            learner = MinecraftLearner(Path(d)/'policy.json')
            session = MinecraftSession(FakeBrain(), learner)
            session.step(observation(2))
            with self.assertRaises(ValueError): session.step(observation(1))
            new = observation(); new['session'] = 'new'
            session.step(new)
            self.assertEqual(learner.updates, 0)
            with self.assertRaises(ValueError): session.step(observation(3))
            for value in (float('nan'), -1, 2):
                bad = observation(); bad['senses'][0] = value
                with self.assertRaises(ValueError): validate_observation(bad)

    def test_http_loop(self):
        with tempfile.TemporaryDirectory() as d:
            session = MinecraftSession(FakeBrain(), MinecraftLearner(Path(d)/'p.json'))
            server = make_server(session,0)
            thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
            try:
                request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/step',
                    data=json.dumps(observation()).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(json.load(response)['seq'],0)
            finally:
                server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__': unittest.main()

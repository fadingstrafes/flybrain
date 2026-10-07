"""Small persistent Q-learning policy over sensory stimulation patterns.

This is an engineered controller around the connectome, not a claim of biological
synaptic plasticity. It never edits the anatomical graph or commands body velocity.
"""

import json
from pathlib import Path

import numpy as np


class ExplorationLearner:
    names = ("balanced", "left bias", "right bias", "front legs", "hind legs", "rest", "takeoff", "land")
    patterns = np.array([
        [1, 1, 1, 1, 1, 1],
        [1.8, 1.4, 1, .15, .3, .4],
        [.15, .3, .4, 1.8, 1.4, 1],
        [1.8, .3, .2, 1.8, .3, .2],
        [.2, .3, 1.8, .2, .3, 1.8],
        [0, 0, 0, 0, 0, 0],
        [.3, .3, .3, .3, .3, .3],
        [1, 1, 1, 1, 1, 1],
    ], dtype=np.float32)

    def __init__(self, path=None, seed=42, training=True):
        self.path = Path(path) if path is not None else None
        self.rng = np.random.default_rng(seed)
        self.training = training
        self.q = np.zeros((256, len(self.names)), dtype=np.float64)
        self.updates = 0
        self.lifetime_reward = 0.0
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text())
            q = np.asarray(data["q"], dtype=np.float64)
            if data.get("version") == 1 and q.shape == (16, 6):
                # Preserve already-learned walking values when adding flight.
                migrated = np.zeros_like(self.q)
                migrated[:16, :6] = q
                q = migrated
            elif data.get("version") == 2 and q.shape == (32, 8):
                migrated = np.zeros_like(self.q)
                migrated[:32] = q
                q = migrated
            if data.get("version") not in (1, 2, 3) or data.get("dataset") != "male-cns:v1.0" or q.shape != self.q.shape or not np.isfinite(q).all():
                raise ValueError(f"Incompatible learning policy: {self.path}")
            self.q = q
            self.updates = int(data["updates"])
            self.lifetime_reward = float(data["lifetime_reward"])
        self.reset_episode()

    @property
    def epsilon(self):
        return max(.08, .35 * np.exp(-self.updates / 2500)) if self.training else 0.0

    def reset_episode(self):
        self.visited = set()
        self.last_state = None
        self.action = 0
        self.last_position = None
        self.last_contacts = 0
        self.last_distance = 0.0
        self.last_reward = 0.0
        self.episode_reward = 0.0
        self.new_tiles = 0

    @staticmethod
    def state(touch, speed, altitude=0, clearances=None):
        visual = 0
        if clearances is not None:
            visual = sum(int(distance < 4.5) << i for i, distance in enumerate(clearances))
        return (int(np.max(touch[:3]) > .25)
                | (int(np.max(touch[3:]) > .25) << 1)
                | (int(max(touch[0], touch[3]) > .5) << 2)
                | (int(speed < .2) << 3) | (int(altitude > .15) << 4) | (visual << 5))

    def visit(self, position):
        tile = tuple(np.floor(np.asarray(position) / .75).astype(int))
        if tile not in self.visited:
            self.visited.add(tile)
            self.new_tiles += 1

    def decide(self, touch, pose):
        state = self.state(touch, pose["speed"], pose.get("altitude", 0), pose.get("clearances"))
        if self.last_state is not None:
            displacement = float(np.linalg.norm(pose["position"] - self.last_position))
            collisions = pose["contacts"] - self.last_contacts
            distance = pose["distance"] - self.last_distance
            reward = self.new_tiles + .08 * distance - .025 * collisions - (.2 if displacement < .05 else 0)
            if pose.get("clearances") is not None:
                reward -= .15*max(0, 1-float(pose["clearances"][1])/3)
            self.last_reward = reward
            self.episode_reward += reward
            if self.training:
                target = reward + .90 * float(self.q[state].max())
                self.q[self.last_state, self.action] += .15 * (target - self.q[self.last_state, self.action])
                self.updates += 1
                self.lifetime_reward += reward
        if self.rng.random() < self.epsilon:
            self.action = int(self.rng.integers(len(self.names)))
        else:
            best = np.flatnonzero(self.q[state] == self.q[state].max())
            self.action = int(self.rng.choice(best))
        self.last_state = state
        self.last_position = pose["position"].copy()
        self.last_contacts = pose["contacts"]
        self.last_distance = pose["distance"]
        self.new_tiles = 0
        if self.training and self.path is not None and self.updates and self.updates % 100 == 0:
            self.save()
        return self.patterns[self.action]

    def save(self, path=None):
        path = Path(path) if path is not None else self.path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 3, "dataset": "male-cns:v1.0", "q": self.q.tolist(),
                "updates": self.updates, "lifetime_reward": self.lifetime_reward,
                "actions": list(self.names),
                "reward": "new .75-unit tiles + .08*distance - .025*contact_steps - .2 when stuck - .15*max(0,1-front_clearance/3)",
                "state": "Contact, low speed, airborne and three visual proximity bits",
                "controller": "Sensory-pattern Q-learning; anatomical weights stay fixed"}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(path)

    def summary(self):
        return {"action": self.names[self.action], "updates": self.updates,
                "reward": self.episode_reward, "last_reward": self.last_reward,
                "tiles": len(self.visited), "epsilon": self.epsilon, "training": self.training}

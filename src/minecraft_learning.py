"""Experimental linear Q readout with eligibility traces and bounded experience replay.

Learns around a fixed connectome, not physiological synaptic plasticity.
"""
from collections import deque
import json
from pathlib import Path
import numpy as np

ACTIONS = ("idle", "forward", "back", "left", "right", "jump", "forward_jump",
           "sneak", "look_left", "look_right", "look_up", "look_down", "attack", "use",
           *(f"hotbar_{i}" for i in range(9)), "inventory", "slot_next", "slot_previous",
           "slot_click", "slot_right_click", "slot_quick_move",
           "takeoff", "land", "ascend", "descend", "crawl", "stand")
NEURAL_FEATURES = 65
FEATURES = 152  # 114 original + 6 foraging/shelter + 32 photoreceptor spike pools


def allowed_actions(x):
    mask=np.ones(len(ACTIONS),dtype=bool)
    inventory=x[65+9]>.5
    for i,name in enumerate(ACTIONS):
        if name.startswith('slot_') and not inventory: mask[i]=False
        if name in ('attack','use') and inventory: mask[i]=False
        if name=='takeoff' and x[65+30]>.5: mask[i]=False
        if name in ('land','ascend','descend') and x[65+30]<=.5: mask[i]=False
        if name=='crawl' and x[65+31]>.5: mask[i]=False
        if name=='stand' and x[65+31]<=.5: mask[i]=False
        if name.startswith('hotbar_'):
            slot=int(name[7:])
            if slot==round(x[65+8]*8) or (x[104]>.5 and x[105+slot]==0): mask[i]=False
    return mask


class MinecraftLearner:
    def __init__(self, path, seed=42, training=True):
        self.path = Path(path)
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((len(ACTIONS), FEATURES), dtype=np.float64)
        self.traces = np.zeros_like(self.weights)
        self.replay = deque(maxlen=512)
        self.updates = self.deaths = self.transitions = 0
        self.total_reward = 0.0
        if self.path.exists():
            data = json.loads(self.path.read_text())
            weights = np.asarray(data['weights'], dtype=float)
            if (data.get('version') not in (1,2,3) or data.get('actions') != list(ACTIONS)
                    or weights.shape not in (self.weights.shape,(len(ACTIONS),NEURAL_FEATURES),(len(ACTIONS),114)) or not np.isfinite(weights).all()):
                raise ValueError('Incompatible Minecraft checkpoint')
            self.weights[:,:weights.shape[1]] = weights
            if data.get('version') in (1,2) and training:
                backup=self.path.with_suffix(f'.v{data["version"]}.backup.json')
                if not backup.exists():
                    with backup.open('x') as stream: stream.write(self.path.read_text())
            for key in ('updates', 'deaths', 'transitions', 'total_reward'):
                setattr(self, key, data[key])
            self.rng.bit_generator.state = data['rng']
            for item in data.get('replay', []):
                x, action, reward, y, terminal = item
                x, y = self._features(x), self._features(y)
                if not 0 <= action < len(ACTIONS) or not np.isfinite(reward):
                    raise ValueError('Invalid replay record')
                self.replay.append((x, action, reward, y, bool(terminal)))

    @staticmethod
    def _features(value):
        value = np.asarray(value, dtype=float)
        if value.shape in ((NEURAL_FEATURES,),(114,)):
            value=np.pad(value,(0,FEATURES-len(value)))
        if value.shape != (FEATURES,) or not np.isfinite(value).all():
            raise ValueError('Invalid neural features')
        return value

    def choose(self, features):
        x = self._features(features)
        valid=np.flatnonzero(allowed_actions(x))
        epsilon = max(.12, .5 * np.exp(-self.transitions / 10000)) if self.training else 0
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        q = self.weights @ x
        return int(self.rng.choice(valid[np.isclose(q[valid],q[valid].max())]))

    def learn(self, x, action, reward, y, terminal=False):
        if not self.training:
            return
        x, y = self._features(x), self._features(y)
        self.traces *= .97 * .8
        self.traces[action] += x
        error = reward + (0 if terminal else .97 * (self.weights @ y)[allowed_actions(y)].max()) - self.weights[action] @ x
        self.weights += .025 * np.clip(error, -5, 5) * self.traces / max(1., np.sum(x*x))
        np.clip(self.weights, -20, 20, out=self.weights)
        self.replay.append((x.copy(), action, float(reward), y.copy(), bool(terminal)))
        self.transitions += 1
        self.updates += 1
        self.total_reward += reward
        if terminal:
            self.deaths += 1
            self.reset_episode()
            # Rehearse real transitions; terminal targets never bootstrap a next life.
            for _ in range(min(64, len(self.replay))):
                sx, a, r, sy, done = self.replay[int(self.rng.integers(len(self.replay)))]
                delta = r + (0 if done else .97 * (self.weights @ sy)[allowed_actions(sy)].max()) - self.weights[a] @ sx
                self.weights[a] += .01 * np.clip(delta, -5, 5) * sx / max(1., sx@sx)
                self.updates += 1
            np.clip(self.weights, -20, 20, out=self.weights)
        if terminal or self.transitions % 100 == 0:
            self.save()

    def reset_episode(self):
        self.traces.fill(0)

    def save(self):
        if not self.training:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(version=3, actions=list(ACTIONS), weights=self.weights.tolist(),
                    updates=self.updates, deaths=self.deaths, transitions=self.transitions,
                    total_reward=self.total_reward, rng=self.rng.bit_generator.state,
                    replay=[(x.tolist(), a, r, y.tolist(), done) for x,a,r,y,done in self.replay],
                    model='Fixed connectome with artificial learned action readout')
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, allow_nan=False))
        temporary.replace(self.path)

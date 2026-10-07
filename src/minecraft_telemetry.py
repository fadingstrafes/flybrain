"""Observer-only game-state estimates; never fed back as rewards or neural input."""
import numpy as np


class BodyTelemetry:
    def __init__(self):
        self.reset()

    def reset(self):
        self.health = None
        self.pain = 0.
        self.last_action = None
        self.repeat = 0

    def update(self, observation, action):
        health = observation['health']
        injury = 0. if self.health is None else max(0., self.health-health)
        self.health = health
        # Explicit engineered scores, decaying in observations, not real seconds.
        self.pain = max(self.pain*.85, min(1., injury/10))
        senses = observation['senses']
        hunger = float(np.clip(1-observation['food']/20, 0, 1))
        stress = float(np.clip(.5*self.pain + .3*max(0,1-health/20)
                               + .15*senses[5] + .05*senses[15], 0, 1))
        state = ('Dead' if observation['dead'] else 'Recent injury' if self.pain>.1
                 else 'On fire' if senses[5] else 'Hungry' if hunger>.35
                 else 'Collision' if senses[15] else 'No acute body signal')
        self.repeat = self.repeat+1 if action == self.last_action else 1
        self.last_action = action
        return dict(body_state=state, pain_proxy=self.pain, stress_proxy=stress,
                    hunger=hunger, injury=injury, action_repeat=self.repeat,
                    inventory_open=bool(senses[9]), flying=bool(senses[30]),
                    crawling=bool(senses[31]), depth_rays=list(senses[16:25]))

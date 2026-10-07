import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.minecraft_drives import SurvivalDrives, state_features
from src.minecraft_learning import MinecraftLearner, allowed_actions, ACTIONS, FEATURES
from src.minecraft_brain import validate_observation
from tests.test_minecraft import observation


def sample(ticks=0, **kwargs):
    result=observation(food=10)
    world=dict(food_stock=0,materials=0,placed=0,broken=0,ticks=ticks,threat=0,
               cover=0,danger=0,position=[0,0,0],hotbar=[0]*9)
    world.update(kwargs)
    result['world']=world
    return result


class DriveTests(unittest.TestCase):
    def test_acquisition_and_shelter_cannot_repeat_by_drop_pickup(self):
        drives=SurvivalDrives(); a=sample(); drives.baseline(a)
        b=sample(20,food_stock=3,materials=4,cover=.8,placed=2,danger=1)
        reward=drives.transition(a,b)
        for key in ('food_acquired','gathering','shelter','shelter_building'):
            self.assertGreater(reward[key],0)
        c=sample(40)
        drives.transition(b,c)
        b['world']['ticks']=60
        repeated=drives.transition(c,b)
        for key in ('food_acquired','gathering','shelter','shelter_building'):
            self.assertEqual(repeated[key],0)

    def test_short_rest_free_long_idle_penalized_recovery_exempt(self):
        drives=SurvivalDrives(); old=sample(); drives.baseline(old)
        for tick in range(20,201,20):
            new=sample(tick); result=drives.transition(old,new); old=new
            self.assertEqual(result['inactivity'],0)
        new=sample(220); result=drives.transition(old,new)
        self.assertLess(result['inactivity'],0)
        old=new; new=sample(240); new.update(health=12,food=20)
        self.assertEqual(drives.transition(old,new)['inactivity'],0)

    def test_invalid_actions_masked_and_world_validation(self):
        o=sample();o['senses']=[0]*32
        x=np.r_[state_features(np.zeros(65),o),np.zeros(32)]
        self.assertEqual(len(x),FEATURES)
        mask=allowed_actions(x)
        for name in ('hotbar_4','slot_click','ascend','stand'):
            self.assertFalse(mask[ACTIONS.index(name)])
        self.assertTrue(mask[ACTIONS.index('forward')])
        o['world']['threat']=float('nan')
        with self.assertRaises(ValueError): validate_observation(o)

    def test_old_checkpoint_keeps_weights_and_backup(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'policy.json'
            learner=MinecraftLearner(path); learner.weights.fill(.25); learner.save()
            old=json.loads(path.read_text());old['version']=1
            old['weights']=[row[:65] for row in old['weights']]
            path.write_text(json.dumps(old)); original=path.read_bytes()
            restored=MinecraftLearner(path)
            np.testing.assert_array_equal(restored.weights[:,:65],.25)
            self.assertFalse(restored.weights[:,65:].any())
            self.assertEqual(path.with_suffix('.v1.backup.json').read_bytes(),original)
            restored.save()
            self.assertEqual(json.loads(path.read_text())['version'],3)

if __name__=='__main__': unittest.main()

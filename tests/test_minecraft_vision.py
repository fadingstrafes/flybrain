import base64
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.minecraft_vision import CompoundEyes,decode_vision


class VisionTests(unittest.TestCase):
    def test_image_input_targets_annotated_eye_and_changes_with_image(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame(dict(bodyId=[1,2,3,4,5,6],type=['R1-R6','R7p','R8p']*2,
                instance=['R1-R6_L','R7p_L','R8p_L','R1-R6_R','R7p_R','R8p_R'])).to_parquet(Path(d)/'metadata.parquet')
            eyes=CompoundEyes(np.arange(1,7),Path(d))
            rgb=np.zeros((2,64,96,3),dtype=np.uint8);rgb[0]=255
            eyes.update(rgb,1)
            self.assertTrue(np.all(eyes.currents[eyes.eyes==0]>0))
            self.assertFalse(eyes.currents[eyes.eyes==1].any())
            rgb[1]=255;eyes.update(rgb,2)
            self.assertTrue(eyes.motion[1].any())
            self.assertTrue(np.all(eyes.currents[eyes.eyes==1]>1.05))
            eyes.update(rgb,2)
            self.assertFalse(eyes.motion.any())
            eyes.update(None,None)
            self.assertFalse(eyes.currents.any())
            self.assertIsNone(eyes.rgb)

    def test_rgb_protocol_roundtrip_and_rejects_bad_payload(self):
        rgb=np.arange(2*64*96*3,dtype=np.uint8).reshape(2,64,96,3)
        packet=dict(width=96,height=64,frame=7,rgb=base64.b64encode(rgb.tobytes()).decode())
        np.testing.assert_array_equal(decode_vision(packet),rgb)
        for change in ({'width':95},{'rgb':'bad'},{'frame':-1}):
            with self.assertRaises(ValueError):decode_vision(dict(packet,**change))

if __name__=='__main__':unittest.main()

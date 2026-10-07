from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from PIL import Image

from src.brain_media import BrainCanvas, MediaPlayer
import test_vision
from src.arena_session import ArenaSession
from src.live_brain import project_graph


class CanvasTests(unittest.TestCase):
    def test_paint_erase_undo_and_save(self):
        canvas = BrainCanvas((100, 100))
        canvas.radius = 3
        canvas.begin((.2, .5))
        canvas.stroke((.8, .5))
        canvas.end()
        self.assertEqual(canvas.image.getpixel((50, 50)), canvas.color)
        canvas.begin((.5, .5), erase=True)
        self.assertEqual(canvas.image.getpixel((50, 50))[3], 0)
        canvas.undo()
        self.assertEqual(canvas.image.getpixel((50, 50)), canvas.color)
        canvas.clear()
        self.assertIsNone(canvas.image.getbbox())
        canvas.undo()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"drawing.png"
            canvas.save(path)
            with Image.open(path) as loaded:
                self.assertEqual(loaded.getpixel((50,50)), canvas.color)

    def test_video_letterboxing_preserves_aspect_and_ink_is_on_top(self):
        canvas = BrainCanvas((100,100))
        frame = np.full((50,100,3),255,dtype=np.uint8)
        composed = canvas.compose(frame, viewport_aspect=1)
        self.assertEqual(composed.getpixel((50,10))[3],0)
        self.assertEqual(composed.getpixel((50,50)),(255,255,255,255))
        canvas.begin((.5,.5))
        self.assertEqual(canvas.compose(frame,1).getpixel((50,50)),canvas.color)


class FiberTests(unittest.TestCase):
    def test_raw_skeletons_share_soma_transform_and_are_deduplicated(self):
        fixture = test_vision.BrainViewTests()
        self.addCleanup(fixture.doCleanups)
        brain = fixture.make_map()
        with tempfile.TemporaryDirectory() as folder:
            raw = np.array([[[0,0,0],[1,1,1]]], dtype="f4")
            np.savez(Path(folder)/"513052.npz",segments=raw)
            fibers = brain.load_fibers([folder,folder])
        self.assertEqual(len(fibers),1)
        np.testing.assert_allclose(fibers[0][1],brain.points)

    def test_image_hits_map_to_neurons_and_propagate_to_a_motor(self):
        fixture = test_vision.BrainViewTests()
        self.addCleanup(fixture.doCleanups)
        brain = fixture.make_map()
        brain.points = np.array([[-.5,0,0],[.5,0,0]],dtype="f4")
        canvas = BrainCanvas((100,100))
        canvas.color = (255,255,255,255)
        canvas.begin((.25,.5))
        indices,levels = brain.stimulus_from_image(canvas.image,np.eye(4),(0,0,100,100))
        np.testing.assert_array_equal(indices,[0])
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            pd.DataFrame(columns=["bodyId","side","system","subclass"]).to_csv(cache/"sensory_map.csv",index=False)
            pd.DataFrame([dict(bodyId=815344,side="L",leg_pair="front",limb="leg",action="extend")]).to_csv(cache/"motor_map.csv",index=False)
            graph = project_graph(csr_matrix([[0,8],[0,0]]),np.array([513052,815344]),np.array([1,0],dtype="f4"),[815344])
            session = ArenaSession(graph,device="cpu",cache=cache,learning=False,vision=False,exploration=False)
            session.brain.abort_fraction = 0
            session.set_art_stimulus(indices,levels)
            for _ in range(8): session.tick()
            self.assertEqual(int(session.counts.sum()),0)
            session.art_enabled = True
            session.art_limit = 1
            for _ in range(8): session.tick()
            self.assertEqual(session.counts[0],8)
            self.assertGreater(session.counts[1],0)  # Driven through W.T, not painted directly.
            self.assertGreater(session.bridge.motor_spikes,0)
            np.testing.assert_array_equal(session.art_target_counts,[8,0])
            session.set_art_stimulus([0,1],[1,1])
            first,_ = session.art_stimulus()
            second,_ = session.art_stimulus()
            self.assertEqual(len(first),1)
            self.assertNotEqual(first[0],second[0])
            session.reset()
            self.assertEqual(len(session.art_indices),0)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"ffmpeg unavailable")
class PlaybackTests(unittest.TestCase):
    def test_decode_pause_resume_and_release_local_video(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"pattern.mkv"
            subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","testsrc2=size=64x48:rate=30",
                            "-t","2","-c:v","ffv1",str(path)],check=True)
            player = MediaPlayer(muted=True)
            self.addCleanup(player.close)
            player.open(path)
            deadline = time.monotonic()+8
            while player.frame is None and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertIsNotNone(player.frame,player.status)
            self.assertEqual(player.frame.shape,(480,640,3))
            player.toggle_pause()
            time.sleep(.08)
            version = player.version
            time.sleep(.1)
            self.assertEqual(player.version,version)
            player.toggle_pause()
            deadline = time.monotonic()+2
            while player.version == version and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertGreater(player.version,version)
            player.close()
            self.assertIsNone(player.decoder)
            self.assertIsNone(player.thread)

    def test_audio_only_file_produces_spectrogram(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"tone.wav"
            subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","sine=frequency=440:duration=0.3",str(path)],check=True)
            player = MediaPlayer(muted=True)
            self.addCleanup(player.close)
            player.open(path)
            player.thread.join(timeout=8)
            self.assertIsNotNone(player.frame,player.status)
            self.assertGreater(int(player.frame.max()),0)
            self.assertTrue(player.status.startswith("Finished"),player.status)


if __name__ == "__main__":
    unittest.main()

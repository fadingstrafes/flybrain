from pathlib import Path
import tempfile
import unittest

import moderngl
import numpy as np
import pandas as pd

from src.fly_brain_opengl import (
    build_trailing_activity, create_skeleton_batch, load_skeletons, motor_group, write_mat4,
)


class ViewerDataTests(unittest.TestCase):
    def test_missing_skeleton_does_not_consume_loading_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cache = run / "skeleton_cache"
            cache.mkdir()
            pd.DataFrame({"bodyId": [513052, 815344]}).to_csv(cache / "selected.csv", index=False)
            np.savez_compressed(cache / "815344.npz", segments=np.array([[[0, 0, 0], [1, 1, 1]]]))
            skeletons, count = load_skeletons(run, 1)
            self.assertEqual(skeletons[0][0], 815344)
            self.assertEqual(count, 1)

    def test_trailing_window_expires_and_ignores_unselected_neurons(self):
        activity = build_trailing_activity(
            np.array([1, 1, 3]), np.array([513052, 815344, 513052]), [513052], 7, 1
        )
        self.assertEqual(activity[:, 0].tolist(), [0, 1, 1, 1, 1, 0, 0])

    def test_motor_map_limb_fallback_handles_nan_system(self):
        self.assertEqual(motor_group({"system": np.nan, "limb": "leg", "side": "L",
                                      "leg_pair": "front"}), "L front")


class ShaderTests(unittest.TestCase):
    def test_batched_activity_colors_and_hiding(self):
        try:
            ctx = moderngl.create_standalone_context(require=330, backend="egl")
        except Exception as error:
            self.skipTest(f"EGL context unavailable: {error}")
        self.addCleanup(ctx.release)
        fbo = ctx.simple_framebuffer((64, 64))
        self.addCleanup(fbo.release)
        fbo.use()
        skeletons = [
            (513052, np.array([[-0.8, -0.5, 0], [0.8, -0.5, 0]], dtype="f4")),
            (815344, np.array([[-0.8, 0.5, 0], [0.8, 0.5, 0]], dtype="f4")),
        ]
        program, vbo, vao, texture = create_skeleton_batch(ctx, skeletons)
        for resource in (program, vbo, vao, texture):
            self.addCleanup(resource.release)
        write_mat4(program["mvp"], np.eye(4, dtype="f4"))
        texture.write(np.array([0, 4], dtype="f4").tobytes())
        texture.use(location=0)
        for hide in (False, True):
            fbo.clear(0, 0, 0, 0)
            program["hide_inactive"].value = hide
            vao.render(mode=moderngl.LINES)
            pixels = np.frombuffer(fbo.read(components=4), dtype=np.uint8).reshape(64, 64, 4)
            self.assertEqual(int(pixels[40:55, :, 0].max()), 255)
            self.assertEqual(bool(pixels[8:24].any()), not hide)
            self.assertEqual(ctx.error, "GL_NO_ERROR")


if __name__ == "__main__":
    unittest.main()

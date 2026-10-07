"""Local media playback and screen-space painting for the CNS presentation layer."""

import json
from contextlib import suppress
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

import numpy as np
import moderngl
from PIL import Image, ImageDraw


class BrainCanvas:
    def __init__(self, size=(960, 720)):
        self.image = Image.new("RGBA", size)
        self.dirty = True
        self.undo_images = []
        self.previous = None
        self.radius = 14
        self.color = (255, 110, 195, 255)

    def begin(self, uv, erase=False):
        self.undo_images.append(self.image.copy())
        self.undo_images = self.undo_images[-12:]
        self.previous = None
        self.stroke(uv, erase)

    def stroke(self, uv, erase=False):
        point = (int(uv[0]*self.image.width), int(uv[1]*self.image.height))
        draw = ImageDraw.Draw(self.image)
        color = (0, 0, 0, 0) if erase else self.color
        if self.previous is not None:
            draw.line([self.previous, point], fill=color, width=self.radius*2)
        x, y = point
        draw.ellipse((x-self.radius, y-self.radius, x+self.radius, y+self.radius), fill=color)
        self.previous = point
        self.dirty = True

    def end(self):
        self.previous = None

    def clear(self):
        self.undo_images.append(self.image.copy())
        self.undo_images = self.undo_images[-12:]
        self.image.paste((0, 0, 0, 0), (0, 0, *self.image.size))
        self.dirty = True

    def undo(self):
        if self.undo_images:
            self.image = self.undo_images.pop()
            self.dirty = True

    def compose(self, frame=None, viewport_aspect=4/3):
        # Letterbox media for the actual viewport; preserve the video's aspect.
        image = Image.new("RGBA", self.image.size)
        if frame is not None:
            media = Image.fromarray(frame).convert("RGBA")
            ratio = (media.width/media.height)/viewport_aspect
            w = self.image.width if ratio >= 1 else round(self.image.width*ratio)
            h = round(self.image.height/ratio) if ratio >= 1 else self.image.height
            media = media.resize((max(1, w), max(1, h)), Image.Resampling.BILINEAR)
            image.paste(media, ((image.width-w)//2, (image.height-h)//2))
        image.alpha_composite(self.image)
        return image

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path)

    def load(self, path):
        with Image.open(path) as source:
            self.image = source.convert("RGBA").resize(self.image.size, Image.Resampling.LANCZOS)
        self.dirty = True


class MediaPlayer:
    """Bounded RGB decoding off the render thread, with optional ffplay audio."""
    width, height, fps = 640, 360, 30

    def __init__(self, muted=False):
        self.muted = muted
        self.frame = None
        self.version = 0
        self.status = "Drop a local video or song here / O open"
        self.path = None
        self.paused = False
        self.playing = False
        self.thread = None

        self.decoder = self.audio = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()

    def open(self, path):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ValueError("Choose a local audio or video file")
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("Playback requires ffmpeg and ffprobe")
        self.close()
        self.path = path
        self.frame = None
        self.version += 1
        self.paused = False
        self.playing = True
        self.status = "Loading " + path.name
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._decode, name="CNS media", daemon=True)
        self.thread.start()

    def _decode(self):
        decoder = audio = None
        try:
            result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(self.path)],
                                    capture_output=True, timeout=10, check=True)
            streams = json.loads(result.stdout)["streams"]
            video = next((s for s in streams if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")), None)
            has_audio = any(s["codec_type"] == "audio" for s in streams)
            if video is None and not has_audio:
                raise ValueError("No playable audio or video stream")
            if self.stop_event.is_set():
                return
            if video is not None:
                width, height = int(video["width"]), int(video["height"])
                sar = video.get("sample_aspect_ratio", "1:1")
                if sar not in ("N/A", "0:1"):
                    a, b = map(int, sar.split(":"))
                    width = width*a/max(1, b)
                scale = min(640/max(1,width),720/max(1,height))
                self.width, self.height = max(2,round(width*scale)),max(2,round(height*scale))
                filters = ["-map", f"0:{video['index']}", "-vf", f"fps={self.fps},scale={self.width}:{self.height},setsar=1"]
            else:
                self.width = 640
                self.height = 360
                filters = ["-filter_complex", f"[0:a:0]showspectrum=s={self.width}x{self.height}:slide=scroll:mode=combined:color=intensity:scale=log:fscale=log:fps={self.fps},format=rgb24"]
            decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", str(self.path), *filters,
                                        "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            with self.lock:
                self.decoder = decoder
            frame_bytes = self.width*self.height*3
            started = None
            frame_number = 0
            while not self.stop_event.is_set():
                data = decoder.stdout.read(frame_bytes)
                if len(data) != frame_bytes:
                    break
                if started is None:
                    started = time.monotonic()
                    if has_audio and not self.muted and shutil.which("ffplay"):
                        audio = subprocess.Popen(["ffplay", "-v", "error", "-nodisp", "-autoexit", "-vn", str(self.path)],
                                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        with self.lock:
                            self.audio = audio
                            if self.paused:
                                audio.send_signal(signal.SIGSTOP)
                while not self.stop_event.is_set():
                    if self.paused:
                        pause_start = time.monotonic()
                        self.stop_event.wait(.02)
                        started += time.monotonic()-pause_start
                    else:
                        remaining = started+frame_number/self.fps-time.monotonic()
                        if remaining <= 0:
                            break
                        self.stop_event.wait(min(.02, remaining))
                if self.stop_event.is_set():
                    break
                self.frame = np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3).copy()
                self.version += 1
                self.status = f"{self.path.name}  ·  {frame_number/self.fps:.1f}s"
                frame_number += 1
            if not self.stop_event.is_set():
                code = decoder.wait(timeout=5)
                self.status = ("Finished · J replay · " if code == 0 else "Decode failed · ") + self.path.name
        except Exception as error:
            if not self.stop_event.is_set():
                self.status = f"Media error: {type(error).__name__}: {error}"[:180]
        finally:
            self.playing = False
            for process in (decoder, audio):
                self._terminate(process)
            with self.lock:
                self.decoder = self.audio = None

    @staticmethod
    def _terminate(process):
        if process is not None:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    process.send_signal(signal.SIGCONT)
                    process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout is not None:
                process.stdout.close()

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused
            if self.audio is not None and self.audio.poll() is None:
                with suppress(ProcessLookupError):
                    self.audio.send_signal(signal.SIGSTOP if self.paused else signal.SIGCONT)

    def close(self):
        self.stop_event.set()
        with self.lock:
            for process in (self.decoder, self.audio):
                if process is not None and process.poll() is None:
                    with suppress(ProcessLookupError):
                        process.send_signal(signal.SIGCONT)
                        process.terminate()
        if self.thread is not None:
            self.thread.join(timeout=15)
        self.thread = None


class ArtworkRenderer:
    def __init__(self, ctx, canvas):
        self.ctx, self.canvas = ctx, canvas
        self.texture = ctx.texture(canvas.image.size, 4)
        self.ink = ctx.texture(canvas.image.size, 4)
        for texture in (self.texture, self.ink):
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = texture.repeat_y = False
        self.program = ctx.program(vertex_shader="""
            #version 330
            in vec2 position; out vec2 uv;
            void main() { gl_Position=vec4(position,0,1); uv=vec2(position.x*.5+.5,.5-position.y*.5); }
        """, fragment_shader="""
            #version 330
            in vec2 uv; uniform sampler2D artwork; uniform float opacity; out vec4 fragColor;
            void main() { fragColor=texture(artwork,uv); fragColor.a*=opacity; }
        """)
        self.vbo = ctx.buffer(np.array([[-1,-1],[1,-1],[-1,1],[1,1]], dtype="f4").tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.vbo, "2f", "position")])
        self.program["artwork"].value = 1
        self.version = -1
        self.aspect = None
        self.enabled = False
        self.composed = canvas.image.copy()

    def update(self, player, aspect):
        if self.canvas.dirty or player.version != self.version or aspect != self.aspect:
            self.composed = self.canvas.compose(player.frame, aspect)
            self.texture.write(self.composed.tobytes())
            self.ink.write(self.canvas.image.tobytes())
            self.enabled = player.frame is not None or self.canvas.image.getbbox() is not None
            self.version, self.aspect = player.version, aspect
            self.canvas.dirty = False

    def draw(self, viewport, mode="neurons"):
        self.ctx.viewport = viewport
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        (self.texture if mode == "overlay" else self.ink).use(location=1)
        self.program["opacity"].value = .65 if mode == "overlay" else .8
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def release(self):
        for resource in (self.vao, self.vbo, self.program, self.texture, self.ink):
            resource.release()

"""Live male-cns:v1.0 sensory / neural / motor / arena experiment."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time
import shutil
import subprocess

import glfw
import moderngl
import numpy as np
from PIL import Image

from src.arena_render import ArenaRenderer, Overlay, SceneHUD
from src.arena_model import DEFAULT_ARENA_SIZE
from src.arena_session import ArenaSession
from src.fly_brain_opengl import create_skeleton_batch, load_skeletons, look_at, perspective, write_mat4
from src.live_brain import load_live_graph
from src.brain_view import BrainMap, BrainRenderer
from src.viewer_camera import arena_overview, pan_camera
from src.brain_media import BrainCanvas, MediaPlayer, ArtworkRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--skeleton-run", type=Path, default=Path("runs/touch_front_left_strong"))
    parser.add_argument("--max-skeletons", type=int, default=40)
    parser.add_argument("--no-exploration", action="store_true")
    parser.add_argument("--no-learning", action="store_true", help="Use fixed exploratory sensory drive")
    parser.add_argument("--evaluate", action="store_true", help="Use the learned policy without updating it")
    parser.add_argument("--policy", type=Path, default=Path("data/cache/arena_policy_v1.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--takeoff", action="store_true", help="Start with a wing-sensory flight request")
    parser.add_argument("--arena-width",type=float,default=DEFAULT_ARENA_SIZE[0])
    parser.add_argument("--arena-height",type=float,default=DEFAULT_ARENA_SIZE[1])
    parser.add_argument("--no-vision",action="store_true",help="Disable eye inputs and visual avoidance")
    parser.add_argument("--no-avoidance",action="store_true",help="Keep eyes enabled without the engineered steering assist")
    parser.add_argument("--view",choices=["arena","brain"],default="arena")
    parser.add_argument("--brain-view", choices=["front", "side", "top", "oblique"], default="front")
    parser.add_argument("--brain-region", choices=["all", "head", "cord"], default="all")
    parser.add_argument("--media", type=Path, help="Local video or audio to play on the CNS map")
    parser.add_argument("--drawing", type=Path, help="Load a saved PNG drawing onto the CNS map")
    parser.add_argument("--media-mode", choices=["neurons", "overlay"], default="neurons")
    parser.add_argument("--mute-media", action="store_true")
    parser.add_argument("--no-fibers", action="store_true", help="Hide cached neuron branches on the CNS map")
    parser.add_argument("--art-current", type=float, default=1.1, help="Media/drawing input current at full brightness")
    parser.add_argument("--art-limit", type=int, default=2048, help="Maximum media/ink targets per neural step")
    parser.add_argument("--max-art-events", type=int, default=2_000_000,
                        help="Maximum recorded media/ink current deliveries; 0 disables their recording")
    parser.add_argument("--no-art-stimulation", action="store_true", help="Display art/media without neural current injection")
    parser.add_argument("--arena-view", choices=["follow", "overview", "chase", "overhead"], default="follow")
    parser.add_argument("--paused", action="store_true", help="Open paused for inspection")
    parser.add_argument("--neural-hz", type=float, default=60)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--frames", type=int, default=0, help="Automatically stop after this many frames")
    parser.add_argument("--headless-steps", type=int, default=0, help="Run fixed simulation steps without a window")
    parser.add_argument("--screenshot", type=Path, help="Save the last rendered frame")
    parser.add_argument("--output", type=Path, help="Session export directory (default: new timestamped run)")
    args = parser.parse_args()
    if (args.width < 700 or args.height < 450 or not 10 <= args.neural_hz <= 240
            or args.frames < 0 or args.headless_steps < 0 or args.max_skeletons < 1
            or not 12 <= args.arena_width <= 2000 or not 12 <= args.arena_height <= 2000):
        parser.error("Use a window of at least 700x450, neural Hz 10–240, room dimensions 12–2000, nonnegative run lengths, and positive skeleton count")
    if args.paused and args.headless_steps:
        parser.error("--paused is for interactive runs; remove it when using --headless-steps")
    if not 0 <= args.art_current <= 3 or not 1 <= args.art_limit <= 8192 or args.max_art_events < 0:
        parser.error("Use art current 0–3, art target limit 1–8192, and a nonnegative art recording limit")
    output = args.output or Path("runs") / datetime.now().strftime("arena_%Y%m%d_%H%M%S_%f")
    graph = load_live_graph()
    print(f"male-cns:v1.0: {len(graph.ids):,} retained bodies from {graph.source_size:,} source IDs", flush=True)
    print("Exploratory input and body mechanics are imposed toy models.", flush=True)
    skeletons = []
    if not args.headless_steps:
        try:
            skeletons, _ = load_skeletons(args.skeleton_run, args.max_skeletons)
        except RuntimeError as error:
            print(f"Morphology panel unavailable: {error}", flush=True)
    session = ArenaSession(graph, [body for body, _ in skeletons], device=args.device,
                           neural_hz=args.neural_hz, exploration=not args.no_exploration,
                           learning=not args.no_learning, policy_path=args.policy,
                           seed=args.seed, training=not args.evaluate,
                           arena_size=(args.arena_width,args.arena_height),
                           vision=not args.no_vision,avoidance=not args.no_avoidance,
                           max_art_events=args.max_art_events)
    session.flight_requested = args.takeoff
    session.flight_override = True if args.takeoff else None
    session.paused = args.paused
    session.art_enabled = not args.no_art_stimulation
    session.art_strength = args.art_current
    session.art_limit = args.art_limit
    if args.headless_steps:
        start = time.perf_counter()
        for step in range(args.headless_steps):
            session.tick()
            if session.brain.halted:
                break
        session.export(output)
        report = {"steps": session.brain.step_count, "simulation_seconds": session.arena.elapsed,
                  "wall_seconds": time.perf_counter() - start, "distance": session.arena.distance,
                  "motor_spikes": session.bridge.motor_spikes, "halted": session.brain.halted,
                  "learning": session.learner.summary() if session.learner else None,
                  "contact_steps": session.arena.contacts,
                  "avoidance_interventions": session.navigator.interventions,
                  "visual_relay_spikes": session.visual_spikes,
                  "output": str(output)}
        print(json.dumps(report, indent=2))
        if session.brain.halted:
            raise SystemExit(2)
        return
    if not glfw.init():
        raise RuntimeError("GLFW could not initialize; use --headless-steps for simulation only")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(args.width, args.height, "FlyBrain — live arena", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Could not create an OpenGL window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    ctx = moderngl.create_context(require=330)
    ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    print("Renderer:", ctx.info.get("GL_RENDERER"), flush=True)
    renderer = ArenaRenderer(ctx, session.arena)
    overlay = Overlay(ctx)
    hud = SceneHUD(ctx)
    brain_map = BrainMap(graph.ids)
    fiber_dirs = [args.skeleton_run / "skeleton_cache", *sorted(Path("runs").glob("*/skeleton_cache"))]
    fibers = brain_map.load_fibers(fiber_dirs)
    brain_renderer = BrainRenderer(ctx,brain_map,fibers)
    canvas = BrainCanvas()
    if args.drawing:
        canvas.load(args.drawing)
    media = MediaPlayer(muted=args.mute_media)
    artwork = ArtworkRenderer(ctx,canvas)
    picker = None
    print(f"Whole CNS map: {len(brain_map.indices):,} anatomical positions",flush=True)
    skeleton_gl = create_skeleton_batch(ctx, skeletons) if skeletons else None
    if skeleton_gl:
        skeleton_gl[0]["hide_inactive"].value = False
    camera = {"yaw": math.radians(-35), "pitch": math.radians(52), "distance": 8.5,
              "follow": True, "drag": None, "target": np.zeros(3), "preset": "follow"}
    aspect = (args.width-overlay.width)/args.height
    brain_camera = dict(brain_map.camera_preset(args.brain_view, aspect, args.brain_region), active_only=False, fibers=not args.no_fibers)
    scene = {"mode":args.view,"click_start":None,"pose":None, "local_map":False,
             "searching":False, "query":"", "matches":[], "match_index":-1,
             "notice":"", "notice_until":0, "capture":False, "checkpoint":False,
             "drawing":False, "painting":False, "eraser":False, "palette":0,
             "media_mode":args.media_mode, "fiber_count":len(fibers), "fly_preview":True}

    def viewport_aspect():
        w, h = glfw.get_framebuffer_size(window)
        return max(.1, (w-overlay.width)/max(1, h))

    def notice(message):
        scene.update(notice=message, notice_until=time.perf_counter()+5)

    def open_media(path):
        try:
            media.open(path)
            scene["mode"] = "brain"
            notice("Playing media on the map · K pause · Y change display")
        except (ValueError, RuntimeError, OSError) as error:
            notice(str(error))

    def paint_at(x, y, begin=False):
        w,h = glfw.get_window_size(window)
        fbw,fbh = glfw.get_framebuffer_size(window)
        if min(w,h,fbw,fbh) <= 0:
            return
        uv = (x*fbw/w/max(1,fbw-overlay.width), y/h)
        if 0 <= uv[0] <= 1 and 0 <= uv[1] <= 1:
            (canvas.begin if begin else canvas.stroke)(uv, scene["eraser"])
        else:
            canvas.end()

    def set_arena_view(name):
        if name == "overview":
            camera.update(arena_overview(session.arena, viewport_aspect()))
        else:
            camera.update(follow=True, preset=name, distance=12 if name=="overhead" else 8.5,
                          yaw=math.pi if name in ("overhead", "chase") else math.radians(-35),
                          pitch=1.45 if name=="overhead" else .32 if name=="chase" else math.radians(52))

    def focus_selection():
        point = brain_map.point_lookup.get(brain_map.selected)
        if point is not None:
            brain_camera.update(target=brain_map.points[point].copy(), distance=.6, preset="selection")
        else:
            notice("This neuron has no cached anatomical position")

    set_arena_view(args.arena_view)

    def key_callback(window, key, scancode, action, mods):
        nonlocal picker
        if scene["searching"] and action in (glfw.PRESS, glfw.REPEAT):
            if key == glfw.KEY_ESCAPE:
                scene["searching"] = False
            elif key == glfw.KEY_BACKSPACE:
                scene["query"] = scene["query"][:-1]
                scene["matches"] = []
            elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
                if not scene["matches"]:
                    scene["matches"] = brain_map.search(scene["query"])
                    scene["match_index"] = -1
                if scene["matches"]:
                    scene["match_index"] = (scene["match_index"]+1)%len(scene["matches"])
                    brain_map.selected = scene["matches"][scene["match_index"]]
                    focus_selection()
                else:
                    notice("No matching neuron in the retained live graph")
            return
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            with session.lock:
                session.paused = not session.paused
        elif key == glfw.KEY_E:
            with session.lock:
                session.exploration = not session.exploration
        elif key == glfw.KEY_R:
            session.reset()
        elif key == glfw.KEY_C:
            if scene["mode"] == "brain":
                brain_camera.update(brain_map.camera_preset("front", viewport_aspect(), brain_camera["region"]))
            else:
                set_arena_view("overview" if camera["follow"] else "follow")
        elif key == glfw.KEY_TAB:
            scene["mode"] = "brain" if scene["mode"] == "arena" else "arena"
            camera["drag"] = None
            scene["click_start"] = None
            scene["painting"] = False
            canvas.end()
        elif key == glfw.KEY_H and scene["mode"] == "brain":
            brain_camera["active_only"] = not brain_camera["active_only"]
        elif key == glfw.KEY_Z and mods & glfw.MOD_CONTROL and scene["mode"] == "brain":
            canvas.undo()
        elif key == glfw.KEY_D and scene["mode"] == "brain":
            scene["drawing"] = not scene["drawing"]
            scene["painting"] = False
            camera["drag"] = None
            canvas.end()
            notice("Draw mode · left paint · Shift erase · right pan" if scene["drawing"] else "Orbit mode")
        elif key == glfw.KEY_X and scene["mode"] == "brain":
            canvas.clear()
        elif key in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET) and scene["mode"] == "brain":
            canvas.radius = int(np.clip(canvas.radius + (3 if key==glfw.KEY_RIGHT_BRACKET else -3), 2, 80))
        elif key == glfw.KEY_T and scene["mode"] == "brain":
            palette = [(255,110,195,255),(80,220,255,255),(255,225,90,255),(255,255,255,255)]
            scene["palette"] = (scene["palette"]+1)%len(palette)
            canvas.color = palette[scene["palette"]]
        elif key == glfw.KEY_L and scene["mode"] == "brain":
            brain_camera["fibers"] = not brain_camera["fibers"]
        elif key == glfw.KEY_U and scene["mode"] == "brain":
            scene["fly_preview"] = not scene["fly_preview"]
        elif key == glfw.KEY_I and scene["mode"] == "brain":
            with session.lock:
                session.art_enabled = not session.art_enabled
            notice("Media/ink neural input ON" if session.art_enabled else "Media/ink neural input OFF")
        elif key in (glfw.KEY_MINUS, glfw.KEY_EQUAL) and scene["mode"] == "brain":
            with session.lock:
                session.art_strength = round(float(np.clip(session.art_strength + (.1 if key==glfw.KEY_EQUAL else -.1), 0, 3)), 1)
        elif key == glfw.KEY_O and picker is None:
            command = (["zenity", "--file-selection", "--title=Play a local video or song"] if shutil.which("zenity") else
                       ["kdialog", "--getopenfilename", str(Path.home())] if shutil.which("kdialog") else None)
            if command:
                picker = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            else:
                notice("Drop a local media file into this window, or launch with --media PATH")
        elif key == glfw.KEY_K:
            media.toggle_pause()
        elif key == glfw.KEY_J and media.path is not None:
            open_media(media.path)
        elif key == glfw.KEY_Y and scene["mode"] == "brain":
            scene["media_mode"] = "overlay" if scene["media_mode"] == "neurons" else "neurons"
        elif key == glfw.KEY_BACKSPACE and scene["mode"] == "brain":
            media.close()
            media.frame = None
            media.version += 1
            media.status = "Media stopped · O open"
        elif key == glfw.KEY_Z and scene["mode"] == "brain" and brain_map.selected is not None:
            focus_selection()
        elif key == glfw.KEY_SLASH and scene["mode"] == "brain":
            scene.update(searching=True, query="", matches=[], match_index=-1, ignore_slash=True)
        elif key == glfw.KEY_B and scene["mode"] == "brain":
            regions = ["all", "head", "cord"]
            region = regions[(regions.index(brain_camera["region"])+1)%3]
            brain_camera.update(brain_map.camera_preset("front", viewport_aspect(), region))
        elif glfw.KEY_3 <= key <= glfw.KEY_6:
            if scene["mode"] == "brain":
                name = ["front", "side", "top", "oblique"][key-glfw.KEY_3]
                brain_camera.update(brain_map.camera_preset(name, viewport_aspect(), brain_camera["region"]))
            elif key in (glfw.KEY_3, glfw.KEY_4):
                set_arena_view("chase" if key==glfw.KEY_3 else "overhead")
        elif key == glfw.KEY_M:
            scene["local_map"] = not scene["local_map"]
        elif key == glfw.KEY_P:
            scene["capture"] = True
        elif key == glfw.KEY_S:
            scene["checkpoint"] = True
        elif key == glfw.KEY_N:
            with session.lock:
                session.paused = False
                try:
                    session.tick()
                finally:
                    session.paused = True
        elif key == glfw.KEY_V:
            with session.lock:
                session.avoidance_enabled = not session.avoidance_enabled
        elif key == glfw.KEY_F:
            with session.lock:
                session.flight_requested = not session.flight_requested
                session.flight_override = session.flight_requested
        elif key == glfw.KEY_G:
            with session.lock:
                session.flight_override = None
        elif key in (glfw.KEY_1, glfw.KEY_2):
            with session.lock:
                offset = 0 if key == glfw.KEY_1 else 3
                session.manual_touch[offset:offset + 3] = 1.0

    def mouse_callback(window, button, action, mods):
        if button in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            x, y = glfw.get_cursor_pos(window)
            width,height = glfw.get_window_size(window)
            fbw,fbh = glfw.get_framebuffer_size(window)
            if min(width, height, fbw, fbh) <= 0:
                return
            panel_x = x*fbw/width-(fbw-overlay.width)
            panel_y = y/height*900
            if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and 82 <= panel_y <= 111 and 18 <= panel_x <= 332:
                scene["mode"] = "arena" if panel_x < 170 else "brain"
                camera["drag"] = scene["click_start"] = None
                scene["searching"] = False
                scene["painting"] = False
                canvas.end()
                return
            if button == glfw.MOUSE_BUTTON_LEFT and scene["mode"] == "brain" and scene["drawing"]:
                if action == glfw.PRESS and x*fbw/width < fbw-overlay.width:
                    scene.update(painting=True, eraser=bool(mods & glfw.MOD_SHIFT))
                    paint_at(x,y,begin=True)
                elif action == glfw.RELEASE:
                    scene["painting"] = False
                    canvas.end()
                return
            if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and scene["mode"] == "arena":
                location = hud.map_position(x*fbw/width, y*fbh/height)
                if location is not None:
                    location = np.clip(location, [-session.arena.half_width, -session.arena.half_height],
                                       [session.arena.half_width, session.arena.half_height])
                    camera.update(target=np.array([*location, 0.0]), follow=False, distance=25, preset="map")
                    camera["drag"] = scene["click_start"] = None
                    return
            if action == glfw.PRESS and x*fbw/width < fbw-overlay.width:
                camera["drag"] = scene["click_start"] = (x,y)
                scene["pan"] = button == glfw.MOUSE_BUTTON_RIGHT
                if scene["mode"] == "arena" and scene["pan"] and camera["follow"]:
                    with session.lock:
                        camera["target"] = np.array([*session.arena.position, session.arena.altitude+.25])
                    camera.update(follow=False, preset="free")
            else:
                if (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE and scene["mode"] == "brain" and scene["click_start"] is not None
                        and np.linalg.norm(np.array([x,y])-scene["click_start"]) < 4 and scene["pose"] is not None):
                    brain_map.pick(x*fbw/width,fbh-y*fbh/height,brain_renderer.mvp,brain_renderer.viewport,
                                   scene["pose"].get("brain_activity") if brain_camera["active_only"] else None,
                                   brain_camera["region"])
                camera["drag"] = scene["click_start"] = None

    def cursor_callback(window, x, y):
        if scene["painting"] and scene["mode"] == "brain":
            paint_at(x,y)
            return
        if camera["drag"] is not None:
            old_x, old_y = camera["drag"]
            target = brain_camera if scene["mode"] == "brain" else camera
            if scene.get("pan"):
                pan_camera(target, x-old_x, y-old_y, glfw.get_window_size(window)[1],
                           42 if scene["mode"]=="brain" else 45)
            else:
                target["yaw"] += (x - old_x) * .006
                target["pitch"] = float(np.clip(target["pitch"] + (y - old_y) * .006, -1.45 if scene["mode"] == "brain" else .15, 1.45))
            target["preset"] = "orbit"
            camera["drag"] = (x, y)

    def scroll_callback(window, dx, dy):
        target = brain_camera if scene["mode"] == "brain" else camera
        target["distance"] = float(np.clip(target["distance"] * math.exp(-dy*.12),
                                          .08 if scene["mode"] == "brain" else 3,
                                          50 if scene["mode"] == "brain" else max(args.arena_width,args.arena_height)*6))

    def char_callback(window, codepoint):
        if scene["searching"]:
            if scene.pop("ignore_slash", False) and chr(codepoint)=="/":
                return
            if chr(codepoint).isprintable() and len(scene["query"]) < 80:
                scene["query"] += chr(codepoint)
                scene["matches"] = []

    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_char_callback(window, char_callback)
    glfw.set_drop_callback(window, lambda window, paths: open_media(paths[0]) if paths else None)
    if args.media:
        open_media(args.media)
    session.start()
    frame_count, fps, interval_frames = 0, 0.0, 0
    last_ui = 0.0
    fps_start = time.perf_counter()
    screenshot = None
    last_size = None
    last_art_input = 0.0
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            if picker is not None and picker.poll() is not None:
                selected_path = picker.communicate()[0].decode().strip()
                if picker.returncode == 0 and selected_path:
                    open_media(selected_path)
                picker = None
            fbw, fbh = glfw.get_framebuffer_size(window)
            if fbw < 700 or fbh < 450:
                session.set_art_stimulus([],[])
                glfw.wait_events_timeout(.05)
                continue
            if last_size != (fbw, fbh):
                if camera["preset"] == "overview":
                    camera.update(arena_overview(session.arena, viewport_aspect()))
                if brain_camera["preset"] in ("front", "side", "top", "oblique"):
                    brain_camera.update(brain_map.camera_preset(brain_camera["preset"], viewport_aspect(), brain_camera["region"]))
                last_size = (fbw, fbh)
                last_ui = 0
            if scene["checkpoint"]:
                checkpoint = output / datetime.now().strftime("checkpoint_%Y%m%d_%H%M%S_%f")
                session.export(checkpoint)
                print(f"Saved checkpoint: {checkpoint}", flush=True)
                notice("Checkpoint saved; session continues")
                scene["checkpoint"] = False
            pose = session.snapshot(whole_brain=scene["mode"]=="brain")
            scene["media_status"] = ("PAUSED · " if media.paused else "") + media.status
            scene["brush_radius"] = canvas.radius
            scene["pose"] = pose
            if pose["error"]:
                raise RuntimeError(pose["error"])
            now = time.perf_counter()
            if now > scene["notice_until"]:
                scene["notice"] = ""
            ctx.viewport = (0, 0, fbw, fbh)
            ctx.clear(.018, .035, .045, 1)
            if scene["mode"] == "brain":
                artwork.update(media, (fbw-overlay.width)/fbh)
                brain_renderer.draw(pose,(0,0,fbw-overlay.width,fbh),brain_camera,
                                    artwork.texture if artwork.enabled and scene["media_mode"]=="neurons" else None)
                artwork.draw((0,0,fbw-overlay.width,fbh),scene["media_mode"])
                if now-last_art_input >= 1/30:
                    input_image = canvas.image if media.paused or not media.playing else artwork.composed
                    targets, levels = brain_map.stimulus_from_image(input_image,brain_renderer.mvp,
                                                                    brain_renderer.viewport,brain_camera["region"])
                    session.set_art_stimulus(targets,levels)
                    last_art_input = now
                if scene["fly_preview"] and fbw-overlay.width >= 850:
                    inset = (fbw-overlay.width-292, fbh-230, 270, 184)
                    ctx.clear(.025,.055,.065,1,viewport=inset)
                    renderer.draw(pose,inset,{"follow":True,"yaw":math.pi-pose["yaw"],"pitch":.55,"distance":5.0})
            else:
                session.set_art_stimulus([],[])
                if camera["preset"] == "chase":
                    camera["yaw"] = math.pi-pose["yaw"]
                renderer.draw(pose, (0, 0, fbw - overlay.width, fbh), camera)
            if skeleton_gl and scene["mode"] == "arena":
                program, vbo, vao, texture = skeleton_gl
                ctx.viewport = (fbw - overlay.width + 8, int(fbh * 185 / 900), overlay.width - 16, int(fbh * 100 / 900))
                mvp = perspective(42, (overlay.width - 16) / max(1, int(fbh * 100 / 900)), .05, 100)
                view = look_at([3.6, -3.5, 2.7], [0, 0, 0], [0, 0, 1])
                write_mat4(program["mvp"], mvp @ view)
                texture.write(pose["watched_activity"].astype("f4").tobytes())
                texture.use(location=0)
                vao.render(moderngl.LINES)
            if now - last_ui > .10 or overlay.height != fbh:
                overlay.update(pose, fbh, fps, len(graph.ids),scene["mode"],brain_map)
                hud.update_scene(pose, fbw-overlay.width, fbh, scene["mode"],
                                 brain_camera if scene["mode"]=="brain" else camera, scene, brain_map,
                                 brain_renderer.mvp if scene["mode"]=="brain" else renderer.mvp)
                last_ui = now
            hud.draw(fbw, fbh)
            overlay.draw(fbw, fbh)
            if scene["capture"]:
                capture_path = output / "screenshots" / datetime.now().strftime("%Y%m%d_%H%M%S_%f.png")
                capture_path.parent.mkdir(parents=True, exist_ok=True)
                Image.frombytes("RGB", (fbw, fbh), ctx.screen.read(components=3)).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(capture_path)
                print(f"Saved screenshot: {capture_path}", flush=True)
                notice("Screenshot saved to this run's screenshots folder")
                scene["capture"] = False
            frame_count += 1
            interval_frames += 1
            if now - fps_start > .5:
                fps = interval_frames / (now - fps_start)
                interval_frames = 0
                fps_start = now
            if args.screenshot and (not args.frames or frame_count >= args.frames):
                screenshot = (fbw, fbh, ctx.screen.read(components=3))
            glfw.swap_buffers(window)
            if args.frames and frame_count >= args.frames:
                break
    finally:
        session.stop()
        media.close()
        if picker is not None:
            picker.terminate()
            picker.communicate(timeout=3)
        canvas.save(output / "brain_drawing.png")
        session.export(output)
        if screenshot is not None:
            width, height, pixels = screenshot
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.frombytes("RGB", (width, height), pixels).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(args.screenshot)
        if skeleton_gl:
            for resource in reversed(skeleton_gl):
                resource.release()
        overlay.release()
        hud.release()
        brain_renderer.release()
        artwork.release()
        renderer.release()
        ctx.release()
        glfw.destroy_window(window)
        glfw.terminate()
        print(f"Saved session: {output}", flush=True)


if __name__ == "__main__":
    main()

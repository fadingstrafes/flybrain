import argparse
import math
import time
from pathlib import Path

import glfw
import moderngl
import numpy as np
import pandas as pd


CACHE = Path("data/cache")


def clean(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def motor_group(row):
    system = clean(row.get("system") or row.get("limb")).lower()
    side = clean(row.get("side")).upper()
    pair = clean(row.get("leg_pair")).lower()

    if system == "leg" and side in {"L", "R"} and pair in {"front", "middle", "hind"}:
        return f"{side} {pair}"
    if system in {"wing", "haltere", "abdomen"}:
        return system
    return "other"


def load_events(run):
    path = run / "spike_events.npz"
    if not path.exists():
        raise RuntimeError(f"Missing {path}")

    data = np.load(path)
    times = data["time"].astype(np.int32)
    body_ids = data["bodyId"].astype(np.int64)

    if len(times) == 0:
        raise RuntimeError("Run contains no spike events.")

    order = np.argsort(times)
    return times[order], body_ids[order]


def build_motor_activation(times, body_ids, motor_map, total_steps, decay=0.90):
    groups = [
        "L front", "L middle", "L hind",
        "R front", "R middle", "R hind",
        "wing", "haltere", "abdomen", "other",
    ]
    actions = ["flex", "extend", "lift", "depress", "generic"]

    gidx = {g: i for i, g in enumerate(groups)}
    aidx = {a: i for i, a in enumerate(actions)}

    group_events = np.zeros((total_steps, len(groups)), dtype=np.float32)
    leg_action_events = np.zeros((total_steps, 6, len(actions)), dtype=np.float32)

    leg_groups = ["L front", "L middle", "L hind", "R front", "R middle", "R hind"]
    lidx = {g: i for i, g in enumerate(leg_groups)}

    lookup = motor_map.set_index("bodyId").to_dict("index")

    # Only 815-ish motor neurons are mapped, so dictionary lookups are cheap.
    for t, body_id in zip(times, body_ids):
        row = lookup.get(int(body_id))
        if row is None:
            continue

        group = motor_group(row)
        group_events[t, gidx[group]] += 1.0

        if group in lidx:
            action = clean(row.get("action")).lower()
            if action not in aidx:
                action = "generic"
            leg_action_events[t, lidx[group], aidx[action]] += 1.0

    group_activation = np.zeros_like(group_events)
    leg_activation = np.zeros_like(leg_action_events)

    glevel = np.zeros(len(groups), dtype=np.float32)
    llevel = np.zeros((6, len(actions)), dtype=np.float32)

    for t in range(total_steps):
        glevel *= decay
        llevel *= decay

        glevel += group_events[t] * 0.12
        llevel += leg_action_events[t] * 0.12

        glevel = np.clip(glevel, 0.0, 1.0)
        llevel = np.clip(llevel, 0.0, 1.0)

        group_activation[t] = glevel
        leg_activation[t] = llevel

    return groups, actions, group_activation, leg_activation


def leg_points(side, pair, activation, actions):
    side_sign = -1.0 if side == "L" else 1.0
    y_anchor = {"front": 0.70, "middle": 0.0, "hind": -0.70}[pair]
    y_dir = {"front": 0.65, "middle": 0.0, "hind": -0.65}[pair]

    action_index = {a: i for i, a in enumerate(actions)}
    flex = float(activation[action_index["flex"]])
    extend = float(activation[action_index["extend"]])
    lift = float(activation[action_index["lift"]])
    depress = float(activation[action_index["depress"]])
    generic = float(activation[action_index["generic"]])

    extension = np.clip(extend - flex, -1.0, 1.0)
    vertical = np.clip(lift - depress, -1.0, 1.0)

    anchor = np.array([0.24 * side_sign, y_anchor, 0.0], dtype=np.float32)

    p1 = anchor + np.array([
        0.48 * side_sign,
        0.34 * y_dir,
        -0.10 + 0.16 * vertical,
    ], dtype=np.float32)

    outward = 0.64 + 0.38 * extension
    bend = 0.34 + 0.58 * flex - 0.32 * extend + 0.12 * generic

    p2 = p1 + np.array([
        outward * side_sign,
        0.56 * y_dir,
        -0.15 + 0.35 * vertical + bend * 0.25,
    ], dtype=np.float32)

    p3 = p2 + np.array([
        (0.58 + 0.32 * extension) * side_sign,
        0.50 * y_dir,
        -0.13 + 0.48 * vertical - bend * 0.30,
    ], dtype=np.float32)

    return np.vstack([anchor, p1, p2, p3])


def chain_segments(points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack([points[:-1], points[1:]], axis=1).reshape(-1, 3)


def build_fly_vertices(frame, groups, actions, motor_activation, leg_activation, motion_gain):
    chunks = []

    body = np.array([
        [0.0, 1.05, 0.05],
        [0.0, 0.25, 0.08],
        [0.0, -0.95, 0.00],
    ], dtype=np.float32)
    chunks.append(chain_segments(body))

    head = np.array([
        [-0.20, 1.20, 0.08],
        [0.20, 1.20, 0.08],
    ], dtype=np.float32)
    chunks.append(chain_segments(head))

    wing_level = float(np.clip(
        motor_activation[frame, groups.index("wing")] * motion_gain,
        0.0, 1.0,
    ))
    wing_span = 1.10 + 0.45 * wing_level
    wing_z = 0.10 + 0.40 * wing_level

    chunks.append(np.array([
        [0.0, 0.35, 0.08],
        [-wing_span, -0.05, wing_z],
        [0.0, 0.35, 0.08],
        [wing_span, -0.05, wing_z],
    ], dtype=np.float32))

    leg_order = [
        ("L", "front"), ("L", "middle"), ("L", "hind"),
        ("R", "front"), ("R", "middle"), ("R", "hind"),
    ]

    for i, (side, pair) in enumerate(leg_order):
        visual_activation = np.clip(
            leg_activation[frame, i] * motion_gain,
            0.0, 1.0,
        )
        chunks.append(chain_segments(leg_points(side, pair, visual_activation, actions)))

    return np.concatenate(chunks, axis=0).astype("f4", copy=False)


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def write_mat4(uniform, matrix):
    # GLSL expects column-major matrices.
    uniform.write(np.asarray(matrix, dtype="f4").T.tobytes())


def build_trailing_activity(times, body_ids, selected_body_ids, total_steps, trail):
    id_to_col = {int(body_id): i for i, body_id in enumerate(selected_body_ids)}
    n = len(selected_body_ids)

    impulses = np.zeros((total_steps, n), dtype=np.int32)

    cols = np.fromiter(
        (id_to_col.get(int(body_id), -1) for body_id in body_ids),
        dtype=np.int32,
        count=len(body_ids),
    )

    keep = cols >= 0
    if np.any(keep):
        np.add.at(impulses, (times[keep], cols[keep]), 1)

    cumulative = np.cumsum(impulses, axis=0, dtype=np.int32)
    activity = cumulative.copy()

    width = trail + 1
    if width < total_steps:
        activity[width:] -= cumulative[:-width]

    return activity


def load_skeletons(run, max_skeletons):
    skeleton_dir = run / "skeleton_cache"
    selected_path = skeleton_dir / "selected.csv"

    if not selected_path.exists():
        raise RuntimeError(
            "No skeleton cache found. Run cache_run_skeletons for this run first."
        )

    selected = pd.read_csv(selected_path)
    selected["bodyId"] = selected["bodyId"].astype(np.int64)

    if max_skeletons > 0:
        selected = selected.head(max_skeletons).copy()

    raw = []
    samples = []

    for body_id in selected["bodyId"]:
        body_id = int(body_id)
        path = skeleton_dir / f"{body_id}.npz"
        if not path.exists():
            continue

        segments = np.load(path)["segments"].astype(np.float32)
        if len(segments) == 0:
            continue

        raw.append((body_id, segments))
        stride = max(1, len(segments) // 1500)
        samples.append(segments[::stride].reshape(-1, 3))

    if not raw:
        raise RuntimeError("No cached skeleton .npz files could be loaded.")

    sample_points = np.concatenate(samples, axis=0)
    low = np.percentile(sample_points, 1, axis=0)
    high = np.percentile(sample_points, 99, axis=0)
    center = (low + high) * 0.5
    extent = float(np.max(high - low))
    scale = 2.5 / max(extent, 1e-6)

    normalized = []
    total_segments = 0

    for body_id, segments in raw:
        verts = segments.reshape(-1, 3)
        verts = ((verts - center) * scale).astype("f4", copy=False)
        normalized.append((body_id, verts))
        total_segments += len(segments)

    return normalized, total_segments


def main():
    parser = argparse.ArgumentParser(
        description="GPU/OpenGL realtime fly + connectome morphology viewer"
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--max-skeletons", type=int, default=40)
    parser.add_argument("--trail", type=int, default=10)
    parser.add_argument("--motion-gain", type=float, default=8.0)
    parser.add_argument("--playback-hz", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--hide-inactive",
        action="store_true",
        help="Do not render faint inactive morphologies",
    )
    args = parser.parse_args()

    run = Path(args.run)

    print("[1/5] Loading spike events...", flush=True)
    times, body_ids = load_events(run)
    total_steps = int(times.max()) + 1
    print(f"      {len(times):,} spike events, {total_steps:,} steps", flush=True)

    print("[2/5] Building motor playback...", flush=True)
    motor_path = CACHE / "motor_map.csv"
    if not motor_path.exists():
        raise RuntimeError(f"Missing {motor_path}")

    motor_map = pd.read_csv(motor_path)
    motor_map["bodyId"] = pd.to_numeric(motor_map["bodyId"], errors="coerce")
    motor_map = motor_map.dropna(subset=["bodyId"]).copy()
    motor_map["bodyId"] = motor_map["bodyId"].astype(np.int64)

    groups, actions, motor_activation, leg_activation = build_motor_activation(
        times, body_ids, motor_map, total_steps
    )

    print("[3/5] Loading real neuron skeletons...", flush=True)
    skeletons, total_segments = load_skeletons(run, args.max_skeletons)
    selected_ids = [body_id for body_id, _ in skeletons]
    print(
        f"      {len(skeletons)} neurons, {total_segments:,} line segments",
        flush=True,
    )

    print("[4/5] Precomputing activity windows...", flush=True)
    skeleton_activity = build_trailing_activity(
        times,
        body_ids,
        selected_ids,
        total_steps,
        args.trail,
    )

    print("[5/5] Starting OpenGL engine...", flush=True)

    if not glfw.init():
        raise RuntimeError("GLFW could not initialize.")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

    window = glfw.create_window(
        args.width,
        args.height,
        "FlyBrain OpenGL",
        None,
        None,
    )

    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create an OpenGL 3.3 window.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = moderngl.create_context(require=330)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (
        moderngl.SRC_ALPHA,
        moderngl.ONE_MINUS_SRC_ALPHA,
    )

    print("      GL vendor:  ", ctx.info.get("GL_VENDOR", "?"), flush=True)
    print("      GL renderer:", ctx.info.get("GL_RENDERER", "?"), flush=True)
    print("      GL version: ", ctx.info.get("GL_VERSION", "?"), flush=True)

    program = ctx.program(
        vertex_shader="""
            #version 330

            in vec3 in_pos;
            uniform mat4 mvp;

            void main() {
                gl_Position = mvp * vec4(in_pos, 1.0);
            }
        """,
        fragment_shader="""
            #version 330

            uniform vec4 u_color;
            out vec4 fragColor;

            void main() {
                fragColor = u_color;
            }
        """,
    )

    skeleton_gl = []

    for body_id, vertices in skeletons:
        vbo = ctx.buffer(vertices.tobytes())
        vao = ctx.vertex_array(
            program,
            [(vbo, "3f", "in_pos")],
        )
        skeleton_gl.append((body_id, vbo, vao, len(vertices)))

    fly_capacity = 4096 * 3 * 4
    fly_vbo = ctx.buffer(reserve=fly_capacity, dynamic=True)
    fly_vao = ctx.vertex_array(
        program,
        [(fly_vbo, "3f", "in_pos")],
    )

    state = {
        "frame": 0,
        "paused": False,
        "playback_hz": max(0.1, float(args.playback_hz)),
        "yaw": math.radians(-55.0),
        "pitch": math.radians(20.0),
        "distance": 4.2,
        "dragging": False,
        "last_mouse": None,
    }

    def key_callback(window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key == glfw.KEY_RIGHT:
            state["paused"] = True
            state["frame"] = min(total_steps - 1, state["frame"] + 1)
        elif key == glfw.KEY_LEFT:
            state["paused"] = True
            state["frame"] = max(0, state["frame"] - 1)
        elif key == glfw.KEY_D:
            state["paused"] = True
            state["frame"] = min(total_steps - 1, state["frame"] + 10)
        elif key == glfw.KEY_A:
            state["paused"] = True
            state["frame"] = max(0, state["frame"] - 10)
        elif key == glfw.KEY_R:
            state["frame"] = 0
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            state["playback_hz"] = min(500.0, state["playback_hz"] * 1.25)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            state["playback_hz"] = max(0.25, state["playback_hz"] / 1.25)

    def mouse_button_callback(window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT:
            return

        if action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(window)
            w, h = glfw.get_window_size(window)

            # Orbit only if the drag starts in the CNS/right viewport.
            if x >= w * 0.5:
                state["dragging"] = True
                state["last_mouse"] = (x, y)
        elif action == glfw.RELEASE:
            state["dragging"] = False
            state["last_mouse"] = None

    def cursor_callback(window, x, y):
        if not state["dragging"] or state["last_mouse"] is None:
            return

        lx, ly = state["last_mouse"]
        dx = x - lx
        dy = y - ly
        state["last_mouse"] = (x, y)

        state["yaw"] += dx * 0.006
        state["pitch"] -= dy * 0.006
        state["pitch"] = float(np.clip(
            state["pitch"],
            math.radians(-85.0),
            math.radians(85.0),
        ))

    def scroll_callback(window, dx, dy):
        x, y = glfw.get_cursor_pos(window)
        w, h = glfw.get_window_size(window)

        if x >= w * 0.5:
            state["distance"] *= math.exp(-dy * 0.12)
            state["distance"] = float(np.clip(state["distance"], 1.5, 12.0))

    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    frame_index = {body_id: i for i, body_id in enumerate(selected_ids)}

    last_clock = time.perf_counter()
    playback_accum = 0.0

    fps_clock = last_clock
    rendered_frames = 0
    fps = 0.0

    while not glfw.window_should_close(window):
        now = time.perf_counter()
        dt = now - last_clock
        last_clock = now

        glfw.poll_events()

        if not state["paused"]:
            playback_accum += dt * state["playback_hz"]
            advance = int(playback_accum)

            if advance > 0:
                state["frame"] = (state["frame"] + advance) % total_steps
                playback_accum -= advance

        frame = state["frame"]

        fbw, fbh = glfw.get_framebuffer_size(window)
        half = max(1, fbw // 2)

        ctx.clear(0.012, 0.016, 0.025, 1.0)

        # ---------------------------------------------------------
        # LEFT VIEWPORT: procedural fly
        # ---------------------------------------------------------

        ctx.viewport = (0, 0, half, fbh)

        fly_proj = perspective(45.0, half / max(1, fbh), 0.05, 100.0)
        fly_view = look_at(
            np.array([4.2, -4.2, 3.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

        write_mat4(program["mvp"], fly_proj @ fly_view)

        fly_vertices = build_fly_vertices(
            frame,
            groups,
            actions,
            motor_activation,
            leg_activation,
            args.motion_gain,
        )

        byte_count = fly_vertices.nbytes
        if byte_count > fly_capacity:
            raise RuntimeError(
                f"Fly dynamic VBO too small: need {byte_count}, have {fly_capacity}"
            )

        fly_vbo.write(fly_vertices.tobytes(), offset=0)
        program["u_color"].value = (0.90, 0.93, 1.00, 1.0)

        try:
            ctx.line_width = 3.0
        except Exception:
            pass

        fly_vao.render(
            mode=moderngl.LINES,
            vertices=len(fly_vertices),
        )

        # ---------------------------------------------------------
        # RIGHT VIEWPORT: real reconstructed neurons
        # ---------------------------------------------------------

        ctx.viewport = (half, 0, fbw - half, fbh)

        cns_aspect = (fbw - half) / max(1, fbh)
        cns_proj = perspective(42.0, cns_aspect, 0.05, 100.0)

        cp = state["pitch"]
        cy = state["yaw"]
        dist = state["distance"]

        eye = np.array([
            dist * math.cos(cp) * math.sin(cy),
            dist * math.cos(cp) * math.cos(cy),
            dist * math.sin(cp),
        ], dtype=np.float32)

        cns_view = look_at(
            eye,
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

        write_mat4(program["mvp"], cns_proj @ cns_view)

        activity_row = skeleton_activity[frame]

        try:
            ctx.line_width = 1.0
        except Exception:
            pass

        active_morphologies = 0

        for body_id, vbo, vao, vertex_count in skeleton_gl:
            col = frame_index[body_id]
            count = int(activity_row[col])

            if count > 0:
                active_morphologies += 1
                strength = min(1.0, 0.45 + 0.12 * count)
                program["u_color"].value = (
                    1.0,
                    0.35 + 0.55 * strength,
                    0.10,
                    1.0,
                )
            else:
                if args.hide_inactive:
                    continue

                program["u_color"].value = (
                    0.16,
                    0.22,
                    0.32,
                    0.10,
                )

            vao.render(
                mode=moderngl.LINES,
                vertices=vertex_count,
            )

        glfw.swap_buffers(window)

        rendered_frames += 1
        if now - fps_clock >= 0.5:
            fps = rendered_frames / (now - fps_clock)
            rendered_frames = 0
            fps_clock = now

            status = "PAUSED" if state["paused"] else "PLAY"
            glfw.set_window_title(
                window,
                (
                    f"FlyBrain OpenGL | {status} | "
                    f"step {frame}/{total_steps - 1} | "
                    f"{fps:5.1f} FPS | "
                    f"{active_morphologies} active morphologies | "
                    f"{state['playback_hz']:.1f} steps/s"
                ),
            )

    for _, vbo, vao, _ in skeleton_gl:
        vao.release()
        vbo.release()

    fly_vao.release()
    fly_vbo.release()
    program.release()
    ctx.release()

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()

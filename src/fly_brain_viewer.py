import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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


def category_from_row(row):
    superclass = clean(row.get("superclass")).lower()
    if "sensory" in superclass:
        return "sensory"
    if "motor" in superclass:
        return "motor"
    if "descending" in superclass:
        return "descending"
    if "ascending" in superclass:
        return "ascending"
    if "intrinsic" in superclass:
        return "intrinsic"
    return "other"


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

    anchor = np.array([0.24 * side_sign, y_anchor, 0.0])
    p1 = anchor + np.array([0.48 * side_sign, 0.34 * y_dir, -0.10 + 0.06 * vertical])

    outward = 0.64 + 0.14 * extension
    bend = 0.34 + 0.24 * flex - 0.10 * extend + 0.06 * generic
    p2 = p1 + np.array([
        outward * side_sign,
        0.56 * y_dir,
        -0.15 + 0.18 * vertical + bend * 0.15,
    ])

    p3 = p2 + np.array([
        (0.58 + 0.12 * extension) * side_sign,
        0.50 * y_dir,
        -0.13 + 0.26 * vertical - bend * 0.18,
    ])

    return np.vstack([anchor, p1, p2, p3])


def build_population(times, body_ids, metadata_lookup, total_steps):
    # Vectorized version: the original did a pandas lookup for every spike,
    # which can take minutes on large whole-CNS runs.
    names = ["sensory", "ascending", "descending", "intrinsic", "motor", "other"]
    idx = {n: i for i, n in enumerate(names)}

    unique_ids, inverse = np.unique(body_ids, return_inverse=True)
    unique_codes = np.full(len(unique_ids), idx["other"], dtype=np.int16)

    for i, body_id in enumerate(unique_ids):
        body_id = int(body_id)
        if body_id in metadata_lookup.index:
            unique_codes[i] = idx[category_from_row(metadata_lookup.loc[body_id])]

    codes = unique_codes[inverse]
    flat = times.astype(np.int64) * len(names) + codes.astype(np.int64)
    counts = np.bincount(flat, minlength=total_steps * len(names))
    pop = counts.reshape(total_steps, len(names)).astype(np.float32)

    kernel = np.ones(10, dtype=np.float32)
    smooth = np.zeros_like(pop)
    for i in range(pop.shape[1]):
        smooth[:, i] = np.convolve(pop[:, i], kernel, mode="same")
    return names, smooth


def main():
    parser = argparse.ArgumentParser(description="Synchronized digital fly + real-neuron activity viewer")
    parser.add_argument("--run", required=True)
    parser.add_argument("--interval", type=int, default=35)
    parser.add_argument("--trail", type=int, default=10)
    parser.add_argument("--inactive-alpha", type=float, default=0.03)
    parser.add_argument("--active-alpha", type=float, default=0.95)
    args = parser.parse_args()

    run = Path(args.run)
    print("[1/6] Loading spike events...", flush=True)
    times, body_ids = load_events(run)
    print(f"      {len(times):,} spike events", flush=True)
    total_steps = int(times.max()) + 1

    print("[2/6] Loading metadata...", flush=True)
    metadata = pd.read_parquet(CACHE / "metadata.parquet")
    metadata["bodyId"] = pd.to_numeric(metadata["bodyId"], errors="coerce")
    metadata = metadata.dropna(subset=["bodyId"]).drop_duplicates("bodyId")
    metadata["bodyId"] = metadata["bodyId"].astype(np.int64)
    metadata_lookup = metadata.set_index("bodyId")

    print("[3/6] Building motor activity...", flush=True)
    motor_path = CACHE / "motor_map.csv"
    if not motor_path.exists():
        raise RuntimeError(f"Missing {motor_path}. Run src.motor_map first.")
    motor_map = pd.read_csv(motor_path)
    motor_map["bodyId"] = pd.to_numeric(motor_map["bodyId"], errors="coerce")
    motor_map = motor_map.dropna(subset=["bodyId"]).copy()
    motor_map["bodyId"] = motor_map["bodyId"].astype(np.int64)

    groups, actions, motor_activation, leg_activation = build_motor_activation(
        times, body_ids, motor_map, total_steps
    )

    print("[4/6] Building population activity...", flush=True)
    pop_names, population = build_population(times, body_ids, metadata_lookup, total_steps)

    print("[5/6] Loading cached skeletons...", flush=True)
    skeleton_dir = run / "skeleton_cache"
    selected_path = skeleton_dir / "selected.csv"
    if not selected_path.exists():
        raise RuntimeError(
            "No skeleton cache found. Run:\n"
            f"python -m src.cache_run_skeletons --run {run} --max-neurons 40"
        )

    selected = pd.read_csv(selected_path)
    selected["bodyId"] = selected["bodyId"].astype(np.int64)
    selected_ids = set(int(x) for x in selected["bodyId"])

    skeletons = {}
    bounds_points = []
    for body_id in selected["bodyId"]:
        body_id = int(body_id)
        path = skeleton_dir / f"{body_id}.npz"
        if not path.exists():
            continue
        segments = np.load(path)["segments"].astype(np.float32)
        if len(segments) == 0:
            continue
        skeletons[body_id] = segments
        stride = max(1, len(segments) // 1000)
        bounds_points.append(segments[::stride].reshape(-1, 3))

    if not skeletons:
        raise RuntimeError("Skeleton cache exists, but no skeleton files could be loaded.")

    bounds_points = np.concatenate(bounds_points, axis=0)
    low = np.percentile(bounds_points, 1, axis=0)
    high = np.percentile(bounds_points, 99, axis=0)

    print(f"      loaded {len(skeletons)} skeletons", flush=True)
    print("[6/6] Creating window...", flush=True)
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.0, 1.25])

    ax_fly = fig.add_subplot(gs[0, 0], projection="3d")
    ax_cns = fig.add_subplot(gs[0, 1], projection="3d")
    ax_motor = fig.add_subplot(gs[1, 0])
    ax_pop = fig.add_subplot(gs[1, 1])

    fig.suptitle(f"Fly + CNS synchronized playback — {run.name}")

    # ---- fly panel ----------------------------------------------------
    ax_fly.set_title("Procedural fly / motor output")
    ax_fly.set_xlim(-2.2, 2.2)
    ax_fly.set_ylim(-2.0, 2.0)
    ax_fly.set_zlim(-1.0, 1.0)
    ax_fly.set_xticks([])
    ax_fly.set_yticks([])
    ax_fly.set_zticks([])
    ax_fly.view_init(elev=28, azim=-70)

    body_line, = ax_fly.plot([], [], [], linewidth=5)
    head_line, = ax_fly.plot([], [], [], linewidth=8)
    wing_l, = ax_fly.plot([], [], [], linewidth=2)
    wing_r, = ax_fly.plot([], [], [], linewidth=2)

    leg_order = [
        ("L", "front"), ("L", "middle"), ("L", "hind"),
        ("R", "front"), ("R", "middle"), ("R", "hind"),
    ]
    leg_lines = []
    for _ in leg_order:
        line, = ax_fly.plot([], [], [], marker="o", markersize=3)
        leg_lines.append(line)

    # ---- CNS panel ----------------------------------------------------
    ax_cns.set_title("Real reconstructed neurons")
    ax_cns.set_xlim(low[0], high[0])
    ax_cns.set_ylim(low[1], high[1])
    ax_cns.set_zlim(low[2], high[2])
    ax_cns.set_xticks([])
    ax_cns.set_yticks([])
    ax_cns.set_zticks([])
    ax_cns.view_init(elev=20, azim=-70)

    skeleton_artists = {}
    for body_id, segments in skeletons.items():
        collection = Line3DCollection(
            segments,
            linewidths=0.35,
            alpha=args.inactive_alpha,
        )
        ax_cns.add_collection3d(collection)
        skeleton_artists[body_id] = collection

    status = ax_cns.text2D(
        0.02, 0.98, "", transform=ax_cns.transAxes,
        va="top", family="monospace", fontsize=8
    )

    # ---- motor panel --------------------------------------------------
    bars = ax_motor.bar(groups, np.zeros(len(groups)))
    ax_motor.set_ylim(0, 1.05)
    ax_motor.set_ylabel("activation")
    ax_motor.set_title("Motor groups")
    ax_motor.tick_params(axis="x", rotation=40)

    # ---- population panel --------------------------------------------
    x = np.arange(total_steps)
    for i, name in enumerate(pop_names):
        ax_pop.plot(x, population[:, i], label=name)
    cursor = ax_pop.axvline(0)
    ax_pop.set_xlim(0, total_steps - 1)
    ax_pop.set_title("CNS population activity")
    ax_pop.set_xlabel("simulation step")
    ax_pop.set_ylabel("spikes / rolling window")
    ax_pop.legend(fontsize=7, ncol=2)

    state = {
        "frame": 0,
        "paused": False,
        "interval": max(5, int(args.interval)),
    }

    def active_cached(frame):
        start = max(0, frame - args.trail)
        left = np.searchsorted(times, start, side="left")
        right = np.searchsorted(times, frame + 1, side="left")
        ids = body_ids[left:right]
        unique, counts = np.unique(ids, return_counts=True)
        return {
            int(body_id): int(count)
            for body_id, count in zip(unique, counts)
            if int(body_id) in selected_ids
        }

    def update_fly(frame):
        body = np.array([
            [0.0, 1.05, 0.05],
            [0.0, 0.25, 0.08],
            [0.0, -0.95, 0.00],
        ])
        body_line.set_data(body[:, 0], body[:, 1])
        body_line.set_3d_properties(body[:, 2])

        head = np.array([
            [-0.18, 1.20, 0.08],
            [0.18, 1.20, 0.08],
        ])
        head_line.set_data(head[:, 0], head[:, 1])
        head_line.set_3d_properties(head[:, 2])

        wing_level = float(motor_activation[frame, groups.index("wing")])
        wing_span = 1.10 + 0.20 * wing_level
        wing_z = 0.10 + 0.18 * wing_level

        wl = np.array([[0.0, 0.35, 0.08], [-wing_span, -0.05, wing_z]])
        wr = np.array([[0.0, 0.35, 0.08], [wing_span, -0.05, wing_z]])
        wing_l.set_data(wl[:, 0], wl[:, 1])
        wing_l.set_3d_properties(wl[:, 2])
        wing_r.set_data(wr[:, 0], wr[:, 1])
        wing_r.set_3d_properties(wr[:, 2])

        for i, ((side, pair), line) in enumerate(zip(leg_order, leg_lines)):
            pts = leg_points(side, pair, leg_activation[frame, i], actions)
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])

    def update(frame):
        frame = int(np.clip(frame, 0, total_steps - 1))
        state["frame"] = frame

        update_fly(frame)

        active = active_cached(frame)
        for body_id, artist in skeleton_artists.items():
            if body_id in active:
                count = active[body_id]
                artist.set_alpha(args.active_alpha)
                artist.set_linewidth(min(2.5, 0.65 + 0.25 * count))
            else:
                artist.set_alpha(args.inactive_alpha)
                artist.set_linewidth(0.35)

        ranking = sorted(active.items(), key=lambda x: x[1], reverse=True)[:10]
        lines = [f"step {frame}/{total_steps - 1}", "", "ACTIVE CACHED NEURONS", "-" * 52]
        if not ranking:
            lines.append("none in current trail")
        for body_id, count in ranking:
            if body_id in metadata_lookup.index:
                row = metadata_lookup.loc[body_id]
                name = clean(row.get("instance")) or clean(row.get("type"))
                superclass = clean(row.get("superclass"))
            else:
                name = ""
                superclass = ""
            lines.append(
                f"{body_id:<9} {name[:21]:<21} {superclass[:14]:<14} x{count}"
            )
        status.set_text("\n".join(lines))

        for bar, value in zip(bars, motor_activation[frame]):
            bar.set_height(float(value))

        cursor.set_xdata([frame, frame])
        fig.canvas.draw_idle()

    update(0)

    timer = fig.canvas.new_timer(interval=state["interval"])

    def on_timer():
        if not state["paused"]:
            next_frame = (state["frame"] + 1) % total_steps
            update(next_frame)

    timer.add_callback(on_timer)
    timer.start()

    def on_key(event):
        key = event.key
        if key == " ":
            state["paused"] = not state["paused"]
        elif key == "right":
            state["paused"] = True
            update(min(total_steps - 1, state["frame"] + 1))
        elif key == "left":
            state["paused"] = True
            update(max(0, state["frame"] - 1))
        elif key in {"d", "D"}:
            state["paused"] = True
            update(min(total_steps - 1, state["frame"] + 10))
        elif key in {"a", "A"}:
            state["paused"] = True
            update(max(0, state["frame"] - 10))
        elif key in {"r", "R"}:
            update(0)
        elif key in {"+", "="}:
            state["interval"] = max(5, int(state["interval"] * 0.8))
            timer.interval = state["interval"]
        elif key in {"-", "_"}:
            state["interval"] = min(1000, int(state["interval"] * 1.25))
            timer.interval = state["interval"]

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    print("Viewer ready.", flush=True)
    plt.show()


if __name__ == "__main__":
    main()

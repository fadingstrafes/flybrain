import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


CACHE = Path("data/cache")


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_run_activation(run_path, impulse=0.20, decay=0.90):
    run = Path(run_path)

    events_file = run / "spike_events.npz"
    if not events_file.exists():
        raise RuntimeError(f"Missing {events_file}")

    motor_map_file = CACHE / "motor_map.csv"
    if not motor_map_file.exists():
        raise RuntimeError(f"Missing {motor_map_file}")

    events = np.load(events_file)
    times = events["time"].astype(int)
    body_ids = events["bodyId"].astype(int)

    motor = pd.read_csv(motor_map_file)
    motor["bodyId"] = motor["bodyId"].astype(int)
    motor = motor.set_index("bodyId")

    if len(times) == 0:
        return {}, 1

    total_steps = int(times.max()) + 1

    groups = defaultdict(
        lambda: np.zeros(total_steps, dtype=np.float32)
    )

    for t, body_id in zip(times, body_ids):
        if body_id not in motor.index:
            continue

        row = motor.loc[body_id]

        limb = str(row.get("limb", "other"))
        side = str(row.get("side", "?"))
        pair = str(row.get("leg_pair", ""))
        segment = str(row.get("segment", "unknown"))
        action = str(row.get("action", "generic"))

        key = (limb, side, pair, segment, action)
        groups[key][t] += 1.0

    activation = {}

    for key, spikes in groups.items():
        act = np.zeros(total_steps, dtype=np.float32)
        level = 0.0

        for t in range(total_steps):
            level *= decay
            level += spikes[t] * impulse
            level = min(level, 1.0)
            act[t] = level

        activation[key] = act

    return activation, total_steps


def get_activation(activation, t, limb, side=None, pair=None, segment=None, action=None):
    total = 0.0

    for key, values in activation.items():
        k_limb, k_side, k_pair, k_segment, k_action = key

        if k_limb != limb:
            continue
        if side is not None and k_side != side:
            continue
        if pair is not None and k_pair != pair:
            continue
        if segment is not None and k_segment != segment:
            continue
        if action is not None and k_action != action:
            continue

        if 0 <= t < len(values):
            total += values[t]

    return min(total, 1.0)


def leg_motor(activation, t, side, pair, action):
    value = get_activation(
        activation,
        t,
        limb="leg",
        side=side,
        pair=pair,
        action=action,
    )

    # let unknown-pair MNs contribute a little
    unknown = get_activation(
        activation,
        t,
        limb="leg",
        side=side,
        pair="unknown",
        action=action,
    )

    return min(1.0, value + 0.25 * unknown)


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------

def ellipsoid_wireframe(center, radii, nu=26, nv=13):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)

    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    x += center[0]
    y += center[1]
    z += center[2]

    return x, y, z


def unit_from_angles(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)

    x = np.cos(az) * np.cos(el)
    y = np.sin(az) * np.cos(el)
    z = np.sin(el)

    vec = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(vec)

    if norm == 0:
        return np.array([1.0, 0.0, 0.0])

    return vec / norm


def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


# ------------------------------------------------------------
# Fly model
# ------------------------------------------------------------

class Fly3D:
    def __init__(self, ax, activation, title):
        self.ax = ax
        self.activation = activation
        self.title = title

        self.leg_defs = {
            ("L", "front"):  {"hip": np.array([-0.28,  0.34, -0.06]), "az": 150},
            ("R", "front"):  {"hip": np.array([ 0.28,  0.34, -0.06]), "az":  30},
            ("L", "middle"): {"hip": np.array([-0.34,  0.00, -0.10]), "az": 180},
            ("R", "middle"): {"hip": np.array([ 0.34,  0.00, -0.10]), "az":   0},
            ("L", "hind"):   {"hip": np.array([-0.28, -0.36, -0.08]), "az": 215},
            ("R", "hind"):   {"hip": np.array([ 0.28, -0.36, -0.08]), "az": -35},
        }

        self.femur_length = 0.48
        self.tibia_length = 0.56
        self.wing_length = 0.95
        self.haltere_length = 0.25

        self.leg_lines = {}
        self.wing_lines = {}
        self.haltere_lines = {}
        self.status = None

        self._setup_axes()
        self._draw_body()
        self._create_artists()

    def _setup_axes(self):
        self.ax.set_title(self.title)
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.3, 1.3)
        self.ax.set_zlim(-1.0, 1.0)
        self.ax.set_box_aspect((1.2, 1.5, 1.0))
        self.ax.view_init(elev=18, azim=-60)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

    def _draw_body(self):
        # thorax
        x, y, z = ellipsoid_wireframe(
            center=(0.0, 0.0, 0.0),
            radii=(0.34, 0.42, 0.28),
        )
        self.ax.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=0.6)

        # abdomen
        x, y, z = ellipsoid_wireframe(
            center=(0.0, -0.72, -0.04),
            radii=(0.26, 0.52, 0.23),
        )
        self.ax.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=0.6)

        # head
        x, y, z = ellipsoid_wireframe(
            center=(0.0, 0.62, 0.02),
            radii=(0.22, 0.23, 0.20),
        )
        self.ax.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=0.6)

        # antennae
        self.ax.plot([0.05, 0.12], [0.79, 0.95], [0.05, 0.12], linewidth=1.0)
        self.ax.plot([-0.05, -0.12], [0.79, 0.95], [0.05, 0.12], linewidth=1.0)

    def _create_artists(self):
        # wings
        for side in ("L", "R"):
            line, = self.ax.plot([], [], [], linewidth=2.5)
            self.wing_lines[side] = line

        # halteres
        for side in ("L", "R"):
            line, = self.ax.plot([], [], [], linewidth=2.0)
            self.haltere_lines[side] = line

        # legs: femur + tibia per leg
        for key in self.leg_defs:
            femur, = self.ax.plot([], [], [], linewidth=2.8)
            tibia, = self.ax.plot([], [], [], linewidth=2.0)
            self.leg_lines[key] = (femur, tibia)

        self.status = self.ax.text2D(
            0.02,
            0.96,
            "",
            transform=self.ax.transAxes,
            fontsize=9,
            va="top",
        )

    def update(self, t):
        # wings
        for side in ("L", "R"):
            wing_act = get_activation(
                self.activation,
                t,
                limb="wing",
                side=side,
            )

            origin = np.array([
                -0.12 if side == "L" else 0.12,
                 0.12,
                 0.18,
            ])

            if side == "L":
                sweep = 150 - 70 * wing_act
            else:
                sweep = 30 + 70 * wing_act

            elev = 20 + 35 * wing_act
            tip = origin + self.wing_length * unit_from_angles(sweep, elev)

            line = self.wing_lines[side]
            line.set_data([origin[0], tip[0]], [origin[1], tip[1]])
            line.set_3d_properties([origin[2], tip[2]])

        # halteres
        for side in ("L", "R"):
            haltere_act = get_activation(
                self.activation,
                t,
                limb="haltere",
                side=side,
            )

            origin = np.array([
                -0.10 if side == "L" else 0.10,
                -0.18,
                 0.05,
            ])

            if side == "L":
                sweep = 210 - 40 * haltere_act
            else:
                sweep = -30 + 40 * haltere_act

            elev = -10 + 20 * haltere_act
            tip = origin + self.haltere_length * unit_from_angles(sweep, elev)

            line = self.haltere_lines[side]
            line.set_data([origin[0], tip[0]], [origin[1], tip[1]])
            line.set_3d_properties([origin[2], tip[2]])

        # legs
        max_leg = 0.0

        for (side, pair), info in self.leg_defs.items():
            hip = info["hip"]
            base_az = info["az"]

            extend = leg_motor(self.activation, t, side, pair, "extend")
            flex = leg_motor(self.activation, t, side, pair, "flex")
            lift = leg_motor(self.activation, t, side, pair, "lift")
            depress = leg_motor(self.activation, t, side, pair, "depress")
            protract = leg_motor(self.activation, t, side, pair, "protract")
            retract = leg_motor(self.activation, t, side, pair, "retract")
            abduct = leg_motor(self.activation, t, side, pair, "abduct")
            adduct = leg_motor(self.activation, t, side, pair, "adduct")
            generic = leg_motor(self.activation, t, side, pair, "generic")

            max_leg = max(
                max_leg,
                extend, flex, lift, depress,
                protract, retract, abduct, adduct, generic
            )

            # hip direction
            az = base_az + 25 * protract - 25 * retract
            if side == "L":
                az += -10 * abduct + 10 * adduct
            else:
                az += 10 * abduct - 10 * adduct

            el = -40 + 20 * lift - 18 * depress - 5 * generic

            femur_dir = unit_from_angles(az, el)
            knee = hip + femur_dir * self.femur_length

            # tibia direction:
            # bias it downward + partly along femur azimuth
            horizontal = normalize(np.array([np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az)), 0.0]))
            down = np.array([0.0, 0.0, -1.0])

            knee_open = 0.55 + 0.35 * flex - 0.30 * extend + 0.10 * generic
            knee_open = np.clip(knee_open, 0.15, 0.95)

            tibia_dir = normalize((1.0 - knee_open) * horizontal + knee_open * down)

            foot = knee + tibia_dir * self.tibia_length

            femur_line, tibia_line = self.leg_lines[(side, pair)]

            femur_line.set_data([hip[0], knee[0]], [hip[1], knee[1]])
            femur_line.set_3d_properties([hip[2], knee[2]])

            tibia_line.set_data([knee[0], foot[0]], [knee[1], foot[1]])
            tibia_line.set_3d_properties([knee[2], foot[2]])

        # status
        left_wing = get_activation(self.activation, t, limb="wing", side="L")
        right_wing = get_activation(self.activation, t, limb="wing", side="R")

        self.status.set_text(
            f"step: {t}\n"
            f"L wing: {left_wing:.2f}\n"
            f"R wing: {right_wing:.2f}\n"
            f"max leg act: {max_leg:.2f}"
        )

        artists = []
        artists.extend(self.wing_lines.values())
        artists.extend(self.haltere_lines.values())
        for pair in self.leg_lines.values():
            artists.extend(pair)
        artists.append(self.status)
        return artists


# ------------------------------------------------------------
# Main compare animation
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side 3D fly comparison"
    )

    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)

    parser.add_argument("--name-a", default="Run A")
    parser.add_argument("--name-b", default="Run B")

    parser.add_argument("--interval", type=int, default=25)
    parser.add_argument("--impulse", type=float, default=0.20)
    parser.add_argument("--decay", type=float, default=0.90)

    args = parser.parse_args()

    activation_a, steps_a = load_run_activation(
        args.run_a,
        impulse=args.impulse,
        decay=args.decay,
    )

    activation_b, steps_b = load_run_activation(
        args.run_b,
        impulse=args.impulse,
        decay=args.decay,
    )

    total_steps = max(steps_a, steps_b)

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    fly_a = Fly3D(ax1, activation_a, args.name_a)
    fly_b = Fly3D(ax2, activation_b, args.name_b)

    fig.suptitle("Male CNS motor-output comparison (3D)")

    def update(frame):
        artists = []
        artists.extend(fly_a.update(frame))
        artists.extend(fly_b.update(frame))
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=total_steps,
        interval=args.interval,
        blit=False,
        repeat=True,
    )

    fig._animation = animation
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

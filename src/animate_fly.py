import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse


CACHE = Path("data/cache")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        required=True,
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--impulse",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--decay",
        type=float,
        default=0.90,
    )

    args = parser.parse_args()

    run = Path(args.run)

    # -----------------------------------------------------
    # Load simulation + motor annotations
    # -----------------------------------------------------

    events = np.load(
        run / "spike_events.npz"
    )

    times = events["time"]
    event_body_ids = events["bodyId"]

    motor = pd.read_csv(
        CACHE / "motor_map.csv"
    )

    motor = motor.set_index(
        "bodyId"
    )

    total_steps = int(times.max()) + 1

    # -----------------------------------------------------
    # Build motor event stream
    # -----------------------------------------------------

    groups = defaultdict(
        lambda: np.zeros(
            total_steps,
            dtype=np.float32,
        )
    )

    motor_events = 0

    for t, body_id in zip(
        times,
        event_body_ids,
    ):

        body_id = int(body_id)

        if body_id not in motor.index:
            continue

        row = motor.loc[body_id]

        limb = str(row["limb"])
        side = str(row["side"])
        pair = str(row["leg_pair"])
        segment = str(row["segment"])
        action = str(row["action"])

        key = (
            limb,
            side,
            pair,
            segment,
            action,
        )

        groups[key][int(t)] += 1

        motor_events += 1

    print(
        f"Motor spike events found: "
        f"{motor_events:,}"
    )

    print(
        f"Active motor groups: "
        f"{len(groups):,}"
    )

    # -----------------------------------------------------
    # Convert spikes -> smooth activation
    # -----------------------------------------------------

    activation = {}

    for key, spikes in groups.items():

        output = np.zeros(
            total_steps,
            dtype=np.float32,
        )

        value = 0.0

        for t in range(total_steps):

            value *= args.decay

            value += (
                spikes[t]
                * args.impulse
            )

            value = min(
                value,
                1.0,
            )

            output[t] = value

        activation[key] = output

    def get_activation(
        t,
        limb,
        side=None,
        pair=None,
        segment=None,
        action=None,
    ):

        total = 0.0

        for key, values in activation.items():

            (
                k_limb,
                k_side,
                k_pair,
                k_segment,
                k_action,
            ) = key

            if k_limb != limb:
                continue

            if side is not None:
                if k_side != side:
                    continue

            if pair is not None:
                if k_pair != pair:
                    continue

            if segment is not None:
                if k_segment != segment:
                    continue

            if action is not None:
                if k_action != action:
                    continue

            total += values[t]

        return min(total, 1.0)

    # -----------------------------------------------------
    # Drawing setup
    # -----------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.set_aspect("equal")

    ax.set_xlim(
        -2.5,
        2.5,
    )

    ax.set_ylim(
        -2.0,
        2.0,
    )

    ax.axis("off")

    ax.set_title(
        "Whole Male CNS motor output"
    )

    # Fly body

    thorax = Ellipse(
        (0, 0),
        0.9,
        1.1,
    )

    abdomen = Ellipse(
        (0, -0.9),
        0.75,
        1.4,
    )

    head = Ellipse(
        (0, 0.8),
        0.7,
        0.55,
    )

    ax.add_patch(thorax)
    ax.add_patch(abdomen)
    ax.add_patch(head)

    # -----------------------------------------------------
    # Wings
    # -----------------------------------------------------

    left_wing, = ax.plot(
        [],
        [],
        linewidth=5,
    )

    right_wing, = ax.plot(
        [],
        [],
        linewidth=5,
    )

    # -----------------------------------------------------
    # Six legs
    #
    # Each has:
    # hip -> knee -> foot
    # -----------------------------------------------------

    leg_defs = {
        ("L", "front"): (
            np.array([-0.30, 0.40]),
            155,
        ),

        ("R", "front"): (
            np.array([0.30, 0.40]),
            25,
        ),

        ("L", "middle"): (
            np.array([-0.40, 0.00]),
            180,
        ),

        ("R", "middle"): (
            np.array([0.40, 0.00]),
            0,
        ),

        ("L", "hind"): (
            np.array([-0.30, -0.40]),
            210,
        ),

        ("R", "hind"): (
            np.array([0.30, -0.40]),
            -30,
        ),
    }

    leg_lines = {}

    for key in leg_defs:

        femur, = ax.plot(
            [],
            [],
            linewidth=4,
        )

        tibia, = ax.plot(
            [],
            [],
            linewidth=3,
        )

        leg_lines[key] = (
            femur,
            tibia,
        )

    status = ax.text(
        -2.35,
        1.75,
        "",
        fontsize=10,
    )

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    def leg_motor(
        t,
        side,
        pair,
        action,
    ):

        # Prefer known leg pair.
        value = get_activation(
            t,
            "leg",
            side=side,
            pair=pair,
            action=action,
        )

        # Some MNs can't yet be assigned to T1/T2/T3.
        # Let them contribute weakly to that side.
        unknown = get_activation(
            t,
            "leg",
            side=side,
            pair="unknown",
            action=action,
        )

        return min(
            1.0,
            value + 0.25 * unknown,
        )

    # -----------------------------------------------------
    # Animation
    # -----------------------------------------------------

    def update(t):

        # -------------------------------------------------
        # Wings
        # -------------------------------------------------

        left_wing_act = get_activation(
            t,
            "wing",
            side="L",
        )

        right_wing_act = get_activation(
            t,
            "wing",
            side="R",
        )

        left_angle = np.deg2rad(
            135
            - 55 * left_wing_act
        )

        right_angle = np.deg2rad(
            45
            + 55 * right_wing_act
        )

        wing_length = 1.35

        left_origin = np.array([
            -0.25,
            0.25,
        ])

        right_origin = np.array([
            0.25,
            0.25,
        ])

        left_tip = (
            left_origin
            + wing_length
            * np.array([
                np.cos(left_angle),
                np.sin(left_angle),
            ])
        )

        right_tip = (
            right_origin
            + wing_length
            * np.array([
                np.cos(right_angle),
                np.sin(right_angle),
            ])
        )

        left_wing.set_data(
            [
                left_origin[0],
                left_tip[0],
            ],
            [
                left_origin[1],
                left_tip[1],
            ],
        )

        right_wing.set_data(
            [
                right_origin[0],
                right_tip[0],
            ],
            [
                right_origin[1],
                right_tip[1],
            ],
        )

        # -------------------------------------------------
        # Legs
        # -------------------------------------------------

        max_leg_activation = 0.0

        for (
            side,
            pair
        ), (
            hip,
            base_angle_deg
        ) in leg_defs.items():

            extend = leg_motor(
                t,
                side,
                pair,
                "extend",
            )

            flex = leg_motor(
                t,
                side,
                pair,
                "flex",
            )

            lift = leg_motor(
                t,
                side,
                pair,
                "lift",
            )

            depress = leg_motor(
                t,
                side,
                pair,
                "depress",
            )

            protract = leg_motor(
                t,
                side,
                pair,
                "protract",
            )

            retract = leg_motor(
                t,
                side,
                pair,
                "retract",
            )

            generic = leg_motor(
                t,
                side,
                pair,
                "generic",
            )

            max_leg_activation = max(
                max_leg_activation,
                extend,
                flex,
                lift,
                depress,
                generic,
            )

            # Hip motion
            hip_delta = (
                25 * protract
                - 25 * retract
                + 8 * generic
            )

            if side == "L":
                hip_delta *= -1

            femur_angle = np.deg2rad(
                base_angle_deg
                + hip_delta
            )

            femur_length = 0.62

            knee = (
                hip
                + femur_length
                * np.array([
                    np.cos(
                        femur_angle
                    ),
                    np.sin(
                        femur_angle
                    ),
                ])
            )

            # Knee motion
            knee_angle_deg = (
                65
                - 55 * extend
                + 45 * flex
                - 20 * lift
                + 20 * depress
            )

            if side == "L":
                tibia_angle = (
                    femur_angle
                    - np.deg2rad(
                        knee_angle_deg
                    )
                )
            else:
                tibia_angle = (
                    femur_angle
                    + np.deg2rad(
                        knee_angle_deg
                    )
                )

            tibia_length = 0.65

            foot = (
                knee
                + tibia_length
                * np.array([
                    np.cos(
                        tibia_angle
                    ),
                    np.sin(
                        tibia_angle
                    ),
                ])
            )

            femur_line, tibia_line = (
                leg_lines[
                    (side, pair)
                ]
            )

            femur_line.set_data(
                [
                    hip[0],
                    knee[0],
                ],
                [
                    hip[1],
                    knee[1],
                ],
            )

            tibia_line.set_data(
                [
                    knee[0],
                    foot[0],
                ],
                [
                    knee[1],
                    foot[1],
                ],
            )

        status.set_text(
            f"simulation step: {t}\n"
            f"left wing: {left_wing_act:.2f}\n"
            f"right wing: {right_wing_act:.2f}\n"
            f"max leg motor activation: "
            f"{max_leg_activation:.2f}"
        )

        artists = [
            left_wing,
            right_wing,
            status,
        ]

        for lines in leg_lines.values():
            artists.extend(lines)

        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=total_steps,
        interval=args.interval,
        repeat=True,
        blit=False,
    )

    fig._animation = animation

    plt.show()


if __name__ == "__main__":
    main()

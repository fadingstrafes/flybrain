import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        required=True,
        help="Simulation run directory",
    )

    parser.add_argument(
        "--motor",
        type=int,
        default=815344,
        help="Motor neuron bodyId",
    )

    parser.add_argument(
        "--spike-gain",
        type=float,
        default=0.18,
        help="Extension added per spike",
    )

    parser.add_argument(
        "--decay",
        type=float,
        default=0.90,
        help="Muscle activation decay",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=25,
        help="Animation frame interval in ms",
    )

    args = parser.parse_args()

    run = Path(args.run)

    events_file = run / "spike_events.npz"

    if not events_file.exists():
        raise RuntimeError(
            f"No spike_events.npz found in {run}"
        )

    events = np.load(events_file)

    times = events["time"]
    body_ids = events["bodyId"]

    # ------------------------------------------------------
    # Extract this motor neuron's spike train
    # ------------------------------------------------------

    motor_times = times[
        body_ids == args.motor
    ]

    if len(motor_times) == 0:
        raise RuntimeError(
            f"Motor neuron {args.motor} never fired in this run."
        )

    total_steps = int(times.max()) + 1

    spikes = np.zeros(
        total_steps,
        dtype=np.int32,
    )

    for t in motor_times:
        spikes[int(t)] += 1

    print(
        f"Motor neuron {args.motor}: "
        f"{len(motor_times)} spikes"
    )

    # ------------------------------------------------------
    # Turn spikes into muscle activation
    #
    # This is a toy muscle model:
    #
    # spike -> contraction impulse -> gradual relaxation
    # ------------------------------------------------------

    activation = np.zeros(
        total_steps,
        dtype=np.float32,
    )

    level = 0.0

    for t in range(total_steps):

        level *= args.decay

        if spikes[t]:
            level += (
                spikes[t]
                * args.spike_gain
            )

        level = min(level, 1.0)

        activation[t] = level

    # ------------------------------------------------------
    # Joint geometry
    # ------------------------------------------------------

    rest_angle = np.deg2rad(65)
    max_extension = np.deg2rad(75)

    # Femur is fixed for this first experiment.
    femur_angle = np.deg2rad(-135)

    hip = np.array([
        -0.45,
        -0.10,
    ])

    femur_length = 0.65
    tibia_length = 0.70

    knee = hip + np.array([
        np.cos(femur_angle),
        np.sin(femur_angle),
    ]) * femur_length

    # ------------------------------------------------------
    # Figure
    # ------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.set_aspect("equal")

    ax.set_xlim(
        -2.0,
        1.3,
    )

    ax.set_ylim(
        -1.8,
        1.2,
    )

    ax.set_title(
        "Male CNS driven virtual leg"
    )

    # crude fly thorax/body
    thorax = Ellipse(
        (0, 0),
        1.0,
        0.65,
    )

    abdomen = Ellipse(
        (0.75, 0.05),
        1.15,
        0.45,
    )

    head = Ellipse(
        (-0.6, 0.05),
        0.45,
        0.45,
    )

    ax.add_patch(thorax)
    ax.add_patch(abdomen)
    ax.add_patch(head)

    # Femur
    femur_line, = ax.plot(
        [
            hip[0],
            knee[0],
        ],
        [
            hip[1],
            knee[1],
        ],
        linewidth=5,
    )

    # Tibia
    tibia_line, = ax.plot(
        [],
        [],
        linewidth=4,
    )

    knee_dot, = ax.plot(
        knee[0],
        knee[1],
        "o",
    )

    status = ax.text(
        -1.9,
        1.0,
        "",
        fontsize=11,
    )

    ax.axis("off")

    # ------------------------------------------------------
    # Animation
    # ------------------------------------------------------

    def update(frame):

        a = activation[frame]

        joint_angle = (
            rest_angle
            - a * max_extension
        )

        tibia_angle = (
            femur_angle
            + joint_angle
        )

        foot = knee + np.array([
            np.cos(tibia_angle),
            np.sin(tibia_angle),
        ]) * tibia_length

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
            f"simulation step: {frame}\n"
            f"Ti extensor activation: {a:.2f}\n"
            f"spikes this step: {spikes[frame]}"
        )

        return (
            tibia_line,
            status,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=total_steps,
        interval=args.interval,
        blit=False,
        repeat=True,
    )

    # prevent animation object from
    # getting garbage collected
    fig._animation = animation

    plt.show()


if __name__ == "__main__":
    main()

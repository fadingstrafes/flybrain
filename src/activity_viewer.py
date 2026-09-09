import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


CACHE = Path("data/cache")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def parse_location(value):
    """
    somaLocation is normally a list/array of [x,y,z].
    Return None when unavailable.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return np.array(
                [
                    float(value[0]),
                    float(value[1]),
                    float(value[2]),
                ]
            )
        except Exception:
            return None

    return None


def neuron_category(row):
    superclass = clean(
        row.get("superclass")
    ).lower()

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


def choose_nt(row):
    consensus = clean(
        row.get("consensus_nt")
    )

    if consensus:
        return consensus

    return clean(
        row.get("predicted_nt")
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Male CNS activity viewer"
    )

    parser.add_argument(
        "--run",
        required=True,
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=35,
        help="Animation delay in milliseconds",
    )

    parser.add_argument(
        "--trail",
        type=int,
        default=12,
        help="How many simulation steps remain visible as active",
    )

    parser.add_argument(
        "--max-background",
        type=int,
        default=15000,
        help="Maximum soma points in background cloud",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Number of active neurons shown in text panel",
    )

    args = parser.parse_args()

    run = Path(args.run)

    # -----------------------------------------------------------------
    # Load spike events
    # -----------------------------------------------------------------

    events_file = (
        run / "spike_events.npz"
    )

    if not events_file.exists():
        raise RuntimeError(
            f"Missing {events_file}"
        )

    events = np.load(
        events_file
    )

    times = (
        events["time"]
        .astype(np.int32)
    )

    event_body_ids = (
        events["bodyId"]
        .astype(np.int64)
    )

    if len(times) == 0:
        raise RuntimeError(
            "This run has no recorded spikes."
        )

    total_steps = int(
        times.max()
    ) + 1

    print(
        f"Spike events: {len(times):,}"
    )

    print(
        f"Simulation steps: {total_steps:,}"
    )

    # Sort events just in case

    order = np.argsort(
        times
    )

    times = times[order]
    event_body_ids = (
        event_body_ids[order]
    )

    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------

    metadata = pd.read_parquet(
        CACHE / "metadata.parquet"
    )

    metadata["bodyId"] = (
        pd.to_numeric(
            metadata["bodyId"],
            errors="coerce",
        )
    )

    metadata = (
        metadata
        .dropna(
            subset=["bodyId"]
        )
        .drop_duplicates(
            "bodyId"
        )
    )

    metadata["bodyId"] = (
        metadata["bodyId"]
        .astype(np.int64)
    )

    metadata_lookup = (
        metadata
        .set_index("bodyId")
    )

    print(
        f"Metadata neurons: "
        f"{len(metadata):,}"
    )

    # -----------------------------------------------------------------
    # Build bodyId -> category lookup
    # -----------------------------------------------------------------

    category_names = [
        "sensory",
        "ascending",
        "descending",
        "intrinsic",
        "motor",
        "other",
    ]

    category_to_index = {
        name: i
        for i, name
        in enumerate(
            category_names
        )
    }

    body_category = {}

    for _, row in metadata.iterrows():

        body_id = int(
            row["bodyId"]
        )

        body_category[
            body_id
        ] = neuron_category(
            row
        )

    # -----------------------------------------------------------------
    # Population activity per timestep
    # -----------------------------------------------------------------

    population = np.zeros(
        (
            total_steps,
            len(category_names),
        ),
        dtype=np.int32,
    )

    for t, body_id in zip(
        times,
        event_body_ids,
    ):

        category = body_category.get(
            int(body_id),
            "other",
        )

        c = category_to_index[
            category
        ]

        population[t, c] += 1

    # Smoothed activity for plotting

    smoothing = 10

    kernel = np.ones(
        smoothing,
        dtype=float,
    )

    population_smooth = np.zeros_like(
        population,
        dtype=float,
    )

    for i in range(
        len(category_names)
    ):

        population_smooth[:, i] = (
            np.convolve(
                population[:, i],
                kernel,
                mode="same",
            )
        )

    # -----------------------------------------------------------------
    # Soma coordinate lookup
    # -----------------------------------------------------------------

    soma_lookup = {}

    soma_points = []

    for _, row in metadata.iterrows():

        body_id = int(
            row["bodyId"]
        )

        location = parse_location(
            row.get(
                "somaLocation"
            )
        )

        if location is None:
            location = parse_location(
                row.get(
                    "tosomaLocation"
                )
            )

        if location is None:
            continue

        soma_lookup[
            body_id
        ] = location

        soma_points.append(
            location
        )

    soma_points = np.array(
        soma_points
    )

    print(
        f"Neurons with soma coordinates: "
        f"{len(soma_points):,}"
    )

    # Sample background cloud

    rng = np.random.default_rng(
        42
    )

    if (
        len(soma_points)
        > args.max_background
    ):

        sample_idx = rng.choice(
            len(soma_points),
            size=args.max_background,
            replace=False,
        )

        background = (
            soma_points[
                sample_idx
            ]
        )

    else:

        background = soma_points

    # Robust axis bounds

    low = np.percentile(
        soma_points,
        1,
        axis=0,
    )

    high = np.percentile(
        soma_points,
        99,
        axis=0,
    )

    # -----------------------------------------------------------------
    # Motor map
    # -----------------------------------------------------------------

    motor_file = (
        CACHE / "motor_map.csv"
    )

    motor_lookup = {}

    if motor_file.exists():

        motor = pd.read_csv(
            motor_file
        )

        motor["bodyId"] = (
            motor["bodyId"]
            .astype(np.int64)
        )

        motor_lookup = (
            motor
            .set_index("bodyId")
            .to_dict("index")
        )

    motor_groups = [
        "L front",
        "L middle",
        "L hind",
        "R front",
        "R middle",
        "R hind",
        "wing",
        "haltere",
        "abdomen",
        "other",
    ]

    motor_group_index = {
        name: i
        for i, name
        in enumerate(
            motor_groups
        )
    }

    motor_spikes = np.zeros(
        (
            total_steps,
            len(motor_groups),
        ),
        dtype=np.float32,
    )

    def get_motor_group(
        body_id
    ):

        row = motor_lookup.get(
            int(body_id)
        )

        if row is None:
            return None

        limb = clean(
            row.get("limb")
        )

        side = clean(
            row.get("side")
        )

        pair = clean(
            row.get("leg_pair")
        )

        if limb == "leg":

            label = (
                f"{side} {pair}"
            )

            if label in motor_group_index:
                return label

            return "other"

        if limb == "wing":
            return "wing"

        if limb == "haltere":
            return "haltere"

        if limb == "abdomen":
            return "abdomen"

        return "other"

    for t, body_id in zip(
        times,
        event_body_ids,
    ):

        group = get_motor_group(
            body_id
        )

        if group is None:
            continue

        motor_spikes[
            t,
            motor_group_index[
                group
            ],
        ] += 1

    # Smooth motor activation

    motor_activation = np.zeros_like(
        motor_spikes,
        dtype=np.float32,
    )

    decay = 0.90

    level = np.zeros(
        len(motor_groups),
        dtype=np.float32,
    )

    for t in range(
        total_steps
    ):

        level *= decay

        level += (
            motor_spikes[t]
            * 0.15
        )

        level = np.clip(
            level,
            0,
            1,
        )

        motor_activation[t] = level

    # -----------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------

    fig = plt.figure(
        figsize=(15, 9)
    )

    ax_brain = fig.add_subplot(
        2,
        2,
        1,
        projection="3d",
    )

    ax_population = (
        fig.add_subplot(
            2,
            2,
            2,
        )
    )

    ax_motor = (
        fig.add_subplot(
            2,
            2,
            3,
        )
    )

    ax_text = (
        fig.add_subplot(
            2,
            2,
            4,
        )
    )

    fig.suptitle(
        f"Male CNS activity — {run.name}"
    )

    # -----------------------------------------------------------------
    # 3D brain/CNS panel
    # -----------------------------------------------------------------

    ax_brain.set_title(
        "Active neuron somata"
    )

    ax_brain.scatter(
        background[:, 0],
        background[:, 1],
        background[:, 2],
        s=1,
        alpha=0.05,
    )

    active_scatter = (
        ax_brain.scatter(
            [],
            [],
            [],
            s=18,
        )
    )

    ax_brain.set_xlim(
        low[0],
        high[0],
    )

    ax_brain.set_ylim(
        low[1],
        high[1],
    )

    ax_brain.set_zlim(
        low[2],
        high[2],
    )

    ax_brain.set_xticks([])
    ax_brain.set_yticks([])
    ax_brain.set_zticks([])

    # -----------------------------------------------------------------
    # Population panel
    # -----------------------------------------------------------------

    ax_population.set_title(
        "Population activity"
    )

    x = np.arange(
        total_steps
    )

    for i, name in enumerate(
        category_names
    ):

        ax_population.plot(
            x,
            population_smooth[
                :, i
            ],
            label=name,
        )

    population_cursor = (
        ax_population.axvline(
            0
        )
    )

    ax_population.set_xlabel(
        "simulation step"
    )

    ax_population.set_ylabel(
        "spikes / rolling window"
    )

    ax_population.legend(
        fontsize=8,
        ncol=2,
    )

    # -----------------------------------------------------------------
    # Motor panel
    # -----------------------------------------------------------------

    ax_motor.set_title(
        "Motor output"
    )

    bars = ax_motor.bar(
        motor_groups,
        np.zeros(
            len(motor_groups)
        ),
    )

    ax_motor.set_ylim(
        0,
        1.05,
    )

    ax_motor.set_ylabel(
        "activation"
    )

    ax_motor.tick_params(
        axis="x",
        rotation=45,
    )

    # -----------------------------------------------------------------
    # Active neuron text
    # -----------------------------------------------------------------

    ax_text.axis(
        "off"
    )

    active_text = ax_text.text(
        0.02,
        0.98,
        "",
        transform=ax_text.transAxes,
        va="top",
        family="monospace",
        fontsize=8,
    )

    # -----------------------------------------------------------------
    # Playback
    # -----------------------------------------------------------------

    state = {
        "paused": False,
        "frame": 0,
    }

    def events_in_window(
        frame
    ):

        start_t = max(
            0,
            frame - args.trail
        )

        left = np.searchsorted(
            times,
            start_t,
            side="left",
        )

        right = np.searchsorted(
            times,
            frame + 1,
            side="left",
        )

        return event_body_ids[
            left:right
        ]

    def update(frame):

        state["frame"] = frame

        ids = events_in_window(
            frame
        )

        # -------------------------------------------------------------
        # Active soma coordinates
        # -------------------------------------------------------------

        coords = []

        for body_id in np.unique(
            ids
        ):

            location = soma_lookup.get(
                int(body_id)
            )

            if location is not None:
                coords.append(
                    location
                )

        if coords:

            coords = np.array(
                coords
            )

            active_scatter._offsets3d = (
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
            )

        else:

            active_scatter._offsets3d = (
                [],
                [],
                [],
            )

        # -------------------------------------------------------------
        # Timeline cursor
        # -------------------------------------------------------------

        population_cursor.set_xdata(
            [frame, frame]
        )

        # -------------------------------------------------------------
        # Motor bars
        # -------------------------------------------------------------

        for bar, value in zip(
            bars,
            motor_activation[
                frame
            ],
        ):

            bar.set_height(
                float(value)
            )

        # -------------------------------------------------------------
        # Current active neurons
        # -------------------------------------------------------------

        if len(ids):

            unique_ids, counts = (
                np.unique(
                    ids,
                    return_counts=True,
                )
            )

            ranking = np.argsort(
                counts
            )[::-1]

            lines = [
                f"step {frame}",
                "",
                "ACTIVE NEURONS",
                "-" * 62,
            ]

            for idx in ranking[
                :args.top
            ]:

                body_id = int(
                    unique_ids[
                        idx
                    ]
                )

                count = int(
                    counts[
                        idx
                    ]
                )

                if (
                    body_id
                    in metadata_lookup.index
                ):

                    row = (
                        metadata_lookup.loc[
                            body_id
                        ]
                    )

                    instance = clean(
                        row.get(
                            "instance"
                        )
                    )

                    superclass = clean(
                        row.get(
                            "superclass"
                        )
                    )

                    nt = choose_nt(
                        row
                    )

                else:

                    instance = ""
                    superclass = ""
                    nt = ""

                lines.append(
                    f"{body_id:<9} "
                    f"{instance[:22]:<22} "
                    f"{superclass[:16]:<16} "
                    f"{nt[:12]:<12} "
                    f"x{count}"
                )

        else:

            lines = [
                f"step {frame}",
                "",
                "No spikes in recent trail.",
            ]

        active_text.set_text(
            "\n".join(
                lines
            )
        )

        return [
            active_scatter,
            population_cursor,
            active_text,
            *bars,
        ]

    animation = FuncAnimation(
        fig,
        update,
        frames=total_steps,
        interval=args.interval,
        repeat=True,
        blit=False,
    )

    # -----------------------------------------------------------------
    # Keyboard controls
    # -----------------------------------------------------------------

    def keypress(event):

        if event.key == " ":

            state["paused"] = (
                not state[
                    "paused"
                ]
            )

            if state["paused"]:
                animation.pause()
            else:
                animation.resume()

        elif event.key == "right":

            animation.pause()

            next_frame = min(
                total_steps - 1,
                state["frame"] + 1,
            )

            update(
                next_frame
            )

            fig.canvas.draw_idle()

        elif event.key == "left":

            animation.pause()

            previous_frame = max(
                0,
                state["frame"] - 1,
            )

            update(
                previous_frame
            )

            fig.canvas.draw_idle()

    fig.canvas.mpl_connect(
        "key_press_event",
        keypress,
    )

    fig._animation = animation

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

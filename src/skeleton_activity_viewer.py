import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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
        default=35,
    )

    parser.add_argument(
        "--trail",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--inactive-alpha",
        type=float,
        default=0.035,
    )

    parser.add_argument(
        "--active-alpha",
        type=float,
        default=0.95,
    )

    args = parser.parse_args()

    run = Path(args.run)

    skeleton_dir = (
        run / "skeleton_cache"
    )

    selected_file = (
        skeleton_dir
        / "selected.csv"
    )

    if not selected_file.exists():

        raise RuntimeError(
            "No skeleton cache found.\n"
            "Run cache_run_skeletons first."
        )

    selected = pd.read_csv(
        selected_file
    )

    selected["bodyId"] = (
        selected["bodyId"]
        .astype(np.int64)
    )

    # --------------------------------------------------
    # Spike events
    # --------------------------------------------------

    events = np.load(
        run / "spike_events.npz"
    )

    times = (
        events["time"]
        .astype(np.int32)
    )

    body_ids = (
        events["bodyId"]
        .astype(np.int64)
    )

    order = np.argsort(
        times
    )

    times = times[order]
    body_ids = body_ids[order]

    total_steps = int(
        times.max()
    ) + 1

    selected_ids = set(
        selected["bodyId"]
        .astype(int)
    )

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

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

    lookup = metadata.set_index(
        "bodyId"
    )

    # --------------------------------------------------
    # Load cached morphology
    # --------------------------------------------------

    skeletons = {}

    all_points = []

    for body_id in selected[
        "bodyId"
    ]:

        body_id = int(body_id)

        path = (
            skeleton_dir
            / f"{body_id}.npz"
        )

        if not path.exists():
            continue

        data = np.load(
            path
        )

        segments = data[
            "segments"
        ].astype(
            np.float32
        )

        if len(segments) == 0:
            continue

        skeletons[
            body_id
        ] = segments

        # sample geometry for determining bounds

        stride = max(
            1,
            len(segments) // 1000,
        )

        all_points.append(
            segments[
                ::stride
            ].reshape(
                -1,
                3,
            )
        )

    if not skeletons:

        raise RuntimeError(
            "No cached skeletons could be loaded."
        )

    points = np.concatenate(
        all_points,
        axis=0,
    )

    low = np.percentile(
        points,
        1,
        axis=0,
    )

    high = np.percentile(
        points,
        99,
        axis=0,
    )

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig = plt.figure(
        figsize=(14, 9)
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
        projection="3d",
    )

    ax.set_title(
        f"Real Male CNS neuron activity — {run.name}"
    )

    ax.set_xlim(
        low[0],
        high[0],
    )

    ax.set_ylim(
        low[1],
        high[1],
    )

    ax.set_zlim(
        low[2],
        high[2],
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.view_init(
        elev=20,
        azim=-70,
    )

    # --------------------------------------------------
    # One Line3DCollection per neuron
    # --------------------------------------------------

    artists = {}

    for body_id, segments in (
        skeletons.items()
    ):

        collection = (
            Line3DCollection(
                segments,
                linewidths=0.35,
                alpha=args.inactive_alpha,
            )
        )

        ax.add_collection3d(
            collection
        )

        artists[
            body_id
        ] = collection

    # --------------------------------------------------
    # Text overlay
    # --------------------------------------------------

    status = ax.text2D(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
    )

    # --------------------------------------------------
    # Activity window
    # --------------------------------------------------

    def active_ids_at(
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

        ids = body_ids[
            left:right
        ]

        unique, counts = np.unique(
            ids,
            return_counts=True,
        )

        result = {}

        for body_id, count in zip(
            unique,
            counts,
        ):

            body_id = int(
                body_id
            )

            if body_id in selected_ids:
                result[
                    body_id
                ] = int(
                    count
                )

        return result

    # --------------------------------------------------
    # Update
    # --------------------------------------------------

    def update(frame):

        active = active_ids_at(
            frame
        )

        # dim everyone

        for body_id, artist in (
            artists.items()
        ):

            if body_id in active:

                count = active[
                    body_id
                ]

                linewidth = min(
                    2.5,
                    0.7
                    + count * 0.25
                )

                artist.set_alpha(
                    args.active_alpha
                )

                artist.set_linewidth(
                    linewidth
                )

            else:

                artist.set_alpha(
                    args.inactive_alpha
                )

                artist.set_linewidth(
                    0.35
                )

        # --------------------------------------------------
        # text
        # --------------------------------------------------

        ranking = sorted(
            active.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        lines = [
            f"step {frame}",
            "",
            "ACTIVE CACHED NEURONS",
            "-" * 60,
        ]

        if not ranking:

            lines.append(
                "none"
            )

        for body_id, count in (
            ranking[:12]
        ):

            if body_id in lookup.index:

                row = lookup.loc[
                    body_id
                ]

                instance = str(
                    row.get(
                        "instance",
                        ""
                    )
                )

                superclass = str(
                    row.get(
                        "superclass",
                        ""
                    )
                )

            else:

                instance = ""
                superclass = ""

            lines.append(
                f"{body_id:<9} "
                f"{instance[:22]:<22} "
                f"{superclass[:18]:<18} "
                f"x{count}"
            )

        status.set_text(
            "\n".join(
                lines
            )
        )

        return [
            *artists.values(),
            status,
        ]

    # --------------------------------------------------
    # Animation
    # --------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=total_steps,
        interval=args.interval,
        repeat=True,
        blit=False,
    )

    fig._animation = animation

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

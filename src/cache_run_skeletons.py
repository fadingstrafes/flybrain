import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from navis.interfaces import neuprint as navis_neuprint


CACHE = Path("data/cache")


def neuron_category(row):
    superclass = str(
        row.get("superclass", "")
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


def tree_to_segments(neuron):
    """
    Convert a navis TreeNeuron into line segments:

    parent ---- child

    Result shape:
        (N, 2, 3)
    """

    nodes = neuron.nodes.copy()

    coords = nodes.set_index(
        "node_id"
    )[
        ["x", "y", "z"]
    ]

    segments = []

    for _, row in nodes.iterrows():

        parent = row["parent_id"]

        if pd.isna(parent):
            continue

        parent = int(parent)

        if parent < 0:
            continue

        if parent not in coords.index:
            continue

        child_xyz = np.array(
            [
                row["x"],
                row["y"],
                row["z"],
            ],
            dtype=np.float32,
        )

        parent_xyz = (
            coords.loc[parent]
            .to_numpy(
                dtype=np.float32
            )
        )

        segments.append(
            [
                parent_xyz,
                child_xyz,
            ]
        )

    if not segments:
        return np.empty(
            (0, 2, 3),
            dtype=np.float32,
        )

    return np.asarray(
        segments,
        dtype=np.float32,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        required=True,
    )

    parser.add_argument(
        "--max-neurons",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--min-spikes",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    run = Path(args.run)

    events = np.load(
        run / "spike_events.npz"
    )

    body_ids = (
        events["bodyId"]
        .astype(np.int64)
    )

    unique_ids, counts = np.unique(
        body_ids,
        return_counts=True,
    )

    metadata = pd.read_parquet(
        CACHE / "metadata.parquet"
    )

    metadata["bodyId"] = pd.to_numeric(
        metadata["bodyId"],
        errors="coerce",
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

    rows = []

    for body_id, count in zip(
        unique_ids,
        counts,
    ):

        if count < args.min_spikes:
            continue

        body_id = int(body_id)

        if body_id in lookup.index:

            row = lookup.loc[
                body_id
            ]

            instance = row.get(
                "instance"
            )

            superclass = row.get(
                "superclass"
            )

            category = neuron_category(
                row
            )

        else:

            instance = ""
            superclass = ""
            category = "other"

        rows.append({
            "bodyId": body_id,
            "spikes": int(count),
            "instance": instance,
            "superclass": superclass,
            "category": category,
        })

    ranked = pd.DataFrame(rows)

    ranked = (
        ranked
        .sort_values(
            "spikes",
            ascending=False,
        )
    )

    # --------------------------------------------------
    # Avoid filling all slots with only one cell class.
    #
    # Reserve some space for sensory / descending /
    # motor cells, then fill remaining slots globally.
    # --------------------------------------------------

    selected_ids = []

    categories = [
        "sensory",
        "descending",
        "motor",
        "ascending",
        "intrinsic",
    ]

    per_category = max(
        2,
        args.max_neurons // 8,
    )

    for category in categories:

        subset = ranked[
            ranked["category"]
            == category
        ]

        for body_id in (
            subset
            .head(per_category)
            ["bodyId"]
        ):

            body_id = int(body_id)

            if body_id not in selected_ids:
                selected_ids.append(
                    body_id
                )

    # fill remaining slots with strongest overall cells

    for body_id in ranked[
        "bodyId"
    ]:

        body_id = int(body_id)

        if body_id not in selected_ids:
            selected_ids.append(
                body_id
            )

        if (
            len(selected_ids)
            >= args.max_neurons
        ):
            break

    selected_ids = selected_ids[
        :args.max_neurons
    ]

    selected = ranked[
        ranked["bodyId"].isin(
            selected_ids
        )
    ].copy()

    # preserve selected order

    selected["order"] = (
        selected["bodyId"]
        .map({
            body_id: i
            for i, body_id
            in enumerate(
                selected_ids
            )
        })
    )

    selected = (
        selected
        .sort_values("order")
        .drop(
            columns=["order"]
        )
    )

    print()
    print("=" * 80)
    print("SKELETON SELECTION")
    print("=" * 80)

    print(
        selected.to_string(
            index=False
        )
    )

    output = (
        run
        / "skeleton_cache"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        output / "selected.csv",
        index=False,
    )

    # --------------------------------------------------
    # neuPrint connection
    # --------------------------------------------------

    token = os.environ[
        "NEUPRINT_APPLICATION_CREDENTIALS"
    ]

    client = navis_neuprint.Client(
        "https://neuprint.janelia.org",
        dataset="male-cns:v1.0",
        token=token,
    )

    # --------------------------------------------------
    # Fetch in small batches
    # --------------------------------------------------

    batch_size = 10

    for start in range(
        0,
        len(selected_ids),
        batch_size,
    ):

        batch = selected_ids[
            start:
            start + batch_size
        ]

        missing = []

        for body_id in batch:

            path = (
                output
                / f"{body_id}.npz"
            )

            if (
                args.force
                or not path.exists()
            ):
                missing.append(
                    body_id
                )

        if not missing:
            print(
                f"Batch {start}: cached"
            )
            continue

        print()
        print(
            f"Fetching skeletons: "
            f"{missing}"
        )

        neurons = (
            navis_neuprint
            .fetch_skeletons(
                missing,
                client=client,
            )
        )

        for neuron in neurons:

            body_id = int(
                neuron.id
            )

            segments = tree_to_segments(
                neuron
            )

            path = (
                output
                / f"{body_id}.npz"
            )

            np.savez_compressed(
                path,
                segments=segments,
            )

            print(
                f"{body_id}: "
                f"{len(segments):,} "
                f"segments"
            )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Skeleton cache: {output}"
    )


if __name__ == "__main__":
    main()

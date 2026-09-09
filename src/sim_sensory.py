import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


CACHE = Path("data/cache")


def body_to_index(neuron_ids, body_id):
    idx = np.searchsorted(neuron_ids, body_id)

    if idx >= len(neuron_ids):
        return None

    if neuron_ids[idx] != body_id:
        return None

    return int(idx)


def main():

    parser = argparse.ArgumentParser(
        description="Stimulate annotated MaleCNS sensory populations"
    )

    parser.add_argument(
        "--system",
        required=True,
        choices=[
            "front_leg",
            "middle_leg",
            "hind_leg",
            "wing",
            "haltere",
            "abdomen",
            "other",
        ],
    )

    parser.add_argument(
        "--side",
        choices=["L", "R", "?"],
        default=None,
    )

    parser.add_argument(
        "--subclass",
        default=None,
        help='Example: "leg bristle" or "chordotonal organ"',
    )

    parser.add_argument(
        "--type",
        default=None,
        help="Optional exact/substring cell-type filter",
    )

    parser.add_argument(
        "--max-neurons",
        type=int,
        default=50,
        help="Maximum sensory neurons to stimulate",
    )

    parser.add_argument(
        "--selection",
        choices=[
            "strongest",
            "random",
            "all",
        ],
        default="strongest",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Parameters forwarded to sim_full.py

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--stim-start",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--stim-end",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--stim-current",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--min-weight",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--synaptic-gain",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--leak",
        type=float,
        default=0.92,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load sensory annotations
    # -----------------------------------------------------

    sensory = pd.read_csv(
        CACHE / "sensory_map.csv"
    )

    selected = sensory[
        sensory["system"] == args.system
    ].copy()

    if args.side is not None:
        selected = selected[
            selected["side"] == args.side
        ]

    if args.subclass is not None:

        selected = selected[
            selected["subclass"]
            .fillna("")
            .astype(str)
            .str.contains(
                args.subclass,
                case=False,
                regex=False,
            )
        ]

    if args.type is not None:

        selected = selected[
            selected["type"]
            .fillna("")
            .astype(str)
            .str.contains(
                args.type,
                case=False,
                regex=False,
            )
        ]

    if selected.empty:
        print("No sensory neurons matched those filters.")

        print("\nAvailable subclasses for this system:")

        base = sensory[
            sensory["system"] == args.system
        ]

        print(
            base["subclass"]
            .fillna("unknown")
            .value_counts()
            .head(30)
            .to_string()
        )

        sys.exit(1)

    print()
    print("=" * 80)
    print("SENSORY POPULATION")
    print("=" * 80)

    print(f"System:        {args.system}")
    print(f"Side:          {args.side}")
    print(f"Subclass:      {args.subclass}")
    print(f"Candidates:    {len(selected):,}")

    # -----------------------------------------------------
    # Select which sensory cells to stimulate
    # -----------------------------------------------------

    if (
        args.selection == "all"
        or len(selected) <= args.max_neurons
    ):

        chosen = selected.copy()

    elif args.selection == "random":

        chosen = selected.sample(
            n=args.max_neurons,
            random_state=args.seed,
        )

    else:
        # -------------------------------------------------
        # Rank sensory neurons by total outgoing anatomical
        # synaptic weight in the complete connectome.
        # -------------------------------------------------

        print(
            "\nRanking candidates by total outgoing "
            "connectome weight..."
        )

        neuron_ids = np.load(
            CACHE / "neuron_ids.npy",
            mmap_mode="r",
        )

        W = load_npz(
            CACHE / "connectome_raw.npz"
        ).tocsr()

        scores = []

        for body_id in selected["bodyId"]:

            idx = body_to_index(
                neuron_ids,
                int(body_id),
            )

            if idx is None:
                scores.append(0)
                continue

            start = W.indptr[idx]
            end = W.indptr[idx + 1]

            score = int(
                W.data[start:end].sum()
            )

            scores.append(score)

        selected["outgoing_weight"] = scores

        chosen = (
            selected
            .sort_values(
                "outgoing_weight",
                ascending=False,
            )
            .head(args.max_neurons)
            .copy()
        )

    chosen_ids = (
        chosen["bodyId"]
        .astype(int)
        .tolist()
    )

    print(
        f"Stimulating:   {len(chosen_ids):,} neurons"
    )

    print("\nSelected sensory neurons:")
    print("-" * 80)

    show_cols = [
        "bodyId",
        "instance",
        "type",
        "side",
        "subclass",
    ]

    if "outgoing_weight" in chosen.columns:
        show_cols.append(
            "outgoing_weight"
        )

    print(
        chosen[show_cols]
        .head(100)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # Save exactly which sensory neurons we used
    # -----------------------------------------------------

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    chosen.to_csv(
        output / "stimulated_sensory_neurons.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Launch the existing whole-CNS simulator
    # -----------------------------------------------------

    command = [
        sys.executable,
        "-m",
        "src.sim_full",

        "--stimulate",
        *[str(x) for x in chosen_ids],

        "--steps",
        str(args.steps),

        "--stim-start",
        str(args.stim_start),

        "--stim-end",
        str(args.stim_end),

        "--stim-current",
        str(args.stim_current),

        "--min-weight",
        str(args.min_weight),

        "--synaptic-gain",
        str(args.synaptic_gain),

        "--threshold",
        str(args.threshold),

        "--leak",
        str(args.leak),

        "--output",
        str(output),
    ]

    print()
    print("=" * 80)
    print("STARTING WHOLE-CNS SIMULATION")
    print("=" * 80)

    subprocess.run(
        command,
        check=True,
    )


if __name__ == "__main__":
    main()

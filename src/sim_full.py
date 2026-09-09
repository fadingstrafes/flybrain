import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


CACHE = Path("data/cache")


def body_to_index(neuron_ids, body_id):
    """
    Convert a real MaleCNS bodyId into our compact matrix index.
    """

    i = np.searchsorted(neuron_ids, body_id)

    if i >= len(neuron_ids):
        return None

    if neuron_ids[i] != body_id:
        return None

    return int(i)


def main():

    parser = argparse.ArgumentParser(
        description="Whole Male CNS connectome activity simulator"
    )

    parser.add_argument(
        "--stimulate",
        type=int,
        nargs="+",
        required=True,
        help="One or more bodyIds to externally stimulate",
    )

    parser.add_argument(
        "--target",
        type=int,
        nargs="*",
        default=[815344],
        help="Neurons to specifically monitor",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--stim-start",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--stim-end",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--stim-current",
        type=float,
        default=1.25,
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
        "--synaptic-gain",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--min-weight",
        type=int,
        default=20,
        help="Ignore connections weaker than this many synapses",
    )

    parser.add_argument(
        "--abort-fraction",
        type=float,
        default=0.20,
        help=(
            "Abort if this fraction of the entire CNS fires "
            "simultaneously. Set 0 to disable."
        ),
    )

    parser.add_argument(
        "--max-recorded-events",
        type=int,
        default=2_000_000,
        help="Maximum individual spike events to save",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/full_run",
    )

    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load complete CNS
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING WHOLE CONNECTOME")
    print("=" * 80)

    neuron_ids = np.load(
        CACHE / "neuron_ids.npy",
        mmap_mode="r",
    )

    nt_sign = np.load(
        CACHE / "nt_sign.npy",
        mmap_mode="r",
    )

    nt_label = np.load(
        CACHE / "nt_label.npy",
        mmap_mode="r",
    )

    W = load_npz(
        CACHE / "connectome_raw.npz"
    ).tocsr()

    W.sort_indices()

    N = len(neuron_ids)

    print(f"Neurons:     {N:,}")
    print(f"Connections: {W.nnz:,}")
    print(
        f"Synapses:    {int(W.sum()):,}"
    )

    print(
        f"Known NT sign: "
        f"{np.count_nonzero(nt_sign):,} / {N:,}"
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = pd.read_parquet(
        CACHE / "metadata.parquet"
    )

    metadata["bodyId"] = pd.to_numeric(
        metadata["bodyId"],
        errors="coerce",
    )

    metadata = (
        metadata
        .dropna(subset=["bodyId"])
        .drop_duplicates("bodyId")
    )

    metadata["bodyId"] = metadata[
        "bodyId"
    ].astype(np.int64)

    metadata_lookup = metadata.set_index(
        "bodyId"
    )

    # ---------------------------------------------------------
    # Find row maximums
    #
    # We normalize each neuron's outgoing connections relative
    # to its own strongest anatomical connection.
    # ---------------------------------------------------------

    print("\nCalculating outgoing normalization...")

    row_sizes = np.diff(W.indptr)

    row_max = np.zeros(
        N,
        dtype=np.float32,
    )

    nonempty = row_sizes > 0

    starts = W.indptr[:-1][nonempty]

    row_max[nonempty] = np.maximum.reduceat(
        W.data,
        starts,
    ).astype(np.float32)

    # ---------------------------------------------------------
    # Stimulated neurons
    # ---------------------------------------------------------

    stim_indices = []

    print()
    print("=" * 80)
    print("STIMULUS")
    print("=" * 80)

    for body_id in args.stimulate:

        idx = body_to_index(
            neuron_ids,
            body_id,
        )

        if idx is None:
            raise RuntimeError(
                f"bodyId {body_id} is not in the graph"
            )

        stim_indices.append(idx)

        if body_id in metadata_lookup.index:

            row = metadata_lookup.loc[body_id]

            name = row.get("instance")
            nt = row.get("consensus_nt")

            if pd.isna(nt):
                nt = row.get("predicted_nt")

        else:

            name = None
            nt = None

        print(
            f"{body_id:<10} "
            f"{str(name):<30} "
            f"NT={nt}"
        )

    stim_indices = np.array(
        stim_indices,
        dtype=np.int32,
    )

    # ---------------------------------------------------------
    # Targets
    # ---------------------------------------------------------

    target_indices = {}

    print("\nTARGETS")
    print("-" * 80)

    for body_id in args.target:

        idx = body_to_index(
            neuron_ids,
            body_id,
        )

        if idx is None:
            print(
                f"{body_id}: not found"
            )

            continue

        target_indices[body_id] = idx

        if body_id in metadata_lookup.index:
            name = metadata_lookup.loc[
                body_id
            ].get("instance")
        else:
            name = None

        print(
            f"{body_id:<10} {name}"
        )

    # ---------------------------------------------------------
    # State
    # ---------------------------------------------------------

    voltage = np.zeros(
        N,
        dtype=np.float32,
    )

    spike_counts = np.zeros(
        N,
        dtype=np.int32,
    )

    previous_spikes = np.empty(
        0,
        dtype=np.int32,
    )

    recorded_times = []
    recorded_indices = []

    recorded_event_count = 0
    recording_enabled = True

    # ---------------------------------------------------------
    # Simulation
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("RUNNING WHOLE-CNS SIMULATION")
    print("=" * 80)

    print(
        f"Steps:            {args.steps}"
    )

    print(
        f"Minimum weight:   {args.min_weight}"
    )

    print(
        f"Synaptic gain:    {args.synaptic_gain}"
    )

    print(
        f"Threshold:        {args.threshold}"
    )

    print(
        f"Leak:             {args.leak}"
    )

    for t in range(args.steps):

        # -----------------------------------------------------
        # Leak
        # -----------------------------------------------------

        voltage *= args.leak

        # -----------------------------------------------------
        # Propagate spikes from previous timestep
        #
        # IMPORTANT:
        # We DO NOT multiply the entire 166k x 166k graph.
        #
        # We only visit rows belonging to neurons that actually
        # fired on the previous timestep.
        # -----------------------------------------------------

        for pre_idx in previous_spikes:

            sign = nt_sign[pre_idx]

            # Unknown transmitter/sign:
            # don't invent an effect.
            if sign == 0:
                continue

            maximum = row_max[pre_idx]

            if maximum <= 0:
                continue

            start = W.indptr[pre_idx]
            end = W.indptr[pre_idx + 1]

            targets = W.indices[
                start:end
            ]

            weights = W.data[
                start:end
            ]

            # Runtime connection filter
            if args.min_weight > 1:

                keep = (
                    weights >= args.min_weight
                )

                if not np.any(keep):
                    continue

                targets = targets[keep]
                weights = weights[keep]

            relative = (
                weights.astype(np.float32)
                / maximum
            )

            current = (
                relative
                * args.synaptic_gain
                * sign
            )

            np.add.at(
                voltage,
                targets,
                current,
            )

        # -----------------------------------------------------
        # External stimulus
        # -----------------------------------------------------

        if (
            args.stim_start
            <= t
            < args.stim_end
        ):
            voltage[stim_indices] += (
                args.stim_current
            )

        # -----------------------------------------------------
        # Threshold
        # -----------------------------------------------------

        fired = np.flatnonzero(
            voltage >= args.threshold
        ).astype(np.int32)

        spike_counts[fired] += 1

        # reset
        voltage[fired] = 0.0

        # -----------------------------------------------------
        # Save events
        # -----------------------------------------------------

        if (
            recording_enabled
            and len(fired) > 0
        ):

            new_total = (
                recorded_event_count
                + len(fired)
            )

            if (
                new_total
                <= args.max_recorded_events
            ):

                recorded_times.append(
                    np.full(
                        len(fired),
                        t,
                        dtype=np.int32,
                    )
                )

                recorded_indices.append(
                    fired.copy()
                )

                recorded_event_count = (
                    new_total
                )

            else:

                recording_enabled = False

                print(
                    "\nEvent recording limit reached; "
                    "continuing simulation without "
                    "recording every spike."
                )

        previous_spikes = fired

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        if (
            t % 25 == 0
            or t == args.steps - 1
        ):

            print(
                f"step {t:>5}/{args.steps}  "
                f"active={len(fired):>7,}  "
                f"ever-active="
                f"{np.count_nonzero(spike_counts):>7,}"
            )

        # -----------------------------------------------------
        # Avalanche protection
        # -----------------------------------------------------

        if (
            args.abort_fraction > 0
            and len(fired)
            > N * args.abort_fraction
        ):

            print()
            print(
                "SIMULATION ABORTED:"
            )

            print(
                f"{len(fired):,} neurons fired "
                "in one timestep."
            )

            print(
                "The toy dynamics entered a "
                "whole-network activity avalanche."
            )

            print(
                "Try a higher --min-weight, "
                "lower --synaptic-gain, or "
                "higher --threshold."
            )

            break

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    active = np.flatnonzero(
        spike_counts
    )

    print(
        f"Neurons that fired: "
        f"{len(active):,} / {N:,}"
    )

    print(
        f"Total spikes: "
        f"{int(spike_counts.sum()):,}"
    )

    # ---------------------------------------------------------
    # Top active neurons
    # ---------------------------------------------------------

    ranked = active[
        np.argsort(
            spike_counts[active]
        )[::-1]
    ]

    print("\nTOP ACTIVE NEURONS")
    print("-" * 80)

    rows = []

    for idx in ranked[:50]:

        body_id = int(
            neuron_ids[idx]
        )

        count = int(
            spike_counts[idx]
        )

        if body_id in metadata_lookup.index:

            row = metadata_lookup.loc[
                body_id
            ]

            instance = row.get(
                "instance"
            )

            superclass = row.get(
                "superclass"
            )

        else:

            instance = None
            superclass = None

        rows.append({
            "bodyId": body_id,
            "instance": instance,
            "superclass": superclass,
            "spikes": count,
        })

        print(
            f"{body_id:<10} "
            f"{str(instance):<32} "
            f"{count:>7} spikes"
        )

    pd.DataFrame(rows).to_csv(
        output / "top_active.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Target activity
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("TARGET ACTIVITY")
    print("=" * 80)

    for body_id, idx in target_indices.items():

        count = int(
            spike_counts[idx]
        )

        if body_id in metadata_lookup.index:

            name = metadata_lookup.loc[
                body_id
            ].get("instance")

        else:

            name = None

        print(
            f"{body_id:<10} "
            f"{str(name):<30} "
            f"{count:>7} spikes"
        )

    # ---------------------------------------------------------
    # Find active motor neurons
    # ---------------------------------------------------------

    instance_text = (
        metadata["instance"]
        .fillna("")
        .astype(str)
    )

    superclass_text = (
        metadata["superclass"]
        .fillna("")
        .astype(str)
    )

    motor_rows = metadata[
        instance_text.str.contains(
            r"\bMN\b",
            case=False,
            regex=True,
        )
        |
        superclass_text.str.contains(
            "motor",
            case=False,
            regex=False,
        )
    ]

    active_motors = []

    for body_id in motor_rows[
        "bodyId"
    ].values:

        idx = body_to_index(
            neuron_ids,
            int(body_id),
        )

        if idx is None:
            continue

        count = int(
            spike_counts[idx]
        )

        if count == 0:
            continue

        row = metadata_lookup.loc[
            int(body_id)
        ]

        active_motors.append({
            "bodyId": int(body_id),
            "instance": row.get("instance"),
            "spikes": count,
        })

    active_motors = pd.DataFrame(
        active_motors
    )

    if not active_motors.empty:

        active_motors = (
            active_motors
            .sort_values(
                "spikes",
                ascending=False,
            )
        )

        print()
        print("=" * 80)
        print("ACTIVE MOTOR NEURONS")
        print("=" * 80)

        print(
            active_motors
            .head(50)
            .to_string(index=False)
        )

        active_motors.to_csv(
            output / "active_motor_neurons.csv",
            index=False,
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    np.savez_compressed(
        output / "spike_counts.npz",
        neuron_ids=np.asarray(neuron_ids),
        spike_counts=spike_counts,
    )

    if recorded_times:

        times = np.concatenate(
            recorded_times
        )

        indices = np.concatenate(
            recorded_indices
        )

        event_body_ids = np.asarray(
            neuron_ids[indices]
        )

        np.savez_compressed(
            output / "spike_events.npz",
            time=times,
            bodyId=event_body_ids,
        )

    params = {
        "stimulate": args.stimulate,
        "targets": args.target,
        "steps": args.steps,
        "stim_start": args.stim_start,
        "stim_end": args.stim_end,
        "stim_current": args.stim_current,
        "threshold": args.threshold,
        "leak": args.leak,
        "synaptic_gain": args.synaptic_gain,
        "min_weight": args.min_weight,
    }

    with open(
        output / "params.json",
        "w",
    ) as f:
        json.dump(
            params,
            f,
            indent=2,
        )

    print()
    print(f"Saved results to: {output}")


if __name__ == "__main__":
    main()

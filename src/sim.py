import os
import argparse
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from neuprint import (
    Client,
    NeuronCriteria,
    fetch_adjacencies,
    fetch_neurons,
)


# ---------------------------------------------------------
# Neurotransmitter -> assumed sign
#
# IMPORTANT:
# These are MODEL assumptions, not guaranteed physiological
# effects at every synapse.
# ---------------------------------------------------------

def transmitter_sign(nt):
    if nt is None:
        return 0.0

    nt = str(nt).lower()

    if nt == "acetylcholine":
        return 1.0

    if nt == "gaba":
        return -1.0

    # Glutamate can have different effects depending on
    # receptor/circuit, so don't guess yet.
    if nt == "glutamate":
        return 0.0

    return 0.0


# ---------------------------------------------------------
# Download a local circuit around one neuron
# ---------------------------------------------------------

def discover_circuit(
    client,
    start_id,
    depth,
    min_weight,
    max_neurons,
):

    discovered = {start_id}
    queue = deque([(start_id, 0)])

    edge_map = {}

    while queue and len(discovered) < max_neurons:

        current, current_depth = queue.popleft()

        if current_depth >= depth:
            continue

        print(
            f"Exploring {current} "
            f"(depth {current_depth}/{depth})"
        )

        criteria = NeuronCriteria(bodyId=current)

        _, connections = fetch_adjacencies(
            sources=criteria,
            targets=None,
            min_total_weight=min_weight,
            client=client,
        )

        if connections.empty:
            continue

        connections = (
            connections
            .groupby(
                ["bodyId_pre", "bodyId_post"],
                as_index=False,
            )["weight"]
            .sum()
            .sort_values(
                "weight",
                ascending=False,
            )
        )

        for _, row in connections.iterrows():

            pre = int(row["bodyId_pre"])
            post = int(row["bodyId_post"])
            weight = int(row["weight"])

            edge_map[(pre, post)] = max(
                weight,
                edge_map.get((pre, post), 0),
            )

            if post not in discovered:

                if len(discovered) >= max_neurons:
                    break

                discovered.add(post)

                queue.append(
                    (post, current_depth + 1)
                )

    return sorted(discovered), edge_map


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Connectome-derived Male CNS activity model"
    )

    parser.add_argument(
        "body_id",
        type=int,
        help="Neuron to externally stimulate",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--min-weight",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max-neurons",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--stim-start",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--stim-end",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--stim-current",
        type=float,
        default=1.25,
    )

    parser.add_argument(
        "--target",
        type=int,
        default=815344,
        help="Motor/output neuron to monitor",
    )

    args = parser.parse_args()

    token = os.environ[
        "NEUPRINT_APPLICATION_CREDENTIALS"
    ]

    client = Client(
        "neuprint.janelia.org",
        dataset="male-cns:v1.0",
        token=token,
    )

    print("\nDISCOVERING CIRCUIT")
    print("=" * 70)

    body_ids, edge_map = discover_circuit(
        client,
        args.body_id,
        args.depth,
        args.min_weight,
        args.max_neurons,
    )

    print(
        f"\nFound {len(body_ids)} neurons "
        f"and {len(edge_map)} connections."
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata, _ = fetch_neurons(
        NeuronCriteria(bodyId=body_ids),
        client=client,
    )

    metadata = metadata.set_index("bodyId")

    id_to_index = {
        body_id: i
        for i, body_id in enumerate(body_ids)
    }

    N = len(body_ids)

    print("\nNEURONS")
    print("-" * 70)

    for body_id in body_ids:

        if body_id not in metadata.index:
            continue

        row = metadata.loc[body_id]

        print(
            f"{body_id:<10} "
            f"{str(row.get('instance')):<25} "
            f"{str(row.get('predictedNt')):<15}"
        )

    # -----------------------------------------------------
    # Connectivity matrix
    # -----------------------------------------------------

    W = np.zeros((N, N), dtype=float)

    # Determine strongest outgoing edge for each source.
    #
    # This lets us scale synapses relatively instead of
    # pretending "1 anatomical synapse = X millivolts".
    max_outgoing = {}

    for (pre, post), weight in edge_map.items():

        if pre not in id_to_index:
            continue

        if post not in id_to_index:
            continue

        max_outgoing[pre] = max(
            weight,
            max_outgoing.get(pre, 0),
        )

    synaptic_gain = 0.85

    for (pre, post), weight in edge_map.items():

        if pre not in id_to_index:
            continue

        if post not in id_to_index:
            continue

        if pre not in metadata.index:
            continue

        nt = metadata.loc[pre].get(
            "predictedNt"
        )

        sign = transmitter_sign(nt)

        # Skip connections whose functional sign
        # we're unwilling to guess.
        if sign == 0:
            continue

        relative_strength = (
            weight / max_outgoing[pre]
        )

        scaled_weight = (
            sign
            * relative_strength
            * synaptic_gain
        )

        i = id_to_index[pre]
        j = id_to_index[post]

        W[i, j] = scaled_weight

    # -----------------------------------------------------
    # Simple LIF-like model
    # -----------------------------------------------------

    voltage = np.zeros(N)

    threshold = 1.0
    leak = 0.90

    spikes = np.zeros(
        (args.steps, N),
        dtype=bool,
    )

    voltage_history = np.zeros(
        (args.steps, N),
        dtype=float,
    )

    stimulus_index = id_to_index[
        args.body_id
    ]

    print("\nSIMULATION")
    print("=" * 70)

    print(
        f"Stimulating {args.body_id} "
        f"from step {args.stim_start} "
        f"to {args.stim_end}"
    )

    if args.target in id_to_index:
        target_index = id_to_index[
            args.target
        ]

        if args.target in metadata.index:

            target_name = metadata.loc[
                args.target
            ].get("instance")

        else:
            target_name = str(args.target)

        print(
            f"Monitoring target: "
            f"{args.target} ({target_name})"
        )

    else:
        target_index = None

        print(
            f"\nWARNING: target {args.target} "
            "is not in this circuit."
        )

        print(
            "Try lowering --min-weight or "
            "increasing --max-neurons."
        )

    # -----------------------------------------------------
    # Run
    # -----------------------------------------------------

    for t in range(args.steps):

        # membrane leak
        voltage *= leak

        # External stimulation
        if args.stim_start <= t < args.stim_end:
            voltage[stimulus_index] += (
                args.stim_current
            )

        # Activity from neurons that fired
        # on the previous step
        if t > 0:

            previous_spikes = (
                spikes[t - 1].astype(float)
            )

            synaptic_input = (
                previous_spikes @ W
            )

            voltage += synaptic_input

        # threshold
        fired = voltage >= threshold

        spikes[t] = fired

        # reset after spike
        voltage[fired] = 0.0

        voltage_history[t] = voltage

    # -----------------------------------------------------
    # Report spike counts
    # -----------------------------------------------------

    counts = spikes.sum(axis=0)

    order = np.argsort(counts)[::-1]

    print("\nSPIKE COUNTS")
    print("-" * 70)

    for i in order:

        if counts[i] == 0:
            continue

        body_id = body_ids[i]

        if body_id in metadata.index:

            name = metadata.loc[
                body_id
            ].get("instance")

        else:
            name = str(body_id)

        print(
            f"{body_id:<10} "
            f"{str(name):<25} "
            f"{int(counts[i]):>5} spikes"
        )

    # -----------------------------------------------------
    # Motor result
    # -----------------------------------------------------

    if target_index is not None:

        motor_spikes = int(
            spikes[:, target_index].sum()
        )

        print("\nMOTOR OUTPUT")
        print("=" * 70)

        print(
            f"{target_name}: "
            f"{motor_spikes} spikes"
        )

        if motor_spikes > 0:
            print(
                "\n*** MOTOR NEURON ACTIVATED ***"
            )
        else:
            print(
                "\nMotor neuron did not reach "
                "threshold in this model."
            )

    # -----------------------------------------------------
    # Raster plot
    # -----------------------------------------------------

    plt.figure(figsize=(12, 7))

    for i, body_id in enumerate(body_ids):

        times = np.where(
            spikes[:, i]
        )[0]

        if len(times) == 0:
            continue

        plt.scatter(
            times,
            np.full_like(times, i),
            marker="|",
            s=100,
        )

    labels = []

    for body_id in body_ids:

        if body_id in metadata.index:

            name = metadata.loc[
                body_id
            ].get("instance")

            labels.append(
                f"{name} ({body_id})"
            )

        else:

            labels.append(str(body_id))

    plt.yticks(
        range(N),
        labels,
        fontsize=7,
    )

    plt.axvspan(
        args.stim_start,
        args.stim_end,
        alpha=0.15,
    )

    plt.xlabel("Simulation step")
    plt.ylabel("Neuron")

    plt.title(
        "Male CNS connectome-derived activity"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

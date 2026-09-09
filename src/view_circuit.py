import os
import argparse
from collections import deque

from neuprint import (
    Client,
    NeuronCriteria,
    fetch_adjacencies,
    fetch_neurons,
)

import navis
from navis.interfaces import neuprint as navis_neuprint


def main():
    parser = argparse.ArgumentParser(
        description="Explore a Male CNS circuit in 3D"
    )

    parser.add_argument("body_id", type=int)

    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Number of synaptic hops to follow",
    )

    parser.add_argument(
        "--min-weight",
        type=int,
        default=50,
        help="Minimum synapse count for an edge",
    )

    parser.add_argument(
        "--max-neurons",
        type=int,
        default=30,
        help="Maximum TOTAL neurons to include",
    )

    parser.add_argument(
        "--direction",
        choices=["downstream", "upstream", "both"],
        default="downstream",
    )

    args = parser.parse_args()

    token = os.environ["NEUPRINT_APPLICATION_CREDENTIALS"]

    client = Client(
        "neuprint.janelia.org",
        dataset="male-cns:v1.0",
        token=token,
    )

    # ---------------------------------------
    # Graph traversal
    # ---------------------------------------

    discovered = {args.body_id}

    # (body_id, current_depth)
    queue = deque([
        (args.body_id, 0)
    ])

    all_edges = []

    while queue and len(discovered) < args.max_neurons:

        current_id, depth = queue.popleft()

        if depth >= args.depth:
            continue

        print(
            f"\nExploring {current_id} "
            f"(depth {depth}/{args.depth})"
        )

        criteria = NeuronCriteria(bodyId=current_id)

        queries = []

        if args.direction in ("downstream", "both"):

            _, downstream = fetch_adjacencies(
                sources=criteria,
                targets=None,
                min_total_weight=args.min_weight,
                client=client,
            )

            if not downstream.empty:

                edges = (
                    downstream
                    .groupby(
                        ["bodyId_pre", "bodyId_post"],
                        as_index=False,
                    )["weight"]
                    .sum()
                )

                queries.append(edges)

        if args.direction in ("upstream", "both"):

            _, upstream = fetch_adjacencies(
                sources=None,
                targets=criteria,
                min_total_weight=args.min_weight,
                client=client,
            )

            if not upstream.empty:

                edges = (
                    upstream
                    .groupby(
                        ["bodyId_pre", "bodyId_post"],
                        as_index=False,
                    )["weight"]
                    .sum()
                )

                queries.append(edges)

        # Combine results
        for edges in queries:

            edges = edges.sort_values(
                "weight",
                ascending=False,
            )

            for _, edge in edges.iterrows():

                pre = int(edge["bodyId_pre"])
                post = int(edge["bodyId_post"])
                weight = int(edge["weight"])

                all_edges.append(
                    (pre, post, weight)
                )

                # Figure out which neuron is the new one
                if current_id == pre:
                    neighbor = post
                else:
                    neighbor = pre

                if neighbor in discovered:
                    continue

                if len(discovered) >= args.max_neurons:
                    break

                discovered.add(neighbor)

                queue.append(
                    (neighbor, depth + 1)
                )

    # ---------------------------------------
    # Results
    # ---------------------------------------

    body_ids = list(discovered)

    print("\n" + "=" * 70)
    print(" CIRCUIT DISCOVERED")
    print("=" * 70)

    print(f"Start neuron : {args.body_id}")
    print(f"Depth        : {args.depth}")
    print(f"Min weight   : {args.min_weight}")
    print(f"Direction    : {args.direction}")
    print(f"Neurons      : {len(body_ids)}")
    print(f"Edges        : {len(all_edges)}")

    # ---------------------------------------
    # Fetch metadata
    # ---------------------------------------

    metadata, _ = fetch_neurons(
        NeuronCriteria(bodyId=body_ids),
        client=client,
    )

    lookup = metadata.set_index("bodyId")

    print("\nNEURONS")
    print("-" * 70)

    for body_id in body_ids:

        if body_id not in lookup.index:
            print(body_id)
            continue

        neuron = lookup.loc[body_id]

        print(
            f"{body_id:<10} "
            f"{str(neuron.get('instance')):<25} "
            f"{str(neuron.get('predictedNt')):<15}"
        )

    print("\nCONNECTIONS")
    print("-" * 70)

    # Remove duplicate edges
    unique_edges = {}

    for pre, post, weight in all_edges:
        key = (pre, post)

        # Keep largest value if duplicated
        unique_edges[key] = max(
            weight,
            unique_edges.get(key, 0)
        )

    for (pre, post), weight in sorted(
        unique_edges.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        # Only display edges where both neurons
        # actually made it into our graph
        if pre not in discovered or post not in discovered:
            continue

        pre_name = (
            lookup.loc[pre].get("instance")
            if pre in lookup.index
            else str(pre)
        )

        post_name = (
            lookup.loc[post].get("instance")
            if post in lookup.index
            else str(post)
        )

        print(
            f"{str(pre_name):<22} "
            f"--{weight:>4}--> "
            f"{str(post_name)}"
        )

    # ---------------------------------------
    # Fetch skeletons
    # ---------------------------------------

    print(
        f"\nFetching {len(body_ids)} "
        "reconstructed skeletons..."
    )

    navis_client = navis_neuprint.Client(
        "https://neuprint.janelia.org",
        dataset="male-cns:v1.0",
        token=token,
    )

    neurons = navis_neuprint.fetch_skeletons(
        body_ids,
        client=navis_client,
    )

    print(neurons)

    # ---------------------------------------
    # 3D
    # ---------------------------------------

    fig = navis.plot3d(
        neurons,
        backend="plotly",
    )

    fig.show()


if __name__ == "__main__":
    main()

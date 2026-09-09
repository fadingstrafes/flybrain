import os
import sys

from neuprint import (
    Client,
    NeuronCriteria,
    fetch_neurons,
    fetch_adjacencies,
)

client = Client(
    "neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=os.environ["NEUPRINT_APPLICATION_CREDENTIALS"],
)


def get_neuron(body_id):

    criteria = NeuronCriteria(bodyId=body_id)

    neurons, _ = fetch_neurons(
        criteria,
        client=client,
    )

    if len(neurons) == 0:
        print(f"No neuron found for bodyId {body_id}")
        return

    n = neurons.iloc[0]

    print()
    print("=" * 70)
    print(f" NEURON {body_id}")
    print("=" * 70)

    fields = [
        ("Type", "type"),
        ("Instance", "instance"),
        ("Superclass", "superclass"),
        ("Class", "class"),
        ("Subclass", "subclass"),
        ("Predicted NT", "predictedNt"),
        ("NT confidence", "predictedNtConfidence"),
        ("Consensus NT", "consensusNt"),
        ("Receptor type", "receptorType"),
        ("Presynaptic sites", "pre"),
        ("Postsynaptic sites", "post"),
        ("Upstream", "upstream"),
        ("Downstream", "downstream"),
        ("Soma side", "somaSide"),
        ("Soma neuromere", "somaNeuromere"),
    ]

    for label, field in fields:
        if field in n.index:
            print(f"{label:20}: {n[field]}")

    print("\nINPUT ROIs")
    print("-" * 70)

    if "inputRois" in n.index:
        for roi in n["inputRois"]:
            print(" ", roi)

    print("\nOUTPUT ROIs")
    print("-" * 70)

    if "outputRois" in n.index:
        for roi in n["outputRois"]:
            print(" ", roi)

    # ---------------------------
    # INPUTS
    # ---------------------------

    _, inputs = fetch_adjacencies(
        sources=None,
        targets=criteria,
        min_total_weight=10,
        client=client,
    )

    inputs = (
        inputs
        .groupby(
            ["bodyId_pre", "bodyId_post"],
            as_index=False
        )["weight"]
        .sum()
        .sort_values("weight", ascending=False)
        .head(15)
    )

    input_ids = inputs["bodyId_pre"].tolist()

    if input_ids:
        input_neurons, _ = fetch_neurons(
            NeuronCriteria(bodyId=input_ids),
            client=client,
        )

        lookup = input_neurons.set_index("bodyId")

        print("\nSTRONGEST INPUTS")
        print("-" * 70)

        for _, edge in inputs.iterrows():

            source = int(edge["bodyId_pre"])
            weight = int(edge["weight"])

            if source in lookup.index:
                x = lookup.loc[source]

                print(
                    f"{source:<10} "
                    f"{str(x.get('instance')):<22} "
                    f"{str(x.get('predictedNt')):<15} "
                    f"{weight:>5} syn"
                )

    # ---------------------------
    # OUTPUTS
    # ---------------------------

    _, outputs = fetch_adjacencies(
        sources=criteria,
        targets=None,
        min_total_weight=10,
        client=client,
    )

    outputs = (
        outputs
        .groupby(
            ["bodyId_pre", "bodyId_post"],
            as_index=False
        )["weight"]
        .sum()
        .sort_values("weight", ascending=False)
        .head(15)
    )

    output_ids = outputs["bodyId_post"].tolist()

    if output_ids:
        output_neurons, _ = fetch_neurons(
            NeuronCriteria(bodyId=output_ids),
            client=client,
        )

        lookup = output_neurons.set_index("bodyId")

        print("\nSTRONGEST OUTPUTS")
        print("-" * 70)

        for _, edge in outputs.iterrows():

            target = int(edge["bodyId_post"])
            weight = int(edge["weight"])

            if target in lookup.index:
                x = lookup.loc[target]

                print(
                    f"{target:<10} "
                    f"{str(x.get('instance')):<22} "
                    f"{str(x.get('predictedNt')):<15} "
                    f"{weight:>5} syn"
                )


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("  python -m src.neuron BODY_ID")
        sys.exit(1)

    get_neuron(int(sys.argv[1]))

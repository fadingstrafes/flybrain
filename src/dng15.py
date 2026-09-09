import os
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

# Our two DNg15 neurons
ids = [10141, 513052]

criteria = NeuronCriteria(bodyId=ids)

neurons, roi_info = fetch_neurons(
    criteria,
    client=client,
)

print("\n=== DNg15 ===")

cols = [
    "bodyId",
    "type",
    "instance",
    "pre",
    "post",
    "inputRois",
    "outputRois",
]

for col in cols:
    if col not in neurons.columns:
        print(f"Dataset doesn't provide column: {col}")

available = [c for c in cols if c in neurons.columns]

print(neurons[available].to_string(index=False))


# ------------------------------------------------
# What does DNg15 strongly connect TO?
# ------------------------------------------------

_, outputs = fetch_adjacencies(
    sources=criteria,
    targets=None,
    min_total_weight=20,
    client=client,
)

outputs = (
    outputs
    .groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]
    .sum()
    .sort_values("weight", ascending=False)
)

print("\n=== Strongest DNg15 outputs ===")
print(outputs.head(30).to_string(index=False))


# ------------------------------------------------
# What strongly connects INTO DNg15?
# ------------------------------------------------

_, inputs = fetch_adjacencies(
    sources=None,
    targets=criteria,
    min_total_weight=20,
    client=client,
)

inputs = (
    inputs
    .groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]
    .sum()
    .sort_values("weight", ascending=False)
)

print("\n=== Strongest DNg15 inputs ===")
print(inputs.head(30).to_string(index=False))

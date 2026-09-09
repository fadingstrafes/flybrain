from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from scipy.sparse import coo_matrix, save_npz


RAW = Path("data/raw")
CACHE = Path("data/cache")
CACHE.mkdir(parents=True, exist_ok=True)

CONNECTOME = RAW / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NT_FILE = RAW / "body-neurotransmitters-male-cns-v1.0.feather"
ANNOTATIONS = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"


print("=" * 80)
print("LOADING FULL CONNECTION GRAPH")
print("=" * 80)

table = feather.read_table(
    CONNECTOME,
    columns=[
        "body_pre",
        "body_post",
        "weight",
    ],
    memory_map=True,
)

print(f"Connections in file: {table.num_rows:,}")


# ------------------------------------------------------------
# Arrow -> NumPy
# ------------------------------------------------------------

print("\nConverting columns...")

pre = (
    table["body_pre"]
    .combine_chunks()
    .to_numpy(zero_copy_only=False)
)

post = (
    table["body_post"]
    .combine_chunks()
    .to_numpy(zero_copy_only=False)
)

weight = (
    table["weight"]
    .combine_chunks()
    .to_numpy(zero_copy_only=False)
    .astype(np.int32, copy=False)
)

# We don't need the Arrow table anymore.
del table


# ------------------------------------------------------------
# Build master neuron-ID list
# ------------------------------------------------------------

print("\nFinding all neurons...")

neuron_ids = np.unique(
    np.concatenate([
        pre,
        post,
    ])
)

N = len(neuron_ids)

print(f"Unique neurons: {N:,}")

np.save(
    CACHE / "neuron_ids.npy",
    neuron_ids,
)


# ------------------------------------------------------------
# Convert real body IDs -> compact array indices
#
# Example:
#
# bodyId 513052  -> index 84623
#
# Sparse matrices want contiguous integer coordinates.
# ------------------------------------------------------------

print("\nMapping body IDs to matrix indices...")

pre_idx = np.searchsorted(
    neuron_ids,
    pre,
).astype(np.int32)

post_idx = np.searchsorted(
    neuron_ids,
    post,
).astype(np.int32)


# Original body IDs are no longer needed for matrix construction.
del pre
del post


# ------------------------------------------------------------
# Sparse connectivity matrix
#
# W[i, j] = anatomical synapse count from neuron i -> neuron j
# ------------------------------------------------------------

print("\nBuilding sparse matrix...")

W = coo_matrix(
    (
        weight,
        (pre_idx, post_idx),
    ),
    shape=(N, N),
    dtype=np.int32,
).tocsr()

W.sum_duplicates()
W.eliminate_zeros()
W.sort_indices()

del pre_idx
del post_idx
del weight


print()
print("=" * 80)
print("CONNECTOME")
print("=" * 80)

print(f"Neurons:       {W.shape[0]:,}")
print(f"Connections:   {W.nnz:,}")
print(f"Total synapses:{int(W.sum()):,}")

memory_bytes = (
    W.data.nbytes
    + W.indices.nbytes
    + W.indptr.nbytes
)

print(
    f"CSR memory:    "
    f"{memory_bytes / 1024**3:.2f} GiB"
)


# ------------------------------------------------------------
# Save graph
# ------------------------------------------------------------

print("\nSaving sparse graph...")

save_npz(
    CACHE / "connectome_raw.npz",
    W,
    compressed=True,
)


# ------------------------------------------------------------
# Neurotransmitter metadata
# ------------------------------------------------------------

print("\nLoading neurotransmitter metadata...")

nt = pd.read_feather(NT_FILE)

annotations = pd.read_feather(
    ANNOTATIONS
)

metadata = annotations.merge(
    nt,
    how="outer",
    left_on="bodyId",
    right_on="body",
)

metadata["bodyId"] = (
    metadata["bodyId"]
    .fillna(metadata["body"])
)

metadata.to_parquet(
    CACHE / "metadata.parquet",
    index=False,
)


# ------------------------------------------------------------
# Build one neurotransmitter sign value per neuron
#
# +1 = excitatory assumption
# -1 = inhibitory assumption
#  0 = unknown / don't infer sign
# ------------------------------------------------------------

nt_sign = np.zeros(
    N,
    dtype=np.float32,
)

nt_label = np.full(
    N,
    "",
    dtype="<U32",
)


def choose_nt(row):
    """
    Prefer consensus_nt when available.
    Fall back to predicted_nt.
    """

    consensus = row.get(
        "consensus_nt"
    )

    if pd.notna(consensus) and str(consensus).strip():
        return str(consensus).lower()

    predicted = row.get(
        "predicted_nt"
    )

    if pd.notna(predicted) and str(predicted).strip():
        return str(predicted).lower()

    return ""


print("\nMapping neurotransmitters...")

for _, row in metadata.iterrows():

    body = row.get("bodyId")

    if pd.isna(body):
        continue

    body = int(body)

    idx = np.searchsorted(
        neuron_ids,
        body,
    )

    if idx >= N:
        continue

    if neuron_ids[idx] != body:
        continue

    label = choose_nt(row)

    nt_label[idx] = label

    # Simplified model assumptions.
    #
    # Receptor-specific effects are NOT represented here.
    if label == "acetylcholine":
        nt_sign[idx] = 1.0

    elif label == "gaba":
        nt_sign[idx] = -1.0

    else:
        # glutamate, dopamine, serotonin,
        # octopamine, unclear, etc.
        #
        # Keep neutral until we explicitly model them.
        nt_sign[idx] = 0.0


np.save(
    CACHE / "nt_sign.npy",
    nt_sign,
)

np.save(
    CACHE / "nt_label.npy",
    nt_label,
)


print()
print("=" * 80)
print("NEUROTRANSMITTERS")
print("=" * 80)

unique, counts = np.unique(
    nt_label[nt_label != ""],
    return_counts=True,
)

for label, count in zip(
    unique,
    counts,
):
    print(
        f"{label:<20} "
        f"{count:>8,}"
    )


print()
print("=" * 80)
print("CACHE COMPLETE")
print("=" * 80)

print("Created:")

for name in [
    "neuron_ids.npy",
    "connectome_raw.npz",
    "nt_sign.npy",
    "nt_label.npy",
    "metadata.parquet",
]:
    path = CACHE / name

    print(
        f"  {path} "
        f"({path.stat().st_size / 1024**2:.1f} MiB)"
    )

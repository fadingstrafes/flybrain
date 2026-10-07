"""Inspect the existing male-cns:v1.0 cache without rebuilding or filtering it."""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--check-edges", action="store_true",
                        help="Load sparse CSR (~1.5 GiB for the current cache) and check known edges")
    parser.add_argument("--output", type=Path, help="Optional JSON report; caches are never modified")
    args = parser.parse_args()
    ids = np.load(args.cache / "neuron_ids.npy", mmap_mode="r")
    signs = np.load(args.cache / "nt_sign.npy", mmap_mode="r")
    if signs.shape != ids.shape:
        raise ValueError("Neuron IDs and transmitter signs have different shapes")
    with zipfile.ZipFile(args.cache / "connectome_raw.npz") as archive:
        with archive.open("shape.npy") as f:
            shape = np.load(f).tolist()
        with archive.open("data.npy") as f:
            version = np.lib.format.read_magic(f)
            reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            data_shape, _, _ = reader(f)
    if shape != [len(ids), len(ids)]:
        raise ValueError("Graph dimensions do not match the body-ID array")
    metadata = pd.read_parquet(args.cache / "metadata.parquet", columns=["bodyId", "superclass"])
    metadata = metadata.dropna(subset=["bodyId"]).drop_duplicates("bodyId")
    body_ids = metadata.bodyId.to_numpy(dtype=np.int64)
    indices = np.searchsorted(ids, body_ids)
    valid = indices < len(ids)
    matched = np.zeros(len(body_ids), dtype=bool)
    matched[valid] = ids[indices[valid]] == body_ids[valid]
    annotated = metadata.superclass.fillna("").str.strip().ne("").to_numpy()
    report = {
        "dataset": "male-cns:v1.0",
        "bulk_source": "gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/",
        "skeleton_source": "https://neuprint.janelia.org",
        "graph_body_ids": len(ids), "connections": data_shape[0],
        "metadata_body_ids_in_graph": int(matched.sum()),
        "graph_ids_with_superclass": int(np.count_nonzero(matched & annotated)),
        "graph_ids_without_metadata": len(ids) - int(matched.sum()),
        "modeled_transmitter_signs": int(np.count_nonzero(signs)),
        "float32_state_vector_mib": len(ids) * 4 / 1024**2,
        "interpretation": (
            "The cache includes every body ID in the bulk edge table. Graph body IDs "
            "must not all be described as annotated neurons. No bodies or edges were removed."
        ),
    }
    if args.check_edges:
        W = load_npz(args.cache / "connectome_raw.npz").tocsr()
        checks = []
        for pre, post, expected in [(12781, 513052, 150), (556329, 10141, 116),
                                    (513052, 800373, 289), (513052, 800206, 188),
                                    (513052, 800002, 151), (513052, 815344, 87)]:
            i, j = np.searchsorted(ids, [pre, post])
            if i >= len(ids) or j >= len(ids) or ids[i] != pre or ids[j] != post:
                actual = None
            else:
                actual = int(W[i, j])
            checks.append({"pre": pre, "post": post, "expected": expected,
                           "actual": actual, "passed": actual == expected})
        report["known_edges"] = checks
    result = json.dumps(report, indent=2) + "\n"
    print(result, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    if args.check_edges and not all(c["passed"] for c in report["known_edges"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

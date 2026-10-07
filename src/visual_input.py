"""Synthetic retinal-sector input to annotated, sign-compatible visual relays."""

from pathlib import Path

import numpy as np
import pandas as pd


class VisualInput:
    def __init__(self, graph, cache=Path("data/cache"), sectors=9, cells_per_sector=8):
        metadata = pd.read_parquet(Path(cache)/"metadata.parquet", columns=["bodyId", "type", "somaSide"])
        candidates = metadata[metadata.type.fillna("").str.match(r"^T[45][abcd]$")].drop_duplicates("bodyId")
        lookup = {int(body): i for i, body in enumerate(graph.ids)}
        self.populations = []
        for side in ("L", "R"):
            rows = candidates[candidates.somaSide.eq(side)].sort_values(["type", "bodyId"])
            indices = [lookup[int(b)] for b in rows.bodyId if int(b) in lookup and graph.signs[lookup[int(b)]] != 0]
            # Spatial retinal assignments are absent from these local annotations.
            # Evenly sample and partition deterministically; do not claim retinotopy.
            count = min(len(indices), sectors*cells_per_sector)
            selected = np.asarray(indices, dtype=np.int64)[np.linspace(0, len(indices)-1, count, dtype=int)] if count else np.empty(0, dtype=np.int64)
            self.populations.extend(np.array_split(selected, sectors))
        self.indices = np.concatenate(self.populations)
        self.sizes = np.array([len(p) for p in self.populations])
        self.body_ids = graph.ids[self.indices]

    def stimulus(self, signals):
        # Only use meaningful visual contrast/proximity; no mandatory tonic firing.
        levels = np.maximum(0, np.asarray(signals).reshape(-1)-.02)*1.1
        return self.indices, np.repeat(levels, self.sizes).astype(np.float32)

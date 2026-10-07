"""Reusable sparse live dynamics for the existing male-cns:v1.0 model.

Retain all nonzero-sign bodies and annotated neurons. An omitted body has zero
outgoing modeled effect, so it cannot affect retained voltages. Normalize against
the ORIGINAL outgoing row maximum, including edges to omitted bodies.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
import torch


CACHE = Path("data/cache")


@dataclass
class LiveGraph:
    ids: np.ndarray
    signs: np.ndarray
    row_max: np.ndarray
    weights: csr_matrix
    source_size: int


def project_graph(W, ids, signs, observed_ids):
    """An exact projection for retained state under the current zero-sign model."""
    observed = np.searchsorted(ids, np.asarray(observed_ids, dtype=np.int64))
    valid = observed < len(ids)
    observed = observed[valid]
    observed = observed[ids[observed] == np.asarray(observed_ids, dtype=np.int64)[valid]]
    keep = np.union1d(np.flatnonzero(signs), observed)
    if not len(keep):
        raise ValueError("Live graph has no retained neurons")
    rows = W[keep].tocsr()
    nonempty = np.diff(rows.indptr) > 0
    row_max = np.zeros(len(keep), dtype=np.float32)
    row_max[nonempty] = np.maximum.reduceat(rows.data, rows.indptr[:-1][nonempty])
    mapped = np.searchsorted(keep, rows.indices)
    valid = mapped < len(keep)
    valid[valid] &= keep[mapped[valid]] == rows.indices[valid]
    rows.data[~valid] = 0
    # Invalid entries carry zero and are eliminated before any arithmetic.
    columns = np.minimum(mapped, len(keep) - 1).astype(np.int32)
    weights = csr_matrix((rows.data, columns, rows.indptr), shape=(len(keep), len(keep)))
    weights.eliminate_zeros()
    weights.sort_indices()
    return LiveGraph(np.asarray(ids[keep]), np.asarray(signs[keep]), row_max, weights, len(ids))


def load_live_graph(cache=CACHE):
    cache = Path(cache)
    derived = cache / "live_brain_v1"
    names = ["connectome_raw.npz", "neuron_ids.npy", "nt_sign.npy", "metadata.parquet"]
    signature = {name: [int((cache / name).stat().st_size), (cache / name).stat().st_mtime_ns]
                 for name in names}
    manifest_path = derived / "manifest.json"
    if manifest_path.exists() and (derived / "state.npz").exists() and (derived / "weights.npz").exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source_signature") == signature and manifest.get("version") == 1:
            with np.load(derived / "state.npz") as state:
                return LiveGraph(state["ids"], state["signs"], state["row_max"],
                                 load_npz(derived / "weights.npz"), manifest["source_size"])
    print("Preparing live projection from the existing male-cns:v1.0 cache...", flush=True)
    ids = np.load(cache / "neuron_ids.npy", mmap_mode="r")
    signs = np.load(cache / "nt_sign.npy", mmap_mode="r")
    metadata = pd.read_parquet(cache / "metadata.parquet", columns=["bodyId", "superclass"])
    annotated = metadata.loc[metadata.superclass.fillna("").str.strip().ne(""), "bodyId"]
    W = load_npz(cache / "connectome_raw.npz").tocsr()
    graph = project_graph(W, ids, signs, annotated.dropna().to_numpy(dtype=np.int64))
    del W
    derived.mkdir(parents=True, exist_ok=True)
    save_npz(derived / "weights.npz", graph.weights)
    np.savez_compressed(derived / "state.npz", ids=graph.ids, signs=graph.signs, row_max=graph.row_max)
    manifest = {
        "version": 1, "dataset": "male-cns:v1.0", "source_signature": signature,
        "source_size": graph.source_size, "retained_bodies": len(graph.ids),
        "selection": "All nonzero transmitter signs plus all nonempty superclass annotations",
        "normalization": "Original full-graph outgoing maximum before projection or filtering",
        "limitations": "Omitted zero-sign bodies are not monitored or counted in live avalanche checks",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Live cache: {len(graph.ids):,} bodies, {graph.weights.nnz:,} anatomical edges", flush=True)
    return graph


class LiveBrain:
    def __init__(self, graph, device="auto", min_weight=5, gain=0.35, leak=0.92,
                 threshold=1.0, abort_fraction=0.2):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.ids = graph.ids
        self.leak, self.threshold, self.abort_fraction = leak, threshold, abort_fraction
        weights = graph.weights.copy().astype(np.float32)
        weights.data[weights.data < min_weight] = 0
        weights.eliminate_zeros()
        incoming = weights.T.tocsr()
        self.W = torch.sparse_csr_tensor(
            torch.as_tensor(incoming.indptr.astype(np.int64), device=self.device),
            torch.as_tensor(incoming.indices.astype(np.int64), device=self.device),
            torch.as_tensor(incoming.data, device=self.device),
            size=incoming.shape, device=self.device, check_invariants=True,
        )
        scale = np.divide(graph.signs * gain, graph.row_max,
                          out=np.zeros(len(self.ids), dtype=np.float32), where=graph.row_max > 0)
        self.scale = torch.as_tensor(scale, device=self.device)
        self.voltage = torch.zeros(len(self.ids), device=self.device)
        self.previous = torch.zeros_like(self.voltage)
        self.reset()

    @torch.inference_mode()
    def reset(self):
        self.voltage.zero_()
        self.previous.zero_()
        self.step_count = 0
        self.active_count = 0
        self.total_spikes = 0
        self.halted = False

    def indices(self, body_ids):
        body_ids = np.asarray(body_ids, dtype=np.int64)
        indices = np.searchsorted(self.ids, body_ids)
        valid = indices < len(self.ids)
        valid[valid] &= self.ids[indices[valid]] == body_ids[valid]
        if not np.all(valid):
            raise ValueError(f"Body IDs absent from live graph: {body_ids[~valid].tolist()}")
        return indices

    @torch.inference_mode()
    def step(self, stimulus_indices=(), currents=()):
        if self.halted:
            return np.empty(0, dtype=np.int64)
        self.voltage.mul_(self.leak)
        if self.active_count:
            self.voltage.add_(torch.sparse.mm(self.W, self.previous[:, None])[:, 0])
        if len(stimulus_indices):
            indices = torch.as_tensor(stimulus_indices, dtype=torch.int64, device=self.device)
            values = torch.as_tensor(currents, dtype=torch.float32, device=self.device)
            self.voltage.index_add_(0, indices, values)
        fired = self.voltage >= self.threshold
        fired_indices = torch.nonzero(fired).flatten()
        self.active_count = fired_indices.numel()
        self.total_spikes += self.active_count
        self.step_count += 1
        self.voltage.masked_fill_(fired, 0)
        self.previous = fired.to(torch.float32) * self.scale
        if self.abort_fraction > 0 and self.active_count > len(self.ids) * self.abort_fraction:
            self.halted = True
        return fired_indices.cpu().numpy()


class SensoryMotorBridge:
    """Annotation-based sensors and a deliberately simple motor decoder."""
    leg_names = ["L front", "L middle", "L hind", "R front", "R middle", "R hind"]

    def __init__(self, graph, cache=CACHE, population_size=24):
        sensory = pd.read_csv(Path(cache) / "sensory_map.csv")
        motor = pd.read_csv(Path(cache) / "motor_map.csv")
        lookup = {int(body): i for i, body in enumerate(graph.ids)}
        strength = np.asarray(graph.weights.sum(axis=1)).ravel()
        self.touch, self.proprio = [], []
        for leg in self.leg_names:
            side, pair = leg.split()
            base = sensory[sensory.side.eq(side) & sensory.system.eq(pair + "_leg")]
            for subclass, result in [("leg bristle", self.touch), ("chordotonal organ", self.proprio)]:
                indices = [lookup[int(b)] for b in base.loc[base.subclass.eq(subclass), "bodyId"]
                           if int(b) in lookup]
                indices = sorted(indices, key=lambda i: (-strength[i], int(graph.ids[i])))[:population_size]
                result.append(np.asarray(indices, dtype=np.int64))
        self.motor_leg = np.full(len(graph.ids), -1, dtype=np.int8)
        self.motor_action = np.zeros(len(graph.ids), dtype=np.int8)
        self.motor_wing = np.full(len(graph.ids), -1, dtype=np.int8)
        self.wing_sensory = []
        for side in ("L", "R"):
            population = sensory[sensory.system.eq("wing") & sensory.side.eq(side) & sensory.subclass.eq("wing bristle")]
            indices = [lookup[int(b)] for b in population.bodyId if int(b) in lookup]
            indices = sorted(indices, key=lambda i: (-strength[i], int(graph.ids[i])))[:population_size]
            self.wing_sensory.append(np.asarray(indices, dtype=np.int64))
        for row in motor.itertuples():
            name = f"{row.side} {row.leg_pair}"
            if row.limb == "leg" and name in self.leg_names and int(row.bodyId) in lookup:
                index = lookup[int(row.bodyId)]
                self.motor_leg[index] = self.leg_names.index(name)
                self.motor_action[index] = {"flex": -1, "extend": 1}.get(row.action, 0)
            if row.limb == "wing" and row.side in ("L", "R") and int(row.bodyId) in lookup:
                self.motor_wing[lookup[int(row.bodyId)]] = 0 if row.side == "L" else 1
        self.reset()

    def reset(self):
        self.activation = np.zeros(6, dtype=np.float32)
        self.extension = np.zeros(6, dtype=np.float32)
        self.motor_spikes = 0
        self.wing_activation = np.zeros(2, dtype=np.float32)
        self.wing_spikes = 0

    def stimulus(self, touch, proprio, time_seconds, exploration=True, pattern=None, flight_requested=False):
        indices, currents = [], []
        for leg in range(6):
            # An imposed exploratory sensory drive, not a discovered neural CPG.
            drive = 0.0
            if exploration:
                phase = time_seconds * 1.2 + leg * 0.83
                drive = 0.42 * (0.6 + 0.4 * np.sin(phase))
                if pattern is not None:
                    drive *= pattern[leg]
            for population, current in [(self.touch[leg], drive + 1.1 * touch[leg]),
                                        (self.proprio[leg], 0.7 * proprio[leg])]:
                if len(population) and current > 0:
                    indices.extend(population)
                    currents.extend([float(current)] * len(population))
        if flight_requested:
            for side, population in enumerate(self.wing_sensory):
                bias = 1.0
                if pattern is not None:
                    bias = float(np.clip((np.mean(pattern[side*3:side*3+3]) + .1) /
                                         (np.mean(pattern) + .1), .15, 1.85))
                indices.extend(population)
                currents.extend([.9 * bias] * len(population))
        return np.asarray(indices, dtype=np.int64), np.asarray(currents, dtype=np.float32)

    def observe(self, fired, dt):
        self.activation *= np.exp(-dt / 0.25)
        self.extension *= np.exp(-dt / 0.25)
        self.wing_activation *= np.exp(-dt / .25)
        legs = self.motor_leg[fired]
        valid = legs >= 0
        self.motor_spikes += int(valid.sum())
        np.add.at(self.activation, legs[valid], 0.12)
        np.add.at(self.extension, legs[valid], self.motor_action[fired[valid]] * 0.12)
        np.clip(self.activation, 0, 1, out=self.activation)
        np.clip(self.extension, -1, 1, out=self.extension)
        wings = self.motor_wing[fired]
        valid = wings >= 0
        self.wing_spikes += int(valid.sum())
        # Preserve bilateral differences instead of saturating both wings constantly.
        np.add.at(self.wing_activation, wings[valid], .006)
        np.clip(self.wing_activation, 0, 1, out=self.wing_activation)

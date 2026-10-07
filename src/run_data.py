"""Read compatible simulator outputs without mistaking missing events for silence."""

from dataclasses import dataclass
import json
from pathlib import Path
import zipfile

import numpy as np


@dataclass(frozen=True)
class RunData:
    times: np.ndarray
    body_ids: np.ndarray
    playback_steps: int
    requested_steps: int
    completed_steps: int | None
    total_spikes: int | None
    recording_complete: bool | None
    notices: tuple[str, ...]


def sum_saved_spikes(path):
    """Stream the count array; legacy whole-body caches have millions of entries."""
    with zipfile.ZipFile(path) as archive, archive.open("spike_counts.npy") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError(f"Unsupported spike-count NPY version: {version}")
        shape, _, dtype = readers[version](stream)
        if len(shape) != 1 or dtype.kind not in "iu":
            raise ValueError("spike_counts must be a one-dimensional integer array")
        remaining = shape[0]
        total = 0
        while remaining:
            count = min(remaining, 1_000_000)
            payload = stream.read(count * dtype.itemsize)
            if len(payload) != count * dtype.itemsize:
                raise ValueError("Truncated spike-count array")
            total += int(np.frombuffer(payload, dtype=dtype).sum(dtype=np.int64))
            remaining -= count
        return total


def load_run(run):
    run = Path(run)
    params_path = run / "params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}
    events_path = run / "spike_events.npz"
    times = np.empty(0, dtype=np.int32)
    body_ids = np.empty(0, dtype=np.int64)
    if events_path.exists():
        with np.load(events_path, allow_pickle=False) as events:
            times = events["time"]
            body_ids = events["bodyId"]
        if (times.ndim != 1 or body_ids.ndim != 1 or times.size != body_ids.size
                or times.dtype.kind not in "iu" or body_ids.dtype.kind not in "iu"):
            raise ValueError("Event times and body IDs must be equal-length integer vectors")
        if np.any(times < 0):
            raise ValueError("Event times cannot be negative")
        order = np.argsort(times, kind="stable")
        times = times[order].astype(np.int64, copy=False)
        body_ids = body_ids[order].astype(np.int64, copy=False)

    count_path = run / "spike_counts.npz"
    total = sum_saved_spikes(count_path) if count_path.exists() else None
    if total is not None and total < times.size:
        raise ValueError("Recorded events exceed saved spike counts; run files disagree")
    complete = (total == times.size) if total is not None else None
    if params.get("recording_complete") is False:
        complete = False
    last_event_end = int(times[-1]) + 1 if times.size else 0
    requested = int(params.get("steps", last_event_end))
    completed = params.get("completed_steps")
    completed = int(completed) if completed is not None else None
    if requested < last_event_end or (completed is not None and completed < last_event_end):
        raise ValueError("Event timestamps exceed the run duration")
    if not events_path.exists() and total != 0:
        raise RuntimeError(f"Missing {events_path}; cannot replay this run")

    notices = []
    if complete is False:
        # New runs record the first omitted timestep. For old runs only the
        # interval through the final saved event is known to be represented.
        steps = int(params.get("recording_stopped_step", last_event_end))
        if steps < last_event_end or (completed is not None and steps > completed):
            raise ValueError("Invalid recording cutoff")
        notices.append(
            f"INCOMPLETE RECORDING: {times.size:,} of {total if total is not None else 'unknown'} "
            f"spikes saved. Playback covers only the recorded prefix ({steps} steps)."
        )
    elif complete is True:
        steps = completed if completed is not None else requested
        if completed is None:
            notices.append(
                "Legacy run: completion time was not saved; using requested duration "
                "for the silent tail."
            )
    else:
        steps = last_event_end
        notices.append("Recording completeness is unknown; playback ends at the last saved event.")
    if params.get("aborted"):
        notices.append(f"Simulation aborted: {params.get('abort_reason', 'unspecified reason')}.")
    return RunData(times, body_ids, max(1, steps), requested, completed, total,
                   complete, tuple(notices))

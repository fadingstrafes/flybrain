"""Bounded, step-aligned recording of imposed media/drawing currents.

This records only artwork input, not environmental sensors or controller state.
The caller serializes append/export/reset with the simulation's session lock.
"""

import operator

import numpy as np


class StimulusRecording:
    # One contiguous buffer avoids per-step Python/NumPy object overhead when
    # a long session injects only a single neuron at a time.
    dtype = np.dtype([("time", "<i8"), ("bodyId", "<i8"), ("current", "<f4")])

    def __init__(self, max_events=2_000_000):
        self.max_events = operator.index(max_events)
        if self.max_events < 0:
            raise ValueError("Stimulus recording limit must be nonnegative")
        self.reset()

    def reset(self):
        self.buffer = bytearray()
        self.recorded_events = 0
        self.stopped_step = None

    def append(self, step, body_ids, currents):
        if self.stopped_step is not None or not len(body_ids):
            return
        # Keep a complete prefix: never save part of one neural step or resume
        # after overflow, even if subsequent steps would fit in the buffer.
        if self.recorded_events + len(body_ids) > self.max_events:
            self.stopped_step = int(step)
            return
        events = np.empty(len(body_ids), dtype=self.dtype)
        events["time"] = step
        events["bodyId"] = body_ids
        events["current"] = currents
        self.buffer.extend(events.tobytes())
        self.recorded_events += len(events)

    def summary(self, completed_steps):
        return {
            "file": "art_input_events.npz", "schema_version": 1,
            "max_events": self.max_events,
            "recorded_events": self.recorded_events,
            "recording_complete": self.stopped_step is None,
            "recording_stopped_step": self.stopped_step,
            "recorded_prefix_steps": completed_steps if self.stopped_step is None else self.stopped_step,
        }

    def save(self, path, completed_steps, neural_hz):
        events = np.frombuffer(self.buffer, dtype=self.dtype)
        summary = self.summary(completed_steps)
        # Explicit empty arrays replace a previous run's file after reset.
        np.savez_compressed(
            path, time=events["time"], bodyId=events["bodyId"], current=events["current"],
            schema_version=np.int64(1), completed_steps=np.int64(completed_steps),
            neural_hz=np.float64(neural_hz), max_events=np.int64(self.max_events),
            recording_complete=np.bool_(summary["recording_complete"]),
            recording_stopped_step=np.int64(-1 if self.stopped_step is None else self.stopped_step),
            recorded_prefix_steps=np.int64(summary["recorded_prefix_steps"]),
        )

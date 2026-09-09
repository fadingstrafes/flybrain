import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz


CACHE = Path("data/cache")


def body_to_index(neuron_ids, body_id):
    """Convert a real MaleCNS bodyId into our compact matrix index."""
    i = np.searchsorted(neuron_ids, body_id)
    if i >= len(neuron_ids):
        return None
    if neuron_ids[i] != body_id:
        return None
    return int(i)


def main():
    parser = argparse.ArgumentParser(
        description="Whole Male CNS connectome activity simulator (PyTorch/ROCm GPU)"
    )

    parser.add_argument(
        "--stimulate",
        type=int,
        nargs="+",
        required=True,
        help="One or more bodyIds to externally stimulate",
    )
    parser.add_argument(
        "--target",
        type=int,
        nargs="*",
        default=[815344],
        help="Neurons to specifically monitor",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--stim-start", type=int, default=50)
    parser.add_argument("--stim-end", type=int, default=200)
    parser.add_argument("--stim-current", type=float, default=1.25)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--leak", type=float, default=0.92)
    parser.add_argument("--synaptic-gain", type=float, default=0.35)
    parser.add_argument(
        "--min-weight",
        type=int,
        default=20,
        help="Ignore connections weaker than this many synapses",
    )
    parser.add_argument(
        "--abort-fraction",
        type=float,
        default=0.20,
        help=(
            "Abort if this fraction of the entire CNS fires "
            "simultaneously. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-recorded-events",
        type=int,
        default=2_000_000,
        help="Maximum individual spike events to save",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/full_run_gpu",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot see a GPU. On AMD/ROCm, torch.cuda.is_available() "
            "must still return True."
        )

    device = torch.device("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load complete CNS
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING WHOLE CONNECTOME")
    print("=" * 80)

    neuron_ids = np.load(CACHE / "neuron_ids.npy", mmap_mode="r")
    nt_sign = np.load(CACHE / "nt_sign.npy", mmap_mode="r")

    W = load_npz(CACHE / "connectome_raw.npz").tocsr()
    W.sort_indices()

    N = len(neuron_ids)
    original_connections = W.nnz
    total_synapses = int(W.sum())

    print(f"Neurons:     {N:,}")
    print(f"Connections: {original_connections:,}")
    print(f"Synapses:    {total_synapses:,}")
    print(f"Known NT sign: {np.count_nonzero(nt_sign):,} / {N:,}")

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = pd.read_parquet(CACHE / "metadata.parquet")
    metadata["bodyId"] = pd.to_numeric(metadata["bodyId"], errors="coerce")
    metadata = metadata.dropna(subset=["bodyId"]).drop_duplicates("bodyId")
    metadata["bodyId"] = metadata["bodyId"].astype(np.int64)
    metadata_lookup = metadata.set_index("bodyId")

    # ---------------------------------------------------------
    # Find row maximums BEFORE min-weight filtering.
    #
    # This matches sim_full.py exactly: each neuron's outgoing
    # weights are normalized by the strongest anatomical edge,
    # even when weaker edges are later removed by --min-weight.
    # ---------------------------------------------------------

    print("\nCalculating outgoing normalization...")

    row_sizes = np.diff(W.indptr)
    row_max = np.zeros(N, dtype=np.float32)
    nonempty = row_sizes > 0
    starts = W.indptr[:-1][nonempty]
    row_max[nonempty] = np.maximum.reduceat(W.data, starts).astype(np.float32)

    # ---------------------------------------------------------
    # Filter weak edges once, instead of checking every edge
    # on every simulation step.
    # ---------------------------------------------------------

    if args.min_weight > 1:
        print(f"Filtering connections below weight {args.min_weight}...")
        W.data[W.data < args.min_weight] = 0
        W.eliminate_zeros()
        W.sort_indices()

    print(f"Connections used: {W.nnz:,}")

    # ---------------------------------------------------------
    # GPU connectivity
    #
    # build_full.py stores:
    #     W[pre, post] = synapse count
    #
    # We need postsynaptic current:
    #     current_post = W.T @ activity_pre
    #
    # so upload W.T as a CSR matrix.
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("UPLOADING CONNECTOME TO GPU")
    print("=" * 80)
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"ROCm: {torch.version.hip}")

    upload_start = time.perf_counter()

    W_post_pre = W.transpose().tocsr().astype(np.float32, copy=False)
    W_post_pre.sort_indices()

    crow = torch.tensor(W_post_pre.indptr, dtype=torch.int64, device=device)
    col = torch.tensor(W_post_pre.indices, dtype=torch.int64, device=device)
    values = torch.tensor(W_post_pre.data, dtype=torch.float32, device=device)

    W_gpu = torch.sparse_csr_tensor(
        crow,
        col,
        values,
        size=(N, N),
        dtype=torch.float32,
        device=device,
    )

    # Fold sign, synaptic gain, and outgoing normalization into
    # one dense presynaptic scale vector. Then every propagation is:
    #
    #   W_gpu @ (fired * pre_scale)
    #
    # which is algebraically the same update as the CPU loop.
    pre_scale = np.zeros(N, dtype=np.float32)
    valid = row_max > 0
    pre_scale[valid] = (
        np.asarray(nt_sign[valid], dtype=np.float32)
        * np.float32(args.synaptic_gain)
        / row_max[valid]
    )
    pre_scale_gpu = torch.tensor(pre_scale, dtype=torch.float32, device=device)

    stim_indices_cpu = []

    # W is no longer needed after the transposed GPU copy is built.
    del W
    del W_post_pre
    del crow
    del col
    del values
    del pre_scale

    torch.cuda.synchronize()
    upload_seconds = time.perf_counter() - upload_start

    print(f"GPU upload/prep: {upload_seconds:.2f} s")
    print(
        "GPU memory allocated: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
    )

    # ---------------------------------------------------------
    # Stimulated neurons
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("STIMULUS")
    print("=" * 80)

    for body_id in args.stimulate:
        idx = body_to_index(neuron_ids, body_id)
        if idx is None:
            raise RuntimeError(f"bodyId {body_id} is not in the graph")

        stim_indices_cpu.append(idx)

        if body_id in metadata_lookup.index:
            row = metadata_lookup.loc[body_id]
            name = row.get("instance")
            nt = row.get("consensus_nt")
            if pd.isna(nt):
                nt = row.get("predicted_nt")
        else:
            name = None
            nt = None

        print(f"{body_id:<10} {str(name):<30} NT={nt}")

    stim_indices_cpu = np.asarray(stim_indices_cpu, dtype=np.int64)
    stim_indices_gpu = torch.tensor(stim_indices_cpu, dtype=torch.int64, device=device)

    # ---------------------------------------------------------
    # Targets
    # ---------------------------------------------------------

    target_indices = {}

    print("\nTARGETS")
    print("-" * 80)

    for body_id in args.target:
        idx = body_to_index(neuron_ids, body_id)
        if idx is None:
            print(f"{body_id}: not found")
            continue

        target_indices[body_id] = idx

        if body_id in metadata_lookup.index:
            name = metadata_lookup.loc[body_id].get("instance")
        else:
            name = None

        print(f"{body_id:<10} {name}")

    # ---------------------------------------------------------
    # GPU state
    # ---------------------------------------------------------

    voltage = torch.zeros(N, dtype=torch.float32, device=device)
    spike_counts_gpu = torch.zeros(N, dtype=torch.int32, device=device)
    previous_activity = torch.zeros(N, dtype=torch.float32, device=device)
    previous_active_count = 0

    # Keep event indices on the GPU during the run and copy them
    # back only once at the end. This avoids a D2H transfer every step.
    recorded_indices_gpu = []
    recorded_steps = []
    recorded_counts = []
    recorded_event_count = 0
    recording_enabled = True

    # ---------------------------------------------------------
    # Simulation
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("RUNNING WHOLE-CNS GPU SIMULATION")
    print("=" * 80)
    print(f"Steps:            {args.steps}")
    print(f"Minimum weight:   {args.min_weight}")
    print(f"Synaptic gain:    {args.synaptic_gain}")
    print(f"Threshold:        {args.threshold}")
    print(f"Leak:             {args.leak}")

    sim_start = time.perf_counter()
    completed_steps = 0

    with torch.inference_mode():
        for t in range(args.steps):
            completed_steps = t + 1

            # -------------------------------------------------
            # Leak
            # -------------------------------------------------
            voltage.mul_(args.leak)

            # -------------------------------------------------
            # Propagate previous timestep on GPU.
            # Skip the sparse matvec entirely when nobody fired.
            # -------------------------------------------------
            if previous_active_count > 0:
                propagated = torch.sparse.mm(
                    W_gpu,
                    previous_activity.unsqueeze(1),
                ).squeeze(1)
                voltage.add_(propagated)

            # -------------------------------------------------
            # External stimulus
            # -------------------------------------------------
            if args.stim_start <= t < args.stim_end:
                voltage[stim_indices_gpu] += args.stim_current

            # -------------------------------------------------
            # Threshold + reset
            # -------------------------------------------------
            fired_mask = voltage >= args.threshold
            active_count = int(fired_mask.sum().item())

            if active_count > 0:
                spike_counts_gpu.add_(fired_mask.to(torch.int32))
                voltage.masked_fill_(fired_mask, 0.0)

            # -------------------------------------------------
            # Save events
            # -------------------------------------------------
            if recording_enabled and active_count > 0:
                new_total = recorded_event_count + active_count

                if new_total <= args.max_recorded_events:
                    fired_indices = torch.nonzero(
                        fired_mask,
                        as_tuple=False,
                    ).flatten().to(torch.int32)

                    recorded_indices_gpu.append(fired_indices)
                    recorded_steps.append(t)
                    recorded_counts.append(active_count)
                    recorded_event_count = new_total
                else:
                    recording_enabled = False
                    print(
                        "\nEvent recording limit reached; "
                        "continuing simulation without recording every spike."
                    )

            # Presynaptic activity used on the next timestep.
            # Unknown transmitter signs are already zero in pre_scale_gpu.
            previous_activity = fired_mask.to(torch.float32)
            previous_activity.mul_(pre_scale_gpu)
            previous_active_count = active_count

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------
            if t % 25 == 0 or t == args.steps - 1:
                ever_active = int(torch.count_nonzero(spike_counts_gpu).item())
                print(
                    f"step {t:>5}/{args.steps}  "
                    f"active={active_count:>7,}  "
                    f"ever-active={ever_active:>7,}"
                )

            # -------------------------------------------------
            # Avalanche protection
            # -------------------------------------------------
            if (
                args.abort_fraction > 0
                and active_count > N * args.abort_fraction
            ):
                print()
                print("SIMULATION ABORTED:")
                print(f"{active_count:,} neurons fired in one timestep.")
                print(
                    "The toy dynamics entered a whole-network activity avalanche."
                )
                print(
                    "Try a higher --min-weight, lower --synaptic-gain, "
                    "or higher --threshold."
                )
                break

    torch.cuda.synchronize()
    sim_seconds = time.perf_counter() - sim_start

    # ---------------------------------------------------------
    # Copy final state back to CPU once
    # ---------------------------------------------------------

    spike_counts = spike_counts_gpu.cpu().numpy()

    print()
    print(f"GPU simulation time: {sim_seconds:.2f} s")
    if completed_steps:
        print(f"Average: {1000.0 * sim_seconds / completed_steps:.2f} ms/step")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    active = np.flatnonzero(spike_counts)

    print(f"Neurons that fired: {len(active):,} / {N:,}")
    print(f"Total spikes: {int(spike_counts.sum()):,}")

    # ---------------------------------------------------------
    # Top active neurons
    # ---------------------------------------------------------

    ranked = active[np.argsort(spike_counts[active])[::-1]]

    print("\nTOP ACTIVE NEURONS")
    print("-" * 80)

    rows = []

    for idx in ranked[:50]:
        body_id = int(neuron_ids[idx])
        count = int(spike_counts[idx])

        if body_id in metadata_lookup.index:
            row = metadata_lookup.loc[body_id]
            instance = row.get("instance")
            superclass = row.get("superclass")
        else:
            instance = None
            superclass = None

        rows.append(
            {
                "bodyId": body_id,
                "instance": instance,
                "superclass": superclass,
                "spikes": count,
            }
        )

        print(f"{body_id:<10} {str(instance):<32} {count:>7} spikes")

    pd.DataFrame(rows).to_csv(output / "top_active.csv", index=False)

    # ---------------------------------------------------------
    # Target activity
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("TARGET ACTIVITY")
    print("=" * 80)

    for body_id, idx in target_indices.items():
        count = int(spike_counts[idx])

        if body_id in metadata_lookup.index:
            name = metadata_lookup.loc[body_id].get("instance")
        else:
            name = None

        print(f"{body_id:<10} {str(name):<30} {count:>7} spikes")

    # ---------------------------------------------------------
    # Find active motor neurons
    # ---------------------------------------------------------

    instance_text = metadata["instance"].fillna("").astype(str)
    superclass_text = metadata["superclass"].fillna("").astype(str)

    motor_rows = metadata[
        instance_text.str.contains(r"\bMN\b", case=False, regex=True)
        | superclass_text.str.contains("motor", case=False, regex=False)
    ]

    active_motors = []

    for body_id in motor_rows["bodyId"].values:
        idx = body_to_index(neuron_ids, int(body_id))
        if idx is None:
            continue

        count = int(spike_counts[idx])
        if count == 0:
            continue

        row = metadata_lookup.loc[int(body_id)]
        active_motors.append(
            {
                "bodyId": int(body_id),
                "instance": row.get("instance"),
                "spikes": count,
            }
        )

    active_motors = pd.DataFrame(active_motors)

    if not active_motors.empty:
        active_motors = active_motors.sort_values("spikes", ascending=False)

        print()
        print("=" * 80)
        print("ACTIVE MOTOR NEURONS")
        print("=" * 80)
        print(active_motors.head(50).to_string(index=False))

        active_motors.to_csv(output / "active_motor_neurons.csv", index=False)

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    np.savez_compressed(
        output / "spike_counts.npz",
        neuron_ids=np.asarray(neuron_ids),
        spike_counts=spike_counts,
    )

    if recorded_indices_gpu:
        all_indices_gpu = torch.cat(recorded_indices_gpu).to(torch.int64)
        indices = all_indices_gpu.cpu().numpy()

        times = np.repeat(
            np.asarray(recorded_steps, dtype=np.int32),
            np.asarray(recorded_counts, dtype=np.int64),
        )

        event_body_ids = np.asarray(neuron_ids[indices])

        np.savez_compressed(
            output / "spike_events.npz",
            time=times,
            bodyId=event_body_ids,
        )

    params = {
        "stimulate": args.stimulate,
        "targets": args.target,
        "steps": args.steps,
        "stim_start": args.stim_start,
        "stim_end": args.stim_end,
        "stim_current": args.stim_current,
        "threshold": args.threshold,
        "leak": args.leak,
        "synaptic_gain": args.synaptic_gain,
        "min_weight": args.min_weight,
    }

    with open(output / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    print()
    print(f"Saved results to: {output}")


if __name__ == "__main__":
    main()

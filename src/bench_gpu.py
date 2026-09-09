import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


path = Path("data/cache/connectome_raw.npz")

print("Loading connectome...")
W = sp.load_npz(path).tocsr().astype(np.float32)

n = W.shape[0]

print(f"Shape: {W.shape}")
print(f"Edges: {W.nnz:,}")
print(f"Density: {W.nnz / (n*n):.8f}")

# ------------------------------------------------------------
# Random activity vector
# ------------------------------------------------------------

rng = np.random.default_rng(42)

x_np = np.zeros(n, dtype=np.float32)

active = rng.choice(
    n,
    size=min(5000, n),
    replace=False,
)

x_np[active] = 1.0


# ------------------------------------------------------------
# CPU benchmark
# ------------------------------------------------------------

print("\nCPU benchmark...")

for _ in range(5):
    y_cpu = W @ x_np

start = time.perf_counter()

iterations = 100

for _ in range(iterations):
    y_cpu = W @ x_np

cpu_time = (
    time.perf_counter() - start
) / iterations

print(
    f"CPU: {cpu_time * 1000:.3f} ms / propagation"
)


# ------------------------------------------------------------
# GPU CSR
# ------------------------------------------------------------

print("\nUploading connectome to GPU...")

device = torch.device("cuda")

crow = torch.tensor(
    W.indptr,
    dtype=torch.int64,
    device=device,
)

col = torch.tensor(
    W.indices,
    dtype=torch.int64,
    device=device,
)

values = torch.tensor(
    W.data,
    dtype=torch.float32,
    device=device,
)

W_gpu = torch.sparse_csr_tensor(
    crow,
    col,
    values,
    size=W.shape,
    device=device,
)

x_gpu = torch.tensor(
    x_np,
    dtype=torch.float32,
    device=device,
).reshape(-1, 1)

print(
    "GPU memory allocated:",
    round(
        torch.cuda.memory_allocated()
        / 1024**2,
        1,
    ),
    "MB",
)


# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------

print("\nGPU warmup...")

for _ in range(20):
    y_gpu = torch.sparse.mm(
        W_gpu,
        x_gpu,
    )

torch.cuda.synchronize()


# ------------------------------------------------------------
# GPU benchmark
# ------------------------------------------------------------

print("GPU benchmark...")

start = time.perf_counter()

for _ in range(iterations):
    y_gpu = torch.sparse.mm(
        W_gpu,
        x_gpu,
    )

torch.cuda.synchronize()

gpu_time = (
    time.perf_counter() - start
) / iterations

print(
    f"GPU: {gpu_time * 1000:.3f} ms / propagation"
)

print(
    f"\nSpeedup: {cpu_time / gpu_time:.2f}x"
)


# ------------------------------------------------------------
# Verify numerical agreement
# ------------------------------------------------------------

gpu_result = (
    y_gpu
    .squeeze(1)
    .cpu()
    .numpy()
)

max_error = np.max(
    np.abs(
        gpu_result - y_cpu
    )
)

print(
    f"Max CPU/GPU difference: {max_error}"
)

print("\nGPU:", torch.cuda.get_device_name(0))

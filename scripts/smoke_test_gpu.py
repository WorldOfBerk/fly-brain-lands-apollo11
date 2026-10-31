"""Loads the weight matrix on the GPU and times one whole-brain sparse mat-vec."""
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"

W = sp.load_npz(PROC / "weights_csr.npz")
dev = torch.device("cuda")
Wt = torch.sparse_csr_tensor(
    torch.from_numpy(W.indptr.astype(np.int64)),
    torch.from_numpy(W.indices.astype(np.int64)),
    torch.from_numpy(W.data),
    size=W.shape,
    device=dev,
)
spikes = (torch.rand(W.shape[0], device=dev) < 0.01).float()

for _ in range(10):
    Wt @ spikes
torch.cuda.synchronize()
steps = 1000
t0 = time.perf_counter()
for _ in range(steps):
    Wt @ spikes
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / steps

print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"per step: {dt * 1e3:.2f} ms")
print(f"1 s of simulation at 0.1 ms steps ~ {dt * 10_000:.1f} s wall time")

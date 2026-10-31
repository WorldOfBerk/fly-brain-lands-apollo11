"""PyTorch/GPU port of the whole-brain LIF model of Shiu et al. (2024, Nature).

Equations and default parameters follow the original Brian2 model:
    dv/dt = (v_0 - v + g) / t_mbr      (when not refractory)
    dg/dt = -g / tau                    (when not refractory)
    threshold: v > v_th  ->  v = v_rst, g = 0, frozen for t_rfc
    a presynaptic spike adds w_syn * (sign * synapse count) to g_post after t_dly
    stimulated neurons get Poisson input at r_poi: v += w_syn * f_poi

Forward Euler, dt = 0.1 ms. Several trials run in parallel: state tensors are (n, B).
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class LIFParams:
    dt: float = 0.1  # ms
    v_0: float = -52.0  # mV
    v_rst: float = -52.0
    v_th: float = -45.0
    t_mbr: float = 20.0  # ms
    tau: float = 5.0
    t_rfc: float = 2.2
    t_dly: float = 1.8
    w_syn: float = 0.275  # mV per synapse
    f_poi: float = 250.0
    # Spike-frequency adaptation (not in Shiu's model; 0 = off). Each spike adds a_inc mV to the
    # adaptation current, which decays with tau_a: dv/dt = (v_0 - v + g - a) / t_mbr. GraphLIFBrain only.
    a_inc: float = 0.0  # mV
    tau_a: float = 300.0  # ms
    # Conductance-based synapses (Shiu's model is current-based; False = original). Weights are scaled so
    # the effect at rest is unchanged: g_e*(E_e - v)/(E_e - v_0), g_i*(E_i - v)/(E_i - v_0).
    conductance: bool = False
    e_exc: float = 0.0  # mV
    e_inh: float = -75.0  # mV (GABA_A / GluCl)


class LIFBrain:
    """mode='event': only outgoing synapses of neurons that fired are processed (~200 spikes x ~150
    synapses per step per trial). mode='spmm': sparse matmul with the full weight matrix every step (slow).
    Both run the same equations; spmm is kept as a reference."""

    def __init__(self, W: sp.csr_matrix, n_trials: int = 1, params: LIFParams | None = None, device: str = "cuda",
                 mode: str = "event"):
        self.p = params or LIFParams()
        self.dev = torch.device(device)
        self.n = W.shape[0]
        self.B = n_trials
        self.mode = mode
        if mode == "spmm":
            self.W = torch.sparse_csr_tensor(
                torch.from_numpy(W.indptr.astype(np.int64)),
                torch.from_numpy(W.indices.astype(np.int64)),
                torch.from_numpy((W.data * self.p.w_syn).astype(np.float32)),
                size=W.shape,
                device=self.dev,
            )
        else:
            Wc = W.tocsc()  # column = presynaptic neuron, outgoing synapses contiguous
            self.colptr = torch.from_numpy(Wc.indptr.astype(np.int64)).to(self.dev)
            self.post = torch.from_numpy(Wc.indices.astype(np.int64)).to(self.dev)
            self.wval = torch.from_numpy((Wc.data * self.p.w_syn).astype(np.float32)).to(self.dev)
        self.delay_steps = round(self.p.t_dly / self.p.dt)
        self.rfc_steps = round(self.p.t_rfc / self.p.dt)
        self.reset()

    def reset(self):
        shape = (self.n, self.B)
        self.v = torch.full(shape, self.p.v_0, device=self.dev)
        self.g = torch.zeros(shape, device=self.dev)
        self.rfc = torch.zeros(shape, dtype=torch.int16, device=self.dev)
        # delay ring buffer
        self.buf = torch.zeros((self.delay_steps, *shape), dtype=torch.bool, device=self.dev)
        self.t = 0
        self.rate_hz = torch.zeros(shape, device=self.dev)  # Poisson input rate (neuron x trial)
        self.inp = torch.zeros(0, dtype=torch.long, device=self.dev)  # neurons with input

    def set_input(self, idx, rate_hz):
        """Poisson drive to the given neurons (0 turns it off).
        rate_hz: scalar for all trials or a (B,) array."""
        rate = torch.as_tensor(rate_hz, device=self.dev, dtype=torch.float32)
        self.rate_hz[torch.as_tensor(idx, device=self.dev, dtype=torch.long)] = rate
        self.inp = torch.nonzero(self.rate_hz.any(dim=1)).squeeze(1)

    def _propagate(self, arriving: torch.Tensor):
        """g[post, b] += w for every arriving spike."""
        if self.mode == "spmm":
            self.g += self.W @ arriving.float()
            return
        nz = torch.nonzero(arriving)
        if nz.shape[0] == 0:
            return
        pre, trial = nz[:, 0], nz[:, 1]
        start = self.colptr[pre]
        cnt = self.colptr[pre + 1] - start
        total = int(cnt.sum())
        if total == 0:
            return
        seg = torch.repeat_interleave(torch.arange(len(pre), device=self.dev), cnt, output_size=total)
        first = torch.cumsum(cnt, 0) - cnt
        pos = start[seg] + torch.arange(total, device=self.dev) - first[seg]
        self.g.view(-1).index_add_(0, self.post[pos] * self.B + trial[seg], self.wval[pos])

    @torch.no_grad()
    def step(self) -> torch.Tensor:
        p = self.p
        active = self.rfc <= 0

        slot = self.t % self.delay_steps
        self._propagate(self.buf[slot])

        dv = (p.v_0 - self.v + self.g) * (p.dt / p.t_mbr)
        dg = -self.g * (p.dt / p.tau)
        self.v = torch.where(active, self.v + dv, self.v)
        self.g = torch.where(active, self.g + dg, self.g)

        # Poisson input also applies while refractory (Brian2's PoissonInput writes to v directly)
        if len(self.inp):
            prob = self.rate_hz[self.inp] * (p.dt * 1e-3)
            poisson = torch.rand_like(prob) < prob
            self.v[self.inp] += poisson * (p.w_syn * p.f_poi)

        spikes = (self.v > p.v_th) & active
        self.v = torch.where(spikes, torch.full_like(self.v, p.v_rst), self.v)
        self.g = torch.where(spikes, torch.zeros_like(self.g), self.g)
        self.rfc = torch.where(spikes, torch.full_like(self.rfc, self.rfc_steps), (self.rfc - 1).clamp_min(0))

        self.buf[slot] = spikes
        self.t += 1
        return spikes.float()

    @torch.no_grad()
    def run_counts(self, duration_ms: float) -> torch.Tensor:
        """Spike counts (neuron x trial) over duration_ms, on the GPU."""
        counts = torch.zeros((self.n, self.B), device=self.dev)
        for _ in range(round(duration_ms / self.p.dt)):
            counts += self.step()
        return counts

    @torch.no_grad()
    def run(self, duration_ms: float, record_idx=None):
        """Returns (mean rate per neuron [Hz] (n,), spike raster (step, len(record_idx), B) or None)."""
        steps = round(duration_ms / self.p.dt)
        counts = torch.zeros((self.n, self.B), device=self.dev)
        rec = None
        if record_idx is not None:
            ridx = torch.as_tensor(record_idx, device=self.dev, dtype=torch.long)
            rec = torch.zeros((steps, len(ridx), self.B), dtype=torch.bool, device=self.dev)
        for k in range(steps):
            s = self.step()
            counts += s
            if rec is not None:
                rec[k] = s[ridx].bool()
        rate = counts.mean(dim=1) / (duration_ms * 1e-3)
        return rate.cpu().numpy(), (rec.cpu().numpy() if rec is not None else None)

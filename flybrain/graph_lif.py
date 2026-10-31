"""CUDA graph version of LIFBrain: same equations, one GPU launch per step.

Event-driven propagation uses fixed-size buffers (nonzero_static + searchsorted) so a whole step can be
captured once and replayed. One graph per slot of the delay ring. If a buffer overflows (too many
spikes/synapses in a step) the `overflow` flag is set and run_counts grows the buffers.
"""
import torch

from .lif import LIFBrain


class GraphLIFBrain(LIFBrain):
    # Measured peaks (LPLC2/LB3c drive, 4 trials): 237 spikes and 87,846 synapses per step per trial.
    def __init__(self, W, n_trials=1, params=None, device="cuda", max_spikes_per_trial=500,
                 max_syn_per_trial=160_000, max_inputs=4096):
        self.K = max_spikes_per_trial * n_trials
        self.S = max_syn_per_trial * n_trials
        self.M = max_inputs
        self.overflow_events = 0
        super().__init__(W, n_trials, params, device, mode="event")

    def reset(self):
        super().reset()
        dev = self.dev
        self.inp_buf = torch.zeros(self.M, dtype=torch.long, device=dev)
        self.inp_valid = torch.zeros(self.M, dtype=torch.bool, device=dev)
        self.counts = torch.zeros((self.n, self.B), device=dev)
        self.a = torch.zeros((self.n, self.B), device=dev)  # adaptation current (mV)
        self.gi = torch.zeros((self.n, self.B), device=dev)  # inhibitory input (g is excitatory only)
        self.overflow = torch.zeros((), dtype=torch.bool, device=dev)
        self._j = torch.arange(self.S, device=dev)
        self._k = torch.arange(self.K, device=dev)
        self.graphs = None

    def set_input(self, idx, rate_hz):
        super().set_input(idx, rate_hz)
        n = len(self.inp)
        if n > self.M:
            raise ValueError(f"at most {self.M} input neurons")
        self.inp_buf.zero_()
        self.inp_buf[:n] = self.inp
        self.inp_valid.zero_()
        self.inp_valid[:n] = True

    def _propagate_static(self, arriving):
        nz = torch.nonzero_static(arriving, size=self.K, fill_value=0)
        nvalid = arriving.sum()
        valid = self._k < nvalid
        pre, trial = nz[:, 0], nz[:, 1]
        start = self.colptr[pre]
        cnt = (self.colptr[pre + 1] - start) * valid
        csum = torch.cumsum(cnt, 0)
        first = csum - cnt
        seg = torch.searchsorted(csum, self._j, right=True).clamp_(max=self.K - 1)
        ok = self._j < csum[-1]
        pos = torch.where(ok, start[seg] + self._j - first[seg], 0)
        w = torch.where(ok, self.wval[pos], 0.0)
        tgt = self.post[pos] * self.B + trial[seg]
        self.g.view(-1).index_add_(0, tgt, w.clamp(min=0.0))  # excitatory
        self.gi.view(-1).index_add_(0, tgt, w.clamp(max=0.0))  # inhibitory
        self.overflow |= (nvalid > self.K) | (csum[-1] > self.S)

    def _step_static(self, slot):
        p = self.p
        active = self.rfc <= 0
        self._propagate_static(self.buf[slot])
        if p.conductance:
            syn = (self.g * ((p.e_exc - self.v) / (p.e_exc - p.v_0))
                   + self.gi * ((p.e_inh - self.v) / (p.e_inh - p.v_0)))
        else:
            syn = self.g + self.gi
        dv = (p.v_0 - self.v + syn - self.a) * (p.dt / p.t_mbr)
        decay = 1.0 - p.dt / p.tau
        self.v.copy_(torch.where(active, self.v + dv, self.v))
        self.g.copy_(torch.where(active, self.g * decay, self.g))
        self.gi.copy_(torch.where(active, self.gi * decay, self.gi))
        self.a.mul_(1.0 - p.dt / p.tau_a)
        prob = self.rate_hz[self.inp_buf] * (p.dt * 1e-3)
        kick = (torch.rand_like(prob) < prob) & self.inp_valid.unsqueeze(1)
        self.v.index_add_(0, self.inp_buf, kick * (p.w_syn * p.f_poi))
        spikes = (self.v > p.v_th) & active
        self.v.masked_fill_(spikes, p.v_rst)
        self.g.masked_fill_(spikes, 0.0)
        self.gi.masked_fill_(spikes, 0.0)
        self.rfc.copy_(torch.where(spikes, torch.full_like(self.rfc, self.rfc_steps), (self.rfc - 1).clamp_min(0)))
        self.a.add_(spikes * p.a_inc)
        self.buf[slot].copy_(spikes)
        self.counts += spikes

    def _capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        state = [t.clone() for t in (self.v, self.g, self.gi, self.rfc, self.buf, self.counts, self.a)]
        with torch.cuda.stream(s):
            for slot in range(self.delay_steps):  # warm-up
                self._step_static(slot)
        torch.cuda.current_stream().wait_stream(s)
        self.graphs = []
        for slot in range(self.delay_steps):
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                self._step_static(slot)
            self.graphs.append(gph)
        for dst, src in zip((self.v, self.g, self.gi, self.rfc, self.buf, self.counts, self.a), state):  # undo warm-up
            dst.copy_(src)
        self.overflow.zero_()

    @torch.no_grad()
    def run_counts(self, duration_ms: float) -> torch.Tensor:
        if self.graphs is None:
            self._capture()
        self.counts.zero_()
        for _ in range(round(duration_ms / self.p.dt)):
            self.graphs[self.t % self.delay_steps].replay()
            self.t += 1
        if bool(self.overflow):
            # some synapses may have been dropped in this window: warn, grow buffers, recapture
            self.overflow_events += 1
            print(f"WARNING: GraphLIFBrain buffer overflow (t={self.t}); growing buffers", flush=True)
            self.K, self.S = int(self.K * 1.5), int(self.S * 1.5)
            self._j = torch.arange(self.S, device=self.dev)
            self._k = torch.arange(self.K, device=self.dev)
            self.overflow.zero_()
            self.graphs = None
        return self.counts.clone()

    @torch.no_grad()
    def run(self, duration_ms: float, record_idx=None):
        if record_idx is not None:
            raise NotImplementedError("use LIFBrain for recording")
        counts = self.run_counts(duration_ms)
        return (counts.mean(dim=1) / (duration_ms * 1e-3)).cpu().numpy(), None

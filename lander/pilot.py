"""Analytic fly pilot (phases 2-3): feeds a looming signal to the brain and turns the Giant Fiber
(DNp01) response into P66 ROD clicks. Vertical axis only.

Encoder: approaching flat ground, image expansion rate = descent rate / altitude (1 / time to contact),
         given to LPLC2 looming neurons as Poisson input.
Brain:   MaleCNS v1.0 LIF model (165,122 neurons), stepped in lockstep with the LM.
Decoder: Giant Fiber above the upper threshold -> ROD up (slow down); below the lower threshold ->
         ROD down (speed up, at most once per second); otherwise no click.

At steady state the Giant Fiber is silent below ~80 Hz LPLC2 input and ramps (~15 -> ~310 Hz) between
80 and 150 Hz; k_enc maps that range to ~0.08-0.15 1/s expansion.

The resulting behaviour resembles the honeybee landing strategy (hold expansion rate constant, so speed
falls in proportion to height). Thresholds and gain are design choices; the brain does looming -> GF.
"""
from dataclasses import dataclass

import numpy as np

from flybrain import Connectome, GraphLIFBrain


@dataclass
class FlyPilotParams:
    tick_ms: float = 200.0  # brain <-> LM update period, at most one ROD click per tick
    k_enc: float = 1000.0  # Hz * s: LPLC2 rate = k_enc * (descent rate / altitude)
    max_rate: float = 150.0  # Hz
    gf_up: float = 100.0  # Hz, above: slow down
    gf_down: float = 30.0  # Hz, below: speed up
    down_every: int = 5  # speed-up click at most every N ticks (5 x 200 ms = 1 s)
    ema: float = 0.5  # EMA factor for the Giant Fiber rate


class FlyPilot:
    def __init__(self, n_trials: int, p: FlyPilotParams | None = None, blind=None, connectome=None):
        self.p = p or FlyPilotParams()
        self.c = connectome or Connectome()
        self.brain = GraphLIFBrain(self.c.W, n_trials=n_trials)
        self.lplc2 = self.c.idx("LPLC2")
        self.gf = self.c.idx("DNp01")
        # blind-fly control: no visual input in these trials
        self.blind = np.zeros(n_trials, bool) if blind is None else np.asarray(blind)
        self.gf_est = np.zeros(n_trials)
        self.n_ticks = 0

    def encode(self, h, v):
        expansion = np.maximum(-v, 0.0) / np.maximum(h, 0.5)
        rate = np.clip(self.p.k_enc * expansion, 0.0, self.p.max_rate)
        return np.where(self.blind, 0.0, rate)

    def tick(self, h, v, active):
        """Run the brain for one tick. Returns (ROD clicks (B,), LPLC2 input rate, spike counts (n, B) on GPU)."""
        rate = np.where(active, self.encode(h, v), 0.0)
        self.brain.set_input(self.lplc2, rate)
        counts = self.brain.run_counts(self.p.tick_ms)
        gf_rate = counts[self.gf].mean(0).cpu().numpy() / (self.p.tick_ms * 1e-3)
        self.gf_est = self.p.ema * gf_rate + (1 - self.p.ema) * self.gf_est
        may_speed_up = self.n_ticks % self.p.down_every == 0
        self.n_ticks += 1
        clicks = np.where(self.gf_est > self.p.gf_up, 1,
                          np.where((self.gf_est < self.p.gf_down) & may_speed_up, -1, 0))
        return np.where(active, clicks, 0), rate, counts

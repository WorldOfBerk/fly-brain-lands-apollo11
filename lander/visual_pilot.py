"""Visual fly pilot: in P66 the fly works both the ROD switch (vertical) and the attitude controller
(horizontal), using motion signals computed from what its own eye sees.

Input (apollo_sim.flyeye), rendered every physics step and passed through Reichardt correlators:
  expansion (looming)     -> LPLC2 Poisson rate
  front-to-back flow      -> T4a + T5a
  back-to-front flow      -> T4b + T5b
Brain: MaleCNS v1.0, calibrated weights (scripts/calibrate_connectome.py), conductance-based LIF.
Output:
  Giant Fiber (DNp01)                       -> ROD: high rate = slow down, low rate = speed up
  brake DNs  (DNp26, DNbe001, DNp20)        selective for front-to-back flow
  accel DNs  (DNg46, DNp53, DNge043)        selective for back-to-front flow
     -> proportional ACA: tilt = -15 deg * (brake - accel) / 200 Hz. A first version used 1 deg clicks
        above a threshold and ended up in a +-4 ft/s limit cycle.
  The DNs were picked by measuring direction selectivity in the model (T4a/T5a vs T4b/T5b drive).

Gains are design choices: calibrate_gains() maps eye signals to Hz so that
  expansion 0.05 1/s     -> LPLC2 13 Hz (GF ~100 Hz, near the ROD threshold)
  lateral flow 0.025 rad/s -> T4a/T5a 15 Hz (brake DNs ~50 Hz)
"""
from dataclasses import dataclass

import numpy as np

from apollo_sim.flyeye import FlyEye
from apollo_sim.terrain import Terrain
from flybrain import Connectome, GraphLIFBrain, LIFParams

BRAKE_DN = ("DNp26", "DNbe001", "DNp20")
ACCEL_DN = ("DNg46", "DNp53", "DNge043")


@dataclass
class VisualPilotParams:
    tick_ms: float = 200.0
    ref_expansion: float = 0.05  # 1/s
    ref_expansion_hz: float = 13.0
    ref_flow: float = 0.025  # rad/s
    ref_flow_hz: float = 15.0
    max_rate: float = 150.0
    gf_up: float = 100.0
    gf_down: float = 40.0
    down_every: int = 5
    dn_sat: float = 200.0  # Hz of (brake - accel) for full tilt
    tilt_max_deg: float = 15.0
    ema: float = 0.5


def calibrate_gains(surface, h=50.0, dt=0.05, T=3.0, p=VisualPilotParams(), x0=-1500.0):
    """Measure the eye's mean signal for reference motions and return gains that map it to Hz.
    Measured over typical mare away from the landing site (x0)."""
    def run(vh, vv):
        eye = FlyEye(surface)
        x, hh, out = x0, h, []
        for k in range(int(T / dt)):
            f, b, ex, _, _ = eye.motion(eye.render(x, surface.height(x) + hh, 0.0), dt)
            if k * dt > 1.0:
                out.append((f, ex))
            x += vh * dt
            hh += vv * dt
        return np.mean(out, 0)

    flow = run(p.ref_flow * h, 0.0)[0]
    expn = run(0.0, -p.ref_expansion * h)[1]
    return {"k_flow": p.ref_flow_hz / flow, "k_expansion": p.ref_expansion_hz / expn,
            "measured_flow": float(flow), "measured_expansion": float(expn)}


class VisualFlyPilot:
    def __init__(self, n_trials, gains, p=None, blind=None, connectome=None, surface=None):
        self.p = p or VisualPilotParams()
        self.c = connectome or Connectome()
        self.brain = GraphLIFBrain(self.c.W, n_trials=n_trials, params=LIFParams(conductance=True), max_inputs=8192)
        self.gains = gains
        self.surface = surface or Terrain()
        self.eyes = [FlyEye(self.surface) for _ in range(n_trials)]
        self.B = n_trials
        self.blind = np.zeros(n_trials, bool) if blind is None else np.asarray(blind)
        self.lplc2 = self.c.idx("LPLC2")
        self.prog = self.c.idx("T4a", "T5a")
        self.reg = self.c.idx("T4b", "T5b")
        self.gf = self.c.idx("DNp01")
        self.brake = self.c.idx(*BRAKE_DN)
        self.accel = self.c.idx(*ACCEL_DN)
        self.gf_est = np.zeros(n_trials)
        self.dn_est = np.zeros(n_trials)
        self.n_ticks = 0
        self._acc = np.zeros((n_trials, 3))
        self._n = np.zeros(n_trials)
        self.last_image = [None] * n_trials

    def observe(self, i, x_downrange, y_abs, pitch, dt):
        """Update trial i's eye each physics step. y_abs: height above the mean radius.

        Head stabilisation: flies keep the head level with neck reflexes while the body pitches, so the
        eye is held level regardless of LM pitch. (With the eye tilted along with the LM, forward motion
        showed up as false expansion and the Giant Fiber fired.)"""
        img = self.eyes[i].render(x_downrange, y_abs, 0.0)
        f, b, ex, _, _ = self.eyes[i].motion(img, dt)
        self._acc[i] += (f, b, ex)
        self._n[i] += 1
        self.last_image[i] = img

    def tick(self, active):
        p, g = self.p, self.gains
        sig = self._acc / np.maximum(self._n, 1)[:, None]
        self._acc[:] = 0
        self._n[:] = 0
        see = active & ~self.blind
        r_loom = np.where(see, np.clip(g["k_expansion"] * sig[:, 2], 0, p.max_rate), 0.0)
        r_prog = np.where(see, np.clip(g["k_flow"] * sig[:, 0], 0, p.max_rate), 0.0)
        r_reg = np.where(see, np.clip(g["k_flow"] * sig[:, 1], 0, p.max_rate), 0.0)
        self.brain.set_input(self.lplc2, r_loom)
        self.brain.set_input(self.prog, r_prog)
        self.brain.set_input(self.reg, r_reg)
        counts = self.brain.run_counts(p.tick_ms)
        sec = p.tick_ms * 1e-3
        gf = counts[self.gf].mean(0).cpu().numpy() / sec
        brake = counts[self.brake].mean(0).cpu().numpy() / sec
        accel = counts[self.accel].mean(0).cpu().numpy() / sec
        self.gf_est = p.ema * gf + (1 - p.ema) * self.gf_est
        self.dn_est = p.ema * (brake - accel) + (1 - p.ema) * self.dn_est
        may_speed_up = self.n_ticks % p.down_every == 0
        self.n_ticks += 1
        rod = np.where(self.gf_est > p.gf_up, 1, np.where((self.gf_est < p.gf_down) & may_speed_up, -1, 0))
        tilt = -np.radians(p.tilt_max_deg) * np.clip(self.dn_est / p.dn_sat, -1.0, 1.0)
        info = {"sig": sig, "r_loom": r_loom, "r_prog": r_prog, "r_reg": r_reg,
                "gf": gf, "brake": brake, "accel": accel}
        return np.where(active, rod, 0), tilt, info, counts

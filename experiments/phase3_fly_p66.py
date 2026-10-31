"""Phase 3: Apollo 11 descent; the AGC flies P63/P64 and the fly works the ROD switch in P66.

Armstrong entered P66 at 102:43:22 at ~470 ft (143 m); control passes to the fly when the AGC's P64
reaches that altitude. Trials run together:
  fly_0..N-1    fly pilot
  blind_fly     same brain, no visual input
Brainless controls:
  agc_auto      automatic P64 -> P65 landing
  no_pilot_p66  P66 entered, nobody clicks ROD

Output: results/phase3/ (telemetry.npz, brain_trial0.npz, summary.json)
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apollo_sim import constants as C  # noqa: E402
from apollo_sim.agc import AGC  # noqa: E402
from apollo_sim.lm import at_ignition  # noqa: E402
from lander.pilot import FlyPilot, FlyPilotParams  # noqa: E402

OUT = ROOT / "results" / "phase3"
HANDOVER_ALT = 470 * C.FT  # where Armstrong entered P66
N_FLY = 2
DT = 0.05
PROBE = 1.7  # m, contact light
SHUTDOWN_DELAY = 1.0  # s
MAX_T = 240.0  # max time in P66
GROUPS = ["LPLC2", "DNp01", "TTMn"]


def fly_to_handover():
    lm = at_ignition()
    agc = AGC(lm)
    t = 0.0
    while lm.altitude > HANDOVER_ALT:
        agc.update(t)
        lm.step(DT)
        t += DT
    return lm, agc, t


class Run:
    """One LM + AGC with contact light and shutdown logic."""

    def __init__(self, lm, agc, name, p66=True):
        self.lm, self.agc, self.name = copy.deepcopy(lm), copy.deepcopy(agc), name
        self.agc.lm = self.lm
        if p66:
            self.agc.start_p66()
        self.contact_t = None
        self.contact_v = None

    def advance(self, t0, duration):
        t = t0
        for _ in range(round(duration / DT)):
            if self.lm.s.landed:
                break
            self.agc.update(t)
            if self.contact_t is None and self.lm.altitude <= PROBE:
                self.contact_t, self.contact_v = t, -self.lm.vertical_speed()
            if self.contact_t is not None and t - self.contact_t >= SHUTDOWN_DELAY:
                self.lm.engine_on = False
            self.lm.step(DT)
            t += DT

    def summary(self, t_handover):
        s = self.lm.s
        vd, vh = s.touchdown_v
        ok = s.landed and vd <= 10 * C.FT and abs(vh) <= 4 * C.FT and s.propellant > 0
        return {"trial": self.name, "outcome": "success" if ok else ("crash" if s.landed else "airborne"),
                "contact_light_speed_mps": self.contact_v, "touchdown_vertical_mps": vd, "touchdown_lateral_mps": vh,
                "fuel_left_kg": s.propellant, "p66_duration_s": (s.t - t_handover) if s.landed else None,
                "landing_time_from_pdi_s": s.t if s.landed else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lm0, agc0, t_h = fly_to_handover()
    print(f"P66 handover: PDI+{t_h:.1f} s, altitude {lm0.altitude:.1f} m, vertical {lm0.vertical_speed():.2f} m/s, "
          f"horizontal {lm0.horizontal_speed():.2f} m/s, fuel {lm0.s.propellant:.0f} kg", flush=True)

    # brainless controls
    controls = [Run(lm0, agc0, "agc_auto", p66=False), Run(lm0, agc0, "no_pilot_p66")]
    for c in controls:
        t = t_h
        while not c.lm.s.landed and t - t_h < MAX_T:
            c.advance(t, 0.2)
            t += 0.2

    # fly trials
    names = [f"fly_{i}" for i in range(N_FLY)] + ["blind_fly"]
    runs = [Run(lm0, agc0, n) for n in names]
    blind = np.array([n == "blind_fly" for n in names])
    pilot = FlyPilot(len(runs), p=FlyPilotParams(k_enc=2000.0), blind=blind)
    tick = pilot.p.tick_ms / 1000
    gidx = {g: torch.as_tensor(pilot.c.idx(g), device=pilot.brain.dev) for g in GROUPS}
    keys = ["t", "h", "vs", "hs", "v_cmd", "thrust", "propellant", "lplc2_in", "clicks", "landed"]
    log = {k: [] for k in keys} | {f"rate_{g}": [] for g in GROUPS}
    b_idx, b_cnt, b_ptr = [], [], [0]

    t = t_h
    t0 = time.perf_counter()
    while not all(r.lm.s.landed for r in runs) and t - t_h < MAX_T:
        h = np.array([max(r.lm.altitude, 0.0) for r in runs])
        vs = np.array([r.lm.vertical_speed() for r in runs])
        active = np.array([not r.lm.s.landed for r in runs])
        clicks, lplc2_in, counts = pilot.tick(h, vs, active)
        for r, c in zip(runs, clicks):
            r.agc.rod_click(int(c))

        for k, v in [("t", t), ("h", h), ("vs", vs), ("hs", [r.lm.horizontal_speed() for r in runs]),
                     ("v_cmd", [r.agc.v_desired for r in runs]), ("thrust", [r.lm.s.thrust for r in runs]),
                     ("propellant", [r.lm.s.propellant for r in runs]), ("lplc2_in", lplc2_in),
                     ("clicks", clicks), ("landed", ~active)]:
            log[k].append(np.array(v))
        for g in GROUPS:
            log[f"rate_{g}"].append((counts[gidx[g]].mean(0) / tick).cpu().numpy())
        nz = torch.nonzero(counts[:, 0]).squeeze(1)
        b_idx.append(nz.cpu().numpy().astype(np.int32))
        b_cnt.append(counts[nz, 0].cpu().numpy().astype(np.uint8))
        b_ptr.append(b_ptr[-1] + len(nz))

        for r in runs:
            r.advance(t, tick)
        t += tick
        if len(log["t"]) % 25 == 0:
            print(f"P66+{t - t_h:5.1f}s ({time.perf_counter() - t0:5.0f}s) h={np.round(h, 1)} v={np.round(vs, 2)} "
                  f"GF={np.round(pilot.gf_est, 0)} fuel={[round(r.lm.s.propellant) for r in runs]}", flush=True)

    summary = {"handover": {"t_from_pdi_s": t_h, "altitude_m": lm0.altitude, "vertical_mps": lm0.vertical_speed(),
                         "horizontal_mps": lm0.horizontal_speed(), "fuel_kg": lm0.s.propellant},
               "trials": [r.summary(t_h) for r in runs + controls]}
    for s in summary["trials"]:
        print(s)
    np.savez_compressed(OUT / "telemetry.npz", names=np.array(names), t_handover=t_h,
                        **{k: np.array(v) for k, v in log.items()})
    np.savez_compressed(OUT / "brain_trial0.npz", idx=np.concatenate(b_idx), cnt=np.concatenate(b_cnt),
                        ptr=np.array(b_ptr), tick_ms=pilot.p.tick_ms)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"total {time.perf_counter() - t0:.0f} s, saved: {OUT}")


if __name__ == "__main__":
    main()

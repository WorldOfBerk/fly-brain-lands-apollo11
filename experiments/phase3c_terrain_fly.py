"""Phase 3c: the fly flies P66 on both axes with its own eye, over Tranquility Base-like terrain
(West crater, boulders, shadows).

Changes from 3b:
  - LM and AGC altitude (landing radar) are relative to the terrain below
  - the eye ray-casts the 3D terrain and samples its shaded luminance
  - touchdown is OK if vertical <= 10 ft/s, lateral <= 4 ft/s and slope under the gear <= 12 deg
  - ACA is proportional (lander/visual_pilot.py)

Output: results/phase3c/
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
from apollo_sim.terrain import Terrain  # noqa: E402
from lander.visual_pilot import VisualFlyPilot, calibrate_gains  # noqa: E402

OUT = ROOT / "results" / "phase3c"
HANDOVER_ALT = 470 * C.FT
N_FLY = 2
DT = 0.05
PROBE = 1.7
SHUTDOWN_DELAY = 1.0
MAX_T = 240.0
MAX_SLOPE = 12.0  # deg, LM landing gear limit


def fly_to_handover(terrain):
    lm = at_ignition(terrain)
    agc = AGC(lm)
    t = 0.0
    while lm.altitude > HANDOVER_ALT:
        agc.update(t)
        lm.step(DT)
        t += DT
    return lm, agc, t


class Run:
    def __init__(self, lm, agc, name, terrain, p66=True, fly=True):
        lm.terrain = None  # don't deepcopy the terrain
        self.lm, self.agc = copy.deepcopy(lm), copy.deepcopy(agc)
        lm.terrain = terrain
        self.lm.terrain = terrain
        self.agc.lm = self.lm
        self.name = name
        if p66:
            self.agc.start_p66()
        self.agc.horizontal_nulling = False  # horizontal: the fly or nobody (ATT HOLD)
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
        slope = self.lm.touchdown_slope
        ok = s.landed and vd <= 10 * C.FT and abs(vh) <= 4 * C.FT and slope <= MAX_SLOPE and s.propellant > 0
        return {"trial": self.name, "outcome": "success" if ok else ("crash" if s.landed else "airborne"),
                "contact_light_speed_mps": self.contact_v, "touchdown_vertical_mps": vd, "touchdown_lateral_mps": vh,
                "slope_deg": slope, "landing_x_m": float(s.pos[0]), "fuel_left_kg": s.propellant,
                "p66_duration_s": (s.t - t_handover) if s.landed else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    terrain = Terrain()
    print(f"terrain ready ({time.perf_counter() - t0:.0f} s)", flush=True)
    gains = calibrate_gains(terrain)
    (OUT / "gains.json").write_text(json.dumps({k: float(v) for k, v in gains.items()}, indent=2))
    lm0, agc0, t_h = fly_to_handover(terrain)
    print(f"P66 handover: PDI+{t_h:.1f} s, x={lm0.s.pos[0]:.0f} m, h={lm0.altitude:.1f} m AGL, "
          f"vs={lm0.vertical_speed():.2f}, vh={lm0.horizontal_speed():.2f} m/s", flush=True)

    controls = [Run(lm0, agc0, "agc_auto", terrain, p66=False), Run(lm0, agc0, "no_pilot_p66", terrain)]
    controls[0].agc.horizontal_nulling = True  # auto P65 nulls horizontal speed itself
    for c in controls:
        t = t_h
        while not c.lm.s.landed and t - t_h < MAX_T:
            c.advance(t, 0.2)
            t += 0.2

    names = [f"fly_{i}" for i in range(N_FLY)] + ["blind_fly"]
    runs = [Run(lm0, agc0, n, terrain) for n in names]
    pilot = VisualFlyPilot(len(runs), gains, blind=np.array([n == "blind_fly" for n in names]), surface=terrain)
    tick = pilot.p.tick_ms / 1000
    keys = ["t", "x", "h", "vs", "hs", "pitch", "tilt_cmd", "v_cmd", "propellant", "landed", "rod",
            "sig", "r_loom", "r_prog", "r_reg", "gf", "brake", "accel"]
    log = {k: [] for k in keys}
    eye_imgs, b_idx, b_cnt, b_ptr = [], [], [], [0]
    t = t_h
    t1 = time.perf_counter()
    while not all(r.lm.s.landed for r in runs) and t - t_h < MAX_T:
        for _ in range(round(tick / DT)):
            for i, r in enumerate(runs):
                if not r.lm.s.landed:
                    pilot.observe(i, r.lm.s.pos[0], r.lm.datum_altitude, r.lm.s.pitch, DT)
                r.advance(t, DT)
            t += DT
        active = np.array([not r.lm.s.landed for r in runs])
        rod, tilt, info, counts = pilot.tick(active)
        for r, a, tl, act in zip(runs, rod, tilt, active):
            if act:
                r.agc.rod_click(int(a))
                if not pilot.blind[runs.index(r)]:
                    r.agc.set_tilt(float(tl))
        snap = {"t": t, "x": [r.lm.s.pos[0] for r in runs], "h": [max(r.lm.altitude, 0) for r in runs],
                "vs": [r.lm.vertical_speed() for r in runs], "hs": [r.lm.horizontal_speed() for r in runs],
                "pitch": [r.lm.s.pitch for r in runs], "tilt_cmd": [r.agc.tilt_cmd for r in runs],
                "v_cmd": [r.agc.v_desired for r in runs], "propellant": [r.lm.s.propellant for r in runs],
                "landed": ~active, "rod": rod}
        snap |= {k: info[k] for k in ["sig", "r_loom", "r_prog", "r_reg", "gf", "brake", "accel"]}
        for k in keys:
            log[k].append(np.array(snap[k]))
        img = pilot.last_image[0]
        eye_imgs.append(np.zeros(pilot.eyes[0].shape, np.float16) if img is None else img.astype(np.float16))
        nz = torch.nonzero(counts[:, 0]).squeeze(1)
        b_idx.append(nz.cpu().numpy().astype(np.int32))
        b_cnt.append(counts[nz, 0].cpu().numpy().astype(np.uint8))
        b_ptr.append(b_ptr[-1] + len(nz))
        if len(log["t"]) % 25 == 0:
            print(f"P66+{t - t_h:5.1f}s ({time.perf_counter() - t1:4.0f}s) x={np.round(snap['x'], 0)} h={np.round(snap['h'], 1)} "
                  f"vs={np.round(snap['vs'], 2)} vh={np.round(snap['hs'], 2)} tilt={np.round(np.degrees(snap['tilt_cmd']), 1)} "
                  f"GF={np.round(pilot.gf_est)} DN={np.round(pilot.dn_est)}", flush=True)

    summary = {"handover": {"t_from_pdi_s": t_h, "x_m": float(lm0.s.pos[0]), "altitude_m": lm0.altitude,
                         "vertical_mps": lm0.vertical_speed(), "horizontal_mps": lm0.horizontal_speed()},
               "gains": {k: float(v) for k, v in gains.items()},
               "trials": [r.summary(t_h) for r in runs + controls]}
    for s in summary["trials"]:
        print(s)
    np.savez_compressed(OUT / "telemetry.npz", names=np.array(names), t_handover=t_h,
                        **{k: np.array(v) for k, v in log.items()})
    np.savez_compressed(OUT / "eye_trial0.npz", images=np.stack(eye_imgs))
    np.savez_compressed(OUT / "brain_trial0.npz", idx=np.concatenate(b_idx), cnt=np.concatenate(b_cnt),
                        ptr=np.array(b_ptr), tick_ms=pilot.p.tick_ms)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"total {time.perf_counter() - t0:.0f} s, saved: {OUT}")


if __name__ == "__main__":
    main()

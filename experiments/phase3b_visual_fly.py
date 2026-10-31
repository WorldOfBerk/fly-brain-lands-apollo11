"""Phase 3b: the fly flies P66 on both axes (ROD + attitude) with its own eye, over flat ground.

Starts where Armstrong took over (470 ft). Trials:
  fly_0..N-1  visual fly pilot (vertical: looming -> Giant Fiber -> ROD, horizontal: T4/T5 -> DNs -> ACA)
  blind_fly   same brain, no visual input
Brainless controls as in phase3_fly_p66.py: agc_auto, no_pilot_p66 (no horizontal nulling; ATT HOLD
keeps the attitude from the handover).

Output: results/phase3b/ telemetry.npz, eye_trial0.npz, brain_trial0.npz, summary.json, gains.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from apollo_sim.flyeye import LunarSurface  # noqa: E402
from lander.visual_pilot import VisualFlyPilot, calibrate_gains  # noqa: E402
from phase3_fly_p66 import DT, MAX_T, Run, fly_to_handover  # noqa: E402

OUT = ROOT / "results" / "phase3b"
N_FLY = 2


def run_control(lm0, agc0, t_h, name, p66):
    r = Run(lm0, agc0, name, p66=p66)
    r.agc.horizontal_nulling = False  # ATT HOLD, nobody touches the ACA
    t = t_h
    while not r.lm.s.landed and t - t_h < MAX_T:
        r.advance(t, 0.2)
        t += 0.2
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    surface = LunarSurface()
    gains = calibrate_gains(surface)
    (OUT / "gains.json").write_text(json.dumps(gains, indent=2))
    print("gains:", gains, flush=True)

    lm0, agc0, t_h = fly_to_handover()
    print(f"P66 handover: PDI+{t_h:.1f} s, h={lm0.altitude:.1f} m, vs={lm0.vertical_speed():.2f}, "
          f"vh={lm0.horizontal_speed():.2f} m/s, pitch={np.degrees(lm0.s.pitch):.1f} deg", flush=True)
    controls = [Run(lm0, agc0, "agc_auto", p66=False), run_control(lm0, agc0, t_h, "no_pilot_p66", True)]
    t = t_h
    while not controls[0].lm.s.landed and t - t_h < MAX_T:
        controls[0].advance(t, 0.2)
        t += 0.2

    names = [f"fly_{i}" for i in range(N_FLY)] + ["blind_fly"]
    runs = [Run(lm0, agc0, n) for n in names]
    for r in runs:
        r.agc.horizontal_nulling = False
    pilot = VisualFlyPilot(len(runs), gains, blind=np.array([n == "blind_fly" for n in names]), surface=surface)
    tick = pilot.p.tick_ms / 1000
    keys = ["t", "h", "vs", "hs", "pitch", "tilt_cmd", "v_cmd", "propellant", "landed", "rod", "aca",
            "sig", "r_loom", "r_prog", "r_reg", "gf", "brake", "accel"]
    log = {k: [] for k in keys}
    eye_imgs, b_idx, b_cnt, b_ptr = [], [], [], [0]
    t = t_h
    t0 = time.perf_counter()
    while not all(r.lm.s.landed for r in runs) and t - t_h < MAX_T:
        # physics + eye for one tick
        for _ in range(round(tick / DT)):
            for i, r in enumerate(runs):
                if not r.lm.s.landed:
                    pilot.observe(i, r.lm.s.pos[0], max(r.lm.altitude, 0.0), r.lm.s.pitch, DT)
                r.advance(t, DT)
            t += DT
        active = np.array([not r.lm.s.landed for r in runs])
        rod, aca, info, counts = pilot.tick(active)
        for r, a, b in zip(runs, rod, aca):
            r.agc.rod_click(int(a))
            r.agc.aca_click(int(b))

        snap = {"t": t, "h": [max(r.lm.altitude, 0) for r in runs], "vs": [r.lm.vertical_speed() for r in runs],
                "hs": [r.lm.horizontal_speed() for r in runs], "pitch": [r.lm.s.pitch for r in runs],
                "tilt_cmd": [r.agc.tilt_cmd for r in runs], "v_cmd": [r.agc.v_desired for r in runs],
                "propellant": [r.lm.s.propellant for r in runs], "landed": ~active, "rod": rod, "aca": aca}
        snap |= {k: info[k] for k in ["sig", "r_loom", "r_prog", "r_reg", "gf", "brake", "accel"]}
        for k in keys:
            log[k].append(np.array(snap[k]))
        eye_imgs.append(pilot.last_image[0].astype(np.float16) if pilot.last_image[0] is not None else None)
        nz = torch.nonzero(counts[:, 0]).squeeze(1)
        b_idx.append(nz.cpu().numpy().astype(np.int32))
        b_cnt.append(counts[nz, 0].cpu().numpy().astype(np.uint8))
        b_ptr.append(b_ptr[-1] + len(nz))
        if len(log["t"]) % 25 == 0:
            print(f"P66+{t - t_h:5.1f}s ({time.perf_counter() - t0:4.0f}s) h={np.round(snap['h'], 1)} "
                  f"vs={np.round(snap['vs'], 2)} vh={np.round(snap['hs'], 2)} tilt={np.round(np.degrees(snap['tilt_cmd']), 0)} "
                  f"GF={np.round(pilot.gf_est)} DN={np.round(pilot.dn_est)}", flush=True)

    summary = {"handover": {"t_from_pdi_s": t_h, "altitude_m": lm0.altitude, "vertical_mps": lm0.vertical_speed(),
                         "horizontal_mps": lm0.horizontal_speed(), "pitch_deg": float(np.degrees(lm0.s.pitch)),
                         "fuel_kg": lm0.s.propellant},
               "gains": gains, "trials": [r.summary(t_h) for r in runs + controls]}
    for s in summary["trials"]:
        print(s)
    np.savez_compressed(OUT / "telemetry.npz", names=np.array(names), t_handover=t_h,
                        **{k: np.array(v) for k, v in log.items()})
    first = next(e for e in eye_imgs if e is not None)
    np.savez_compressed(OUT / "eye_trial0.npz", images=np.stack([e if e is not None else np.zeros_like(first) for e in eye_imgs]))
    np.savez_compressed(OUT / "brain_trial0.npz", idx=np.concatenate(b_idx), cnt=np.concatenate(b_cnt),
                        ptr=np.array(b_ptr), tick_ms=pilot.p.tick_ms)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"total {time.perf_counter() - t0:.0f} s, saved: {OUT}")


if __name__ == "__main__":
    main()

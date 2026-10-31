"""Phase 2: the fly brain lands a toy LM in P66.

Trials run together:
  0..N_FLY-1  fly pilot (independent stochastic brains)
  N_FLY       blind fly (no visual input, same brain)
Brainless control: nobody clicks ROD.

Output (results/phase2/):
  telemetry.npz      - per tick h, v, v_cmd, thrust, fuel, LPLC2 input, GF/DNp04/TTMn rates, ROD clicks
  brain_trial0.npz   - sparse whole-brain spike counts per tick for trial 0
  summary.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lander import FlyPilot, LunarModule  # noqa: E402

OUT = ROOT / "results" / "phase2"
N_FLY = 3
MAX_T = 240.0
GROUPS = ["LPLC2", "DNp01", "DNp04", "TTMn"]


def summarize(lm, names):
    out = lm.outcome()
    return [
        {"trial": names[i], "outcome": str(out[i]), "contact_speed_mps": float(lm.contact_v[i]),
         "touchdown_speed_mps": float(lm.touchdown_v[i]), "fuel_left_kg": float(lm.fuel[i]),
         "landing_time_s": float(lm.contact_t[i] + lm.p.shutdown_delay) if lm.landed[i] else None}
        for i in range(lm.n)
    ]


def no_pilot(tick_s):
    lm = LunarModule(1)
    while not lm.landed.all() and lm.t < MAX_T:
        lm.advance(tick_s)
    return summarize(lm, ["no_pilot"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    B = N_FLY + 1
    blind = np.zeros(B, bool)
    blind[N_FLY] = True
    pilot = FlyPilot(B, blind=blind)
    lm = LunarModule(B)
    tick_s = pilot.p.tick_ms / 1000
    gidx = {g: torch.as_tensor(pilot.c.idx(g), device=pilot.brain.dev) for g in GROUPS}

    log = {k: [] for k in ["t", "h", "v", "v_cmd", "thrust", "fuel", "lplc2_in", "clicks", "landed"]}
    log.update({f"rate_{g}": [] for g in GROUPS})
    brain_idx, brain_cnt, brain_ptr = [], [], [0]

    t0 = time.perf_counter()
    while not lm.landed.all() and lm.t < MAX_T:
        active = ~lm.landed
        clicks, lplc2_in, counts = pilot.tick(lm.h, lm.v, active)

        for k, val in [("t", lm.t), ("h", lm.h), ("v", lm.v), ("v_cmd", lm.v_cmd), ("thrust", lm.thrust),
                       ("fuel", lm.fuel), ("lplc2_in", lplc2_in), ("clicks", clicks), ("landed", lm.landed)]:
            log[k].append(np.copy(val))
        for g in GROUPS:
            log[f"rate_{g}"].append((counts[gidx[g]].mean(0) / tick_s).cpu().numpy())
        nz = torch.nonzero(counts[:, 0]).squeeze(1)
        brain_idx.append(nz.cpu().numpy().astype(np.int32))
        brain_cnt.append(counts[nz, 0].cpu().numpy().astype(np.uint8))
        brain_ptr.append(brain_ptr[-1] + len(nz))

        lm.rod_click(clicks)
        lm.advance(tick_s)

        if len(log["t"]) % 25 == 0:
            print(f"t={lm.t:5.1f}s wall={time.perf_counter() - t0:5.0f}s h={np.round(lm.h, 1)} "
                  f"v={np.round(lm.v, 2)} GF={np.round(pilot.gf_est, 0)}", flush=True)

    names = [f"fly_{i}" for i in range(N_FLY)] + ["blind_fly"]
    summary = summarize(lm, names) + no_pilot(tick_s)
    for s in summary:
        print(s)

    np.savez_compressed(OUT / "telemetry.npz", names=np.array(names), **{k: np.array(v) for k, v in log.items()})
    np.savez_compressed(OUT / "brain_trial0.npz", idx=np.concatenate(brain_idx), cnt=np.concatenate(brain_cnt),
                        ptr=np.array(brain_ptr), tick_ms=pilot.p.tick_ms)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"total {time.perf_counter() - t0:.0f} s, saved: {OUT}")


if __name__ == "__main__":
    main()

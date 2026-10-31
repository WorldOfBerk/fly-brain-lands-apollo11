"""Phase 1 validation: does the model reproduce known circuits?

A) No input           -> the brain should stay silent.
B) Looming (LPLC2)    -> do the Giant Fiber (DNp01) and jump motor neurons (TTMn) fire?
C) Taste neurons      -> which labellar GRN types drive the proboscis extension motor neuron MN9?
                         (Shiu et al.: sugar GRNs drive MN9, bitter GRNs do not)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flybrain import Connectome, LIFBrain  # noqa: E402

OUT = ROOT / "results" / "phase1"
DURATION_MS = 1000
N_TRIALS = 10
RATE_HZ = 150

READOUT = ["DNp01", "DNp02", "DNp04", "DNp11", "TTMn", "PSI", "MN9", "MDN", "DNa02"]


def mean_rate(rate, c, t):
    idx = c.idx(t)
    return float(rate[idx].mean()) if len(idx) else float("nan")


def trial(brain, c, stim_idx, label):
    brain.reset()
    if len(stim_idx):
        brain.set_input(stim_idx, RATE_HZ)
    t0 = time.perf_counter()
    rate, _ = brain.run(DURATION_MS)
    el = time.perf_counter() - t0
    readout = {t: mean_rate(rate, c, t) for t in READOUT}
    active = int((rate > 1).sum())
    print(f"[{label}] {el:.0f} s | active neurons (>1 Hz): {active:,} | "
          + " ".join(f"{t}={v:.1f}" for t, v in readout.items()))
    return {"label": label, "n_stim": int(len(stim_idx)), "active_neurons": active, "readout_hz": readout}, rate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = Connectome()
    brain = LIFBrain(c.W, n_trials=N_TRIALS)
    results = {}

    results["A_baseline"], _ = trial(brain, c, [], "A no input")

    results["B_looming_LPLC2"], rate_b = trial(brain, c, c.idx("LPLC2"), "B LPLC2")
    results["B_looming_LPLC2_LC4"], _ = trial(brain, c, c.idx("LPLC2", "LC4"), "B LPLC2+LC4")
    np.save(OUT / "rate_LPLC2.npy", rate_b)

    lb_types = sorted(t for t in c.neurons.loc[c.neurons["class"] == "gustatory", "type"].dropna().unique()
                      if t.startswith("LB"))
    results["C_gustatory"] = []
    for t in lb_types:
        r, _ = trial(brain, c, c.idx(t), f"C {t}")
        results["C_gustatory"].append(r)

    (OUT / "validation.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("saved:", OUT / "validation.json")


if __name__ == "__main__":
    main()

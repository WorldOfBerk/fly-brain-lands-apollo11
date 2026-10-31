"""Checks the fix for runaway activity: raw Shiu model vs calibrated weights + conductance synapses.

Criteria from the literature:
  - silence without input
  - activity dies out when the stimulus stops (no persistent state)
  - sparse Kenyon cell responses to odour (~5-10 %; Turner et al. 2008, Honegger et al. 2011)
  - the Giant Fiber escape circuit and the MN9 (proboscis extension) circuit still work
Also: steady-state Giant Fiber response vs LPLC2 input (the pilot thresholds come from this).

Output: results/kc_fix/report.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flybrain import Connectome, GraphLIFBrain, LIFParams  # noqa: E402

OUT = ROOT / "results" / "kc_fix"
TESTS = [("none", None, 0), ("LPLC2", "LPLC2", 150), ("LB3c (taste)", "LB3c", 150),
         ("ORN_DA1 (smell)", "ORN_DA1", 100), ("ORN_DM2 (smell)", "ORN_DM2", 100), ("ORN_VA2 (smell)", "ORN_VA2", 100)]


def evaluate(c, params):
    t = c.neurons["type"].fillna("")
    groups = {"KC": t.str.startswith("KC").values, "PN": c.neurons["class"].fillna("").str.contains("ALPN").values,
              "GF": (t == "DNp01").values, "TTMn": (t == "TTMn").values, "MN9": (t == "MN9").values}
    rows = []
    for name, typ, rate in TESTS:
        b = GraphLIFBrain(c.W, n_trials=4, params=params)
        if typ:
            b.set_input(c.idx(typ), float(rate))
        on = b.run_counts(300).mean(1).cpu().numpy() / 0.3
        if typ:
            b.set_input(c.idx(typ), 0.0)
        b.run_counts(200)
        off = b.run_counts(300).mean(1).cpu().numpy() / 0.3
        row = {"test": name, "active_neurons_during_stim": int((on > 1).sum()),
               "active_neurons_after_stim": int((off > 1).sum())}
        for g, m in groups.items():
            row[f"{g}_hz"] = round(float(on[m].mean()), 1)
            row[f"{g}_active_pct"] = round(float((on[m] > 1).mean() * 100), 1)
        rows.append(row)
        print(row, flush=True)
    return rows


def gf_transfer(c, params, rates=(40, 60, 70, 80, 90, 100, 110, 120, 150)):
    b = GraphLIFBrain(c.W, n_trials=len(rates), params=params)
    b.set_input(c.idx("LPLC2"), np.array(rates, dtype=np.float32))
    b.run_counts(300)  # transient
    r = b.run_counts(700)
    gf = r[c.idx("DNp01")].mean(0).cpu().numpy() / 0.7
    return {int(k): round(float(v), 1) for k, v in zip(rates, gf)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    print("== raw MaleCNS + Shiu (current-based)")
    report["raw_shiu"] = evaluate(Connectome(weights="raw"), LIFParams())
    print("== calibrated + conductance-based")
    c = Connectome(weights="calibrated")
    p = LIFParams(conductance=True)
    report["calibrated_conductance"] = evaluate(c, p)
    report["gf_transfer_lplc2_hz_to_gf_hz"] = gf_transfer(c, p)
    print("GF transfer:", report["gf_transfer_lplc2_hz_to_gf_hz"])
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

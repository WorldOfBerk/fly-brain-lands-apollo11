"""Phase 1 figures:
  escape_raster.png  - looming (LPLC2) input from 200 ms; Giant Fiber and jump motor neuron responses
  taste_mn9.png      - MN9 response to each labellar taste neuron type (from validation.json)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flybrain import Connectome, LIFBrain  # noqa: E402

OUT = ROOT / "results" / "phase1"
ONSET_MS, OFF_MS, TOTAL_MS = 200, 700, 900
GROUPS = [("LPLC2", "#4C78A8"), ("DNp01", "#E45756"), ("DNp04", "#F58518"), ("TTMn", "#54A24B")]
LABELS = {"LPLC2": "LPLC2 (looming detector)", "DNp01": "DNp01 (Giant Fiber)",
          "DNp04": "DNp04 (descending neuron)", "TTMn": "TTMn (jump motor neuron)"}


def escape_raster(c):
    idx_groups = [(t, col, c.idx(t)[:12] if t == "LPLC2" else c.idx(t)) for t, col in GROUPS]
    rec_idx = np.concatenate([g[2] for g in idx_groups])

    brain = LIFBrain(c.W, n_trials=1)
    brain.reset()
    _, r1 = brain.run(ONSET_MS, rec_idx)
    brain.set_input(c.idx("LPLC2"), 150)
    _, r2 = brain.run(OFF_MS - ONSET_MS, rec_idx)
    brain.set_input(c.idx("LPLC2"), 0)
    _, r3 = brain.run(TOTAL_MS - OFF_MS, rec_idx)
    spikes = np.concatenate([r1, r2, r3])[:, :, 0]
    t = np.arange(spikes.shape[0]) * brain.p.dt

    fig, ax = plt.subplots(figsize=(10, 5))
    row, yticks, ylabels = 0, [], []
    for name, col, idx in idx_groups:
        start = row
        for _ in idx:
            st = t[spikes[:, row]]
            ax.vlines(st, row + 0.1, row + 0.9, color=col, lw=0.6)
            row += 1
        yticks.append((start + row) / 2)
        ylabels.append(LABELS[name])
        ax.axhline(row, color="#ddd", lw=0.5)
    ax.axvspan(ONSET_MS, OFF_MS, color="#999", alpha=0.12, label="looming stimulus")
    ax.set_yticks(yticks, ylabels)
    ax.set_xlabel("time (ms)")
    ax.set_xlim(0, TOTAL_MS)
    ax.set_ylim(0, row)
    ax.set_title("MaleCNS v1.0 whole-CNS LIF model: looming -> Giant Fiber -> escape")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "escape_raster.png", dpi=150)


def taste_bars():
    res = json.loads((OUT / "validation.json").read_text())["C_gustatory"]
    names = [r["label"].split()[-1] for r in res]
    mn9 = [r["readout_hz"]["MN9"] for r in res]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(names, mn9, color="#72B7B2")
    ax.set_ylabel("MN9 firing rate (Hz)")
    ax.set_title("Proboscis extension motor neuron (MN9) when each labellar taste neuron type is driven")
    fig.tight_layout()
    fig.savefig(OUT / "taste_mn9.png", dpi=150)


if __name__ == "__main__":
    escape_raster(Connectome())
    taste_bars()
    print("saved:", OUT)

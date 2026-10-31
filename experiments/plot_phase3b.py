"""Plot for phases 3b/3c: two-axis visual fly pilot vs Armstrong. Usage: plot_phase3b.py [phase3c]"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apollo_sim import reference as ref  # noqa: E402
from apollo_sim.constants import FT  # noqa: E402

OUT = ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "phase3b")
FLY, BLIND, REAL = "#54A24B", "#B279A2", "#E45756"


def main():
    d = np.load(OUT / "telemetry.npz")
    summary = json.loads((OUT / "summary.json").read_text())
    names = list(d["names"])
    t = d["t"] - d["t_handover"]
    p66 = ref.EVENTS["P66"] - ref.PDI
    co = [c for c in ref.callouts_si() if c[0] >= p66 - 10]
    ct = [c[0] - p66 for c in co]

    fig, ax = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    for i, n in enumerate(names):
        a = ~d["landed"][:, i]
        col, lw, al = (BLIND, 1, 0.6) if n == "blind_fly" else (FLY, 2 if i == 0 else 1, 1 if i == 0 else 0.5)
        lab = {0: "fly pilot (MaleCNS v1.0, own eye)", len(names) - 1: "blind fly"}.get(i)
        ax[0].plot(t[a], d["h"][a, i] / FT, color=col, lw=lw, alpha=al, label=lab)
        ax[1].plot(t[a], -d["vs"][a, i] / FT, color=col, lw=lw, alpha=al)
        ax[2].plot(t[a], d["hs"][a, i] / FT, color=col, lw=lw, alpha=al)
        if n != "blind_fly":
            ax[3].plot(t[a], np.degrees(d["tilt_cmd"][a, i]), color=col, lw=lw * 0.7, alpha=al)
    a = ~d["landed"][:, 0]
    ax[4].plot(t[a], d["gf"][a, 0], color="#4C78A8", lw=1, label="Giant Fiber (DNp01) -> ROD")
    ax[4].plot(t[a], d["brake"][a, 0], color="#F58518", lw=1, label="brake DNs (DNp26/DNbe001/DNp20)")
    ax[4].plot(t[a], d["accel"][a, 0], color="#72B7B2", lw=1, label="accel DNs (DNg46/DNp53/DNge043)")
    ax[0].plot([x for x, c in zip(ct, co) if c[1]], [c[1] / FT for c in co if c[1]], "o-", color=REAL, ms=4,
               label="Armstrong (Apollo 11, Aldrin's callouts)")
    ax[1].plot([x for x, c in zip(ct, co) if c[2] is not None], [-c[2] / FT for c in co if c[2] is not None], "o-", color=REAL, ms=4)
    ax[2].plot([x for x, c in zip(ct, co) if c[3] is not None], [c[3] / FT for c in co if c[3] is not None], "o-", color=REAL, ms=4)
    ax[0].set_ylabel("altitude (ft)")
    ax[0].legend(fontsize=8)
    ax[1].set_ylabel("descent rate (ft/s)")
    ax[1].set_ylim(-5, 40)
    ax[2].set_ylabel("forward speed (ft/s)")
    ax[2].axhline(0, color="#999", lw=0.6)
    ax[3].set_ylabel("tilt command (deg)\n- = brake")
    ax[4].set_ylabel("Hz (trial 0)")
    ax[4].legend(fontsize=8)
    ax[4].set_xlabel("time since P66 (s)")
    rows = {s["trial"]: s for s in summary["trials"]}
    fl = [rows[n] for n in names if n.startswith("fly")]
    ax[0].set_title(
        "P66 on both axes: fruit fly brain on the ROD switch and attitude controller, using its own eye, vs Neil Armstrong\n"
        f"fly {sum(s['outcome'] == 'success' for s in fl)}/{len(fl)} landed | P66 "
        + ", ".join(f"{s['p66_duration_s']:.0f} s" for s in fl)
        + " | lateral at touchdown " + ", ".join(f"{abs(s['touchdown_lateral_mps']) / FT:.1f}" for s in fl) + " ft/s"
        + f"  --  Armstrong: P66 {ref.EVENTS['contact_light'] - ref.EVENTS['P66']} s, lateral {ref.TOUCHDOWN_LATERAL / FT:.1f} ft/s",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "p66_two_axis_vs_armstrong.png", dpi=140)
    print("saved:", OUT / "p66_two_axis_vs_armstrong.png")


if __name__ == "__main__":
    main()

"""Phase 3 plots: simulation vs the real Apollo 11 landing (Aldrin's callouts).

  descent_vs_apollo11.png  - altitude and sink rate from PDI to touchdown (AGC sim + real callouts)
  p66_fly_vs_armstrong.png - P66: fly pilot vs Armstrong
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
from apollo_sim import reference as ref  # noqa: E402
from apollo_sim.constants import FT  # noqa: E402
from apollo_sim.run import fly  # noqa: E402

OUT = ROOT / "results" / "phase3"
SIM, REAL, FLY, BLIND = "#4C78A8", "#E45756", "#54A24B", "#B279A2"


def descent_plot():
    lm, agc, r = fly()
    t, h, vs = r[:, 0], r[:, 1], r[:, 2]
    prog = r[:, 7]
    co = ref.callouts_si()
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t, h / FT, color=SIM, label="simulation: Luminary guidance (P63/P64/P65)")
    ax[0].scatter([c[0] for c in co if c[1]], [c[1] / FT for c in co if c[1]], color=REAL, s=18, zorder=3,
                  label="Apollo 11: Aldrin's callouts")
    ax[1].plot(t, -vs / FT, color=SIM)
    ax[1].scatter([c[0] for c in co if c[2]], [-c[2] / FT for c in co if c[2]], color=REAL, s=18, zorder=3)
    for name, get in ref.EVENTS.items():
        if name == "PDI":
            continue
        x = get - ref.PDI
        for a in ax:
            a.axvline(x, color=REAL, lw=0.6, ls=":")
        ax[0].text(x, ax[0].get_ylim()[1] * 0.97 if name != "shutdown" else 0, name, rotation=90, fontsize=7,
                   color=REAL, va="top")
    for p in (64, 65):
        i = np.argmax(prog == p)
        if prog[i] == p:
            for a in ax:
                a.axvline(t[i], color=SIM, lw=0.6, ls="--")
            ax[1].text(t[i], 200, f"sim P{p}", rotation=90, fontsize=7, color=SIM)
    ax[0].set_ylabel("altitude (ft)")
    ax[0].set_yscale("log")
    ax[0].set_ylim(10, 60000)
    ax[0].legend()
    ax[1].set_ylabel("descent rate (ft/s)")
    ax[1].set_ylim(-20, 250)
    ax[1].set_xlabel("time since PDI (s)")
    ax[0].set_title(f"PDI -> touchdown: simulation {lm.s.t:.0f} s / {lm.s.propellant:.0f} kg fuel left; "
                    f"Apollo 11 {ref.EVENTS['contact_light'] - ref.PDI:.0f} s / ~{ref.PROPELLANT_LEFT:.0f} kg")
    fig.tight_layout()
    fig.savefig(OUT / "descent_vs_apollo11.png", dpi=150)


def p66_plot():
    f = OUT / "telemetry.npz"
    if not f.exists():
        return
    d = np.load(f)
    summary = json.loads((OUT / "summary.json").read_text())
    names = list(d["names"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for i, n in enumerate(names):
        alive = ~d["landed"][:, i]
        col = BLIND if n == "blind_fly" else FLY
        lab = "fly pilot" if i == 0 else ("blind fly" if n == "blind_fly" else None)
        h, v = d["h"][alive, i], -d["vs"][alive, i]
        ax[0].plot(v / FT, h / FT, color=col, lw=2 if i == 0 else 1, alpha=1 if i == 0 else 0.5, label=lab)
        ax[1].plot(d["t"][alive] - d["t_handover"], h / FT, color=col, lw=2 if i == 0 else 1,
                   alpha=1 if i == 0 else 0.5, label=lab)
    co = [c for c in ref.callouts_si() if c[0] >= ref.EVENTS["P66"] - ref.PDI - 10]
    ax[0].plot([-c[2] / FT for c in co if c[1] and c[2] is not None], [c[1] / FT for c in co if c[1] and c[2] is not None],
               "o-", color=REAL, ms=4, label="Armstrong (Apollo 11)")
    ax[1].plot([c[0] - (ref.EVENTS["P66"] - ref.PDI) for c in co if c[1]], [c[1] / FT for c in co if c[1]],
               "o-", color=REAL, ms=4, label="Armstrong (Apollo 11)")
    ax[0].set_xlabel("descent rate (ft/s)")
    ax[0].set_ylabel("altitude (ft)")
    ax[0].set_ylim(0, 520)
    ax[0].set_xlim(0, 40)
    ax[0].axvline(10, color="#999", ls=":", lw=0.8)
    ax[0].text(10.3, 480, "landing gear limit 10 ft/s", fontsize=8, color="#666")
    ax[1].set_xlabel("time since P66 (s)")
    ax[1].set_ylabel("altitude (ft)")
    ax[1].set_ylim(0, 520)
    for a in ax:
        a.legend()
    rows = {s["trial"]: s for s in summary["trials"]}
    fly_ok = [s for k, s in rows.items() if k.startswith("fly")]
    txt = (f"fly: {sum(s['outcome'] == 'success' for s in fly_ok)}/{len(fly_ok)} landed, "
           f"P66 {np.mean([s['p66_duration_s'] or np.nan for s in fly_ok]):.0f} s, "
           f"contact light {np.mean([s['contact_light_speed_mps'] for s in fly_ok]) / FT:.1f} ft/s | "
           f"Armstrong: P66 {ref.EVENTS['contact_light'] - ref.EVENTS['P66']} s, touchdown {ref.TOUCHDOWN_VERTICAL / FT:.1f} ft/s")
    fig.suptitle("P66: fruit fly brain (MaleCNS v1.0) vs Neil Armstrong\n" + txt)
    fig.tight_layout()
    fig.savefig(OUT / "p66_fly_vs_armstrong.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    descent_plot()
    p66_plot()
    print("saved:", OUT)

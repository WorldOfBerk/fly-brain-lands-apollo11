"""Phase 2 plot: altitude, sink rate, brain input/output and ROD clicks."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase2"
FLY, BLIND = "#4C78A8", "#E45756"


def main():
    d = np.load(OUT / "telemetry.npz")
    names = list(d["names"])
    t = d["t"]
    summary = {s["trial"]: s for s in json.loads((OUT / "summary.json").read_text())}

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for i, name in enumerate(names):
        col = BLIND if name == "blind_fly" else FLY
        alive = ~d["landed"][:, i]
        lw, a = (2.0, 1.0) if i == 0 else (1.0, 0.45)
        lab = None
        if i == 0:
            lab = "fly pilot"
        elif name == "blind_fly":
            lab = f"blind fly (crashed at {summary[name]['touchdown_speed_mps']:.1f} m/s)"
        axes[0].plot(t[alive], d["h"][alive, i], color=col, lw=lw, alpha=a, label=lab)
        axes[1].plot(t[alive], -d["v"][alive, i], color=col, lw=lw, alpha=a)
        axes[1].plot(t[alive], -d["v_cmd"][alive, i], color=col, lw=0.6, alpha=a * 0.6, ls="--")
        if name != "blind_fly":
            axes[2].plot(t[alive], d["rate_DNp01"][alive, i], color=col, lw=lw * 0.7, alpha=a)
        if i == 0:
            axes[2].plot(t[alive], d["lplc2_in"][alive, i], color="#999", lw=1.2, label="LPLC2 input (looming)")
            c = d["clicks"][alive, 0]
            axes[3].vlines(t[alive][c > 0], 0, 1, color="#54A24B", lw=1, label="ROD up (slower)")
            axes[3].vlines(t[alive][c < 0], -1, 0, color="#F58518", lw=1, label="ROD down (faster)")

    axes[0].set_ylabel("altitude (m)")
    axes[0].legend()
    axes[1].set_ylabel("descent rate (m/s)\n-- command")
    axes[1].axhline(3.05, color="#999", lw=0.8, ls=":")
    axes[1].text(t[0], 3.15, "landing gear limit 3.05 m/s", fontsize=8, color="#666")
    axes[2].set_ylabel("Hz")
    axes[2].plot([], [], color=FLY, label="Giant Fiber (DNp01) output")
    axes[2].legend(loc="upper left")
    axes[3].set_ylabel("ROD switch\n(fly 0)")
    axes[3].set_yticks([])
    axes[3].legend(loc="upper left")
    axes[3].set_xlabel("time since P66 (s)")

    fly_ok = [s for k, s in summary.items() if k.startswith("fly")]
    ok = sum(s["outcome"] == "success" for s in fly_ok)
    vtd = np.mean([s["touchdown_speed_mps"] for s in fly_ok])
    vc = np.mean([s["contact_speed_mps"] for s in fly_ok])
    axes[0].set_title(f"Fruit fly brain (MaleCNS v1.0, 165,122 neurons) landing the LM in P66\n"
                      f"fly: {ok}/{len(fly_ok)} landed, {vc:.2f} m/s at contact light, "
                      f"touchdown {vtd:.2f} m/s | "
                      f"no pilot: {summary['no_pilot']['touchdown_speed_mps']:.1f} m/s ({summary['no_pilot']['outcome']})")
    fig.tight_layout()
    fig.savefig(OUT / "landing_telemetry.png", dpi=150)
    print("saved:", OUT / "landing_telemetry.png")


if __name__ == "__main__":
    main()

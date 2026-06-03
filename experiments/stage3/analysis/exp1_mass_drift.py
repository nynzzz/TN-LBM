"""
Exp 1 figure: mass drift vs chi.

Same 3-subplot layout as chi_vs_error: one column per N, two Re curves per
subplot. Y-axis = final relative mass drift |M - M_0| / M_0 on a log scale.

Restricts to the Exp 1 standard chi sweep [6..64].

Usage:
    python -m experiments.stage3.analysis.exp1_mass_drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.stage3.analysis.exp1_chi_vs_error import (
    EXP1_CHI_SWEEP, EXP1_N_VALUES, EXP1_RE_VALUES, _RE_COLORS, _OUT_DIR,
)

import glob, json

_RESULTS_DIR = _THIS_DIR.parent / "results"


def _load_mass_drift(test_case: str, n: int, re: float, cutoff: float,
                     chi_min: int = min(EXP1_CHI_SWEEP)):
    """Return [(chi, |mass_drift_rel|), ...] sorted by chi."""
    pts = []
    for f in glob.glob(str(_RESULTS_DIR / "job_*.json")):
        d = json.load(open(f))
        p = d["params"]
        if (p["test_case"] == test_case and p["n"] == n
                and abs(p["re"] - re) < 1e-9 and abs(p["cutoff"] - cutoff) < 1e-15):
            chi = p["chi"]
            if chi is None or not (chi_min <= chi <= max(EXP1_CHI_SWEEP)):
                continue
            md = d["mps"]["final"].get("final_mass_drift_rel")
            if md is not None:
                pts.append((chi, abs(md)))
    return sorted(pts)


def plot_test_case(test_case: str, cutoff: float = 1e-10):
    chi_min = 10 if test_case == "tg" else min(EXP1_CHI_SWEEP)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, n in zip(axes, EXP1_N_VALUES):
        for re in EXP1_RE_VALUES:
            pts = _load_mass_drift(test_case, n, float(re), cutoff, chi_min=chi_min)
            if not pts:
                continue
            chis = np.array([c for c, _ in pts])
            md = np.array([e for _, e in pts])
            ax.plot(chis, md, "o-", color=_RE_COLORS[re], markersize=5,
                    linewidth=1.4, label=f"Re={re}")
        ax.set_title(f"N = {n}", fontsize=11)
        ax.set_xlabel(r"bond dimension $\chi$")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.15)
        ax.legend(loc="best", frameon=False, fontsize=9)
    axes[0].set_ylabel(r"relative mass drift  $|M - M_0| / M_0$")
    fig.suptitle(f"{test_case.upper()} — Exp 1 mass drift vs chi (cutoff={cutoff:.0e}, taylor=2)",
                 fontsize=12)
    fig.tight_layout()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"mass_drift_{test_case}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cutoff", type=float, default=1e-10)
    args = p.parse_args()
    plot_test_case("tg", cutoff=args.cutoff)
    plot_test_case("cavity", cutoff=args.cutoff)


if __name__ == "__main__":
    main()

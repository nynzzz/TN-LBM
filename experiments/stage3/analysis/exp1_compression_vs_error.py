"""
Exp 1 figure: compression ratio vs L2 error.

Compression ratio = dense_floats / mps_floats, computed via Gross Eq 46
(nvps_per_population from common.py). Higher CR = more compression.

Layout: 3 subplots (one per N), each with Re=100/500 curves. Scatter plot
with chi as the implicit varying parameter (marker labels could be added
later if needed).

Usage:
    python -m experiments.stage3.analysis.exp1_compression_vs_error
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

from experiments.stage3.common import nvps_mps_field
from experiments.stage3.analysis.exp1_chi_vs_error import (
    EXP1_CHI_SWEEP, EXP1_N_VALUES, EXP1_RE_VALUES, _RE_COLORS, _OUT_DIR,
    _add_threshold, load_sweep,
)


def plot_test_case(test_case: str, cutoff: float = 1e-10):
    chi_min = 10 if test_case == "tg" else 6  # drop tg pre-stability chi
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, n in zip(axes, EXP1_N_VALUES):
        dense_floats = 9 * n * n
        for re in EXP1_RE_VALUES:
            pts = load_sweep(test_case, n, float(re), cutoff, chi_min=chi_min)
            if not pts:
                continue
            chis = np.array([c for c, *_ in pts])
            l2 = np.array([e for _, e, *_ in pts])
            cr = np.array([dense_floats / nvps_mps_field(n, c, "snake")
                           for c in chis])
            ax.plot(cr, l2, "o-", color=_RE_COLORS[re], markersize=5,
                    linewidth=1.4, label=f"Re={re}")
        _add_threshold(ax)
        ax.set_title(f"N = {n}", fontsize=11)
        ax.set_xlabel("compression ratio (dense / MPS)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.15)
        ax.legend(loc="best", frameon=False, fontsize=9)
    axes[0].set_ylabel(r"relative $L_2$ velocity error vs vanilla LBM")
    fig.suptitle(f"{test_case.upper()} — Exp 1 compression ratio vs error (cutoff={cutoff:.0e}, taylor=2)",
                 fontsize=12)
    fig.tight_layout()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"compression_vs_error_{test_case}.pdf"
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

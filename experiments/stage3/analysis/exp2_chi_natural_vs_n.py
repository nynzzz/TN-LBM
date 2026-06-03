"""
Exp 2: chi_natural vs N — the headline thesis claim.

For each (test_case, N, Re), the simulation runs with chi cap = 512
(effectively unbounded; the natural chi for our grid sizes is << 512). The
"natural" bond dimension is the max chi reported across populations at the
final time step.

If the natural chi scales as O(log N), MPS-LBM achieves logarithmic memory
scaling and the method is viable for high-resolution CFD. This script
plots chi_natural vs N (with both linear and log-N x-axes) per Re, per
test case.

Restricts to cutoff=1e-10 by default; pass --cutoff for the aggressive 1e-7
variant.

Outputs to results/stage3/figures/exp2/:
  - chi_natural_vs_n.pdf       2 panels (tg | cavity), linear y, log2(N) x
  - chi_natural_vs_n_logy.pdf  same but log y (for power-law spotting)

Usage:
    python -m experiments.stage3.analysis.exp2_chi_natural_vs_n
    python -m experiments.stage3.analysis.exp2_chi_natural_vs_n --cutoff 1e-7
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_RESULTS_DIR = _THIS_DIR.parent / "results"
_OUT_DIR = _ROOT / "results" / "stage3" / "figures" / "exp2"

# Exp 2 anchor: chi cap = 512 (effectively unbounded for our N range)
ANCHOR_CHI = 512


def load_chi_natural(test_case: str, cutoff: float):
    """Return {Re: [(N, max_chi, mean_chi), ...]}  for chi_cap=ANCHOR_CHI runs."""
    by_re = defaultdict(list)
    for f in glob.glob(str(_RESULTS_DIR / "job_*.json")):
        d = json.load(open(f))
        p = d["params"]
        if (p["test_case"] != test_case or p["chi"] != ANCHOR_CHI
                or abs(p["cutoff"] - cutoff) > 1e-15):
            continue
        final = d["mps"]["final"]
        mean_chi = final.get("final_mean_chi")
        # max_chi is in the last timeseries entry
        ts = d["mps"]["timeseries"]
        if not ts:
            continue
        max_chi = ts[-1].get("max_chi")
        if mean_chi is None or max_chi is None:
            continue
        by_re[int(p["re"])].append((p["n"], int(max_chi), float(mean_chi)))
    for re in by_re:
        by_re[re].sort()
    return by_re


def _add_reference_lines(ax, n_range, anchor_n: int, anchor_chi: float,
                          log_y: bool):
    """Add the two meaningful reference scalings: constant and log_2(N).
    Anchored at (anchor_n, anchor_chi)."""
    ns = np.array(sorted(set(n_range)))
    refs = [
        ("const",        np.full_like(ns, anchor_chi, dtype=float), "black", ":"),
        (r"$\log_2 N$",  anchor_chi * np.log2(ns) / np.log2(anchor_n),
                                                                    "gray",  "--"),
    ]
    for label, ys, color, ls in refs:
        ax.plot(ns, ys, color=color, linestyle=ls, linewidth=1.2,
                alpha=0.85, label=label)


def plot(cutoff: float, log_y: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, tc in zip(axes, ("tg", "cavity")):
        by_re = load_chi_natural(tc, cutoff)
        if not by_re:
            ax.set_title(f"{tc.upper()} — no data")
            continue
        re_sorted = sorted(by_re)
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(re_sorted)))
        all_ns = []
        anchor_chi = None
        for color, re in zip(cmap, re_sorted):
            pts = by_re[re]
            ns = np.array([n for n, _, _ in pts])
            max_chis = np.array([mc for _, mc, _ in pts])
            all_ns.extend(ns.tolist())
            if anchor_chi is None and len(ns) > 0:
                # anchor reference lines at the smallest N's chi for the lowest Re
                anchor_chi = float(max_chis[0])
                anchor_n = int(ns[0])
            ax.plot(ns, max_chis, "o-", color=color, markersize=6,
                    linewidth=1.4, label=f"Re={re}")
        if anchor_chi is not None:
            _add_reference_lines(ax, all_ns, anchor_n, anchor_chi, log_y)
        ax.set_xlabel(r"grid size $N$")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.grid(True, which="both", alpha=0.15)
        ax.set_title(f"{tc.upper()}", fontsize=11)
        if log_y:
            ax.set_yscale("log")
        ax.legend(loc="best", frameon=False, fontsize=8, ncol=2)
    axes[0].set_ylabel(r"natural bond dimension $\chi_{\max}$ (final time)")
    fig.suptitle(
        rf"Exp 2 — natural $\chi$ vs grid size  (chi_cap={ANCHOR_CHI}, cutoff={cutoff:.0e}, taylor=2)",
        fontsize=12,
    )
    fig.tight_layout()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_logy" if log_y else ""
    out_path = _OUT_DIR / f"chi_natural_vs_n_cut{cutoff:.0e}{suffix}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    args = p.parse_args()
    for cutoff in (1e-10, 1e-7):
        plot(cutoff, log_y=False)
        plot(cutoff, log_y=True)


if __name__ == "__main__":
    main()

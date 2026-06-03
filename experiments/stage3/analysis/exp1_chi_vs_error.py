"""
Exp 1 figure: chi vs final L2 error.

Per experiment_plan.md: "one subplot per test case, curves for each (N, Re)".
Layout: 1 row × 3 columns (N=64, N=128, N=256). Two curves per subplot
(Re=100, Re=500) shown in distinct colors.

For TG (analytical solution available), generates a second figure with
2 rows × 3 columns (rows = Re, cols = N), each subplot showing:
  - MPS vs vanilla
  - MPS vs analytical
  - vanilla vs analytical (horizontal LBM floor)

Restricts to the Exp 1 standard chi sweep:
    [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64]

Usage:
    python -m experiments.stage3.analysis.exp1_chi_vs_error
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lbm import D2Q9
from lbm.collision import compute_moments
from tn.compression import decompress_populations
from simulations.taylor_green import analytical_taylor_green

_RESULTS_DIR = _THIS_DIR.parent / "results"
_CKPT_DIR = _THIS_DIR.parent / "checkpoints"
_BASELINE_DIR = _THIS_DIR.parent / "baselines"
_OUT_DIR = _ROOT / "results" / "stage3" / "figures" / "exp1"

# Standard Exp 1 chi sweep (per experiment_plan.md):
EXP1_CHI_SWEEP = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64]
ERROR_THRESHOLD = 0.01

# Standard Exp 1 grid
EXP1_N_VALUES = [64, 128, 256]
EXP1_RE_VALUES = [100, 500]

# Distinct colors per Re
_RE_COLORS = {100: "C0", 500: "C3"}


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def _rel_l2(u_test: np.ndarray, u_ref: np.ndarray) -> float:
    d = u_test - u_ref
    return float(np.sqrt(np.sum(d ** 2) / (np.sum(u_ref ** 2) + 1e-30)))


def load_sweep(test_case: str, n: int, re: float, cutoff: float,
               chi_min: int = min(EXP1_CHI_SWEEP),
               chi_max: int = max(EXP1_CHI_SWEEP)):
    """Return [(chi, l2_vs_vanilla, nt_completed, job_id), ...] sorted by chi."""
    pts = []
    for f in glob.glob(str(_RESULTS_DIR / "job_*.json")):
        d = json.load(open(f))
        p = d["params"]
        if (p["test_case"] == test_case
                and p["n"] == n
                and abs(p["re"] - re) < 1e-9
                and abs(p["cutoff"] - cutoff) < 1e-15):
            chi = p["chi"]
            if chi is None or not (chi_min <= chi <= chi_max):
                continue
            l2 = d["mps"]["final"].get("final_l2_err_vs_baseline")
            nt = d["mps"]["final"]["nt_completed"]
            if l2 is not None:
                pts.append((chi, l2, nt, p["job_id"]))
    return sorted(pts)


def tg_analytical_errors(n: int, re: float, u: float, nt: int,
                         job_ids: list[int]):
    """For TG only: return (list of mps_vs_analytical per job_id, vanilla_vs_analytical scalar)."""
    nu = u * n / re
    u_an = analytical_taylor_green(n, t=nt, nu=nu, U0=u)

    baseline_path = _BASELINE_DIR / f"tg_n{n}_re{int(re)}_u{u:.3g}.npz"
    van_vs_an = None
    if baseline_path.exists():
        npz = np.load(baseline_path)
        van_vs_an = _rel_l2(npz["u_final"], u_an)

    mps_vs_an = []
    for jid in job_ids:
        ckpt = _CKPT_DIR / f"job_{jid:05d}_final.pkl"
        if not ckpt.exists():
            mps_vs_an.append(None)
            continue
        with open(ckpt, "rb") as f:
            d = pickle.load(f)
        f_mps = decompress_populations(d["mps_state"], d["metadata"])
        _, u_mps = compute_moments(D2Q9, f_mps)
        mps_vs_an.append(_rel_l2(u_mps, u_an))
    return mps_vs_an, van_vs_an


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

def _add_threshold(ax):
    """1% threshold reference line. Label goes in the axhline call so it
    appears in the legend rather than as floating text outside the panel."""
    ax.axhline(ERROR_THRESHOLD, color="black", linestyle="--",
               linewidth=1.3, alpha=0.95, label="1% threshold")


# ----------------------------------------------------------------------
# Simple per-test-case plot (1 row × 3 cols, Re curves overlaid)
# ----------------------------------------------------------------------

def plot_test_case_simple(test_case: str, cutoff: float = 1e-10):
    # For TG, drop the very-small chi values that diverge at higher Re —
    # they squash the y-axis. The chi=6, 8 points are pre-stability for TG
    # at Re=500 (mass blows up, L2 > 100%).
    chi_min = 10 if test_case == "tg" else min(EXP1_CHI_SWEEP)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, n in zip(axes, EXP1_N_VALUES):
        for re in EXP1_RE_VALUES:
            pts = load_sweep(test_case, n, float(re), cutoff, chi_min=chi_min)
            if not pts:
                continue
            chis = np.array([c for c, *_ in pts])
            l2 = np.array([e for _, e, *_ in pts])
            ax.plot(chis, l2, "o-", color=_RE_COLORS[re], markersize=5,
                    linewidth=1.4, label=f"Re={re}")
        _add_threshold(ax)
        ax.set_title(f"N = {n}", fontsize=11)
        ax.set_xlabel(r"bond dimension $\chi$")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.15)
        ax.legend(loc="best", frameon=False, fontsize=9)
    axes[0].set_ylabel(r"relative $L_2$ velocity error vs vanilla LBM")
    fig.suptitle(f"{test_case.upper()} — Exp 1 chi-vs-error (cutoff={cutoff:.0e}, taylor=2)",
                 fontsize=12)
    fig.tight_layout()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"chi_vs_error_{test_case}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}")
    return out_path


# ----------------------------------------------------------------------
# TG with analytical comparison (2 rows × 3 cols, rows = Re, cols = N)
# ----------------------------------------------------------------------

def plot_tg_with_analytical(cutoff: float = 1e-10, u: float = 0.1):
    chi_min = 10   # see comment in plot_test_case_simple
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), sharey=True)
    for row, re in enumerate(EXP1_RE_VALUES):
        for col, n in enumerate(EXP1_N_VALUES):
            ax = axes[row, col]
            pts = load_sweep("tg", n, float(re), cutoff, chi_min=chi_min)
            if not pts:
                ax.set_title(f"N={n}, Re={re}  (no data)", fontsize=10)
                continue
            chis = np.array([c for c, *_ in pts])
            l2_van = np.array([e for _, e, *_ in pts])
            nt = pts[0][2]
            job_ids = [jid for *_, jid in pts]
            mps_vs_an, van_vs_an = tg_analytical_errors(n, float(re), u, nt, job_ids)

            ax.plot(chis, l2_van, "o-", color="C0", markersize=5, linewidth=1.4,
                    label="MPS vs vanilla LBM")
            valid = [(c, e) for c, e in zip(chis, mps_vs_an) if e is not None]
            if valid:
                cs = np.array([c for c, _ in valid])
                es = np.array([e for _, e in valid])
                ax.plot(cs, es, "s-", color="C2", markersize=4, linewidth=1.2,
                        label="MPS vs analytical")
            if van_vs_an is not None:
                ax.axhline(van_vs_an, color="C1", linestyle="-", linewidth=1.5,
                           alpha=0.95, label="vanilla vs analytical")
            _add_threshold(ax)
            ax.set_title(f"N = {n}, Re = {re}", fontsize=10)
            if row == 1:
                ax.set_xlabel(r"bond dimension $\chi$")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.15)
            if row == 0 and col == 2:
                ax.legend(loc="best", frameon=False, fontsize=8)
    axes[0, 0].set_ylabel(r"relative $L_2$ error")
    axes[1, 0].set_ylabel(r"relative $L_2$ error")
    fig.suptitle(f"TG — chi-vs-error with analytical reference (cutoff={cutoff:.0e}, taylor=2)",
                 fontsize=12)
    fig.tight_layout()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "chi_vs_error_tg_with_analytical.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}")
    return out_path


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cutoff", type=float, default=1e-10)
    args = p.parse_args()
    plot_test_case_simple("tg", cutoff=args.cutoff)
    plot_test_case_simple("cavity", cutoff=args.cutoff)
    plot_tg_with_analytical(cutoff=args.cutoff)


if __name__ == "__main__":
    main()

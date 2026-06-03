"""
Exp 4: memory-equivalent comparison — MPS-native vs coarse vanilla.

Per experiment_plan.md: at N_fine = 256, sweep chi for MPS and the
memory-equivalent N_coarse for vanilla, then compare both to the *fine*
vanilla baseline at N=256.

For TG, also compares against the analytical solution (bonus curves on a
separate plot, since the analytical exists at any time).

Outputs (in results/stage3/figures/exp4/):
  - memory_vs_error_tg.pdf       3 panels (one per Re)
  - memory_vs_error_cavity.pdf   3 panels (one per Re)
  - memory_vs_error_tg_vs_analytical.pdf   same layout, vs analytical reference

Required upstream data:
  - results/job_*.json — MPS chi sweeps at N=256 (Exp 4 jobs)
  - baselines/tg_n256_re*.npz, cavity_n256_re*.npz — fine vanilla
  - baselines_exp4_coarse/*.npz — coarse vanilla at memory-equivalent N
    (produced by exp4_run_coarse_vanilla.py)
  - checkpoints/job_*_final.pkl — MPS final states (for TG analytical compare)

Usage:
    python -m experiments.stage3.analysis.exp4_memory_comparison
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
from scipy.ndimage import zoom

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lbm import D2Q9
from lbm.collision import compute_moments
from tn.compression import decompress_populations
from simulations.taylor_green import analytical_taylor_green
from experiments.stage3.common import nvps_mps_field

_RESULTS_DIR  = _THIS_DIR.parent / "results"
_CKPT_DIR     = _THIS_DIR.parent / "checkpoints"
_BASELINE_DIR = _THIS_DIR.parent / "baselines"
_COARSE_DIR   = _THIS_DIR.parent / "baselines_exp4_coarse"
_OUT_DIR      = _ROOT / "results" / "stage3" / "figures" / "exp4"

N_FINE = 256
U = 0.1
EXP4_RE_VALUES = [100, 500, 1000]
TG_CHI_SWEEP  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
CAV_CHI_SWEEP = [6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel_l2(u_test: np.ndarray, u_ref: np.ndarray) -> float:
    d = u_test - u_ref
    return float(np.sqrt(np.sum(d ** 2) / (np.sum(u_ref ** 2) + 1e-30)))


def _interp_to_fine(u_coarse: np.ndarray, n_fine: int) -> np.ndarray:
    """Bilinear interpolation of (2, N_coarse, N_coarse) → (2, n_fine, n_fine)."""
    factor = n_fine / u_coarse.shape[-1]
    return np.stack([zoom(u_coarse[c], factor, order=1) for c in range(2)])


def _fine_vanilla(test_case: str, re: int) -> np.ndarray:
    p = _BASELINE_DIR / f"{test_case}_n{N_FINE}_re{re}_u{U:.3g}.npz"
    return np.load(p)["u_final"]


def _coarse_vanilla(test_case: str, n_coarse: int, re: int):
    p = _COARSE_DIR / f"{test_case}_n{n_coarse}_re{re}_u{U:.3g}.npz"
    if not p.exists():
        return None
    return np.load(p)["u_final"]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_mps_curve(test_case: str, re: int, chi_sweep: list[int]) -> list[tuple]:
    """Return [(chi, memory_floats, l2_vs_fine_van, l2_vs_analytical_or_None, job_id), ...]."""
    pts = []
    for f in glob.glob(str(_RESULTS_DIR / "job_*.json")):
        d = json.load(open(f))
        p = d["params"]
        if (p["test_case"] != test_case or p["n"] != N_FINE
                or abs(p["re"] - re) > 1e-9
                or p["chi"] is None or p["chi"] not in chi_sweep
                or p["cutoff"] != 1e-10):
            continue
        l2_van = d["mps"]["final"].get("final_l2_err_vs_baseline")
        if l2_van is None:
            continue
        chi = p["chi"]
        mem = nvps_mps_field(N_FINE, chi, "snake")
        l2_an = None
        if test_case == "tg":
            ckpt = _CKPT_DIR / f"job_{p['job_id']:05d}_final.pkl"
            if ckpt.exists():
                with open(ckpt, "rb") as fh:
                    ck = pickle.load(fh)
                f_mps = decompress_populations(ck["mps_state"], ck["metadata"])
                _, u_mps = compute_moments(D2Q9, f_mps)
                nu = U * N_FINE / re
                u_an = analytical_taylor_green(N_FINE, t=d["mps"]["final"]["nt_completed"],
                                                nu=nu, U0=U)
                l2_an = _rel_l2(u_mps, u_an)
        pts.append((chi, mem, l2_van, l2_an, p["job_id"]))
    return sorted(pts)


def load_vanilla_curve(test_case: str, re: int, chi_sweep: list[int]) -> list[tuple]:
    """Return [(n_coarse, memory_floats, l2_vs_fine_van, l2_vs_analytical_or_None), ...]."""
    from experiments.stage3.common import memory_equivalent_n_coarse
    u_fine = _fine_vanilla(test_case, re)
    pts = []
    seen_n = set()
    for chi in chi_sweep:
        n_c = memory_equivalent_n_coarse(N_FINE, chi, "snake")
        if n_c in seen_n:
            continue
        seen_n.add(n_c)
        u_c = _coarse_vanilla(test_case, n_c, re)
        if u_c is None:
            continue
        u_c_up = _interp_to_fine(u_c, N_FINE)
        l2_van = _rel_l2(u_c_up, u_fine)
        l2_an = None
        if test_case == "tg":
            # Fair comparison: interpolate coarse vanilla up to N_fine, then
            # compare to analytical at N_fine and the SAME final time as the
            # MPS run. This matches the MPS vs analytical comparison and
            # penalises coarse vanilla for missing high-frequency info.
            nu_fine = U * N_FINE / re
            # nt at fine grid (this is what MPS jobs use); use first MPS job's nt
            # as the canonical comparison time.
            # Fall back to coarse-native nt if no MPS job at this Re yet.
            nt_fine = None
            try:
                from experiments.stage3.test_cases import setup_test_case
                from experiments.stage3.common import Job as _Job
                _j = _Job(job_id=0, test_case="tg", n=N_FINE, re=float(re),
                          u=U, cutoff=1e-10, chi=512, taylor_order=2,
                          mapping="snake")
                nt_fine = setup_test_case(_j).nt
            except Exception:
                pass
            if nt_fine is None:
                # last-resort: derive from physics
                tau_decay_fine = 1.0 / (2 * nu_fine * (2 * np.pi / N_FINE) ** 2)
                nt_fine = int(2 * tau_decay_fine)
            u_an_fine = analytical_taylor_green(N_FINE, t=nt_fine, nu=nu_fine, U0=U)
            u_c_up = _interp_to_fine(u_c, N_FINE)
            l2_an = _rel_l2(u_c_up, u_an_fine)
        mem = 9 * n_c * n_c
        pts.append((n_c, mem, l2_van, l2_an))
    return sorted(pts)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_test_case(test_case: str, vs_analytical: bool = False):
    chi_sweep = TG_CHI_SWEEP if test_case == "tg" else CAV_CHI_SWEEP
    if vs_analytical and test_case != "tg":
        return None

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, re in zip(axes, EXP4_RE_VALUES):
        mps  = load_mps_curve(test_case, re, chi_sweep)
        van  = load_vanilla_curve(test_case, re, chi_sweep)
        idx_mps = 3 if vs_analytical else 2
        idx_van = 3 if vs_analytical else 2

        if mps:
            mem  = [m[1] for m in mps]
            err  = [m[idx_mps] for m in mps]
            keep = [(x, y) for x, y in zip(mem, err) if y is not None]
            if keep:
                xs, ys = zip(*keep)
                ax.plot(xs, ys, "o-", color="C0", markersize=5,
                        linewidth=1.4, label="MPS-native @ N=256, varying χ")
        if van:
            mem  = [v[1] for v in van]
            err  = [v[idx_van] for v in van]
            keep = [(x, y) for x, y in zip(mem, err) if y is not None]
            if keep:
                xs, ys = zip(*keep)
                ax.plot(xs, ys, "s--", color="C3", markersize=5,
                        linewidth=1.4, label="vanilla @ N_coarse")

        ax.axhline(0.01, color="black", linestyle="--", linewidth=1.0,
                   alpha=0.6, label="1% threshold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("memory (floats)")
        ax.set_title(f"Re = {re}", fontsize=11)
        ax.grid(True, which="both", alpha=0.15)
        ax.legend(loc="best", frameon=False, fontsize=8)

    ref_label = "analytical" if vs_analytical else "fine vanilla (N=256)"
    axes[0].set_ylabel(f"relative $L_2$ error vs {ref_label}")
    fig.suptitle(f"{test_case.upper()} — Exp 4 memory-equivalent comparison "
                 f"({'vs analytical' if vs_analytical else 'vs fine vanilla'})",
                 fontsize=12)
    fig.tight_layout()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_vs_analytical" if vs_analytical else ""
    out_path = _OUT_DIR / f"memory_vs_error_{test_case}{suffix}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-cavity", action="store_true",
                   help="skip cavity (useful if coarse vanilla still running)")
    args = p.parse_args()
    plot_test_case("tg")
    plot_test_case("tg", vs_analytical=True)
    if not args.skip_cavity:
        plot_test_case("cavity")


if __name__ == "__main__":
    main()

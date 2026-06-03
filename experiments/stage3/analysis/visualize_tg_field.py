"""
Visualize TG velocity field: MPS final vs vanilla vs analytical.

For one (N, Re, chi) tuple, load:
  - MPS final state (decompress to velocity field)
  - Vanilla baseline
  - Analytical TG solution at the same final time

Plot side-by-side velocity magnitude maps + a delta-from-analytical map for each.

Usage:
    python -m experiments.stage3.analysis.visualize_tg_field --n 256 --re 100 --chi 12
"""

from __future__ import annotations

import argparse
import json
import glob
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

_RESULTS_DIR  = _THIS_DIR.parent / "results"
_CKPT_DIR     = _THIS_DIR.parent / "checkpoints"
_BASELINE_DIR = _THIS_DIR.parent / "baselines"
_OUT_DIR      = _ROOT / "results" / "stage3" / "figures" / "viz"


def find_job(test_case: str, n: int, re: float, chi: int, cutoff: float = 1e-10):
    for f in glob.glob(str(_RESULTS_DIR / "job_*.json")):
        d = json.load(open(f))
        p = d["params"]
        if (p["test_case"] == test_case and p["n"] == n
                and abs(p["re"] - re) < 1e-9
                and p["chi"] == chi
                and abs(p["cutoff"] - cutoff) < 1e-15):
            return d
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--chi", type=int, required=True)
    ap.add_argument("--u", type=float, default=0.1)
    ap.add_argument("--cutoff", type=float, default=1e-10)
    args = ap.parse_args()

    d = find_job("tg", args.n, args.re, args.chi, args.cutoff)
    if d is None:
        raise SystemExit(f"No job found for tg N={args.n} Re={args.re} chi={args.chi}")
    jid = d["params"]["job_id"]
    nt = d["mps"]["final"]["nt_completed"]

    # Vanilla
    vp = _BASELINE_DIR / f"tg_n{args.n}_re{int(args.re)}_u{args.u:.3g}.npz"
    u_van = np.load(vp)["u_final"]

    # MPS
    ck = _CKPT_DIR / f"job_{jid:05d}_final.pkl"
    if not ck.exists():
        raise SystemExit(f"No checkpoint at {ck}")
    with open(ck, "rb") as f:
        cd = pickle.load(f)
    f_mps = decompress_populations(cd["mps_state"], cd["metadata"])
    _, u_mps = compute_moments(D2Q9, f_mps)

    # Analytical
    nu = args.u * args.n / args.re
    u_an = analytical_taylor_green(args.n, t=nt, nu=nu, U0=args.u)

    # Plot: top row = |u| for MPS, vanilla, analytical
    # bottom row = |u_X - u_analytical| heatmap for MPS and vanilla, and a histogram of differences
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))

    speeds = {
        "MPS chi=" + str(args.chi): np.sqrt(u_mps[0]**2 + u_mps[1]**2),
        "vanilla (N=" + str(args.n) + ")": np.sqrt(u_van[0]**2 + u_van[1]**2),
        "analytical (t=" + str(nt) + ")": np.sqrt(u_an[0]**2 + u_an[1]**2),
    }
    vmax = max(s.max() for s in speeds.values())
    for ax, (title, s) in zip(axes[0], speeds.items()):
        im = ax.imshow(s.T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.85)

    # bottom row: error maps and histogram
    delta_mps = np.sqrt((u_mps[0]-u_an[0])**2 + (u_mps[1]-u_an[1])**2)
    delta_van = np.sqrt((u_van[0]-u_an[0])**2 + (u_van[1]-u_an[1])**2)
    dmax = max(delta_mps.max(), delta_van.max())

    im = axes[1, 0].imshow(delta_mps.T, origin="lower", cmap="magma", vmin=0, vmax=dmax)
    axes[1, 0].set_title(f"|u_MPS - u_analytical|  max={delta_mps.max():.2e}", fontsize=10)
    plt.colorbar(im, ax=axes[1, 0], shrink=0.85)

    im = axes[1, 1].imshow(delta_van.T, origin="lower", cmap="magma", vmin=0, vmax=dmax)
    axes[1, 1].set_title(f"|u_vanilla - u_analytical|  max={delta_van.max():.2e}", fontsize=10)
    plt.colorbar(im, ax=axes[1, 1], shrink=0.85)

    ax = axes[1, 2]
    ax.hist(delta_mps.flatten(), bins=50, alpha=0.6, label="MPS error", color="C0")
    ax.hist(delta_van.flatten(), bins=50, alpha=0.6, label="vanilla error", color="C3")
    ax.set_xlabel("|u - u_an| per cell")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("pixel-wise error distribution")

    fig.suptitle(f"TG N={args.n} Re={int(args.re)} chi={args.chi} u={args.u}  (t={nt})",
                 fontsize=12)
    fig.tight_layout()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / f"tg_n{args.n}_re{int(args.re)}_chi{args.chi}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote: {out}")
    # Print stats
    rel_mps = float(np.sqrt(np.sum(delta_mps**2) / np.sum(np.sqrt(u_an[0]**2+u_an[1]**2)**2)))
    rel_van = float(np.sqrt(np.sum(delta_van**2) / np.sum(np.sqrt(u_an[0]**2+u_an[1]**2)**2)))
    print(f"  MPS vs analytical  rel-L2: {rel_mps:.3e}")
    print(f"  vanilla vs analytical rel-L2: {rel_van:.3e}")
    print(f"  speed range (analytical): [{u_an.min():.3e}, {u_an.max():.3e}]")


if __name__ == "__main__":
    main()

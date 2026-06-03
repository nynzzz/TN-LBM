"""
Plot SV spectra from one or more SV-capture runs.

For each loaded run, plot σ_k vs k (log-y) per snapshot. The supervisor's
"stairs/spikes" pattern would show as flat plateaus or clusters of nearly
equal σ values — those are degeneracies, and chi values that cut inside
a degenerate cluster are the candidates for the wrong-attractor failure.

Usage:
    python -m experiments.stage3.exp_sv_spectrum.plot_spectrum \
        --runs cluster_chi17 cluster_chi12_repro local_chi12 \
        --population 1 --bond 7

Default: one figure per run, panels per snapshot, single bond at the
middle of one representative population.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve()
_DATA_DIR = _THIS.parent / "data"
_OUT_DIR = _THIS.parent / "plots"


def load_run(run_name: str):
    """Return sorted list of (t, snap_dict) loaded from disk."""
    d = _DATA_DIR / run_name
    if not d.exists():
        raise SystemExit(f"no data dir at {d}")
    snaps = []
    for p in sorted(d.glob("snap_t*.pkl")):
        with open(p, "rb") as f:
            snaps.append(pickle.load(f))
    snaps.sort(key=lambda s: s["t"])
    return snaps


def plot_run(run_name: str, population: int, bond: int):
    snaps = load_run(run_name)
    if not snaps:
        print(f"no snapshots in {run_name}")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.95, len(snaps)))
    for color, snap in zip(cmap, snaps):
        svs = snap["svs_per_pop_per_bond"][population][bond]
        ax.plot(np.arange(1, len(svs) + 1), svs, "o-",
                color=color, markersize=4, linewidth=1.0,
                label=f"t={snap['t']}")
    chi = snaps[0]["chi"]
    ax.set_xlabel(r"singular value index $k$")
    ax.set_ylabel(r"$\sigma_k$")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, ncol=2, loc="best", frameon=False)
    ax.set_title(f"{run_name}  (pop={population}, bond={bond}, chi={chi})")
    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / f"{run_name}_pop{population}_bond{bond}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote: {out}")


def plot_comparison(run_names: list[str], population: int, bond: int, t_idx: int = -1):
    """Overlay SVs from several runs at one snapshot index. Useful for
    chi=12 cluster vs chi=12 local comparison at the same t."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for run_name in run_names:
        snaps = load_run(run_name)
        if not snaps:
            continue
        snap = snaps[t_idx]
        svs = snap["svs_per_pop_per_bond"][population][bond]
        ax.plot(np.arange(1, len(svs) + 1), svs, "o-",
                markersize=4, linewidth=1.0,
                label=f"{run_name} (chi={snap['chi']}, t={snap['t']})")
    ax.set_xlabel(r"singular value index $k$")
    ax.set_ylabel(r"$\sigma_k$")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=9, loc="best", frameon=False)
    ax.set_title(f"SV spectrum comparison (pop={population}, bond={bond})")
    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / f"compare_pop{population}_bond{bond}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--population", type=int, default=1,
                   help="D2Q9 population index (0=rest, 1-8=directions)")
    p.add_argument("--bond", type=int, default=None,
                   help="bond index to plot; default = middle bond")
    p.add_argument("--compare", action="store_true",
                   help="overlay all runs at one snapshot (last by default)")
    args = p.parse_args()

    for run in args.runs:
        snaps = load_run(run)
        if not snaps:
            print(f"skipping {run}: no snapshots")
            continue
        bond = args.bond
        if bond is None:
            n_bonds = len(snaps[0]["svs_per_pop_per_bond"][args.population])
            bond = n_bonds // 2
            print(f"{run}: defaulting to bond={bond} (middle of {n_bonds})")
        plot_run(run, args.population, bond)

    if args.compare and len(args.runs) > 1:
        bond = args.bond if args.bond is not None else len(
            load_run(args.runs[0])[0]["svs_per_pop_per_bond"][args.population]
        ) // 2
        plot_comparison(args.runs, args.population, bond)


if __name__ == "__main__":
    main()

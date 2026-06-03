"""
Plot decompressed velocity fields (|u|) at every snapshot of an SV-capture run.

Useful for visually confirming whether the MPS state at each snapshot is
correctly reproducing the physical flow (e.g., TG 4x4 vortex grid). Pair
with `plot_spectrum.py` to correlate flow appearance with SV structure.

Usage:
    python -m experiments.stage3.exp_sv_spectrum.plot_fields --runs smoke_test
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lbm import D2Q9
from lbm.collision import compute_moments

_DATA_DIR = _THIS.parent / "data"
_OUT_DIR = _THIS.parent / "plots"


def load_run(run_name: str):
    d = _DATA_DIR / run_name
    snaps = []
    for p in sorted(d.glob("snap_t*.pkl")):
        with open(p, "rb") as f:
            snaps.append(pickle.load(f))
    snaps.sort(key=lambda s: s["t"])
    return snaps


def plot_fields(run_name: str):
    snaps = load_run(run_name)
    if not snaps:
        print(f"no snapshots for {run_name}")
        return
    n_cols = min(len(snaps), 6)
    n_rows = (len(snaps) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.6 * n_cols, 2.6 * n_rows),
                              squeeze=False)
    for i, snap in enumerate(snaps):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        f = snap["f_decompressed"]
        _, u = compute_moments(D2Q9, f)
        speed = np.sqrt(u[0] ** 2 + u[1] ** 2)
        im = ax.imshow(speed.T, origin="lower", cmap="viridis")
        ax.set_title(f"t={snap['t']}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.85)
    # hide unused axes
    for j in range(len(snaps), n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].axis("off")
    fig.suptitle(f"{run_name}: |u| at each snapshot  "
                 f"(chi={snaps[0]['chi']}, N={snaps[0]['n']}, Re={int(snaps[0]['re'])})",
                 fontsize=11)
    fig.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / f"{run_name}_fields.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args()
    for run in args.runs:
        plot_fields(run)


if __name__ == "__main__":
    main()

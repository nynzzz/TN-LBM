"""
Run MPS-native LBM on TG (or any periodic test case) while capturing the SVD
spectrum of each MPS bond at chosen snapshot times. Used to investigate the
chi-dependent wrong-attractor failure mode observed on the cluster.

For each snapshot, we save (one pickle file per snapshot):
  - the full MPS state (9 populations)
  - the SVD spectrum at every bond of every population
  - the decompressed populations (for L2 / visualisation later)
  - metadata (chi, t, env, physics params, wall time)

Files written to: experiments/stage3/exp_sv_spectrum/data/{run_name}/snap_t{t:07d}.pkl

Usage (typical TG attractor experiment):
    python -m experiments.stage3.exp_sv_spectrum.run_sv_capture \
        --chi 17 --run-name cluster_chi17 --n 256 --re 1000 \
        --snapshot-times 0 100 500 2000 7000 15000 22000 30000 40000 50000 64845

Designed to run identically on local Mac and Snellius (no env-specific code).
"""

from __future__ import annotations

import argparse
import os
import pickle
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lbm import D2Q9
from tn.compression import compress_populations, decompress_populations
from tn_lbm.collision import collide_bgk_mps
from tn_lbm.streaming import stream_all_populations
from experiments.stage3.common import Job
from experiments.stage3.test_cases import setup_test_case


def get_singular_values(mps) -> list[np.ndarray]:
    """Return SVs at every bond of an MPS. One numpy array per bond.

    Quimb's singular_values(i) wants i in [1, L-1] (bond i sits between
    sites i-1 and i). We iterate over that range.
    """
    psi = mps.copy()
    L = psi.L
    svs_per_bond = []
    for i in range(1, L):
        try:
            sv = psi.singular_values(i)
            svs_per_bond.append(np.asarray(sv))
        except Exception as e:
            # Manual fallback via SVD on the bond tensor
            try:
                psi2 = mps.copy()
                psi2.left_canonize(stop=i)
                psi2.right_canonize(start=i)
                ti = psi2.tensors[i - 1]
                if len(ti.inds) == 3:
                    left = [ti.inds[0], ti.inds[1]]
                    right = [ti.inds[2]]
                else:
                    left = list(ti.inds[:-1])
                    right = [ti.inds[-1]]
                mat = ti.to_dense(left, right)
                sv = np.linalg.svd(mat, compute_uv=False)
                svs_per_bond.append(np.asarray(sv))
            except Exception:
                svs_per_bond.append(np.array([np.nan]))
    return svs_per_bond


def env_info() -> dict:
    import numpy as np
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy_version": np.__version__,
        "blas": _detect_blas(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def _detect_blas() -> str:
    """Best-effort BLAS detection from numpy's build info."""
    try:
        import numpy as np
        cfg = np.show_config(mode="dicts")
        if isinstance(cfg, dict):
            blas = cfg.get("Build Dependencies", {}).get("blas", {})
            return blas.get("name", "unknown")
    except Exception:
        pass
    return "unknown"


def run(chi: int, run_name: str, snapshot_times: list[int],
        n: int = 256, re: float = 1000.0, u: float = 0.1,
        cutoff: float = 1e-10, taylor_order: int = 2,
        save_svs: bool = True, save_decompressed: bool = True):
    out_dir = _THIS.parent / "data" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")

    job = Job(job_id=0, test_case="tg", n=n, re=float(re), u=u,
              cutoff=cutoff, chi=chi, taylor_order=taylor_order, mapping="snake")
    spec = setup_test_case(job)
    print(f"physics: TG N={n} Re={int(re)} u={u}  tau={spec.tau:.4f}  nt={spec.nt}")
    print(f"snapshot times: {snapshot_times}")

    mps_list, metadata = compress_populations(spec.f_init, mapping="snake")
    env = env_info()

    t0 = time.time()
    last_snap = max(snapshot_times)

    def save_snapshot(t: int):
        snap = {
            "run_name": run_name,
            "env": env,
            "chi": chi,
            "t": t,
            "n": n, "re": re, "u": u,
            "cutoff": cutoff, "taylor_order": taylor_order,
            "mps_state": [m.copy() for m in mps_list],
            "metadata": metadata,
            "wall_time_cumulative_s": time.time() - t0,
        }
        if save_svs:
            snap["svs_per_pop_per_bond"] = [
                get_singular_values(m) for m in mps_list
            ]
        if save_decompressed:
            snap["f_decompressed"] = decompress_populations(mps_list, metadata)
        path = out_dir / f"snap_t{t:07d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(snap, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  saved {path.name} (elapsed {time.time() - t0:.1f}s)")

    if 0 in snapshot_times:
        save_snapshot(0)

    for t in range(1, spec.nt + 1):
        mps_list = collide_bgk_mps(
            mps_list, D2Q9, spec.tau, spec.n,
            max_bond=chi, cutoff=cutoff, taylor_order=taylor_order)
        mps_list = stream_all_populations(
            mps_list, metadata, D2Q9,
            max_bond=chi, cyclic=True)
        if t in snapshot_times:
            save_snapshot(t)
        if t >= last_snap:
            break

    print(f"done. total wall: {(time.time()-t0)/60:.1f} min")
    print(f"output: {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chi", type=int, required=True)
    p.add_argument("--run-name", required=True,
                   help="subfolder of data/ to write to (e.g. 'cluster_chi17')")
    p.add_argument("--snapshot-times", type=int, nargs="+", required=True)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--re", type=float, default=1000.0)
    p.add_argument("--u", type=float, default=0.1)
    p.add_argument("--cutoff", type=float, default=1e-10)
    p.add_argument("--taylor-order", type=int, default=2)
    p.add_argument("--no-svs", action="store_true",
                   help="skip SV spectrum extraction (faster, smaller files)")
    p.add_argument("--no-decompressed", action="store_true",
                   help="skip decompressed f save (smaller files)")
    args = p.parse_args()

    run(chi=args.chi, run_name=args.run_name, snapshot_times=args.snapshot_times,
        n=args.n, re=args.re, u=args.u, cutoff=args.cutoff,
        taylor_order=args.taylor_order,
        save_svs=not args.no_svs, save_decompressed=not args.no_decompressed)


if __name__ == "__main__":
    main()

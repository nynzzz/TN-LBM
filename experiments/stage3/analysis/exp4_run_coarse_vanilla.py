"""
Run vanilla LBM at memory-equivalent coarse N values for Exp 4.

For each Exp 4 chi sweep value, the memory-equivalent dense grid has:
    N_coarse = floor(sqrt(NVPS_mps(N_fine, chi) / 9))

where NVPS_mps is the per-population memory of the MPS at (N_fine, chi)
under our snake mapping (Gross Eq 46 adapted for snake; see
common.nvps_per_population).

This script enumerates all unique (test_case, N_coarse, Re) combos required
by the TG and cavity Exp 4 chi sweeps, then runs vanilla LBM at each. Cached
baselines in experiments/stage3/baselines/ are skipped automatically (the
runner short-circuits on cache hit), so re-running is safe and incremental.

Usage:
    python -m experiments.stage3.analysis.exp4_run_coarse_vanilla
    python -m experiments.stage3.analysis.exp4_run_coarse_vanilla --test-case tg     # only TG
    python -m experiments.stage3.analysis.exp4_run_coarse_vanilla --re 100           # only one Re
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.stage3 import common
from experiments.stage3.common import (
    Job, baseline_exists, memory_equivalent_n_coarse,
)
from experiments.stage3.runners import run_vanilla

# Redirect baseline storage to a separate subfolder so we don't mix the
# coarse-vanilla outputs with the existing N=16..512 baselines.
_COARSE_DIR = _THIS_DIR.parent / "baselines_exp4_coarse"
_COARSE_DIR.mkdir(exist_ok=True)
common.BASELINES_DIR = _COARSE_DIR

# Exp 4 chi sweeps (per experiment_plan.md):
TG_CHI_SWEEP  = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
CAV_CHI_SWEEP = [6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80]
EXP4_RE_VALUES = [100, 500, 1000]
N_FINE = 256
U = 0.1


def build_job_list(test_cases: list[str], re_values: list[int]) -> list[Job]:
    """Build list of (test_case, N_coarse, Re) Job objects, deduped."""
    jobs = []
    seen = set()
    next_jid = 900000   # synthetic IDs, well outside master_jobs.csv range
    for tc in test_cases:
        chi_sweep = TG_CHI_SWEEP if tc == "tg" else CAV_CHI_SWEEP
        # collect unique N_coarse values for this test case
        n_coarse_set = sorted({
            memory_equivalent_n_coarse(N_FINE, chi, "snake")
            for chi in chi_sweep
        })
        for n_c in n_coarse_set:
            for re in re_values:
                key = (tc, n_c, re)
                if key in seen:
                    continue
                seen.add(key)
                # Default nt computation in test_cases.py uses integer division
                # `50000 * (n // 64)`, which is 0 for any N < 64. Pass nt
                # explicitly to avoid that. Scale generously; cavity will exit
                # early via convergence_tol once du<1e-6.
                nt_override = None
                if tc == "cavity" and n_c < 64:
                    nt_override = max(5000, int(20000 * n_c / 64))
                elif tc == "cavity":
                    nt_override = int(20000 * n_c / 64)
                # TG uses physics-derived nt = 2*tau_decay (computed in test_cases.py
                # from spec.nu and k); that path works for any N. Leave nt=None.
                jobs.append(Job(
                    job_id=next_jid,
                    test_case=tc,
                    n=n_c,
                    re=float(re),
                    u=U,
                    cutoff=1e-10,
                    chi=512,
                    taylor_order=2,
                    mapping="snake",
                    nt=nt_override,
                ))
                next_jid += 1
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-case", choices=["tg", "cavity", "both"], default="both")
    p.add_argument("--re", type=int, choices=EXP4_RE_VALUES, default=None,
                   help="restrict to one Re (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="print job list and exit, don't run")
    args = p.parse_args()

    test_cases = ["tg", "cavity"] if args.test_case == "both" else [args.test_case]
    re_values = [args.re] if args.re else EXP4_RE_VALUES
    jobs = build_job_list(test_cases, re_values)

    cached = [j for j in jobs if baseline_exists(j)]
    todo   = [j for j in jobs if not baseline_exists(j)]

    print(f"Exp 4 coarse vanilla baselines")
    print(f"  test cases: {test_cases}")
    print(f"  Re values:  {re_values}")
    print(f"  total:      {len(jobs)}  (cached: {len(cached)}, to run: {len(todo)})")
    print()

    if args.dry_run or not todo:
        print("Jobs to run:")
        for j in todo:
            print(f"  {j.test_case:<7} N={j.n:>4} Re={int(j.re):>5}")
        return

    t0 = time.time()
    for i, j in enumerate(todo, 1):
        t_job = time.time()
        print(f"[{i:>3}/{len(todo)}] {j.test_case:<7} N={j.n:>4} Re={int(j.re):>5}  ... ",
              end="", flush=True)
        try:
            run_vanilla(j, verbose=False)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        wall = time.time() - t_job
        cumulative_h = (time.time() - t0) / 3600
        print(f"done in {wall:.1f}s   (cumulative {cumulative_h:.2f}h)")

    print()
    print(f"Total wall time: {(time.time() - t0)/3600:.2f}h")


if __name__ == "__main__":
    main()

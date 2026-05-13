"""
Pre-generate all unique vanilla baselines before submitting the MPS array.

Without pre-gen, many MPS jobs sharing the same (test_case, N, Re, u)
would race on baseline write. This script populates the cache once.

Two modes:
  - Default (no --task-id): run all unique baselines sequentially.
  - With --task-id N: run only the Nth unique baseline (for SLURM array).

Usage:
    # Sequential (small total cost):
    python -m experiments.stage3.build_baselines

    # SLURM array (parallel — see jobs/submit_baselines.sh):
    python -m experiments.stage3.build_baselines --task-id $SLURM_ARRAY_TASK_ID

    # Show how many unique baselines exist:
    python -m experiments.stage3.build_baselines --count
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.stage3.common import JOBS_DIR, jobs_from_csv, baseline_exists
from experiments.stage3.runners import run_vanilla


def unique_baseline_jobs(master_csv: Path) -> list:
    """Return list of Job objects, deduplicated by baseline_key, in stable order."""
    jobs = jobs_from_csv(master_csv)
    seen: dict[str, "Job"] = {}
    for j in sorted(jobs, key=lambda x: x.job_id):
        if j.baseline_key not in seen:
            seen[j.baseline_key] = j
    return list(seen.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master-csv", type=Path,
                   default=JOBS_DIR / "master_jobs.csv")
    p.add_argument("--task-id", type=int, default=None,
                   help="Run only the Nth unique baseline (for SLURM array)")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cache exists")
    p.add_argument("--count", action="store_true",
                   help="Print the number of unique baselines and exit")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    unique = unique_baseline_jobs(args.master_csv)

    if args.count:
        print(len(unique))
        sys.exit(0)

    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(unique):
            print(f"ERROR: task-id {args.task_id} out of range "
                  f"[0, {len(unique)-1}]", file=sys.stderr)
            sys.exit(1)
        j = unique[args.task_id]
        print(f"Baseline {args.task_id}/{len(unique)-1}: {j.baseline_key}")
        if not args.force and baseline_exists(j):
            print(f"  cached — skipping")
            sys.exit(0)
        t0 = time.time()
        run_vanilla(j, force=args.force, verbose=args.verbose)
        print(f"  done in {time.time()-t0:.1f}s")
        sys.exit(0)

    # Sequential mode (run all)
    print(f"Unique baselines: {len(unique)}")
    cached = sum(1 for j in unique if baseline_exists(j))
    todo = [j for j in unique if args.force or not baseline_exists(j)]
    print(f"  already cached: {cached}")
    print(f"  to compute:     {len(todo)}")
    print()

    t_start = time.time()
    for i, j in enumerate(todo, 1):
        t_job = time.time()
        print(f"[{i:>3}/{len(todo)}] {j.baseline_key}")
        try:
            run_vanilla(j, force=args.force, verbose=args.verbose)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            continue
        print(f"  ✓ done in {time.time()-t_job:.1f}s  "
              f"(total: {(time.time()-t_start)/60:.1f}min)")

    print(f"\nAll baselines done in {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()

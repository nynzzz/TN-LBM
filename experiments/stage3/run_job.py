#!/usr/bin/env python
"""
CLI entry point for running a single stage 3 MPS job.

Used by SLURM array script (`jobs/submit.sh`). Idempotent — skips if the
result JSON already exists, so resubmits are safe.

Exit codes:
    0  — job completed (result JSON written)
    99 — wall-clock limit hit; checkpoint saved; needs resubmit
    1  — error (see traceback)

Usage:
    python -m experiments.stage3.run_job --job-id 42
    python -m experiments.stage3.run_job --job-id 42 --wall-clock-limit-s 169500
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.stage3.common import JOBS_DIR, get_job
from experiments.stage3.runners import run_mps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-id", type=int, required=True,
                   help="Job id (row index in master_jobs.csv)")
    p.add_argument("--master-csv", type=Path,
                   default=JOBS_DIR / "master_jobs.csv",
                   help="Path to master jobs CSV")
    p.add_argument("--wall-clock-limit-s", type=float, default=None,
                   help="Stop and checkpoint after this many seconds (default: no limit)")
    p.add_argument("--checkpoint-every", type=int, default=5000,
                   help="Save checkpoint every N steps (default: 5000)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if result JSON exists")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Load job spec
    try:
        job = get_job(args.job_id, args.master_csv)
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Job {job.job_id}: {job.test_case} N={job.n} Re={job.re} u={job.u} "
          f"chi={job.chi} cutoff={job.cutoff} taylor_order={job.taylor_order}")
    print(f"  result_path: {job.result_path}")

    # Idempotency — skip if result already exists
    if not args.force and job.result_path.exists():
        print(f"  result already exists, skipping. Use --force to re-run.")
        sys.exit(0)

    t0 = time.time()
    try:
        result = run_mps(
            job,
            wall_clock_limit_s=args.wall_clock_limit_s,
            checkpoint_every=args.checkpoint_every,
            verbose=args.verbose,
        )
    except Exception:
        elapsed = time.time() - t0
        print(f"ERROR after {elapsed:.0f}s:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    status = result["status"]
    print(f"  status={status}  t_reached={result['t_reached']}  wall={elapsed:.0f}s")

    if status == "complete":
        sys.exit(0)
    elif status == "incomplete":
        print(f"  checkpoint saved — resubmit to resume.")
        sys.exit(99)
    else:
        print(f"  unexpected status: {status}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

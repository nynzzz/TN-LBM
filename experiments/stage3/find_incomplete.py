"""
Scan the master CSV and classify each job's state.

Categories:
    complete:    result_path exists  (nothing to do)
    incomplete:  no result, but checkpoint exists  (wall-clock hit; can resume)
    not_started: no result, no checkpoint          (never ran or crashed before any checkpoint)

Output: counts + comma-separated job_id lists ready to paste into sbatch:
    sbatch --array=<paste-here> experiments/stage3/jobs/submit.sh

Usage:
    python -m experiments.stage3.find_incomplete
    python -m experiments.stage3.find_incomplete --include-not-started
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.stage3.common import JOBS_DIR, jobs_from_csv


def _to_ranges(ids: list[int]) -> str:
    """Compress sorted ints to ranges, e.g. [1,2,3,5,7,8] -> '1-3,5,7-8'."""
    if not ids:
        return ""
    parts = []
    start = end = ids[0]
    for x in ids[1:]:
        if x == end + 1:
            end = x
        else:
            parts.append(f"{start}-{end}" if end > start else f"{start}")
            start = end = x
    parts.append(f"{start}-{end}" if end > start else f"{start}")
    return ",".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master-csv", type=Path,
                   default=JOBS_DIR / "master_jobs.csv")
    p.add_argument("--include-not-started", action="store_true",
                   help="Also print not-started jobs (in addition to incomplete)")
    args = p.parse_args()

    jobs = jobs_from_csv(args.master_csv)

    complete, incomplete, not_started = [], [], []
    for j in jobs:
        if j.result_path.exists():
            complete.append(j.job_id)
        elif j.checkpoint_path.exists():
            incomplete.append(j.job_id)
        else:
            not_started.append(j.job_id)

    total = len(jobs)
    print(f"Master CSV: {args.master_csv}")
    print(f"Total jobs: {total}")
    print(f"  complete:    {len(complete):>4}  ({len(complete)/total*100:5.1f}%)")
    print(f"  incomplete:  {len(incomplete):>4}  ({len(incomplete)/total*100:5.1f}%)  (have checkpoint — resume)")
    print(f"  not started: {len(not_started):>4}  ({len(not_started)/total*100:5.1f}%)  (no checkpoint either)")
    print()

    if incomplete:
        print("Incomplete (wall-clock-hit, has checkpoint) — resubmit to resume:")
        print(f"  sbatch --array={_to_ranges(sorted(incomplete))} experiments/stage3/jobs/submit.sh")
        print()

    if args.include_not_started and not_started:
        print("Not started — submit fresh:")
        print(f"  sbatch --array={_to_ranges(sorted(not_started))} experiments/stage3/jobs/submit.sh")
        print()

    # Summary line for grep-friendly automation
    print(f"summary: complete={len(complete)} incomplete={len(incomplete)} "
          f"not_started={len(not_started)} total={total}")


if __name__ == "__main__":
    main()

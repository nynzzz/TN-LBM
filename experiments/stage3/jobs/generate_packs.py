"""
Generate `packs.txt` for the packed submission scheme.

Reads master_jobs.csv, filters out:
  - completed jobs (have result JSON)
  - known-failing jobs (Taylor overflow at taylor_order=2)

Groups remaining by N, then splits into "packs" of tasks that share a slot:
  - N <= 128:  16 per slot  (~1.5 GB each in 24 GB slot)
  - N == 256:  8 per slot   (~3 GB each)
  - N == 512:  2 per slot   (~12 GB each)

Writes one line per pack to packs.txt (comma-separated task IDs).

Usage:
    python -m experiments.stage3.jobs.generate_packs
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.stage3.common import JOBS_DIR, RESULTS_DIR, jobs_from_csv

# Known-failing jobs (Taylor 1/rho overflow at taylor_order=2).
# These should NOT be packed — they'd just crash again. Retry later with taylor_order=3.
KNOWN_FAILURES = {80, 221, 233, 235, 237, 243, 245, 247, 249, 251, 257, 259, 319}


def pack_size_for(n: int) -> int:
    if n <= 128:
        return 16
    if n == 256:
        return 8
    if n == 512:
        return 2
    raise ValueError(f"unknown N={n}, no pack size rule")


def main():
    csv = JOBS_DIR / "master_jobs.csv"
    jobs = jobs_from_csv(csv)

    # Find completed (result JSON exists)
    done_ids = set()
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.iterdir():
            if f.name.startswith("job_") and f.suffix == ".json":
                try:
                    done_ids.add(int(f.stem.split("_")[1]))
                except (ValueError, IndexError):
                    pass

    # Filter remaining
    remaining = [j for j in jobs
                 if j.job_id not in done_ids and j.job_id not in KNOWN_FAILURES]
    print(f"Total master jobs: {len(jobs)}")
    print(f"  completed:        {len(done_ids)}")
    print(f"  known failures:   {len(KNOWN_FAILURES & set(j.job_id for j in jobs))}")
    print(f"  remaining to pack: {len(remaining)}")
    print()

    # Group by N
    by_n: dict[int, list] = {}
    for j in remaining:
        by_n.setdefault(j.n, []).append(j)

    # Within each N, sort by job_id for deterministic packing
    for n in by_n:
        by_n[n].sort(key=lambda j: j.job_id)

    # Build packs
    packs: list[list[int]] = []
    for n in sorted(by_n):
        ps = pack_size_for(n)
        tasks = [j.job_id for j in by_n[n]]
        print(f"  N={n:4d}: {len(tasks)} tasks → pack size {ps}", end="")
        # Chunk into pack-sized groups
        n_packs = 0
        for i in range(0, len(tasks), ps):
            chunk = tasks[i:i + ps]
            packs.append(chunk)
            n_packs += 1
        print(f"  → {n_packs} packs")

    print()
    print(f"TOTAL PACKS: {len(packs)} (slots needed)")
    print(f"Estimated SBU cost vs unpacked: {len(packs)}/16 = {len(packs)*100//(len(remaining)//16+1):>3d}% of slot count, "
          f"{len(packs)*16:>5d} cores vs {len(remaining)*16:>5d} cores billed.")

    # Write packs.txt
    out_path = JOBS_DIR / "packs.txt"
    with out_path.open("w") as f:
        for pack in packs:
            f.write(",".join(str(t) for t in pack) + "\n")
    print(f"\nWritten: {out_path}")
    print(f"Submit with:")
    print(f"  sbatch --array=0-{len(packs)-1} experiments/stage3/jobs/submit_packed.sh")
    print(f"Or with throttle (e.g. max 20 packs concurrent):")
    print(f"  sbatch --array=0-{len(packs)-1}%20 experiments/stage3/jobs/submit_packed.sh")


if __name__ == "__main__":
    main()

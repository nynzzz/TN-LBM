"""
Generate the master jobs CSV by enumerating all 5 experiments and deduplicating.

Run once:
    python -m experiments.stage3.build_master_jobs

Writes: experiments/stage3/master_jobs.csv

Analysis scripts (analyze_exp{1..5}.py) define filters over this CSV to select
the relevant jobs for each experiment. The deduplication ensures that overlapping
parameter combinations across experiments only run once.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.stage3.common import Job, JOBS_DIR, jobs_to_csv


# ============================================================================
# Shared constants
# ============================================================================

# Standard chi sweep (Exp 1, 3, parts of 4)
CHI_SWEEP = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64]

# TG-specific Exp 4 chi sweep (TG saturates early)
CHI_TG_EXP4 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

# Cavity Exp 4 chi sweep (extended to break-even at ~80 for N=256 snake)
CHI_CAV_EXP4 = [6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80]


# ============================================================================
# Exp 1: Chi sweep at fixed (N, Re) per test case
# ============================================================================

def gen_exp1() -> Iterable[Job]:
    """Chi sweep — maps accuracy vs compression at multiple (N, Re).

    No cylinder here: cylinder appears only in Exp 5 (its unique value is
    drag/lift on the immersed body, not the chi/error curve).
    """
    cases = []
    # TG: 6 (N, Re) pairs
    for n in (64, 128, 256):
        for re in (100, 500):
            cases.append(("tg", n, re, 0.1))
    # Cavity: 6 (N, Re) pairs
    for n in (64, 128, 256):
        for re in (100, 500):
            cases.append(("cavity", n, re, 0.1))

    for (tc, n, re, u) in cases:
        for chi in CHI_SWEEP:
            yield Job(
                job_id=-1,                      # filled by build()
                test_case=tc, n=n, re=re, u=u,
                cutoff=1e-10, chi=chi, taylor_order=2,
                mapping="snake",
            )


# ============================================================================
# Exp 2: Grid scaling — natural chi growth (max_bond=512)
# ============================================================================

# (N, Re) pairs for Exp 2 — extended for cleaner log/power-law fits.
# Constraint: tau = 0.3·N/Re + 0.5 must stay above ~0.51 for stability.
# Original plan: 23 pairs. Additions (+11): low-end N=16, Re=50 column,
# Re=800, Re=2000 — all with tau ≥ 0.51.
EXP2_GRID = [
    # N=16 — new (cheap, low-end of log(N) axis)
    (16, 50),   (16, 100),  (16, 200),
    # N=32
    (32, 50),   (32, 100),  (32, 200),  (32, 400),  (32, 1000),
    # N=64
    (64, 50),   (64, 100),  (64, 200),  (64, 400),  (64, 800),
    (64, 1000), (64, 3200),
    # N=128
    (128, 50),  (128, 100), (128, 200), (128, 400), (128, 800),
    (128, 1000),(128, 2000),(128, 3200),
    # N=256
    (256, 100), (256, 200), (256, 400), (256, 800),
    (256, 1000),(256, 2000),(256, 3200),
    # N=512
    (512, 200), (512, 400), (512, 1000), (512, 3200),
]

EXP2_CUTOFFS = [1e-10, 1e-7]


def gen_exp2() -> Iterable[Job]:
    """Natural chi growth — let chi grow to 512 cap and observe."""
    for tc in ("tg", "cavity"):
        for (n, re) in EXP2_GRID:
            for cutoff in EXP2_CUTOFFS:
                yield Job(
                    job_id=-1,
                    test_case=tc, n=n, re=re, u=0.1,
                    cutoff=cutoff, chi=512, taylor_order=2,
                    mapping="snake",
                )


# ============================================================================
# Exp 3: Error decomposition (Taylor, truncation, BC)
# ============================================================================

# 3a: Ma sweep at N=64 Re=100 — vary u and Taylor order. Full rank, no cutoff.
EXP3A_U_LIST = [0.01, 0.02, 0.05, 0.1, 0.15]
EXP3A_TAYLOR_ORDERS = [1, 2, 3]
EXP3A_N_GRID_INDEP = [32, 64, 128, 256]   # N sweep at u=0.1, 3 orders


def gen_exp3a() -> Iterable[Job]:
    """Taylor error scaling: Ma sweep + grid independence, all 3 orders, full rank."""
    # Ma sweep at fixed N=64
    for u in EXP3A_U_LIST:
        for order in EXP3A_TAYLOR_ORDERS:
            yield Job(
                job_id=-1,
                test_case="tg", n=64, re=100, u=u,
                cutoff=0.0, chi=None, taylor_order=order,    # full rank
                mapping="snake",
            )
    # Grid independence at fixed u=0.1
    for n in EXP3A_N_GRID_INDEP:
        for order in EXP3A_TAYLOR_ORDERS:
            yield Job(
                job_id=-1,
                test_case="tg", n=n, re=100, u=0.1,
                cutoff=0.0, chi=None, taylor_order=order,
                mapping="snake",
            )


def gen_exp3b() -> Iterable[Job]:
    """Truncation: TG at N=64 Re=100, chi sweep × 3 cutoffs. (cutoff=1e-10 overlaps Exp 1.)"""
    for chi in CHI_SWEEP:
        for cutoff in (1e-10, 1e-7, 0.0):
            yield Job(
                job_id=-1,
                test_case="tg", n=64, re=100, u=0.1,
                cutoff=cutoff, chi=chi, taylor_order=2,
                mapping="snake",
            )


def gen_exp3c() -> Iterable[Job]:
    """BC: cavity at N=64 Re=100, chi sweep × 2 cutoffs. (cutoff=1e-10 overlaps Exp 1.)"""
    for chi in CHI_SWEEP:
        for cutoff in (1e-10, 1e-7):
            yield Job(
                job_id=-1,
                test_case="cavity", n=64, re=100, u=0.1,
                cutoff=cutoff, chi=chi, taylor_order=2,
                mapping="snake",
            )


def gen_exp3() -> Iterable[Job]:
    yield from gen_exp3a()
    yield from gen_exp3b()
    yield from gen_exp3c()


# ============================================================================
# Exp 4: Memory comparison at N=256
# ============================================================================

EXP4_RES = [100, 500, 1000]


def gen_exp4() -> Iterable[Job]:
    """Memory comparison: fine MPS vs coarse vanilla at fixed memory budget."""
    for re in EXP4_RES:
        # TG with its own chi list
        for chi in CHI_TG_EXP4:
            yield Job(
                job_id=-1,
                test_case="tg", n=256, re=re, u=0.1,
                cutoff=1e-10, chi=chi, taylor_order=2,
                mapping="snake",
            )
        # Cavity with its own chi list
        for chi in CHI_CAV_EXP4:
            yield Job(
                job_id=-1,
                test_case="cavity", n=256, re=re, u=0.1,
                cutoff=1e-10, chi=chi, taylor_order=2,
                mapping="snake",
            )


# ============================================================================
# Exp 5: Observables — reuses Exp 1/2 final states, plus cylinder demo runs
# ============================================================================

def gen_exp5() -> Iterable[Job]:
    """Observable extraction reuses MPS final states from Exp 1/2, except for
    cylinder (not in any other experiment) — produce its 4 demo runs here.

    Per plan: cylinder at N=128, N=256, each at chi=32 and chi=64.
    """
    for n in (128, 256):
        for chi in (32, 64):
            yield Job(
                job_id=-1,
                test_case="cylinder", n=n, re=100, u=0.1,
                cutoff=1e-10, chi=chi, taylor_order=2,
                mapping="snake",
            )


# ============================================================================
# Build master CSV
# ============================================================================

def _job_key(j: Job):
    """Dedup key. job_id excluded since it's assigned post-dedup."""
    return (j.test_case, j.n, j.re, j.u, j.cutoff, j.chi,
            j.taylor_order, j.mapping, j.nt, j.snapshot_every)


def build() -> list[Job]:
    """Run all generators, dedup, assign sequential job_ids."""
    all_jobs: list[Job] = []
    counts = {}
    for name, gen in [
        ("exp1", gen_exp1),
        ("exp2", gen_exp2),
        ("exp3", gen_exp3),
        ("exp4", gen_exp4),
        ("exp5", gen_exp5),
    ]:
        batch = list(gen())
        counts[name] = len(batch)
        all_jobs.extend(batch)

    print(f"Raw counts per experiment (before dedup):")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  TOTAL (raw): {len(all_jobs)}")

    # Dedup
    seen = {}
    unique: list[Job] = []
    for j in all_jobs:
        k = _job_key(j)
        if k in seen:
            continue
        seen[k] = j
        unique.append(j)

    print(f"\nAfter dedup: {len(unique)} unique jobs "
          f"({len(all_jobs) - len(unique)} duplicates removed)")

    # Assign sequential ids
    final = [replace(j, job_id=i) for i, j in enumerate(unique)]

    # Summary by test_case + cutoff for sanity
    print("\nBreakdown by test_case / cutoff:")
    breakdown = {}
    for j in final:
        k = (j.test_case, f"{j.cutoff:g}")
        breakdown[k] = breakdown.get(k, 0) + 1
    for (tc, cf), n in sorted(breakdown.items()):
        print(f"  {tc:>9} cutoff={cf:>7}: {n}")

    return final


def main():
    jobs = build()
    out_path = JOBS_DIR / "master_jobs.csv"
    jobs_to_csv(jobs, out_path)
    print(f"\nWritten: {out_path} ({len(jobs)} jobs)")


if __name__ == "__main__":
    main()

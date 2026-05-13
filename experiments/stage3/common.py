"""
Stage 3 shared infrastructure.

Provides:
- Job spec (dataclass) + master CSV I/O
- Per-test-case setup (TG / cavity / cylinder)
- Vanilla baseline runner with disk cache
- MPS-native runner with snapshot collection
- MPS state save/load + checkpoint resume
- JSON result writer

Design: one Job → one JSON output file. Experiments are views over the
result set, defined by filters in analysis/analyze_exp{N}.py scripts.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Literal

import numpy as np

# Project root on sys.path
import sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ============================================================================
# Paths
# ============================================================================

STAGE3_DIR = Path(__file__).resolve().parent
BASELINES_DIR = STAGE3_DIR / "baselines"
CHECKPOINTS_DIR = STAGE3_DIR / "checkpoints"
RESULTS_DIR = STAGE3_DIR / "results"
JOBS_DIR = STAGE3_DIR / "jobs"

for d in (BASELINES_DIR, CHECKPOINTS_DIR, RESULTS_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Job spec
# ============================================================================

TestCase = Literal["tg", "cavity", "cylinder"]


@dataclass(frozen=True)
class Job:
    """
    One MPS-native simulation job. Uniquely identified by all fields.

    Physics derivation (tau, nt, geometry) is NOT stored here — it lives in
    test_cases.py and is computed from (test_case, n, re, u) on demand.
    """
    job_id: int
    test_case: TestCase
    n: int                  # grid side
    re: float               # Reynolds number
    u: float                # characteristic velocity (u_lid / u_inlet / u_max)
    cutoff: float           # SVD cutoff (1e-10 standard, 0 = none, 1e-7 aggressive)
    chi: Optional[int] = None       # max bond dim cap; None = unbounded (Exp 2)
    taylor_order: int = 2           # 1, 2, or 3
    mapping: str = "snake"
    nt: Optional[int] = None        # if None, derived from test case
    snapshot_every: Optional[int] = None  # if None, nt // 200

    @property
    def baseline_key(self) -> str:
        """Identifier for the vanilla baseline (test_case, N, Re, u)."""
        return f"{self.test_case}_n{self.n}_re{int(self.re)}_u{self.u:.3g}"

    @property
    def result_path(self) -> Path:
        return RESULTS_DIR / f"job_{self.job_id:05d}.json"

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINTS_DIR / f"job_{self.job_id:05d}.pkl"


CSV_FIELDS = ["job_id", "test_case", "n", "re", "u", "cutoff",
              "chi", "taylor_order", "mapping", "nt", "snapshot_every"]


def jobs_to_csv(jobs: list[Job], path: Path) -> None:
    """Write list of jobs to CSV. Optional ints (chi, nt, snapshot_every) → empty string when None."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for j in jobs:
            row = {k: asdict(j)[k] for k in CSV_FIELDS}
            # None → empty string for clean CSV
            row = {k: ("" if v is None else v) for k, v in row.items()}
            w.writerow(row)


def jobs_from_csv(path: Path) -> list[Job]:
    """Read list of jobs from CSV. Empty strings in optional int fields → None."""
    import csv

    def _opt_int(s: Optional[str]) -> Optional[int]:
        return int(s) if s not in (None, "") else None

    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append(Job(
                job_id=int(row["job_id"]),
                test_case=row["test_case"],
                n=int(row["n"]),
                re=float(row["re"]),
                u=float(row["u"]),
                cutoff=float(row["cutoff"]),
                chi=_opt_int(row.get("chi")),
                taylor_order=int(row.get("taylor_order") or 2),
                mapping=row.get("mapping") or "snake",
                nt=_opt_int(row.get("nt")),
                snapshot_every=_opt_int(row.get("snapshot_every")),
            ))
    return out


def get_job(job_id: int, csv_path: Path) -> Job:
    """Fetch a single job by id."""
    for j in jobs_from_csv(csv_path):
        if j.job_id == job_id:
            return j
    raise KeyError(f"job_id={job_id} not found in {csv_path}")


# ============================================================================
# Baseline cache (vanilla final velocity field per (test_case, N, Re, u))
# ============================================================================

def baseline_path(job: Job) -> Path:
    return BASELINES_DIR / f"{job.baseline_key}.npz"


def save_baseline(job: Job, u_final: np.ndarray,
                  t_converged: Optional[int] = None,
                  wall_time_s: float = 0.0,
                  extra: Optional[dict] = None) -> None:
    """Save vanilla baseline + metadata to .npz."""
    meta = {
        "test_case": job.test_case, "n": job.n, "re": job.re, "u": job.u,
        "t_converged": int(t_converged) if t_converged is not None else -1,
        "wall_time_s": float(wall_time_s),
    }
    if extra:
        meta.update({k: (str(v) if not isinstance(v, (int, float, bool)) else v)
                     for k, v in extra.items()})
    np.savez(baseline_path(job), u_final=u_final, meta=json.dumps(meta))


def load_baseline(job: Job) -> tuple[np.ndarray, dict]:
    """Load (u_final, meta_dict) for the baseline matching this job."""
    p = baseline_path(job)
    if not p.exists():
        raise FileNotFoundError(f"no baseline at {p} — run baseline first")
    data = np.load(p, allow_pickle=False)
    return data["u_final"], json.loads(str(data["meta"]))


def baseline_exists(job: Job) -> bool:
    return baseline_path(job).exists()


# ============================================================================
# MPS state + checkpoint persistence
# ============================================================================

def save_checkpoint(job: Job, t: int, mps_state: list, snapshots: list,
                    rng_state: Optional[bytes] = None) -> None:
    """Pickle MPS state + accumulated snapshots at step t. Enables resume."""
    payload = {
        "job_id": job.job_id,
        "t": t,
        "mps_state": mps_state,   # quimb MPS objects pickle cleanly
        "snapshots": snapshots,
        "rng_state": rng_state,
    }
    tmp = job.checkpoint_path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(job.checkpoint_path)


def load_checkpoint(job: Job) -> Optional[dict]:
    if not job.checkpoint_path.exists():
        return None
    with open(job.checkpoint_path, "rb") as f:
        return pickle.load(f)


def clear_checkpoint(job: Job) -> None:
    if job.checkpoint_path.exists():
        job.checkpoint_path.unlink()


# ============================================================================
# Snapshot + result JSON
# ============================================================================

@dataclass
class Snapshot:
    t: int
    mean_chi: float
    max_chi: int
    per_pop_chi: list[int]
    mass: float
    mass_drift_rel: float
    du: float
    l2_err_vs_baseline: Optional[float] = None
    drag: Optional[float] = None
    lift: Optional[float] = None
    wall_time_cumulative_s: float = 0.0
    mps_memory_floats: int = 0


def write_result_json(job: Job, snapshots: list[Snapshot],
                       vanilla_info: dict, final: dict,
                       mps_final_state_path: Optional[Path] = None) -> None:
    """Write the final per-job JSON. One file per job."""
    result = {
        "experiment": "stage3",
        "job_id": job.job_id,
        "params": asdict(job),
        "vanilla": vanilla_info,
        "mps": {
            "timeseries": [asdict(s) for s in snapshots],
            "final": final,
            "final_state_path": str(mps_final_state_path) if mps_final_state_path else None,
        },
    }
    with open(job.result_path, "w") as f:
        json.dump(result, f, indent=2)


# ============================================================================
# Memory accounting — Gross Eq 46 (NVPS)
# ============================================================================

def nvps_per_population(n: int, chi: int, mapping: str = "snake") -> int:
    """
    Exact MPS storage per scalar field (Gross Eq 46).

    Snake (D=2):     K = 2·log2(N), p = 2
    Interleaved:     K = log2(N),   p = 4

    NVPS = Σ_{k=1..K} min(p^(k-1), p^(K-k+1), χ) · p · min(p^k, p^(K-k), χ)
    """
    if mapping == "snake":
        K = 2 * int(np.log2(n))
        p = 2
    elif mapping == "interleaved":
        K = int(np.log2(n))
        p = 4
    else:
        raise ValueError(f"unknown mapping: {mapping}")
    total = 0
    for k in range(1, K + 1):
        left = min(p ** (k - 1), p ** (K - k + 1), chi)
        right = min(p ** k, p ** (K - k), chi)
        total += left * p * right
    return total


def nvps_mps_field(n: int, chi: int, mapping: str = "snake") -> int:
    """Total NVPS for the 9-population D2Q9 LBM state."""
    return 9 * nvps_per_population(n, chi, mapping)


def memory_equivalent_n_coarse(n_fine: int, chi: int, mapping: str = "snake") -> int:
    """Return N_coarse such that vanilla(N_coarse) has same memory as MPS(n_fine, chi)."""
    nvps = nvps_mps_field(n_fine, chi, mapping)
    return int(np.floor(np.sqrt(nvps / 9)))


# ============================================================================
# L2 error helper
# ============================================================================

def l2_velocity_error(u_test: np.ndarray, u_ref: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> float:
    """Relative L2 norm of velocity difference. mask: where to compute (e.g. fluid)."""
    if mask is not None:
        d = u_test[:, mask] - u_ref[:, mask]
        denom = np.sum(u_ref[:, mask] ** 2)
    else:
        d = u_test - u_ref
        denom = np.sum(u_ref ** 2)
    return float(np.sqrt(np.sum(d ** 2) / (denom + 1e-30)))

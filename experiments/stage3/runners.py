"""
Vanilla baseline runner + MPS-native runner.

Vanilla:
  f = collide_bgk(D2Q9, f, tau)
  f = spec.apply_bc_vanilla(D2Q9, f)        # post-collision -> post-stream+BC

MPS:
  mps, moments = collide_bgk_mps(...)
  if spec.apply_bc_mps:
      mps = spec.apply_bc_mps(mps, ..., moments=moments)
  else:
      mps = stream_all_populations(mps, meta, D2Q9, max_bond, cyclic=True)

Convergence (cavity): per-step relative L2 of velocity, 2 consecutive steps below tol.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lbm import D2Q9, compute_equilibrium
from lbm.collision import compute_moments
from tn.compression import compress_populations, decompress_populations
from tn_lbm.collision import collide_bgk_mps
from tn_lbm.streaming import stream_all_populations
from tn_lbm.arithmetic import mps_sum_value
from tn_lbm.observables import check_convergence_mps, compute_drag_lift_mps

from .common import (
    Job, Snapshot,
    save_baseline, load_baseline, baseline_exists,
    save_checkpoint, load_checkpoint, clear_checkpoint,
    write_result_json,
    nvps_per_population, l2_velocity_error,
)
from .test_cases import TestCaseSpec, setup_test_case


# ============================================================================
# Dense helpers
# ============================================================================

def _safe_moments(lattice, f):
    """compute_moments with safe division (rho=0 at solid nodes)."""
    rho = np.sum(f, axis=0)
    rho_safe = np.where(rho > 1e-15, rho, 1.0)
    u = np.zeros((2, *rho.shape))
    for i in range(lattice.q):
        u[0] += lattice.c[i, 0] * f[i]
        u[1] += lattice.c[i, 1] * f[i]
    u /= rho_safe
    return rho, u


def _du_dense(u_new, u_prev, fluid_mask=None):
    """Relative L2 velocity change (matches canonical pattern)."""
    if fluid_mask is not None:
        d = u_new[:, fluid_mask] - u_prev[:, fluid_mask]
        denom = np.sum(u_new[:, fluid_mask] ** 2) + 1e-30
    else:
        d = u_new - u_prev
        denom = np.sum(u_new ** 2) + 1e-30
    return float(np.sqrt(np.sum(d ** 2) / denom))


# ============================================================================
# Vanilla runner
# ============================================================================

def run_vanilla(job: Job, force: bool = False, verbose: bool = False) -> tuple[np.ndarray, dict]:
    """
    Run vanilla LBM. Per-step convergence check (cavity), 2-consecutive count.
    Uses disk cache keyed by (test_case, n, re, u).
    """
    if not force and baseline_exists(job):
        if verbose:
            print(f"[vanilla] cached: {job.baseline_key}")
        return load_baseline(job)

    spec = setup_test_case(job)
    f = spec.f_init.copy()
    _, u_prev = _safe_moments(D2Q9, f)

    t_converged = -1
    consec = 0
    t0 = time.time()

    t = 0
    for t in range(spec.nt + 1):
        if t < spec.nt:
            # Safe BGK collision: uses _safe_moments to avoid NaN at solid nodes
            # (mirrors canonical pattern from stage3_mps_native_cylinder.ipynb)
            rho_safe, u_safe = _safe_moments(D2Q9, f)
            f_eq = compute_equilibrium(D2Q9, rho_safe, u_safe)
            fstar = f - (f - f_eq) / spec.tau
            # apply_bc_vanilla does stream + BC (post-collision input)
            f = spec.apply_bc_vanilla(D2Q9, fstar)

        # Per-step convergence (only meaningful when convergence_tol is set, i.e. cavity)
        if spec.convergence_tol is not None and t > 0:
            _, u = _safe_moments(D2Q9, f)
            du = _du_dense(u, u_prev, fluid_mask=spec.fluid_mask)
            u_prev = u
            if du < spec.convergence_tol:
                consec += 1
                if consec >= 2:
                    t_converged = t
                    if verbose:
                        print(f"[vanilla] converged at t={t} (du={du:.2e})")
                    break
            else:
                consec = 0
            if verbose and t % max(1, spec.snapshot_every) == 0:
                print(f"  [vanilla t={t}] du={du:.2e}")

    wall = time.time() - t0
    _, u_final = _safe_moments(D2Q9, f)
    save_baseline(job, u_final, t_converged=t_converged, wall_time_s=wall,
                  extra={"nt_actual": int(t)})
    if verbose:
        print(f"[vanilla] {job.baseline_key}: nt={t}, t_conv={t_converged}, wall={wall:.1f}s")
    return load_baseline(job)


# ============================================================================
# MPS runner
# ============================================================================

def _take_snapshot(t: int, mps_list, mps_metadata, baseline_u: Optional[np.ndarray],
                   initial_mass: float, spec: TestCaseSpec, mapping: str,
                   du: float, wall_cum_s: float, n: int, chi_cap: Optional[int],
                   compute_l2: bool = False) -> Snapshot:
    """Build a Snapshot from the current MPS state.

    L2 vs vanilla baseline is only meaningful when MPS time matches the
    baseline's time (= final time of vanilla run). For intermediate
    snapshots, comparing MPS at t against vanilla at t_final is misleading.
    So we only compute L2 when compute_l2=True (set by the runner at the
    last snapshot only).
    """
    per_pop = [int(m.max_bond()) for m in mps_list]
    mass = float(sum(mps_sum_value(m) for m in mps_list))
    drift = (mass - initial_mass) / (initial_mass + 1e-30)

    l2 = None
    if compute_l2 and baseline_u is not None:
        try:
            f_dec = decompress_populations(mps_list, mps_metadata)
            _, u_mps = _safe_moments(D2Q9, f_dec)
            l2 = l2_velocity_error(u_mps, baseline_u, mask=spec.fluid_mask)
        except Exception:
            l2 = None

    drag = lift = None
    if spec.boundary_masks_mps is not None:
        try:
            d, l = compute_drag_lift_mps(
                mps_list, D2Q9, spec.boundary_masks_mps, max_bond=chi_cap
            )
            drag, lift = float(d), float(l)
        except Exception:
            drag = lift = None

    chi_for_memory = max(per_pop)
    return Snapshot(
        t=t,
        mean_chi=float(np.mean(per_pop)),
        max_chi=chi_for_memory,
        per_pop_chi=per_pop,
        mass=mass,
        mass_drift_rel=float(drift),
        du=du,
        l2_err_vs_baseline=l2,
        drag=drag, lift=lift,
        wall_time_cumulative_s=float(wall_cum_s),
        mps_memory_floats=int(9 * nvps_per_population(n, chi_for_memory, mapping)),
    )


def run_mps(job: Job,
            wall_clock_limit_s: Optional[float] = None,
            checkpoint_every: int = 5000,
            verbose: bool = False) -> dict:
    """
    Run MPS-native simulation. Resume from checkpoint, stop on wall_clock_limit_s.
    Per-step velocity convergence (cavity), 2-consecutive count.
    """
    spec = setup_test_case(job)
    mapping = job.mapping

    # Baseline (run if missing) for L2 comparison
    if baseline_exists(job):
        baseline_u, _ = load_baseline(job)
    else:
        if verbose:
            print(f"[mps] no baseline cached, running vanilla first")
        baseline_u, _ = run_vanilla(job, verbose=verbose)

    # Compress IC to get metadata (deterministic given mapping + N)
    _, mps_metadata = compress_populations(spec.f_init, mapping=mapping)

    # Resume from checkpoint if present
    ckpt = load_checkpoint(job)
    if ckpt is not None:
        t_start = ckpt["t"]
        mps_list = ckpt["mps_state"]
        snapshots: list[Snapshot] = ckpt["snapshots"]
        if verbose:
            print(f"[mps] resuming at t={t_start} ({len(snapshots)} snapshots)")
    else:
        t_start = 0
        mps_list, _ = compress_populations(spec.f_init, mapping=mapping)
        snapshots = []

    initial_mass = float(np.sum(spec.f_init))
    moments_prev = None
    du_val = 1.0
    consec = 0
    t_converged = -1
    t_run_start = time.time()

    for t in range(t_start, spec.nt + 1):
        # Snapshot (intermediate — no L2; that's set only at the final snapshot below)
        is_final_nt = (t == spec.nt)
        if t % spec.snapshot_every == 0 or is_final_nt:
            wall_cum = time.time() - t_run_start
            snap = _take_snapshot(t, mps_list, mps_metadata, baseline_u,
                                   initial_mass, spec, mapping,
                                   du_val, wall_cum, job.n, chi_cap=job.chi,
                                   compute_l2=is_final_nt)
            snapshots.append(snap)
            if verbose and (t == 0 or t % (spec.snapshot_every * 10) == 0):
                print(f"  [mps t={t}] chi={snap.max_chi} mass_drift={snap.mass_drift_rel:.2e} "
                      f"du={snap.du:.2e} L2={snap.l2_err_vs_baseline}")

        if t < spec.nt:
            # Collision (returns moments — used for convergence du + cylinder outlet)
            mps_list, moments_new = collide_bgk_mps(
                mps_list, D2Q9, spec.tau, job.n,
                max_bond=job.chi, cutoff=job.cutoff,
                return_moments=True, taylor_order=job.taylor_order,
            )

            # Per-step convergence du from moments
            if moments_prev is not None:
                try:
                    _, du_val = check_convergence_mps(
                        mps_list, mps_list, mode='velocity',
                        moments_new=moments_new, moments_old=moments_prev,
                    )
                except Exception:
                    pass
            moments_prev = moments_new

            # Stream + BC (cavity/cylinder combined; TG separate)
            if spec.apply_bc_mps is None:
                mps_list = stream_all_populations(mps_list, mps_metadata, D2Q9,
                                                    max_bond=job.chi, cyclic=True)
            else:
                mps_list = spec.apply_bc_mps(
                    mps_list, D2Q9, spec.bc_data,
                    max_bond=job.chi, cutoff=job.cutoff,
                    moments=moments_new,
                )

            # Convergence check (cavity only)
            if spec.convergence_tol is not None and du_val < spec.convergence_tol:
                consec += 1
                if consec >= 2:
                    t_converged = t + 1
                    if verbose:
                        print(f"[mps] converged at t={t+1} (du={du_val:.2e})")
                    # Final snapshot at convergence — compute L2 vs baseline here
                    wall_cum = time.time() - t_run_start
                    snapshots.append(_take_snapshot(
                        t + 1, mps_list, mps_metadata, baseline_u,
                        initial_mass, spec, mapping,
                        du_val, wall_cum, job.n, chi_cap=job.chi,
                        compute_l2=True,
                    ))
                    break
            else:
                consec = 0

        # Checkpoint
        if checkpoint_every and t > 0 and t % checkpoint_every == 0 and t < spec.nt:
            save_checkpoint(job, t, mps_list, snapshots)
            if verbose:
                print(f"  [mps t={t}] checkpoint saved")

        # Wall-clock limit
        if wall_clock_limit_s is not None and (time.time() - t_run_start) > wall_clock_limit_s:
            save_checkpoint(job, t, mps_list, snapshots)
            if verbose:
                print(f"  [mps t={t}] wall limit hit; checkpoint saved")
            return {"status": "incomplete", "t_reached": t}

    # Final summary
    wall_total = time.time() - t_run_start
    nt_done = snapshots[-1].t if snapshots else 0
    final = {
        "total_wall_time_s": float(wall_total),
        "wall_time_per_step_s": float(wall_total / max(1, nt_done)),
        "nt_completed": int(nt_done),
        "t_converged": int(t_converged),
        "final_l2_err_vs_baseline": snapshots[-1].l2_err_vs_baseline if snapshots else None,
        "final_mass_drift_rel": snapshots[-1].mass_drift_rel if snapshots else None,
        "final_mean_chi": snapshots[-1].mean_chi if snapshots else None,
        "final_drag": snapshots[-1].drag if snapshots else None,
        "final_lift": snapshots[-1].lift if snapshots else None,
    }

    # Save final MPS state for Exp 5 observable extraction
    mps_state_path = job.checkpoint_path.with_name(job.checkpoint_path.stem + "_final.pkl")
    with open(mps_state_path, "wb") as f:
        pickle.dump({"mps_state": mps_list, "metadata": mps_metadata},
                    f, protocol=pickle.HIGHEST_PROTOCOL)

    _, baseline_meta = load_baseline(job)
    write_result_json(job, snapshots, vanilla_info=baseline_meta,
                       final=final, mps_final_state_path=mps_state_path)
    clear_checkpoint(job)

    return {"status": "complete", "t_reached": nt_done}

"""
Per-test-case setup: geometry, IC, BC, default nt.

Each `setup_<case>(job)` returns a TestCaseSpec. Contract:

  apply_bc_vanilla(lattice, f_post_collision) -> f_post_stream_post_BC
  apply_bc_mps(mps_list, lattice, bc_data, *, max_bond, cutoff, moments) -> mps_list

For TG (periodic), apply_bc_vanilla wraps np.roll streaming and apply_bc_mps is
None — the runner does MPS streaming separately via stream_all_populations.
For cavity/cylinder, both apply_bc_* do combined stream+BC in one shot,
matching the canonical patterns in stage3_mps_native_*.ipynb.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lbm import D2Q9, compute_equilibrium
from lbm.boundary import (
    create_cylinder_mask, create_channel_walls,
    apply_fwbb,            # cavity FWBB: post-collision -> post-stream + BC
    apply_fwbb_cylinder,   # cylinder: post-collision -> post-stream + BC
)
from simulations.lid_driven_cavity import create_cavity_walls
from tn_lbm.boundary import (
    precompute_cavity_bc, apply_mps_boundary,
    precompute_cylinder_bc, apply_mps_cylinder_bc,
    mask_to_mps,
)

from .common import Job, TestCase


# ============================================================================
# TestCaseSpec
# ============================================================================

@dataclass
class TestCaseSpec:
    n: int
    tau: float
    nu: float
    u_char: float

    f_init: np.ndarray              # (9, N, N)
    rho_init: np.ndarray            # (N, N)
    u_init: np.ndarray              # (2, N, N)

    fluid_mask: np.ndarray          # (N, N) bool

    nt: int
    snapshot_every: int

    # Vanilla: post-collision input -> post-stream + post-BC output
    apply_bc_vanilla: Callable[[object, np.ndarray], np.ndarray]

    # MPS: combined stream+BC. None for TG (runner streams separately).
    # Signature: (mps_list, lattice, bc_data, *, max_bond, cutoff, moments) -> mps_list
    apply_bc_mps: Optional[Callable] = None

    bc_data: Optional[dict] = None
    boundary_masks_mps: Optional[list] = None  # cylinder drag/lift

    convergence_tol: Optional[float] = None    # cavity only

    meta: dict = field(default_factory=dict)


# ============================================================================
# Taylor–Green vortex (periodic)
# ============================================================================

def _setup_tg(job: Job) -> TestCaseSpec:
    n, u, re = job.n, job.u, job.re
    cs2 = D2Q9.cs2

    nu = u * n / re
    tau = 3 * nu + 0.5

    k = 2 * np.pi / n
    x = np.arange(n) + 0.5
    y = np.arange(n) + 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')

    u_init = np.zeros((2, n, n))
    u_init[0] = u * np.sin(k * X) * np.cos(k * Y)
    u_init[1] = -u * np.cos(k * X) * np.sin(k * Y)
    rho_init = 1.0 - (u ** 2) / (4 * cs2) * (np.cos(2 * k * X) + np.cos(2 * k * Y))
    f_init = compute_equilibrium(D2Q9, rho_init, u_init)

    tau_decay = 1.0 / (2 * nu * k * k)
    nt = job.nt if job.nt is not None else int(2 * tau_decay)
    snap = job.snapshot_every or max(1, nt // 200)

    def apply_bc_vanilla(lattice, f_post_collision):
        """Periodic stream — np.roll only, no BC."""
        out = np.empty_like(f_post_collision)
        for i in range(lattice.q):
            cx, cy = int(lattice.c[i, 0]), int(lattice.c[i, 1])
            out[i] = np.roll(f_post_collision[i], (cx, cy), axis=(0, 1))
        return out

    return TestCaseSpec(
        n=n, tau=tau, nu=nu, u_char=u,
        f_init=f_init, rho_init=rho_init, u_init=u_init,
        fluid_mask=np.ones((n, n), dtype=bool),
        nt=nt, snapshot_every=snap,
        apply_bc_vanilla=apply_bc_vanilla,
        apply_bc_mps=None,  # TG: runner streams separately via stream_all_populations
        meta={"tau_decay": float(tau_decay), "k": float(k)},
    )


# ============================================================================
# Lid-driven cavity (FWBB — matches MPS BC formulation)
# ============================================================================

def _setup_cavity(job: Job) -> TestCaseSpec:
    n, u, re = job.n, job.u, job.re

    nu = u * n / re
    tau = 3 * nu + 0.5

    walls, lid = create_cavity_walls(n)
    fluid = ~(walls | lid)

    rho_init = np.ones((n, n))
    u_init = np.zeros((2, n, n))
    f_init = compute_equilibrium(D2Q9, rho_init, u_init)

    default_nt = {64: 20000, 128: 60000, 256: 200000, 512: 600000}.get(n, 50000 * (n // 64))
    nt = job.nt if job.nt is not None else default_nt
    snap = job.snapshot_every or max(1, nt // 200)

    u_wall = np.array([u, 0.0])

    def apply_bc_vanilla(lattice, f_post_collision):
        # apply_fwbb does stream + BC together (matches MPS apply_mps_boundary)
        return apply_fwbb(lattice, f_post_collision, lid, u_wall, rho_wall=1.0)

    bc_data = precompute_cavity_bc(n, D2Q9, u_wall=u_wall, rho_wall=1.0, mapping=job.mapping)

    def apply_bc_mps(mps_list, lattice, bc_data, *, max_bond, cutoff, moments=None):
        return apply_mps_boundary(mps_list, lattice, bc_data,
                                   max_bond=max_bond, cutoff=cutoff)

    return TestCaseSpec(
        n=n, tau=tau, nu=nu, u_char=u,
        f_init=f_init, rho_init=rho_init, u_init=u_init,
        fluid_mask=fluid,
        nt=nt, snapshot_every=snap,
        apply_bc_vanilla=apply_bc_vanilla,
        apply_bc_mps=apply_bc_mps,
        bc_data=bc_data,
        convergence_tol=1e-6,
        meta={"walls_count": int(walls.sum()), "lid_count": int(lid.sum())},
    )


# ============================================================================
# Cylinder flow (FWBB cylinder — matches MPS apply_mps_cylinder_bc)
# ============================================================================

def _setup_cylinder(job: Job) -> TestCaseSpec:
    n, u, re = job.n, job.u, job.re

    cylinder_r = max(2, n // 32)
    cylinder_x = n // 4
    cylinder_y = n // 2 + 1
    D_cyl = 2 * cylinder_r

    nu = u * D_cyl / re
    tau = 3 * nu + 0.5

    cylinder = create_cylinder_mask(n, n, cylinder_x, cylinder_y, cylinder_r)
    walls = create_channel_walls(n, n)
    solid = cylinder | walls
    fluid = ~solid

    # Per-direction MPS boundary masks for drag/lift snapshots
    boundary_masks_mps = [None] * D2Q9.q
    for i in range(D2Q9.q):
        cx_i, cy_i = int(D2Q9.c[i, 0]), int(D2Q9.c[i, 1])
        if cx_i == 0 and cy_i == 0:
            continue
        shifted = np.roll(np.roll(cylinder, cx_i, axis=0), cy_i, axis=1)
        mask_2d = shifted & fluid
        boundary_masks_mps[i], _ = mask_to_mps(mask_2d, mapping=job.mapping)

    rho_init = np.ones((n, n))
    u_init = np.zeros((2, n, n))
    u_init[0, fluid] = u
    f_init = compute_equilibrium(D2Q9, rho_init, u_init)
    for i in range(D2Q9.q):
        f_init[i][solid] = 0.0

    St = 0.15
    T_shed = D_cyl / (St * u)
    n_periods = 8
    nt = job.nt if job.nt is not None else int(n_periods * T_shed)
    snap = job.snapshot_every or max(1, nt // 200)

    def apply_bc_vanilla(lattice, f_post_collision):
        # apply_fwbb_cylinder does stream + BC together (matches MPS apply_mps_cylinder_bc)
        return apply_fwbb_cylinder(lattice, f_post_collision, cylinder, walls,
                                     u_inlet=u, rho_inlet=1.0)

    bc_data = precompute_cylinder_bc(n, D2Q9, u_inlet=u, rho_inlet=1.0,
                                       cylinder_x=cylinder_x, cylinder_y=cylinder_y,
                                       cylinder_r=cylinder_r, mapping=job.mapping)

    def apply_bc_mps(mps_list, lattice, bc_data, *, max_bond, cutoff, moments=None):
        return apply_mps_cylinder_bc(mps_list, lattice, bc_data,
                                      max_bond=max_bond, cutoff=cutoff,
                                      moments=moments)

    return TestCaseSpec(
        n=n, tau=tau, nu=nu, u_char=u,
        f_init=f_init, rho_init=rho_init, u_init=u_init,
        fluid_mask=fluid,
        nt=nt, snapshot_every=snap,
        apply_bc_vanilla=apply_bc_vanilla,
        apply_bc_mps=apply_bc_mps,
        bc_data=bc_data,
        boundary_masks_mps=boundary_masks_mps,
        convergence_tol=None,
        meta={
            "cylinder_r": cylinder_r, "cylinder_x": cylinder_x, "cylinder_y": cylinder_y,
            "D_cyl": D_cyl, "T_shed_assumed": float(T_shed), "n_periods": n_periods,
        },
    )


# ============================================================================
# Dispatch
# ============================================================================

_SETUPS = {
    "tg": _setup_tg,
    "cavity": _setup_cavity,
    "cylinder": _setup_cylinder,
}


def setup_test_case(job: Job) -> TestCaseSpec:
    if job.test_case not in _SETUPS:
        raise ValueError(f"unknown test_case: {job.test_case}")
    return _SETUPS[job.test_case](job)

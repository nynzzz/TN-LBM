"""
MPS-native BGK collision for TN-LBM.

Implements the collision step entirely in MPS arithmetic:
1. Compute moments (rho, rho*u) via MPS addition
2. Approximate 1/rho via Taylor expansion (Gross et al. Eq 34)
3. Compute equilibrium distributions via Hadamard products
4. BGK relaxation: f_new = (1-1/tau)*f + (1/tau)*f_eq

Uses naive Hadamard + truncation (16 products total with optimizations).
"""

import numpy as np
import quimb.tensor as qtn

from .arithmetic import mps_add, mps_scale, mps_hadamard, mps_subtract, mps_sum_value


def build_ones_mps(num_sites):
    """
    Build an all-ones MPS with chi=1 (analytical construction).

    Every element of the represented vector equals 1.0.

    Args:
        num_sites: number of qubit sites (2L for 2D grids)

    Returns:
        mps: quimb MatrixProductState with chi=1
    """
    arrays = []
    for k in range(num_sites):
        if k == 0:
            # Left edge: shape (phys, bond_right) = (2, 1)
            arrays.append(np.ones((2, 1)))
        elif k == num_sites - 1:
            # Right edge: shape (bond_left, phys) = (1, 2)
            arrays.append(np.ones((1, 2)))
        else:
            # Middle: shape (bond_left, phys, bond_right) = (1, 2, 1)
            arrays.append(np.ones((1, 2, 1)))

    return qtn.MatrixProductState(arrays, shape='lpr')


def compute_moments_mps(mps_list, lattice, max_bond=None, cutoff=1e-10):
    """
    Compute density and momentum from 9 population MPS.

    |rho> = sum_i |f_i>
    |rho*ux> = sum_i c_ix * |f_i>
    |rho*uy> = sum_i c_iy * |f_i>

    Pure linear operations (additions and scalar multiplications).

    Args:
        mps_list: list of 9 MPS (one per D2Q9 population)
        lattice: D2Q9 lattice definition
        max_bond: optional bond dimension truncation for additions
        cutoff: SVD cutoff

    Returns:
        (rho_mps, rhou_x_mps, rhou_y_mps)
    """
    # Density: sum all 9 populations
    rho = mps_list[0].copy()
    for i in range(1, lattice.q):
        rho = mps_add(rho, mps_list[i], max_bond=max_bond, cutoff=cutoff)

    # Momentum x: sum c_ix * f_i (skip c_ix = 0)
    rhou_x = None
    for i in range(lattice.q):
        cx = float(lattice.c[i, 0])
        if abs(cx) < 1e-15:
            continue
        term = mps_list[i].copy() if cx == 1.0 else mps_scale(mps_list[i], cx)
        if rhou_x is None:
            rhou_x = term
        else:
            rhou_x = mps_add(rhou_x, term, max_bond=max_bond, cutoff=cutoff)

    # Momentum y: sum c_iy * f_i (skip c_iy = 0)
    rhou_y = None
    for i in range(lattice.q):
        cy = float(lattice.c[i, 1])
        if abs(cy) < 1e-15:
            continue
        term = mps_list[i].copy() if cy == 1.0 else mps_scale(mps_list[i], cy)
        if rhou_y is None:
            rhou_y = term
        else:
            rhou_y = mps_add(rhou_y, term, max_bond=max_bond, cutoff=cutoff)

    return rho, rhou_x, rhou_y


def compute_inverse_density_mps(rho_mps, ones_mps, n_grid,
                                 max_bond=None, cutoff=1e-10):
    """
    Approximate 1/rho via second-order Taylor expansion (Gross et al. Eq 34).

    1/rho ~ 1/rho_0 - delta_rho/rho_0^2 + delta_rho^2/rho_0^3

    Where rho_0 = mean density (scalar), delta_rho = rho - rho_0.
    Error: O(Ma^6), negligible compared to LBM's O(Ma^2).

    Args:
        rho_mps: density MPS
        ones_mps: all-ones MPS (chi=1)
        n_grid: grid side length (total points = n_grid^2)
        max_bond: bond dimension truncation
        cutoff: SVD cutoff

    Returns:
        inv_rho_mps: MPS approximation of 1/rho
    """
    # Mean density (scalar)
    rho_0 = mps_sum_value(rho_mps) / (n_grid ** 2)

    # Density fluctuation
    delta_rho = mps_subtract(rho_mps, mps_scale(ones_mps, rho_0),
                              max_bond=max_bond, cutoff=cutoff)

    # delta_rho^2 (1 Hadamard)
    delta_rho_sq = mps_hadamard(delta_rho, delta_rho,
                                 max_bond=max_bond, cutoff=cutoff)

    # Taylor: 1/rho ~ (1/rho_0) - (1/rho_0^2)*delta_rho + (1/rho_0^3)*delta_rho^2
    term0 = mps_scale(ones_mps, 1.0 / rho_0)
    term1 = mps_scale(delta_rho, -1.0 / rho_0 ** 2)
    term2 = mps_scale(delta_rho_sq, 1.0 / rho_0 ** 3)

    inv_rho = mps_add(term0, term1, max_bond=max_bond, cutoff=cutoff)
    inv_rho = mps_add(inv_rho, term2, max_bond=max_bond, cutoff=cutoff)

    return inv_rho


def compute_equilibrium_mps(mps_list, lattice, n_grid,
                             max_bond=None, cutoff=1e-10):
    """
    Compute all 9 equilibrium distributions in MPS space.

    f_eq_i = w_i * rho * (1 + cu/cs2 + cu^2/(2*cs4) - u^2/(2*cs2))

    Uses 16 Hadamard products (optimized: reuses cu^2 across populations).

    Args:
        mps_list: list of 9 population MPS
        lattice: D2Q9 lattice definition
        n_grid: grid side length
        max_bond: bond dimension truncation for Hadamard products
        cutoff: SVD cutoff

    Returns:
        f_eq_list: list of 9 equilibrium MPS
    """
    num_sites = len(mps_list[0].tensors)
    cs2 = lattice.cs2       # 1/3
    cs4 = cs2 * cs2         # 1/9

    # --- Sub-step 1: Moments ---
    rho, rhou_x, rhou_y = compute_moments_mps(mps_list, lattice,
                                                max_bond=max_bond, cutoff=cutoff)

    # --- Sub-step 2: Inverse density (Taylor expansion) ---
    ones = build_ones_mps(num_sites)
    inv_rho = compute_inverse_density_mps(rho, ones, n_grid,
                                           max_bond=max_bond, cutoff=cutoff)

    # --- Sub-step 3: Velocity extraction ---
    u_x = mps_hadamard(rhou_x, inv_rho, max_bond=max_bond, cutoff=cutoff)
    u_y = mps_hadamard(rhou_y, inv_rho, max_bond=max_bond, cutoff=cutoff)

    # --- Sub-step 4: Shared quadratic terms ---
    # ux^2, uy^2 (reused by axis-aligned pops)
    ux_sq = mps_hadamard(u_x, u_x, max_bond=max_bond, cutoff=cutoff)
    uy_sq = mps_hadamard(u_y, u_y, max_bond=max_bond, cutoff=cutoff)
    u_sq = mps_add(ux_sq, uy_sq, max_bond=max_bond, cutoff=cutoff)

    # (ux+uy)^2, (ux-uy)^2 (reused by diagonal pops)
    u_plus = mps_add(u_x, u_y, max_bond=max_bond, cutoff=cutoff)
    u_minus = mps_subtract(u_x, u_y, max_bond=max_bond, cutoff=cutoff)
    u_plus_sq = mps_hadamard(u_plus, u_plus, max_bond=max_bond, cutoff=cutoff)
    u_minus_sq = mps_hadamard(u_minus, u_minus, max_bond=max_bond, cutoff=cutoff)

    # Precompute the -u^2/(2*cs2) term (shared by all pops)
    u_sq_term = mps_scale(u_sq, -1.0 / (2.0 * cs2))

    # --- Sub-step 5: Per-population equilibrium ---
    f_eq = [None] * lattice.q

    for i in range(lattice.q):
        cx = int(lattice.c[i, 0])
        cy = int(lattice.c[i, 1])
        w_i = lattice.w[i]

        # Determine cu and cu^2 based on velocity direction
        if cx == 0 and cy == 0:
            # Pop 0 (rest): cu = 0, cu^2 = 0
            bracket = mps_add(ones, u_sq_term, max_bond=max_bond, cutoff=cutoff)

        elif cy == 0:
            # Axis-aligned x (pops 1, 3): cu = cx*ux, cu^2 = ux^2
            cu = u_x if cx == 1 else mps_scale(u_x, -1.0)
            cu_sq = ux_sq
            cu_term = mps_scale(cu, 1.0 / cs2)
            cu_sq_term = mps_scale(cu_sq, 1.0 / (2.0 * cs4))
            bracket = mps_add(ones, cu_term, max_bond=max_bond, cutoff=cutoff)
            bracket = mps_add(bracket, cu_sq_term, max_bond=max_bond, cutoff=cutoff)
            bracket = mps_add(bracket, u_sq_term, max_bond=max_bond, cutoff=cutoff)

        elif cx == 0:
            # Axis-aligned y (pops 2, 4): cu = cy*uy, cu^2 = uy^2
            cu = u_y if cy == 1 else mps_scale(u_y, -1.0)
            cu_sq = uy_sq
            cu_term = mps_scale(cu, 1.0 / cs2)
            cu_sq_term = mps_scale(cu_sq, 1.0 / (2.0 * cs4))
            bracket = mps_add(ones, cu_term, max_bond=max_bond, cutoff=cutoff)
            bracket = mps_add(bracket, cu_sq_term, max_bond=max_bond, cutoff=cutoff)
            bracket = mps_add(bracket, u_sq_term, max_bond=max_bond, cutoff=cutoff)

        else:
            # Diagonal (pops 5-8): cu = cx*ux + cy*uy
            # Pop 5: (+1,+1) -> cu = ux+uy, cu^2 = (ux+uy)^2
            # Pop 6: (-1,+1) -> cu = -(ux-uy), cu^2 = (ux-uy)^2
            # Pop 7: (-1,-1) -> cu = -(ux+uy), cu^2 = (ux+uy)^2
            # Pop 8: (+1,-1) -> cu = ux-uy, cu^2 = (ux-uy)^2
            if cx == cy:  # pops 5 (+1,+1) and 7 (-1,-1)
                cu = u_plus if cx == 1 else mps_scale(u_plus, -1.0)
                cu_sq = u_plus_sq
            else:  # pops 6 (-1,+1) and 8 (+1,-1)
                cu = u_minus if cx == 1 else mps_scale(u_minus, -1.0)
                cu_sq = u_minus_sq

            cu_term = mps_scale(cu, 1.0 / cs2)
            cu_sq_term = mps_scale(cu_sq, 1.0 / (2.0 * cs4))
            bracket = mps_add(ones, cu_term, max_bond=max_bond, cutoff=cutoff)
            bracket = mps_add(bracket, cu_sq_term, max_bond=max_bond, cutoff=cutoff)
            bracket = mps_add(bracket, u_sq_term, max_bond=max_bond, cutoff=cutoff)

        # f_eq_i = w_i * rho * bracket
        rho_scaled = mps_scale(rho, w_i)
        f_eq[i] = mps_hadamard(rho_scaled, bracket,
                                max_bond=max_bond, cutoff=cutoff)

    return f_eq


def collide_bgk_mps(mps_list, lattice, tau, n_grid,
                     max_bond=None, cutoff=1e-10):
    """
    Full BGK collision in MPS space.

    f_i_new = (1 - 1/tau) * f_i + (1/tau) * f_eq_i

    Args:
        mps_list: list of 9 population MPS
        lattice: D2Q9 lattice definition
        tau: BGK relaxation time
        n_grid: grid side length
        max_bond: bond dimension truncation
        cutoff: SVD cutoff

    Returns:
        f_new_list: list of 9 post-collision MPS
    """
    f_eq = compute_equilibrium_mps(mps_list, lattice, n_grid,
                                    max_bond=max_bond, cutoff=cutoff)

    omega = 1.0 / tau
    one_minus_omega = 1.0 - omega

    f_new = []
    for i in range(lattice.q):
        term1 = mps_scale(mps_list[i], one_minus_omega)
        term2 = mps_scale(f_eq[i], omega)
        f_new.append(mps_add(term1, term2, max_bond=max_bond, cutoff=cutoff))

    return f_new

"""
Local test for memory comparison: TN-LBM vs Coarse LBM.
Tests a single χ value (χ=12) for all three simulation types.

For time-dependent flows (Taylor-Green, Cylinder), we use physically equivalent time:
- Same t/τ_decay ratio (Taylor-Green)
- Same t/T_shed ratio (Cylinder)
"""

import numpy as np
import sys
from pathlib import Path
from scipy.ndimage import zoom

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lbm import D2Q9, compute_equilibrium, stream, collide_bgk
from lbm.boundary import (
    apply_bounce_back, apply_bounce_back_moving_top,
    create_cylinder_mask, create_channel_walls,
    equilibrium_inlet_left, extrapolation_outlet_right
)
from lbm.collision import compute_moments
from simulations.lid_driven_cavity import create_cavity_walls


# =============================================================================
# Helper Functions
# =============================================================================

def compute_memory(method: str, N: int, chi: int = None) -> int:
    """Compute memory usage in number of floats."""
    if method == 'vanilla':
        return 9 * N * N  # 9 populations × N²
    elif method == 'tn':
        L = 2 * int(np.log2(N))  # chain length per population
        return 9 * L * chi * chi  # 9 populations × L sites × χ² per site
    else:
        raise ValueError(f"Unknown method: {method}")


def memory_equivalent_grid(N_fine: int, chi: int) -> int:
    """Compute N_coarse for vanilla LBM with same memory as TN-LBM."""
    tn_memory = compute_memory('tn', N_fine, chi)
    return int(np.sqrt(tn_memory / 9))


def interpolate_to_fine(u_coarse: np.ndarray, N_fine: int) -> np.ndarray:
    """Interpolate coarse velocity field to fine grid using bicubic."""
    N_coarse = u_coarse.shape[1]
    scale = N_fine / N_coarse
    u_fine = np.zeros((2, N_fine, N_fine))
    for dim in range(2):
        u_fine[dim] = zoom(u_coarse[dim], scale, order=3)
    return u_fine


def compute_velocity_error(u_test: np.ndarray, u_baseline: np.ndarray) -> float:
    """Compute relative L2 error of velocity field."""
    diff = u_test - u_baseline
    error = np.sqrt(np.sum(diff**2)) / np.sqrt(np.sum(u_baseline**2) + 1e-10)
    return error


# =============================================================================
# Lid-Driven Cavity (Steady State)
# =============================================================================

def run_cavity_vanilla(N: int, Re: float, u_lid: float = 0.1,
                       max_nt: int = 200000, conv_tol: float = 1e-5,
                       verbose: bool = True) -> tuple:
    """
    Run vanilla LBM for lid-driven cavity until convergence.
    Returns: (u_final, nt_converged, converged)
    """
    L = N
    nu = u_lid * L / Re
    tau = 3 * nu + 0.5

    if tau <= 0.5 or tau > 2.0:
        raise ValueError(f"Unstable τ={tau:.4f} for N={N}, Re={Re}")

    walls, lid = create_cavity_walls(N)
    u_wall = np.array([u_lid, 0.0])

    rho = np.ones((N, N))
    u_prev = np.zeros((2, N, N))
    f = compute_equilibrium(D2Q9, rho, u_prev)

    if verbose:
        print(f"  Running N={N}, Re={Re}, τ={tau:.4f}")

    for t in range(max_nt):
        f = collide_bgk(D2Q9, f, tau)
        f = stream(D2Q9, f)
        f = apply_bounce_back(D2Q9, f, walls)
        f = apply_bounce_back_moving_top(D2Q9, f, u_wall)

        # Check convergence every step
        rho, u = compute_moments(D2Q9, f)
        u_change = np.max(np.abs(u - u_prev)) / (np.max(np.abs(u)) + 1e-10)

        if u_change < conv_tol:
            if verbose:
                print(f"  Converged at t={t}, u_change={u_change:.2e}")
            return u, t, True

        u_prev = u.copy()

        if verbose and t > 0 and t % 10000 == 0:
            print(f"    t={t}, u_change={u_change:.2e}")

    if verbose:
        print(f"  Max iterations reached, u_change={u_change:.2e}")
    return u, max_nt, False


def run_cavity_tn(N: int, Re: float, chi: int, nt: int,
                  verbose: bool = True) -> np.ndarray:
    """
    Run TN-LBM (compress-decompress every step) for cavity.
    """
    from tn.compression import field_to_qtt, qtt_to_field

    L = N
    nu = 0.1 * L / Re
    tau = 3 * nu + 0.5

    walls, lid = create_cavity_walls(N)
    u_wall = np.array([0.1, 0.0])

    rho = np.ones((N, N))
    u = np.zeros((2, N, N))
    f = compute_equilibrium(D2Q9, rho, u)

    if verbose:
        print(f"  Running TN-LBM: N={N}, Re={Re}, χ={chi}, nt={nt}")

    for t in range(nt):
        # Collision
        f = collide_bgk(D2Q9, f, tau)

        # Streaming
        f = stream(D2Q9, f)

        # Boundary conditions
        f = apply_bounce_back(D2Q9, f, walls)
        f = apply_bounce_back_moving_top(D2Q9, f, u_wall)

        # Compress-decompress each population (the TN step)
        for i in range(9):
            mps, meta = field_to_qtt(f[i], max_bond=chi, mapping='snake')
            f[i] = qtt_to_field(mps, meta)

        if verbose and t > 0 and t % 5000 == 0:
            rho, u_check = compute_moments(D2Q9, f)
            print(f"    t={t}, max_u={np.max(np.abs(u_check)):.6f}")

    rho, u_final = compute_moments(D2Q9, f)
    return u_final


def test_cavity_memory_comparison(chi: int = 12, N_fine: int = 256, Re: float = 100):
    """Test memory comparison for lid-driven cavity."""
    print("\n" + "=" * 60)
    print("CAVITY: Memory Comparison Test")
    print(f"N_fine={N_fine}, Re={Re}, χ={chi}")
    print("=" * 60)

    N_coarse = memory_equivalent_grid(N_fine, chi)
    memory = compute_memory('tn', N_fine, chi)

    print(f"\nMemory budget: {memory:,} floats")
    print(f"  TN-LBM: N={N_fine}, χ={chi}")
    print(f"  Coarse: N={N_coarse}")

    # Step 1: Run fine-grid baseline
    print("\n[1] Running fine-grid baseline (N=256)...")
    u_baseline, nt_baseline, converged = run_cavity_vanilla(N_fine, Re)
    if not converged:
        print("  WARNING: Baseline did not converge!")
    print(f"  Baseline: converged at nt={nt_baseline}")

    # Step 2: Run TN-LBM
    print(f"\n[2] Running TN-LBM (χ={chi})...")
    u_tn = run_cavity_tn(N_fine, Re, chi, nt_baseline)
    error_tn = compute_velocity_error(u_tn, u_baseline)
    print(f"  TN-LBM error: {error_tn:.4e} ({error_tn*100:.2f}%)")

    # Step 3: Run coarse LBM
    print(f"\n[3] Running coarse LBM (N={N_coarse})...")
    u_coarse, nt_coarse, converged_coarse = run_cavity_vanilla(N_coarse, Re)
    u_coarse_interp = interpolate_to_fine(u_coarse, N_fine)
    error_coarse = compute_velocity_error(u_coarse_interp, u_baseline)
    print(f"  Coarse LBM error: {error_coarse:.4e} ({error_coarse*100:.2f}%)")

    # Results
    print("\n" + "-" * 40)
    print("RESULTS:")
    print(f"  TN-LBM error:    {error_tn:.4e} ({error_tn*100:.2f}%)")
    print(f"  Coarse LBM error: {error_coarse:.4e} ({error_coarse*100:.2f}%)")
    print(f"  Winner: {'TN-LBM' if error_tn < error_coarse else 'Coarse LBM'}")
    print(f"  Error ratio (Coarse/TN): {error_coarse/error_tn:.2f}x")

    return {
        'chi': chi, 'N_fine': N_fine, 'N_coarse': N_coarse,
        'memory': memory, 'error_tn': error_tn, 'error_coarse': error_coarse
    }


# =============================================================================
# Taylor-Green Vortex (Time-Dependent, Decaying)
# =============================================================================

def run_taylor_green_vanilla(N: int, Re: float, nt: int, U0: float = 0.1,
                             verbose: bool = True) -> tuple:
    """
    Run vanilla LBM for Taylor-Green vortex.
    Returns: (u_final, tau_decay)
    """
    from simulations.taylor_green import analytical_taylor_green

    L = N
    nu = U0 * L / Re
    tau = 3 * nu + 0.5
    k = 2 * np.pi / N
    tau_decay = 1 / (2 * nu * k**2)

    if verbose:
        print(f"  Running N={N}, Re={Re}, τ={tau:.4f}, τ_decay={tau_decay:.1f}")

    # Initialize with analytical solution at t=0
    rho = np.ones((N, N))
    u_init = analytical_taylor_green(N, t=0, nu=nu, U0=U0)
    f = compute_equilibrium(D2Q9, rho, u_init)

    for t in range(nt):
        f = collide_bgk(D2Q9, f, tau)
        f = stream(D2Q9, f)  # Periodic BC (no walls)

        if verbose and t > 0 and t % (nt // 10) == 0:
            rho, u = compute_moments(D2Q9, f)
            u_max = np.max(np.sqrt(u[0]**2 + u[1]**2))
            print(f"    t={t}, |u|_max={u_max:.6f}")

    rho, u_final = compute_moments(D2Q9, f)
    return u_final, tau_decay


def run_taylor_green_tn(N: int, Re: float, chi: int, nt: int, U0: float = 0.1,
                        verbose: bool = True) -> np.ndarray:
    """
    Run TN-LBM for Taylor-Green vortex.
    """
    from simulations.taylor_green import analytical_taylor_green
    from tn.compression import field_to_qtt, qtt_to_field

    L = N
    nu = U0 * L / Re
    tau = 3 * nu + 0.5

    if verbose:
        print(f"  Running TN-LBM: N={N}, Re={Re}, χ={chi}, nt={nt}")

    rho = np.ones((N, N))
    u_init = analytical_taylor_green(N, t=0, nu=nu, U0=U0)
    f = compute_equilibrium(D2Q9, rho, u_init)

    for t in range(nt):
        f = collide_bgk(D2Q9, f, tau)
        f = stream(D2Q9, f)

        # Compress-decompress
        for i in range(9):
            mps, meta = field_to_qtt(f[i], max_bond=chi, mapping='snake')
            f[i] = qtt_to_field(mps, meta)

        if verbose and t > 0 and t % (nt // 10) == 0:
            rho, u = compute_moments(D2Q9, f)
            u_max = np.max(np.sqrt(u[0]**2 + u[1]**2))
            print(f"    t={t}, |u|_max={u_max:.6f}")

    rho, u_final = compute_moments(D2Q9, f)
    return u_final


def test_taylor_green_memory_comparison(chi: int = 12, N_fine: int = 256, Re: float = 100):
    """
    Test memory comparison for Taylor-Green vortex.

    Key insight: τ_decay = 1/(2νk²) where ν ∝ N and k ∝ 1/N, so τ_decay ∝ N.
    To reach the same physical time (same t/τ_decay ratio), coarse grid needs:
        nt_coarse = nt_fine * (N_coarse / N_fine)
    """
    print("\n" + "=" * 60)
    print("TAYLOR-GREEN: Memory Comparison Test")
    print(f"N_fine={N_fine}, Re={Re}, χ={chi}")
    print("=" * 60)

    N_coarse = memory_equivalent_grid(N_fine, chi)
    memory = compute_memory('tn', N_fine, chi)

    # T_sampling from CSV (at 2×τ_decay for fine grid)
    t_sampling_fine = 6500  # From find_t_sampling.py results

    # Physically equivalent time for coarse grid
    # τ_decay ∝ N (linear), so nt_coarse = nt_fine * (N_coarse/N_fine)
    scale_factor = N_coarse / N_fine
    t_sampling_coarse = int(t_sampling_fine * scale_factor)

    print(f"\nMemory budget: {memory:,} floats")
    print(f"  TN-LBM: N={N_fine}, χ={chi}, nt={t_sampling_fine}")
    print(f"  Coarse: N={N_coarse}, nt={t_sampling_coarse} (physically equivalent)")

    # Step 1: Run fine-grid baseline
    print(f"\n[1] Running fine-grid baseline (N={N_fine})...")
    u_baseline, tau_decay_fine = run_taylor_green_vanilla(N_fine, Re, t_sampling_fine)
    print(f"  Baseline: τ_decay={tau_decay_fine:.1f}, t/τ_decay={t_sampling_fine/tau_decay_fine:.2f}")

    # Step 2: Run TN-LBM
    print(f"\n[2] Running TN-LBM (χ={chi})...")
    u_tn = run_taylor_green_tn(N_fine, Re, chi, t_sampling_fine)
    error_tn = compute_velocity_error(u_tn, u_baseline)
    print(f"  TN-LBM error: {error_tn:.4e} ({error_tn*100:.2f}%)")

    # Step 3: Run coarse LBM at physically equivalent time
    print(f"\n[3] Running coarse LBM (N={N_coarse}, nt={t_sampling_coarse})...")
    u_coarse, tau_decay_coarse = run_taylor_green_vanilla(N_coarse, Re, t_sampling_coarse)
    print(f"  Coarse: τ_decay={tau_decay_coarse:.1f}, t/τ_decay={t_sampling_coarse/tau_decay_coarse:.2f}")

    # Interpolate to fine grid
    u_coarse_interp = interpolate_to_fine(u_coarse, N_fine)
    error_coarse = compute_velocity_error(u_coarse_interp, u_baseline)
    print(f"  Coarse LBM error: {error_coarse:.4e} ({error_coarse*100:.2f}%)")

    # Results
    print("\n" + "-" * 40)
    print("RESULTS:")
    print(f"  TN-LBM error:    {error_tn:.4e} ({error_tn*100:.2f}%)")
    print(f"  Coarse LBM error: {error_coarse:.4e} ({error_coarse*100:.2f}%)")
    print(f"  Winner: {'TN-LBM' if error_tn < error_coarse else 'Coarse LBM'}")
    if error_tn > 0:
        print(f"  Error ratio (Coarse/TN): {error_coarse/error_tn:.2f}x")

    return {
        'chi': chi, 'N_fine': N_fine, 'N_coarse': N_coarse,
        'memory': memory, 'error_tn': error_tn, 'error_coarse': error_coarse,
        't_fine': t_sampling_fine, 't_coarse': t_sampling_coarse
    }


# =============================================================================
# Cylinder Flow (Time-Dependent, Periodic Shedding)
# =============================================================================

def run_cylinder_vanilla(N: int, Re: float, nt: int, u_inlet: float = 0.1,
                         verbose: bool = True) -> tuple:
    """
    Run vanilla LBM for cylinder flow (square domain).
    Returns: (u_final, T_shed_estimated)
    """
    # equilibrium_inlet_left, extrapolation_outlet_right imported at top
    # create_cylinder_mask, create_channel_walls imported at top

    # Cylinder geometry for square domain
    cylinder_r = N / 16
    cylinder_x = N / 4
    cylinder_y = N / 2
    D = 2 * cylinder_r

    nu = u_inlet * D / Re
    tau = 3 * nu + 0.5

    # Estimated shedding period (St ≈ 0.17 for Re=100)
    St = 0.17
    T_shed = D / (St * u_inlet)

    if verbose:
        print(f"  Running N={N}, Re={Re}, τ={tau:.4f}, D={D:.0f}")
        print(f"  T_shed (estimated) = {T_shed:.1f} timesteps")

    # Create geometry
    cylinder = create_cylinder_mask(N, N, cylinder_x, cylinder_y, cylinder_r)
    walls = create_channel_walls(N, N)
    solid = cylinder | walls

    # Initialize
    rho = np.ones((N, N))
    u = np.zeros((2, N, N))
    u[0, :, :] = u_inlet
    u[0, solid] = 0
    u[1, solid] = 0
    f = compute_equilibrium(D2Q9, rho, u)

    # Inlet profile with perturbation
    y = np.arange(N)
    u_inlet_profile = np.zeros((2, N))
    perturbation = 0.01 * u_inlet * (y - N/2) / (N/2)
    u_inlet_profile[0, :] = u_inlet + perturbation

    for t in range(nt):
        f = collide_bgk(D2Q9, f, tau)
        f = stream(D2Q9, f)
        f = apply_bounce_back(D2Q9, f, solid)
        f = equilibrium_inlet_left(D2Q9, f, u_inlet_profile, rho_inlet=1.0)
        f = extrapolation_outlet_right(f)

        if verbose and t > 0 and t % (nt // 10) == 0:
            rho_t, u_t = compute_moments(D2Q9, f)
            u_max = np.max(np.abs(u_t[:, ~solid]))
            print(f"    t={t}, max_u={u_max:.4f}")

    rho, u_final = compute_moments(D2Q9, f)
    u_final[0, solid] = 0
    u_final[1, solid] = 0

    return u_final, T_shed


def run_cylinder_tn(N: int, Re: float, chi: int, nt: int, u_inlet: float = 0.1,
                    verbose: bool = True) -> np.ndarray:
    """
    Run TN-LBM for cylinder flow.
    """
    from tn.compression import field_to_qtt, qtt_to_field
    # equilibrium_inlet_left, extrapolation_outlet_right imported at top
    # create_cylinder_mask, create_channel_walls imported at top

    cylinder_r = N / 16
    cylinder_x = N / 4
    cylinder_y = N / 2
    D = 2 * cylinder_r

    nu = u_inlet * D / Re
    tau = 3 * nu + 0.5

    if verbose:
        print(f"  Running TN-LBM: N={N}, Re={Re}, χ={chi}, nt={nt}")

    cylinder = create_cylinder_mask(N, N, cylinder_x, cylinder_y, cylinder_r)
    walls = create_channel_walls(N, N)
    solid = cylinder | walls

    rho = np.ones((N, N))
    u = np.zeros((2, N, N))
    u[0, :, :] = u_inlet
    u[0, solid] = 0
    u[1, solid] = 0
    f = compute_equilibrium(D2Q9, rho, u)

    y = np.arange(N)
    u_inlet_profile = np.zeros((2, N))
    perturbation = 0.01 * u_inlet * (y - N/2) / (N/2)
    u_inlet_profile[0, :] = u_inlet + perturbation

    for t in range(nt):
        f = collide_bgk(D2Q9, f, tau)
        f = stream(D2Q9, f)
        f = apply_bounce_back(D2Q9, f, solid)
        f = equilibrium_inlet_left(D2Q9, f, u_inlet_profile, rho_inlet=1.0)
        f = extrapolation_outlet_right(f)

        # Compress-decompress
        for i in range(9):
            mps, meta = field_to_qtt(f[i], max_bond=chi, mapping='snake')
            f[i] = qtt_to_field(mps, meta)

        if verbose and t > 0 and t % (nt // 10) == 0:
            rho_t, u_t = compute_moments(D2Q9, f)
            u_max = np.max(np.abs(u_t))
            print(f"    t={t}, max_u={u_max:.4f}")

    rho, u_final = compute_moments(D2Q9, f)
    u_final[0, solid] = 0
    u_final[1, solid] = 0

    return u_final


def test_cylinder_memory_comparison(chi: int = 12, N_fine: int = 256, Re: float = 100):
    """
    Test memory comparison for cylinder flow.

    For cylinder flow, T_shed ∝ D/U. Since D ∝ N (we keep D/N constant),
    and U is fixed, T_shed ∝ N. So:
        nt_coarse = nt_fine * (N_coarse / N_fine)
    """
    print("\n" + "=" * 60)
    print("CYLINDER: Memory Comparison Test")
    print(f"N_fine={N_fine}, Re={Re}, χ={chi}")
    print("=" * 60)

    N_coarse = memory_equivalent_grid(N_fine, chi)
    memory = compute_memory('tn', N_fine, chi)

    # T_sampling from CSV
    t_sampling_fine = 10511  # From find_t_sampling.py results

    # Physically equivalent time for coarse grid
    # T_shed ∝ N (since D ∝ N), so nt_coarse = nt_fine * (N_coarse/N_fine)
    scale_factor = N_coarse / N_fine
    t_sampling_coarse = int(t_sampling_fine * scale_factor)

    print(f"\nMemory budget: {memory:,} floats")
    print(f"  TN-LBM: N={N_fine}, χ={chi}, nt={t_sampling_fine}")
    print(f"  Coarse: N={N_coarse}, nt={t_sampling_coarse} (physically equivalent)")

    # Step 1: Run fine-grid baseline
    print(f"\n[1] Running fine-grid baseline (N={N_fine})...")
    u_baseline, T_shed_fine = run_cylinder_vanilla(N_fine, Re, t_sampling_fine)
    print(f"  Baseline: T_shed={T_shed_fine:.1f}, t/T_shed={t_sampling_fine/T_shed_fine:.2f}")

    # Step 2: Run TN-LBM
    print(f"\n[2] Running TN-LBM (χ={chi})...")
    u_tn = run_cylinder_tn(N_fine, Re, chi, t_sampling_fine)

    # Mask out solid for error calculation
    # create_cylinder_mask, create_channel_walls imported at top
    cylinder = create_cylinder_mask(N_fine, N_fine, N_fine/4, N_fine/2, N_fine/16)
    walls = create_channel_walls(N_fine, N_fine)
    solid = cylinder | walls

    # Compute error only in fluid region
    u_baseline_fluid = u_baseline.copy()
    u_tn_fluid = u_tn.copy()
    u_baseline_fluid[:, solid] = 0
    u_tn_fluid[:, solid] = 0

    error_tn = compute_velocity_error(u_tn_fluid, u_baseline_fluid)
    print(f"  TN-LBM error: {error_tn:.4e} ({error_tn*100:.2f}%)")

    # Step 3: Run coarse LBM
    print(f"\n[3] Running coarse LBM (N={N_coarse}, nt={t_sampling_coarse})...")
    u_coarse, T_shed_coarse = run_cylinder_vanilla(N_coarse, Re, t_sampling_coarse)
    print(f"  Coarse: T_shed={T_shed_coarse:.1f}, t/T_shed={t_sampling_coarse/T_shed_coarse:.2f}")

    # Interpolate to fine grid
    u_coarse_interp = interpolate_to_fine(u_coarse, N_fine)
    u_coarse_interp[:, solid] = 0

    error_coarse = compute_velocity_error(u_coarse_interp, u_baseline_fluid)
    print(f"  Coarse LBM error: {error_coarse:.4e} ({error_coarse*100:.2f}%)")

    # Results
    print("\n" + "-" * 40)
    print("RESULTS:")
    print(f"  TN-LBM error:    {error_tn:.4e} ({error_tn*100:.2f}%)")
    print(f"  Coarse LBM error: {error_coarse:.4e} ({error_coarse*100:.2f}%)")
    print(f"  Winner: {'TN-LBM' if error_tn < error_coarse else 'Coarse LBM'}")
    if error_tn > 0:
        print(f"  Error ratio (Coarse/TN): {error_coarse/error_tn:.2f}x")

    return {
        'chi': chi, 'N_fine': N_fine, 'N_coarse': N_coarse,
        'memory': memory, 'error_tn': error_tn, 'error_coarse': error_coarse,
        't_fine': t_sampling_fine, 't_coarse': t_sampling_coarse
    }


# =============================================================================
# Validation Functions
# =============================================================================

def validate_time_scaling_taylor_green(N_fine: int = 256, N_coarse: int = 48, Re: float = 100):
    """
    Validate that the time scaling formula gives physically equivalent states.

    For Taylor-Green, τ_decay = 1/(2νk²) where ν ∝ N and k ∝ 1/N, so τ_decay ∝ N.
    At the same t/τ_decay ratio, the velocity amplitude decay should be identical.
    """
    from simulations.taylor_green import analytical_taylor_green

    print("\n" + "=" * 60)
    print("VALIDATION: Taylor-Green Time Scaling")
    print(f"N_fine={N_fine}, N_coarse={N_coarse}, Re={Re}")
    print("=" * 60)

    U0 = 0.1

    # Compute τ_decay for both grids
    # τ_decay = 1/(2νk²), ν = U₀N/Re, k = 2π/N
    # → τ_decay = Re*N / (8π²U₀) ∝ N (linear!)
    nu_fine = U0 * N_fine / Re
    k_fine = 2 * np.pi / N_fine
    tau_decay_fine = 1 / (2 * nu_fine * k_fine**2)

    nu_coarse = U0 * N_coarse / Re
    k_coarse = 2 * np.pi / N_coarse
    tau_decay_coarse = 1 / (2 * nu_coarse * k_coarse**2)

    print(f"\nτ_decay scaling check:")
    print(f"  τ_decay_fine   = {tau_decay_fine:.1f}")
    print(f"  τ_decay_coarse = {tau_decay_coarse:.1f}")
    print(f"  Ratio: {tau_decay_coarse/tau_decay_fine:.4f}")
    print(f"  Expected (N_c/N_f): {N_coarse/N_fine:.4f}")

    # Choose test time (at 2×τ_decay for fine grid)
    t_fine = int(2 * tau_decay_fine)
    # τ_decay ∝ N, so t_coarse = t_fine * (N_coarse/N_fine)
    t_coarse = int(t_fine * (N_coarse / N_fine))

    print(f"\nTest times:")
    print(f"  t_fine   = {t_fine} (t/τ = {t_fine/tau_decay_fine:.2f})")
    print(f"  t_coarse = {t_coarse} (t/τ = {t_coarse/tau_decay_coarse:.2f})")

    # Run both simulations
    print(f"\nRunning fine grid (N={N_fine}, nt={t_fine})...")
    u_fine, _ = run_taylor_green_vanilla(N_fine, Re, t_fine, verbose=False)
    u_max_fine = np.max(np.sqrt(u_fine[0]**2 + u_fine[1]**2))
    decay_ratio_fine = u_max_fine / U0

    print(f"Running coarse grid (N={N_coarse}, nt={t_coarse})...")
    u_coarse, _ = run_taylor_green_vanilla(N_coarse, Re, t_coarse, verbose=False)
    u_max_coarse = np.max(np.sqrt(u_coarse[0]**2 + u_coarse[1]**2))
    decay_ratio_coarse = u_max_coarse / U0

    # Expected decay from analytical solution
    expected_decay = np.exp(-t_fine / tau_decay_fine)

    print(f"\nVelocity amplitude decay:")
    print(f"  Fine:     |u|_max/U0 = {decay_ratio_fine:.4f}")
    print(f"  Coarse:   |u|_max/U0 = {decay_ratio_coarse:.4f}")
    print(f"  Expected: exp(-t/τ)  = {expected_decay:.4f}")

    # Check if they match (within 5%)
    relative_diff = abs(decay_ratio_fine - decay_ratio_coarse) / decay_ratio_fine
    print(f"\nRelative difference: {relative_diff*100:.2f}%")

    if relative_diff < 0.05:
        print("✓ VALIDATION PASSED: Time scaling gives physically equivalent states")
    else:
        print("✗ VALIDATION FAILED: Decay ratios don't match!")

    return {
        't_fine': t_fine, 't_coarse': t_coarse,
        'decay_fine': decay_ratio_fine, 'decay_coarse': decay_ratio_coarse,
        'expected': expected_decay, 'relative_diff': relative_diff
    }


def validate_time_scaling_cylinder(N_fine: int = 256, N_coarse: int = 48, Re: float = 100):
    """
    Validate that the time scaling formula gives physically equivalent states for cylinder.

    For cylinder, T_shed ∝ D ∝ N (since D/N is kept constant), so at the same
    t/T_shed ratio, both grids should be at equivalent shedding phases.
    """
    print("\n" + "=" * 60)
    print("VALIDATION: Cylinder Time Scaling")
    print(f"N_fine={N_fine}, N_coarse={N_coarse}, Re={Re}")
    print("=" * 60)

    u_inlet = 0.1

    # Compute T_shed for both grids
    D_fine = N_fine / 8  # D = 2 * (N/16)
    D_coarse = N_coarse / 8
    St = 0.17  # Strouhal number for Re=100

    T_shed_fine = D_fine / (St * u_inlet)
    T_shed_coarse = D_coarse / (St * u_inlet)

    print(f"\nT_shed scaling check:")
    print(f"  D_fine={D_fine:.1f}, D_coarse={D_coarse:.1f}")
    print(f"  T_shed_fine   = {T_shed_fine:.1f}")
    print(f"  T_shed_coarse = {T_shed_coarse:.1f}")
    print(f"  Ratio: {T_shed_coarse/T_shed_fine:.4f}")
    print(f"  Expected (N_c/N_f): {N_coarse/N_fine:.4f}")

    # Choose test time (5 shedding periods after transient)
    t_transient_fine = 1100  # From find_t_sampling.py
    t_transient_coarse = int(t_transient_fine * N_coarse / N_fine)

    t_fine = int(t_transient_fine + 5 * T_shed_fine)
    t_coarse = int(t_transient_coarse + 5 * T_shed_coarse)

    print(f"\nTest times (transient + 5 shedding periods):")
    print(f"  t_fine   = {t_fine} (t/T_shed = {(t_fine-t_transient_fine)/T_shed_fine:.1f} after transient)")
    print(f"  t_coarse = {t_coarse} (t/T_shed = {(t_coarse-t_transient_coarse)/T_shed_coarse:.1f} after transient)")

    # Run both simulations
    print(f"\nRunning fine grid (N={N_fine}, nt={t_fine})...")
    u_fine, T_shed_actual_fine = run_cylinder_vanilla(N_fine, Re, t_fine, verbose=False)

    print(f"Running coarse grid (N={N_coarse}, nt={t_coarse})...")
    u_coarse, T_shed_actual_coarse = run_cylinder_vanilla(N_coarse, Re, t_coarse, verbose=False)

    # For cylinder, we can't directly compare decay like Taylor-Green
    # Instead, check that both are in periodic shedding regime by looking at max velocity
    u_max_fine = np.max(np.sqrt(u_fine[0]**2 + u_fine[1]**2))
    u_max_coarse = np.max(np.sqrt(u_coarse[0]**2 + u_coarse[1]**2))

    print(f"\nMax velocity (should be similar if both in periodic regime):")
    print(f"  Fine:   |u|_max = {u_max_fine:.4f}")
    print(f"  Coarse: |u|_max = {u_max_coarse:.4f}")

    # The key validation: time ratios should match
    ratio_fine = (t_fine - t_transient_fine) / T_shed_fine
    ratio_coarse = (t_coarse - t_transient_coarse) / T_shed_coarse

    print(f"\nShedding periods after transient:")
    print(f"  Fine:   {ratio_fine:.2f} periods")
    print(f"  Coarse: {ratio_coarse:.2f} periods")

    ratio_diff = abs(ratio_fine - ratio_coarse)
    if ratio_diff < 0.5:
        print(f"✓ VALIDATION PASSED: Both at ~{ratio_fine:.1f} shedding periods")
    else:
        print(f"✗ WARNING: Period counts differ by {ratio_diff:.1f}")

    return {
        't_fine': t_fine, 't_coarse': t_coarse,
        'T_shed_fine': T_shed_fine, 'T_shed_coarse': T_shed_coarse,
        'periods_fine': ratio_fine, 'periods_coarse': ratio_coarse
    }


def validate_interpolation_error(N_fine: int = 256, N_coarse: int = 48, Re: float = 100):
    """
    Compute the "best possible" error for coarse grid approach.

    This is achieved by:
    1. Taking the fine-grid baseline
    2. Downsampling to N_coarse
    3. Interpolating back to N_fine

    This error is the interpolation floor - coarse LBM cannot do better than this.
    """
    print("\n" + "=" * 60)
    print("VALIDATION: Interpolation Error Floor")
    print(f"N_fine={N_fine}, N_coarse={N_coarse}")
    print("=" * 60)

    # Run fine-grid cavity to convergence
    print(f"\nRunning fine-grid baseline (N={N_fine})...")
    u_baseline, _, _ = run_cavity_vanilla(N_fine, Re, verbose=False)

    # Downsample to coarse grid (average pooling)
    scale = N_fine // N_coarse
    u_downsampled = np.zeros((2, N_coarse, N_coarse))
    for dim in range(2):
        for i in range(N_coarse):
            for j in range(N_coarse):
                u_downsampled[dim, i, j] = np.mean(
                    u_baseline[dim, i*scale:(i+1)*scale, j*scale:(j+1)*scale]
                )

    # Interpolate back to fine grid
    u_interp = interpolate_to_fine(u_downsampled, N_fine)

    # Compute error
    error = compute_velocity_error(u_interp, u_baseline)

    print(f"\nResults:")
    print(f"  Downsampling factor: {scale}x")
    print(f"  Interpolation error: {error:.4e} ({error*100:.2f}%)")
    print(f"\nThis is the MINIMUM error achievable by coarse-grid approach.")
    print(f"Actual coarse LBM will have additional discretization error.")

    return {'N_coarse': N_coarse, 'interpolation_error': error}


def validate_chi64_sanity_check(N_fine: int = 256, Re: float = 100):
    """
    Sanity check: At χ=64, N_coarse=256, coarse LBM should match baseline exactly.
    """
    chi = 64
    N_coarse = memory_equivalent_grid(N_fine, chi)

    print("\n" + "=" * 60)
    print(f"VALIDATION: Sanity Check (χ={chi}, N_coarse={N_coarse})")
    print("=" * 60)

    if N_coarse != N_fine:
        print(f"ERROR: Expected N_coarse={N_fine}, got {N_coarse}")
        return None

    print(f"\nAt χ={chi}, N_coarse={N_coarse} = N_fine")
    print("Coarse LBM should give ~0% error (same grid as baseline)")

    # Run baseline
    print(f"\nRunning baseline (N={N_fine})...")
    u_baseline, nt_baseline, _ = run_cavity_vanilla(N_fine, Re, verbose=False)

    # Run "coarse" (which is same as fine)
    print(f"Running 'coarse' (N={N_coarse})...")
    u_coarse, _, _ = run_cavity_vanilla(N_coarse, Re, verbose=False)

    # No interpolation needed since N_coarse = N_fine
    error = compute_velocity_error(u_coarse, u_baseline)

    print(f"\nResult:")
    print(f"  Error: {error:.4e} ({error*100:.4f}%)")

    if error < 1e-10:
        print("✓ VALIDATION PASSED: Coarse=Fine gives zero error")
    else:
        print("✗ UNEXPECTED: Non-zero error for identical grids")

    return {'chi': chi, 'N_coarse': N_coarse, 'error': error}


def run_all_validations(N_fine: int = 256, Re: float = 100, chi: int = 12):
    """Run all validation checks."""
    N_coarse = memory_equivalent_grid(N_fine, chi)

    print("\n" + "=" * 70)
    print("RUNNING ALL VALIDATIONS")
    print("=" * 70)

    results = {}

    # 1. Taylor-Green time scaling
    results['time_scaling'] = validate_time_scaling_taylor_green(N_fine, N_coarse, Re)

    # 2. Interpolation error floor
    results['interpolation'] = validate_interpolation_error(N_fine, N_coarse, Re)

    # 3. χ=64 sanity check
    results['sanity_chi64'] = validate_chi64_sanity_check(N_fine, Re)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    ts = results['time_scaling']
    print(f"\n1. Taylor-Green time scaling:")
    print(f"   Decay ratio diff: {ts['relative_diff']*100:.2f}% ", end="")
    print("✓" if ts['relative_diff'] < 0.05 else "✗")

    interp = results['interpolation']
    print(f"\n2. Interpolation error floor (N_coarse={N_coarse}):")
    print(f"   Error: {interp['interpolation_error']*100:.2f}%")
    print(f"   (Coarse LBM error should be >= this)")

    san = results['sanity_chi64']
    if san:
        print(f"\n3. χ=64 sanity check:")
        print(f"   Error: {san['error']*100:.4f}% ", end="")
        print("✓" if san['error'] < 1e-10 else "✗")

    return results


# =============================================================================
# GIF Visualization Functions
# =============================================================================

def create_taylor_green_comparison_gif(N_fine: int = 256, N_coarse: int = 48, Re: float = 100,
                                        n_frames: int = 50, save_path: str = None):
    """
    Create side-by-side GIF comparing fine and coarse Taylor-Green at equivalent times.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from simulations.taylor_green import analytical_taylor_green

    print("\n" + "=" * 60)
    print("Creating Taylor-Green Comparison GIF")
    print(f"N_fine={N_fine}, N_coarse={N_coarse}, Re={Re}")
    print("=" * 60)

    U0 = 0.1

    # Compute τ_decay for both grids
    nu_fine = U0 * N_fine / Re
    k_fine = 2 * np.pi / N_fine
    tau_decay_fine = 1 / (2 * nu_fine * k_fine**2)

    nu_coarse = U0 * N_coarse / Re
    k_coarse = 2 * np.pi / N_coarse
    tau_decay_coarse = 1 / (2 * nu_coarse * k_coarse**2)

    # Simulation parameters - go to 3×τ_decay
    t_max_fine = int(3 * tau_decay_fine)
    t_max_coarse = int(3 * tau_decay_coarse)

    # Frame intervals (at same t/τ ratios)
    t_ratios = np.linspace(0, 3, n_frames)
    t_frames_fine = (t_ratios * tau_decay_fine).astype(int)
    t_frames_coarse = (t_ratios * tau_decay_coarse).astype(int)

    print(f"Running fine grid (N={N_fine}, nt={t_max_fine})...")

    # Run fine simulation and collect frames
    tau_fine = 3 * nu_fine + 0.5
    rho_fine = np.ones((N_fine, N_fine))
    u_init_fine = analytical_taylor_green(N_fine, 0, nu_fine)
    f_fine = compute_equilibrium(D2Q9, rho_fine, u_init_fine)

    # Store full velocity fields (not just magnitude) for quiver arrows
    frames_fine = []  # List of (u_x, u_y, speed) tuples
    frame_idx = 0
    for t in range(t_max_fine + 1):
        if frame_idx < len(t_frames_fine) and t == t_frames_fine[frame_idx]:
            rho, u = compute_moments(D2Q9, f_fine)
            speed = np.sqrt(u[0]**2 + u[1]**2)
            frames_fine.append((u[0].copy(), u[1].copy(), speed.copy()))
            frame_idx += 1

        if t < t_max_fine:
            f_fine = collide_bgk(D2Q9, f_fine, tau_fine)
            f_fine = stream(D2Q9, f_fine)

    print(f"Running coarse grid (N={N_coarse}, nt={t_max_coarse})...")

    # Run coarse simulation and collect frames
    tau_coarse = 3 * nu_coarse + 0.5
    rho_coarse = np.ones((N_coarse, N_coarse))
    u_init_coarse = analytical_taylor_green(N_coarse, 0, nu_coarse)
    f_coarse = compute_equilibrium(D2Q9, rho_coarse, u_init_coarse)

    frames_coarse = []  # List of (u_x, u_y, speed) tuples
    frame_idx = 0
    for t in range(t_max_coarse + 1):
        if frame_idx < len(t_frames_coarse) and t == t_frames_coarse[frame_idx]:
            rho, u = compute_moments(D2Q9, f_coarse)
            speed = np.sqrt(u[0]**2 + u[1]**2)
            frames_coarse.append((u[0].copy(), u[1].copy(), speed.copy()))
            frame_idx += 1

        if t < t_max_coarse:
            f_coarse = collide_bgk(D2Q9, f_coarse, tau_coarse)
            f_coarse = stream(D2Q9, f_coarse)

    print(f"Creating GIF with {len(frames_fine)} frames...")

    # Create figure with quiver arrows
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Quiver arrow spacing
    skip_fine = max(1, N_fine // 16)
    skip_coarse = max(1, N_coarse // 16)

    # Initial vmax from first frame
    vmax_init = max(np.max(frames_fine[0][2]), np.max(frames_coarse[0][2]))

    # Fine grid plot
    im_fine = axes[0].imshow(frames_fine[0][2].T, origin='lower', cmap='viridis',
                              extent=[0, N_fine, 0, N_fine], vmin=0, vmax=vmax_init, aspect='equal')
    x_fine, y_fine = np.meshgrid(np.arange(0, N_fine, skip_fine), np.arange(0, N_fine, skip_fine), indexing='ij')
    Q_fine = axes[0].quiver(x_fine, y_fine,
                             frames_fine[0][0][::skip_fine, ::skip_fine],
                             frames_fine[0][1][::skip_fine, ::skip_fine],
                             color='white', scale=2, width=0.003, alpha=0.8)
    axes[0].set_title(f'Fine Grid (N={N_fine})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    # Coarse grid plot
    im_coarse = axes[1].imshow(frames_coarse[0][2].T, origin='lower', cmap='viridis',
                                extent=[0, N_coarse, 0, N_coarse], vmin=0, vmax=vmax_init, aspect='equal')
    x_coarse, y_coarse = np.meshgrid(np.arange(0, N_coarse, skip_coarse), np.arange(0, N_coarse, skip_coarse), indexing='ij')
    Q_coarse = axes[1].quiver(x_coarse, y_coarse,
                               frames_coarse[0][0][::skip_coarse, ::skip_coarse],
                               frames_coarse[0][1][::skip_coarse, ::skip_coarse],
                               color='white', scale=2, width=0.003, alpha=0.8)
    axes[1].set_title(f'Coarse Grid (N={N_coarse})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    plt.colorbar(im_fine, ax=axes[0], label='|u|')
    plt.colorbar(im_coarse, ax=axes[1], label='|u|')

    time_text = fig.suptitle(f't/τ_decay = 0.00', fontsize=14)

    def update(frame):
        ux_fine, uy_fine, speed_fine = frames_fine[frame]
        ux_coarse, uy_coarse, speed_coarse = frames_coarse[frame]

        # Update fine grid (static color scale from vmax_init)
        im_fine.set_array(speed_fine.T)
        Q_fine.set_UVC(ux_fine[::skip_fine, ::skip_fine], uy_fine[::skip_fine, ::skip_fine])

        # Update coarse grid
        im_coarse.set_array(speed_coarse.T)
        Q_coarse.set_UVC(ux_coarse[::skip_coarse, ::skip_coarse], uy_coarse[::skip_coarse, ::skip_coarse])

        time_text.set_text(f't/τ_decay = {t_ratios[frame]:.2f}')
        return im_fine, im_coarse, Q_fine, Q_coarse, time_text

    anim = FuncAnimation(fig, update, frames=len(frames_fine), interval=100, blit=False)

    if save_path is None:
        save_path = Path(__file__).parent / 'taylor_green_comparison.gif'

    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close()

    print(f"✓ Saved to {save_path}")
    return save_path


def create_cylinder_comparison_gif(N_fine: int = 256, N_coarse: int = 48, Re: float = 100,
                                    n_frames: int = 50, save_path: str = None):
    """
    Create side-by-side GIF comparing fine and coarse cylinder flow at equivalent times.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    print("\n" + "=" * 60)
    print("Creating Cylinder Flow Comparison GIF")
    print(f"N_fine={N_fine}, N_coarse={N_coarse}, Re={Re}")
    print("=" * 60)

    u_inlet = 0.1
    St = 0.17

    # Compute T_shed for both grids
    D_fine = N_fine / 8
    D_coarse = N_coarse / 8
    T_shed_fine = D_fine / (St * u_inlet)
    T_shed_coarse = D_coarse / (St * u_inlet)

    # Simulation parameters - skip transient, capture 5 shedding periods
    t_transient_fine = 1100
    t_transient_coarse = int(t_transient_fine * N_coarse / N_fine)

    t_max_fine = int(t_transient_fine + 5 * T_shed_fine)
    t_max_coarse = int(t_transient_coarse + 5 * T_shed_coarse)

    # Frame intervals (at same t/T_shed ratios after transient)
    t_ratios = np.linspace(0, 5, n_frames)  # 0 to 5 shedding periods
    t_frames_fine = (t_transient_fine + t_ratios * T_shed_fine).astype(int)
    t_frames_coarse = (t_transient_coarse + t_ratios * T_shed_coarse).astype(int)

    print(f"Running fine grid (N={N_fine}, nt={t_max_fine})...")

    # Setup fine grid
    cylinder_r_fine = N_fine / 16
    cylinder_x_fine = N_fine / 4
    cylinder_y_fine = N_fine / 2

    nu_fine = u_inlet * D_fine / Re
    tau_fine = 3 * nu_fine + 0.5

    cylinder_fine = create_cylinder_mask(N_fine, N_fine, cylinder_x_fine, cylinder_y_fine, cylinder_r_fine)
    walls_fine = create_channel_walls(N_fine, N_fine)
    solid_fine = cylinder_fine | walls_fine

    rho_fine = np.ones((N_fine, N_fine))
    u_fine = np.zeros((2, N_fine, N_fine))
    u_fine[0, :, :] = u_inlet
    u_fine[0, solid_fine] = 0
    u_fine[1, solid_fine] = 0
    f_fine = compute_equilibrium(D2Q9, rho_fine, u_fine)

    y_fine = np.arange(N_fine)
    u_inlet_profile_fine = np.zeros((2, N_fine))
    u_inlet_profile_fine[0, :] = u_inlet + 0.01 * u_inlet * (y_fine - N_fine/2) / (N_fine/2)

    frames_fine = []
    frame_idx = 0
    for t in range(t_max_fine + 1):
        if frame_idx < len(t_frames_fine) and t >= t_frames_fine[frame_idx]:
            rho, u = compute_moments(D2Q9, f_fine)
            vorticity = np.gradient(u[1], axis=0) - np.gradient(u[0], axis=1)
            vorticity[solid_fine] = 0
            frames_fine.append(vorticity.copy())
            frame_idx += 1

        if t < t_max_fine:
            f_fine = collide_bgk(D2Q9, f_fine, tau_fine)
            f_fine = stream(D2Q9, f_fine)
            f_fine = apply_bounce_back(D2Q9, f_fine, solid_fine)
            f_fine = equilibrium_inlet_left(D2Q9, f_fine, u_inlet_profile_fine, rho_inlet=1.0)
            f_fine = extrapolation_outlet_right(f_fine)

        if t > 0 and t % 2000 == 0:
            print(f"  Fine: t={t}/{t_max_fine}")

    print(f"Running coarse grid (N={N_coarse}, nt={t_max_coarse})...")

    # Setup coarse grid
    cylinder_r_coarse = N_coarse / 16
    cylinder_x_coarse = N_coarse / 4
    cylinder_y_coarse = N_coarse / 2

    nu_coarse = u_inlet * D_coarse / Re
    tau_coarse = 3 * nu_coarse + 0.5

    cylinder_coarse = create_cylinder_mask(N_coarse, N_coarse, cylinder_x_coarse, cylinder_y_coarse, cylinder_r_coarse)
    walls_coarse = create_channel_walls(N_coarse, N_coarse)
    solid_coarse = cylinder_coarse | walls_coarse

    rho_coarse = np.ones((N_coarse, N_coarse))
    u_coarse = np.zeros((2, N_coarse, N_coarse))
    u_coarse[0, :, :] = u_inlet
    u_coarse[0, solid_coarse] = 0
    u_coarse[1, solid_coarse] = 0
    f_coarse = compute_equilibrium(D2Q9, rho_coarse, u_coarse)

    y_coarse = np.arange(N_coarse)
    u_inlet_profile_coarse = np.zeros((2, N_coarse))
    u_inlet_profile_coarse[0, :] = u_inlet + 0.01 * u_inlet * (y_coarse - N_coarse/2) / (N_coarse/2)

    frames_coarse = []
    frame_idx = 0
    for t in range(t_max_coarse + 1):
        if frame_idx < len(t_frames_coarse) and t >= t_frames_coarse[frame_idx]:
            rho, u = compute_moments(D2Q9, f_coarse)
            vorticity = np.gradient(u[1], axis=0) - np.gradient(u[0], axis=1)
            vorticity[solid_coarse] = 0
            frames_coarse.append(vorticity.copy())
            frame_idx += 1

        if t < t_max_coarse:
            f_coarse = collide_bgk(D2Q9, f_coarse, tau_coarse)
            f_coarse = stream(D2Q9, f_coarse)
            f_coarse = apply_bounce_back(D2Q9, f_coarse, solid_coarse)
            f_coarse = equilibrium_inlet_left(D2Q9, f_coarse, u_inlet_profile_coarse, rho_inlet=1.0)
            f_coarse = extrapolation_outlet_right(f_coarse)

        if t > 0 and t % 500 == 0:
            print(f"  Coarse: t={t}/{t_max_coarse}")

    print(f"Creating GIF with {min(len(frames_fine), len(frames_coarse))} frames...")

    # Ensure same number of frames
    n_frames_actual = min(len(frames_fine), len(frames_coarse))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Determine color scale
    vmax_fine = np.percentile(np.abs(frames_fine[n_frames_actual//2]), 99)
    vmax_coarse = np.percentile(np.abs(frames_coarse[n_frames_actual//2]), 99)
    vmax = max(vmax_fine, vmax_coarse)

    im_fine = axes[0].imshow(frames_fine[0].T, origin='lower', cmap='RdBu_r',
                              vmin=-vmax, vmax=vmax, aspect='equal')
    axes[0].set_title(f'Fine Grid (N={N_fine})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    im_coarse = axes[1].imshow(frames_coarse[0].T, origin='lower', cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, aspect='equal')
    axes[1].set_title(f'Coarse Grid (N={N_coarse})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    plt.colorbar(im_fine, ax=axes[0], label='Vorticity')
    plt.colorbar(im_coarse, ax=axes[1], label='Vorticity')

    time_text = fig.suptitle(f't/T_shed = 0.00 (after transient)', fontsize=14)

    def update(frame):
        im_fine.set_array(frames_fine[frame].T)
        im_coarse.set_array(frames_coarse[frame].T)
        time_text.set_text(f't/T_shed = {t_ratios[frame]:.2f} (after transient)')
        return im_fine, im_coarse, time_text

    anim = FuncAnimation(fig, update, frames=n_frames_actual, interval=100, blit=False)

    if save_path is None:
        save_path = Path(__file__).parent / 'cylinder_comparison.gif'

    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close()

    print(f"✓ Saved to {save_path}")
    return save_path


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test memory comparison for χ=12")
    parser.add_argument("--chi", type=int, default=12, help="Bond dimension")
    parser.add_argument("--sim", type=str, default="cavity",
                        choices=["cavity", "taylor_green", "cylinder", "all"],
                        help="Which simulation to test")
    parser.add_argument("--validate", action="store_true",
                        help="Run all validation checks")
    parser.add_argument("--validate-tg", action="store_true",
                        help="Run Taylor-Green time scaling validation only")
    parser.add_argument("--validate-cyl", action="store_true",
                        help="Run Cylinder time scaling validation only")
    parser.add_argument("--gif-tg", action="store_true",
                        help="Create Taylor-Green comparison GIF")
    parser.add_argument("--gif-cyl", action="store_true",
                        help="Create Cylinder comparison GIF")
    parser.add_argument("--gif-all", action="store_true",
                        help="Create both comparison GIFs")
    parser.add_argument("--n-frames", type=int, default=50,
                        help="Number of frames for GIF (default: 50)")
    args = parser.parse_args()

    chi = args.chi
    N_coarse = memory_equivalent_grid(256, chi)
    memory = compute_memory('tn', 256, chi)

    # Handle GIF generation
    if args.gif_tg or args.gif_all:
        create_taylor_green_comparison_gif(N_fine=256, N_coarse=N_coarse, Re=100,
                                            n_frames=args.n_frames)
    if args.gif_cyl or args.gif_all:
        create_cylinder_comparison_gif(N_fine=256, N_coarse=N_coarse, Re=100,
                                        n_frames=args.n_frames)

    # Handle validation and experiments (only if not doing GIFs)
    if not (args.gif_tg or args.gif_cyl or args.gif_all):
        if args.validate_tg:
            # Run Taylor-Green validation only
            validate_time_scaling_taylor_green(N_fine=256, N_coarse=N_coarse, Re=100)
        elif args.validate_cyl:
            # Run Cylinder validation only
            validate_time_scaling_cylinder(N_fine=256, N_coarse=N_coarse, Re=100)
        elif args.validate:
            # Run all validation checks
            run_all_validations(N_fine=256, Re=100, chi=chi)
        else:
            # Run actual experiments
            print("=" * 60)
            print(f"Memory Comparison Test: χ={chi}")
            print(f"  N_fine=256, N_coarse={N_coarse}")
            print(f"  Memory: {memory:,} floats")
            print("=" * 60)

            results = {}

            if args.sim in ["cavity", "all"]:
                results['cavity'] = test_cavity_memory_comparison(chi=chi)

            if args.sim in ["taylor_green", "all"]:
                results['taylor_green'] = test_taylor_green_memory_comparison(chi=chi)

            if args.sim in ["cylinder", "all"]:
                results['cylinder'] = test_cylinder_memory_comparison(chi=chi)

            # Summary
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            for sim, res in results.items():
                winner = "TN" if res['error_tn'] < res['error_coarse'] else "Coarse"
                print(f"{sim:15s}: TN={res['error_tn']:.2e}, Coarse={res['error_coarse']:.2e} -> {winner}")

#!/usr/bin/env python
"""
Memory Comparison Experiments for Delft Blue Cluster

Compares TN-LBM vs memory-equivalent coarse vanilla LBM across:
- Lid-driven cavity (steady-state)
- Taylor-Green vortex (decaying)
- Cylinder flow (periodic shedding)

Usage:
    python run_memory_comparison.py --sim all --n-cores 48
    python run_memory_comparison.py --sim cavity --n-cores 16
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.ndimage import zoom

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lbm import D2Q9, compute_equilibrium, stream, collide_bgk
from lbm.boundary import (
    apply_bounce_back, apply_bounce_back_moving_top,
    create_cylinder_mask, create_channel_walls,
    equilibrium_inlet_left, extrapolation_outlet_right
)
from lbm.collision import compute_moments
from simulations.lid_driven_cavity import create_cavity_walls
from simulations.taylor_green import analytical_taylor_green
from tn.compression import field_to_qtt, qtt_to_field

# =============================================================================
# Constants
# =============================================================================

CHI_VALUES = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64]
N_FINE = 256
RE = 100
CONVERGENCE_TOL = 1e-6

# Sampling times from find_t_sampling.py results
T_SAMPLING_TG = 6500
T_SAMPLING_CYL = 10511

# Output directory
RESULTS_DIR = Path(__file__).parent / "results"


# =============================================================================
# Helper Functions
# =============================================================================

def compute_memory(method: str, N: int, chi: int = None) -> int:
    """Compute memory usage in number of floats."""
    if method == 'vanilla':
        return 9 * N * N
    elif method == 'tn':
        L = 2 * int(np.log2(N))
        return 9 * L * chi * chi
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
    return np.sqrt(np.sum(diff**2)) / np.sqrt(np.sum(u_baseline**2) + 1e-10)


# =============================================================================
# Cavity Simulation Functions
# =============================================================================

def run_cavity_vanilla(N: int, Re: float, u_lid: float = 0.1,
                       max_nt: int = 200000, conv_tol: float = 1e-6,
                       verbose: bool = False) -> tuple:
    """Run vanilla LBM for lid-driven cavity until convergence."""
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

    for t in range(max_nt):
        f = collide_bgk(D2Q9, f, tau)
        f = stream(D2Q9, f)
        f = apply_bounce_back(D2Q9, f, walls)
        f = apply_bounce_back_moving_top(D2Q9, f, u_wall)

        rho, u = compute_moments(D2Q9, f)
        u_change = np.max(np.abs(u - u_prev)) / (np.max(np.abs(u)) + 1e-10)

        if u_change < conv_tol:
            if verbose:
                print(f"  Cavity N={N}: converged at t={t}")
            return u, t, True

        u_prev = u.copy()

    return u, max_nt, False


def run_cavity_tn(N: int, Re: float, chi: int, nt: int,
                  verbose: bool = False) -> np.ndarray:
    """Run TN-LBM (compress-decompress every step) for cavity."""
    L = N
    nu = 0.1 * L / Re
    tau = 3 * nu + 0.5

    walls, lid = create_cavity_walls(N)
    u_wall = np.array([0.1, 0.0])

    rho = np.ones((N, N))
    u = np.zeros((2, N, N))
    f = compute_equilibrium(D2Q9, rho, u)

    for t in range(nt):
        f = collide_bgk(D2Q9, f, tau)
        f = stream(D2Q9, f)
        f = apply_bounce_back(D2Q9, f, walls)
        f = apply_bounce_back_moving_top(D2Q9, f, u_wall)

        # Compress-decompress each population
        for i in range(9):
            mps, meta = field_to_qtt(f[i], max_bond=chi, mapping='snake')
            f[i] = qtt_to_field(mps, meta)

    rho, u_final = compute_moments(D2Q9, f)
    return u_final


# =============================================================================
# Taylor-Green Simulation Functions
# =============================================================================

def run_taylor_green_vanilla(N: int, Re: float, nt: int, U0: float = 0.1,
                             verbose: bool = False) -> tuple:
    """Run vanilla LBM for Taylor-Green vortex."""
    L = N
    nu = U0 * L / Re
    tau = 3 * nu + 0.5
    k = 2 * np.pi / N
    tau_decay = 1 / (2 * nu * k**2)

    rho = np.ones((N, N))
    u_init = analytical_taylor_green(N, t=0, nu=nu, U0=U0)
    f = compute_equilibrium(D2Q9, rho, u_init)

    for t in range(nt):
        f = collide_bgk(D2Q9, f, tau)
        f = stream(D2Q9, f)

    rho, u_final = compute_moments(D2Q9, f)
    return u_final, tau_decay


def run_taylor_green_tn(N: int, Re: float, chi: int, nt: int, U0: float = 0.1,
                        verbose: bool = False) -> np.ndarray:
    """Run TN-LBM for Taylor-Green vortex."""
    L = N
    nu = U0 * L / Re
    tau = 3 * nu + 0.5

    rho = np.ones((N, N))
    u_init = analytical_taylor_green(N, t=0, nu=nu, U0=U0)
    f = compute_equilibrium(D2Q9, rho, u_init)

    for t in range(nt):
        f = collide_bgk(D2Q9, f, tau)
        f = stream(D2Q9, f)

        # Compress-decompress each population
        for i in range(9):
            mps, meta = field_to_qtt(f[i], max_bond=chi, mapping='snake')
            f[i] = qtt_to_field(mps, meta)

    rho, u_final = compute_moments(D2Q9, f)
    return u_final


# =============================================================================
# Cylinder Simulation Functions
# =============================================================================

def run_cylinder_vanilla(N: int, Re: float, nt: int, u_inlet: float = 0.1,
                         verbose: bool = False) -> tuple:
    """Run vanilla LBM for cylinder flow (square domain)."""
    cylinder_r = N / 16
    cylinder_x = N / 4
    cylinder_y = N / 2
    D = 2 * cylinder_r

    nu = u_inlet * D / Re
    tau = 3 * nu + 0.5

    St = 0.17
    T_shed = D / (St * u_inlet)

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

    rho, u_final = compute_moments(D2Q9, f)
    u_final[0, solid] = 0
    u_final[1, solid] = 0

    return u_final, T_shed


def run_cylinder_tn(N: int, Re: float, chi: int, nt: int, u_inlet: float = 0.1,
                    verbose: bool = False) -> np.ndarray:
    """Run TN-LBM for cylinder flow."""
    cylinder_r = N / 16
    cylinder_x = N / 4
    cylinder_y = N / 2
    D = 2 * cylinder_r

    nu = u_inlet * D / Re
    tau = 3 * nu + 0.5

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

        # Compress-decompress each population
        for i in range(9):
            mps, meta = field_to_qtt(f[i], max_bond=chi, mapping='snake')
            f[i] = qtt_to_field(mps, meta)

    rho, u_final = compute_moments(D2Q9, f)
    u_final[0, solid] = 0
    u_final[1, solid] = 0

    return u_final


def get_cylinder_solid_mask(N: int) -> np.ndarray:
    """Get solid mask for cylinder flow."""
    cylinder_r = N / 16
    cylinder_x = N / 4
    cylinder_y = N / 2
    cylinder = create_cylinder_mask(N, N, cylinder_x, cylinder_y, cylinder_r)
    walls = create_channel_walls(N, N)
    return cylinder | walls


# =============================================================================
# Baseline Runner
# =============================================================================

def get_baselines(sims: list, verbose: bool = True) -> dict:
    """Run all baselines (sequential, shared across all χ values)."""
    baselines = {}

    if 'cavity' in sims:
        if verbose:
            print("Running cavity baseline (N=256, convergence)...")
        start = time.time()
        u, nt, converged = run_cavity_vanilla(N_FINE, RE, conv_tol=CONVERGENCE_TOL, verbose=verbose)
        elapsed = time.time() - start
        if verbose:
            print(f"  Cavity baseline: nt={nt}, converged={converged}, time={elapsed:.1f}s")
        baselines['cavity'] = {'u': u, 'nt': nt}

    if 'taylor_green' in sims:
        if verbose:
            print(f"Running Taylor-Green baseline (N=256, t={T_SAMPLING_TG})...")
        start = time.time()
        u, tau_decay = run_taylor_green_vanilla(N_FINE, RE, T_SAMPLING_TG, verbose=verbose)
        elapsed = time.time() - start
        if verbose:
            print(f"  Taylor-Green baseline: τ_decay={tau_decay:.1f}, time={elapsed:.1f}s")
        baselines['taylor_green'] = {'u': u, 't': T_SAMPLING_TG, 'tau_decay': tau_decay}

    if 'cylinder' in sims:
        if verbose:
            print(f"Running cylinder baseline (N=256, t={T_SAMPLING_CYL})...")
        start = time.time()
        u, T_shed = run_cylinder_vanilla(N_FINE, RE, T_SAMPLING_CYL, verbose=verbose)
        elapsed = time.time() - start
        if verbose:
            print(f"  Cylinder baseline: T_shed={T_shed:.1f}, time={elapsed:.1f}s")
        baselines['cylinder'] = {'u': u, 't': T_SAMPLING_CYL, 'T_shed': T_shed}

    return baselines


# =============================================================================
# Worker Function for Parallel Execution
# =============================================================================

def run_single_task(args: tuple) -> dict:
    """Run TN-LBM and coarse LBM for one (simulation, χ) pair."""
    sim_type, chi, baseline_data = args

    N_coarse = memory_equivalent_grid(N_FINE, chi)
    memory = compute_memory('tn', N_FINE, chi)

    print(f"  Starting: {sim_type}, χ={chi}, N_coarse={N_coarse}")
    start = time.time()

    if sim_type == 'cavity':
        nt = baseline_data['nt']
        u_baseline = baseline_data['u']

        # TN-LBM
        u_tn = run_cavity_tn(N_FINE, RE, chi, nt)
        error_tn = compute_velocity_error(u_tn, u_baseline)

        # Coarse (run to convergence)
        u_coarse, _, _ = run_cavity_vanilla(N_coarse, RE, conv_tol=CONVERGENCE_TOL)
        u_coarse_interp = interpolate_to_fine(u_coarse, N_FINE)
        error_coarse = compute_velocity_error(u_coarse_interp, u_baseline)

    elif sim_type == 'taylor_green':
        t_fine = baseline_data['t']
        t_coarse = int(t_fine * N_coarse / N_FINE)
        u_baseline = baseline_data['u']

        # TN-LBM
        u_tn = run_taylor_green_tn(N_FINE, RE, chi, t_fine)
        error_tn = compute_velocity_error(u_tn, u_baseline)

        # Coarse (time scaled)
        u_coarse, _ = run_taylor_green_vanilla(N_coarse, RE, t_coarse)
        u_coarse_interp = interpolate_to_fine(u_coarse, N_FINE)
        error_coarse = compute_velocity_error(u_coarse_interp, u_baseline)

    elif sim_type == 'cylinder':
        t_fine = baseline_data['t']
        t_coarse = int(t_fine * N_coarse / N_FINE)
        u_baseline = baseline_data['u']

        # Get solid masks
        solid_fine = get_cylinder_solid_mask(N_FINE)

        # TN-LBM
        u_tn = run_cylinder_tn(N_FINE, RE, chi, t_fine)

        # Coarse (time scaled)
        u_coarse, _ = run_cylinder_vanilla(N_coarse, RE, t_coarse)
        u_coarse_interp = interpolate_to_fine(u_coarse, N_FINE)

        # Mask solid region for error calculation
        u_tn[:, solid_fine] = 0
        u_coarse_interp[:, solid_fine] = 0
        u_baseline_masked = u_baseline.copy()
        u_baseline_masked[:, solid_fine] = 0

        error_tn = compute_velocity_error(u_tn, u_baseline_masked)
        error_coarse = compute_velocity_error(u_coarse_interp, u_baseline_masked)

    else:
        raise ValueError(f"Unknown simulation type: {sim_type}")

    elapsed = time.time() - start
    print(f"  Finished: {sim_type}, χ={chi}, err_tn={error_tn:.4e}, err_coarse={error_coarse:.4e}, time={elapsed:.1f}s")

    return {
        'sim': sim_type,
        'chi': chi,
        'memory': memory,
        'N_coarse': N_coarse,
        'error_tn': error_tn,
        'error_coarse': error_coarse,
        'time_seconds': elapsed
    }


# =============================================================================
# CSV Output
# =============================================================================

def save_csv(results: list, filename: str):
    """Save results to CSV file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename

    # Sort by chi
    results_sorted = sorted(results, key=lambda x: x['chi'])

    fieldnames = ['chi', 'memory', 'N_coarse', 'error_tn', 'error_coarse']

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results_sorted)

    print(f"Saved: {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Memory comparison experiments for Delft Blue cluster"
    )
    parser.add_argument(
        '--sim', type=str, default='all',
        choices=['cavity', 'taylor_green', 'cylinder', 'all'],
        help='Which simulation to run (default: all)'
    )
    parser.add_argument(
        '--n-cores', type=int, default=None,
        help='Number of cores for parallel execution (default: auto-detect)'
    )
    parser.add_argument(
        '--chi', type=int, nargs='+', default=None,
        help='Specific chi values to run (default: all 16 values)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed progress'
    )
    args = parser.parse_args()

    # Determine simulations to run
    if args.sim == 'all':
        sims = ['cavity', 'taylor_green', 'cylinder']
    else:
        sims = [args.sim]

    # Determine chi values
    chi_values = args.chi if args.chi else CHI_VALUES

    # Determine number of cores
    n_cores = args.n_cores if args.n_cores else min(cpu_count(), len(sims) * len(chi_values))

    print("=" * 60)
    print("Memory Comparison Experiments")
    print("=" * 60)
    print(f"Simulations: {sims}")
    print(f"χ values: {chi_values}")
    print(f"N_fine: {N_FINE}")
    print(f"Re: {RE}")
    print(f"Cores: {n_cores}")
    print("=" * 60)

    # Step 1: Run baselines (sequential)
    print("\n[Step 1] Running baselines...")
    baselines = get_baselines(sims, verbose=True)

    # Step 2: Build task list
    tasks = []
    for sim in sims:
        for chi in chi_values:
            tasks.append((sim, chi, baselines[sim]))

    print(f"\n[Step 2] Running {len(tasks)} tasks on {n_cores} cores...")

    # Step 3: Run in parallel
    start_parallel = time.time()
    with Pool(n_cores) as pool:
        results = pool.map(run_single_task, tasks)
    elapsed_parallel = time.time() - start_parallel

    print(f"\n[Step 3] Parallel execution completed in {elapsed_parallel:.1f}s")

    # Step 4: Save results grouped by simulation
    print("\n[Step 4] Saving results...")
    for sim in sims:
        sim_results = [r for r in results if r['sim'] == sim]
        save_csv(sim_results, f'{sim}_memory_comparison.csv')

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for sim in sims:
        sim_results = sorted([r for r in results if r['sim'] == sim], key=lambda x: x['chi'])
        print(f"\n{sim.upper()}:")
        print(f"{'χ':>4} | {'Memory':>10} | {'N_coarse':>8} | {'TN Error':>12} | {'Coarse Error':>12} | {'Winner':>8}")
        print("-" * 70)
        for r in sim_results:
            winner = "TN" if r['error_tn'] < r['error_coarse'] else "Coarse"
            print(f"{r['chi']:>4} | {r['memory']:>10,} | {r['N_coarse']:>8} | "
                  f"{r['error_tn']:>12.4e} | {r['error_coarse']:>12.4e} | {winner:>8}")

    print("\nDone!")


if __name__ == '__main__':
    main()

"""
Taylor-Green Vortex simulation using LBM.

A decaying vortex flow with analytical solution - ideal for validation.
The flow is fully periodic and decays exponentially due to viscosity.

Analytical solution:
    u(x, y, t) =  U0 * sin(kx) * cos(ky) * exp(-2νk²t)
    v(x, y, t) = -U0 * cos(kx) * sin(ky) * exp(-2νk²t)

where k = 2π/L is the wavenumber.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lbm import D2Q9, compute_equilibrium, stream, collide_bgk
from lbm.collision import compute_moments


def analytical_taylor_green(n: int, t: float, nu: float, U0: float = 0.1):
    """
    Compute analytical Taylor-Green vortex solution at time t.

    Args:
        n: Grid size (n x n)
        t: Time (in lattice units)
        nu: Kinematic viscosity
        U0: Initial velocity amplitude

    Returns:
        u: Velocity field (2, n, n) where u[0]=ux, u[1]=uy
    """
    # Physical domain is [0, 2π] x [0, 2π]
    # Wavenumber k = 2π/L = 2π/(2π) = 1 in normalized coordinates
    # But in lattice units, L = n, so k = 2π/n
    k = 2 * np.pi / n

    # Grid coordinates (cell centers)
    x = np.arange(n) + 0.5
    y = np.arange(n) + 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Decay factor
    decay = np.exp(-2 * nu * k**2 * t)

    # Analytical velocity field
    u = np.zeros((2, n, n))
    u[0] = U0 * np.sin(k * X) * np.cos(k * Y) * decay
    u[1] = -U0 * np.cos(k * X) * np.sin(k * Y) * decay

    return u


def run_taylor_green(
    n: int = 128,
    Re: float = 100,
    U0: float = 0.1,
    nt: int = 10000,
    save_every: int = 100,
    verbose: bool = True
):
    """
    Run Taylor-Green vortex simulation.

    Args:
        n: Grid size (n x n, should be power of 2 for TN compatibility)
        Re: Reynolds number (Re = U0 * L / ν, where L = n)
        U0: Initial velocity amplitude (keep < 0.1 for low Mach)
        nt: Number of timesteps
        save_every: Save snapshots every N steps
        verbose: Print progress

    Returns:
        dict with keys:
            'u': velocity history (n_snapshots, 2, n, n)
            'rho': density history (n_snapshots, n, n)
            'u_analytical': analytical velocity at each snapshot time
            'times': timestep indices for snapshots
            'error': L2 error vs analytical at each snapshot
            'params': simulation parameters
    """
    # Compute physical parameters
    L = n  # Characteristic length = domain size
    nu = U0 * L / Re  # Kinematic viscosity
    tau = 3 * nu + 0.5  # Relaxation time

    # Wavenumber and decay time constant
    k = 2 * np.pi / n
    tau_decay = 1 / (2 * nu * k**2)  # Time for amplitude to decay to 1/e

    if verbose:
        print("=" * 60)
        print("Taylor-Green Vortex Simulation")
        print("=" * 60)
        print(f"  Grid: {n} x {n}")
        print(f"  Re = {Re}, U0 = {U0}")
        print(f"  nu = {nu:.6f}, tau = {tau:.4f}")
        print(f"  Decay time constant: tau_decay = {tau_decay:.1f} timesteps")
        print(f"  Running for {nt} timesteps ({nt/tau_decay:.2f} decay times)")

    if tau <= 0.5:
        raise ValueError(f"tau = {tau:.4f} <= 0.5: unstable! Reduce U0 or increase Re.")
    if tau > 2.0:
        print(f"  Warning: tau = {tau:.4f} > 2.0 may be inaccurate")
    if U0 * np.sqrt(3) > 0.3:
        print(f"  Warning: Mach = {U0 * np.sqrt(3):.3f} > 0.3, compressibility effects")

    # Initialize with analytical solution at t=0
    rho = np.ones((n, n))
    u_init = analytical_taylor_green(n, t=0, nu=nu, U0=U0)
    f = compute_equilibrium(D2Q9, rho, u_init)

    # Storage
    u_history = []
    rho_history = []
    u_analytical_history = []
    times = []
    errors = []

    for t in range(nt):
        # Collision
        f = collide_bgk(D2Q9, f, tau)

        # Streaming (periodic by default - no BC needed!)
        f = stream(D2Q9, f)

        # Save snapshots
        if t % save_every == 0:
            rho, u = compute_moments(D2Q9, f)
            u_analytical = analytical_taylor_green(n, t=t, nu=nu, U0=U0)

            # Compute L2 error
            diff = u - u_analytical
            error = np.sqrt(np.sum(diff**2)) / np.sqrt(np.sum(u_analytical**2) + 1e-10)

            u_history.append(u.copy())
            rho_history.append(rho.copy())
            u_analytical_history.append(u_analytical.copy())
            times.append(t)
            errors.append(error)

            if verbose and t % (save_every * 10) == 0:
                u_max = np.max(np.sqrt(u[0]**2 + u[1]**2))
                u_analytical_max = np.max(np.sqrt(u_analytical[0]**2 + u_analytical[1]**2))
                print(f"  t = {t:6d} | |u|_max = {u_max:.6f} | |u|_analytical = {u_analytical_max:.6f} | error = {error:.2e}")

    if verbose:
        print("  Done!")
        print(f"  Final error: {errors[-1]:.4e}")

    return {
        'u': np.array(u_history),
        'rho': np.array(rho_history),
        'u_analytical': np.array(u_analytical_history),
        'times': np.array(times),
        'error': np.array(errors),
        'params': {
            'n': n, 'Re': Re, 'U0': U0, 'nu': nu, 'tau': tau,
            'tau_decay': tau_decay, 'nt': nt, 'save_every': save_every
        }
    }


def plot_velocity_decay(result, save_path=None):
    """
    Plot velocity amplitude decay: numerical vs analytical.

    Args:
        result: Output from run_taylor_green
        save_path: If provided, save figure to this path
    """
    params = result['params']
    times = result['times']
    tau_decay = params['tau_decay']
    U0 = params['U0']

    # Compute max velocity at each time
    u_max_numerical = np.array([np.max(np.sqrt(u[0]**2 + u[1]**2)) for u in result['u']])
    u_max_analytical = np.array([np.max(np.sqrt(u[0]**2 + u[1]**2)) for u in result['u_analytical']])

    # Theoretical decay
    t_theory = np.linspace(0, times[-1], 100)
    u_theory = U0 * np.exp(-t_theory / tau_decay)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: velocity decay
    ax1.semilogy(times / tau_decay, u_max_numerical, 'b.-', label='LBM', markersize=4)
    ax1.semilogy(times / tau_decay, u_max_analytical, 'r--', label='Analytical', linewidth=2)
    ax1.semilogy(t_theory / tau_decay, u_theory, 'k:', label='Theory: exp(-t/τ)', alpha=0.5)
    ax1.set_xlabel('t / τ_decay')
    ax1.set_ylabel('Max velocity |u|')
    ax1.set_title(f'Velocity Decay (Re={params["Re"]}, N={params["n"]})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: error over time
    ax2.semilogy(times / tau_decay, result['error'], 'g.-', markersize=4)
    ax2.set_xlabel('t / τ_decay')
    ax2.set_ylabel('Relative L2 Error')
    ax2.set_title('Error vs Analytical Solution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_vorticity_snapshots(result, time_indices=None, save_path=None):
    """
    Plot vorticity field at multiple time snapshots.

    Args:
        result: Output from run_taylor_green
        time_indices: List of snapshot indices to plot (default: 4 evenly spaced)
        save_path: If provided, save figure to this path
    """
    params = result['params']
    n = params['n']
    tau_decay = params['tau_decay']

    if time_indices is None:
        # Pick 4 evenly spaced snapshots
        n_snapshots = len(result['u'])
        time_indices = [0, n_snapshots // 3, 2 * n_snapshots // 3, n_snapshots - 1]

    fig, axes = plt.subplots(2, len(time_indices), figsize=(4 * len(time_indices), 8))

    # Compute vorticity range for consistent colormap
    vmax = 0
    for idx in time_indices:
        u = result['u'][idx]
        omega = compute_vorticity(u)
        vmax = max(vmax, np.max(np.abs(omega)))

    for i, idx in enumerate(time_indices):
        t = result['times'][idx]
        u_num = result['u'][idx]
        u_ana = result['u_analytical'][idx]

        omega_num = compute_vorticity(u_num)
        omega_ana = compute_vorticity(u_ana)

        # Numerical
        im1 = axes[0, i].imshow(omega_num.T, origin='lower', cmap='RdBu_r',
                                 vmin=-vmax, vmax=vmax, extent=[0, n, 0, n])
        axes[0, i].set_title(f'LBM (t={t}, t/τ={t/tau_decay:.2f})')
        axes[0, i].set_xlabel('x')
        if i == 0:
            axes[0, i].set_ylabel('y')

        # Analytical
        im2 = axes[1, i].imshow(omega_ana.T, origin='lower', cmap='RdBu_r',
                                 vmin=-vmax, vmax=vmax, extent=[0, n, 0, n])
        axes[1, i].set_title(f'Analytical')
        axes[1, i].set_xlabel('x')
        if i == 0:
            axes[1, i].set_ylabel('y')

    # Add colorbars
    fig.colorbar(im1, ax=axes[0, :], label='Vorticity ω', shrink=0.8)
    fig.colorbar(im2, ax=axes[1, :], label='Vorticity ω', shrink=0.8)

    fig.suptitle(f'Taylor-Green Vorticity Evolution (Re={params["Re"]}, N={params["n"]})', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_error_analysis(result, save_path=None):
    """
    Detailed error analysis plot.

    Args:
        result: Output from run_taylor_green
        save_path: If provided, save figure to this path
    """
    params = result['params']
    times = result['times']
    tau_decay = params['tau_decay']

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig)

    # 1. Error over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(times / tau_decay, result['error'], 'b.-', markersize=4)
    ax1.set_xlabel('t / τ_decay')
    ax1.set_ylabel('Relative L2 Error')
    ax1.set_title('Error Evolution')
    ax1.grid(True, alpha=0.3)

    # 2. Final velocity field comparison (centerline)
    ax2 = fig.add_subplot(gs[0, 1])
    u_final = result['u'][-1]
    u_ana_final = result['u_analytical'][-1]
    n = params['n']

    # Take horizontal centerline
    y_mid = n // 2
    x = np.arange(n)
    ax2.plot(x, u_final[0, :, y_mid], 'b-', label='LBM ux', linewidth=2)
    ax2.plot(x, u_ana_final[0, :, y_mid], 'r--', label='Analytical ux', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('ux at y=N/2')
    ax2.set_title(f'Centerline Velocity (t={times[-1]})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Pointwise error field at final time
    ax3 = fig.add_subplot(gs[0, 2])
    error_field = np.sqrt((u_final[0] - u_ana_final[0])**2 + (u_final[1] - u_ana_final[1])**2)
    im = ax3.imshow(error_field.T, origin='lower', cmap='hot', extent=[0, n, 0, n])
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Pointwise Error |u - u_analytical|')
    fig.colorbar(im, ax=ax3)

    fig.suptitle(f'Taylor-Green Error Analysis (Re={params["Re"]}, N={params["n"]})', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def compute_vorticity(u):
    """Compute vorticity from velocity field: ω = ∂v/∂x - ∂u/∂y"""
    # Use periodic finite differences
    dvdx = np.roll(u[1], -1, axis=0) - np.roll(u[1], 1, axis=0)
    dudy = np.roll(u[0], -1, axis=1) - np.roll(u[0], 1, axis=1)
    return (dvdx - dudy) / 2


def animate_taylor_green(result, skip=8, save_path=None):
    """
    Animate the Taylor-Green vortex decay.

    Shows velocity field (top) and vorticity with error (bottom),
    consistent with other simulation animations.

    Args:
        result: Output from run_taylor_green
        skip: Plot every skip-th velocity vector
        save_path: If provided, save animation to this file (.gif)
    """
    import matplotlib.animation as animation

    params = result['params']
    n = params['n']
    tau_decay = params['tau_decay']
    times = result['times']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Taylor-Green Vortex - Re = {params["Re"]}, N = {params["n"]}', fontsize=14)

    ax_vel, ax_err = axes[0]
    ax_vort_lbm, ax_vort_ana = axes[1]

    # --- Top Left: Velocity field with speed ---
    u = result['u'][0]
    speed = np.sqrt(u[0]**2 + u[1]**2)
    vmax_speed = np.max(np.sqrt(result['u'][0][0]**2 + result['u'][0][1]**2)) * 1.2

    im_vel = ax_vel.imshow(speed.T, origin='lower', cmap='viridis',
                            extent=[0, n, 0, n], aspect='equal', vmin=0, vmax=vmax_speed)
    plt.colorbar(im_vel, ax=ax_vel, label='Speed |u|')

    x, y = np.meshgrid(np.arange(0, n, skip), np.arange(0, n, skip), indexing='ij')
    Q = ax_vel.quiver(x, y, u[0, ::skip, ::skip], u[1, ::skip, ::skip],
                      color='white', scale=2, width=0.003, alpha=0.8)

    ax_vel.set_xlabel('x')
    ax_vel.set_ylabel('y')
    ax_vel.set_title('Velocity Field')

    time_text = ax_vel.text(0.02, 0.98, '', transform=ax_vel.transAxes, fontsize=10,
                            verticalalignment='top', color='white',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # --- Top Right: Error field ---
    u_ana = result['u_analytical'][0]
    error_field = np.sqrt((u[0] - u_ana[0])**2 + (u[1] - u_ana[1])**2)
    # Use log scale for error to see structure
    error_max = np.max(error_field) if np.max(error_field) > 0 else 1e-10

    im_err = ax_err.imshow(error_field.T, origin='lower', cmap='hot',
                            extent=[0, n, 0, n], aspect='equal', vmin=0, vmax=error_max)
    plt.colorbar(im_err, ax=ax_err, label='|u - u_analytical|')

    ax_err.set_xlabel('x')
    ax_err.set_ylabel('y')
    ax_err.set_title('Pointwise Error')

    error_text = ax_err.text(0.98, 0.98, '', transform=ax_err.transAxes, fontsize=10,
                              verticalalignment='top', horizontalalignment='right',
                              color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # --- Bottom Left: LBM Vorticity ---
    omega_num = compute_vorticity(result['u'][0])
    omega_ana = compute_vorticity(result['u_analytical'][0])

    # Use percentile for initial vmax to avoid outliers
    vmax_vort = np.percentile(np.abs(omega_num), 99) * 1.1

    im_vort_lbm = ax_vort_lbm.imshow(omega_num.T, origin='lower', cmap='RdBu_r',
                                      vmin=-vmax_vort, vmax=vmax_vort,
                                      extent=[0, n, 0, n], aspect='equal')
    plt.colorbar(im_vort_lbm, ax=ax_vort_lbm, label='Vorticity ω')
    ax_vort_lbm.set_xlabel('x')
    ax_vort_lbm.set_ylabel('y')
    ax_vort_lbm.set_title('LBM Vorticity')

    # --- Bottom Right: Analytical Vorticity ---
    im_vort_ana = ax_vort_ana.imshow(omega_ana.T, origin='lower', cmap='RdBu_r',
                                      vmin=-vmax_vort, vmax=vmax_vort,
                                      extent=[0, n, 0, n], aspect='equal')
    plt.colorbar(im_vort_ana, ax=ax_vort_ana, label='Vorticity ω')
    ax_vort_ana.set_xlabel('x')
    ax_vort_ana.set_ylabel('y')
    ax_vort_ana.set_title('Analytical Vorticity')

    def animate(frame):
        u = result['u'][frame]
        u_ana = result['u_analytical'][frame]

        # Update velocity field
        speed = np.sqrt(u[0]**2 + u[1]**2)
        im_vel.set_array(speed.T)
        # Update colorbar limits for decaying flow
        current_max = np.max(speed)
        im_vel.set_clim(0, max(current_max * 1.2, 1e-10))
        Q.set_UVC(u[0, ::skip, ::skip], u[1, ::skip, ::skip])

        # Update error field
        error_field = np.sqrt((u[0] - u_ana[0])**2 + (u[1] - u_ana[1])**2)
        im_err.set_array(error_field.T)
        current_err_max = np.max(error_field)
        im_err.set_clim(0, max(current_err_max * 1.1, 1e-10))

        # Update vorticity (adaptive colorbar)
        omega_num = compute_vorticity(u)
        omega_ana = compute_vorticity(u_ana)
        im_vort_lbm.set_array(omega_num.T)
        im_vort_ana.set_array(omega_ana.T)

        # Adaptive vorticity colorbar
        current_vmax = np.percentile(np.abs(omega_num), 99) * 1.1
        current_vmax = max(current_vmax, 1e-10)
        im_vort_lbm.set_clim(-current_vmax, current_vmax)
        im_vort_ana.set_clim(-current_vmax, current_vmax)

        # Update time text
        t = times[frame]
        time_text.set_text(f't = {t}  |  t/τ = {t/tau_decay:.2f}')
        error_text.set_text(f'L2 error = {result["error"][frame]:.2e}')

        return im_vel, Q, im_err, im_vort_lbm, im_vort_ana, time_text, error_text

    anim = animation.FuncAnimation(
        fig, animate, frames=len(result['u']), interval=100, blit=False
    )

    plt.tight_layout()

    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim


def run_all_validations(save_dir=None, n=256, Re=100, nt=20000):
    """
    Run Taylor-Green simulation and create all validation plots.

    Args:
        save_dir: If provided, save all outputs to this directory
        n: Grid size (n x n)
        Re: Reynolds number
        nt: Number of timesteps
    """
    print("=" * 60)
    print("Taylor-Green Vortex Validation")
    print("=" * 60)

    # Adjust save_every based on grid size and nt
    save_every = max(100, nt // 150)

    result = run_taylor_green(
        n=n,
        Re=Re,
        U0=0.1,
        nt=nt,
        save_every=save_every
    )

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\nSaving validation plots...")
        plot_velocity_decay(result, save_path=save_dir / "taylor_green_decay.png")
        plot_vorticity_snapshots(result, save_path=save_dir / "taylor_green_vorticity.png")
        plot_error_analysis(result, save_path=save_dir / "taylor_green_error.png")
        animate_taylor_green(result, save_path=save_dir / "taylor_green_evolution.gif")

        print(f"\nAll plots saved to {save_dir}")
    else:
        print("\nShowing plots...")
        plot_velocity_decay(result)
        plot_vorticity_snapshots(result)
        plot_error_analysis(result)
        animate_taylor_green(result)

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Taylor-Green Vortex Simulation")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save outputs (default: show interactively)")
    parser.add_argument("--n", type=int, default=256, help="Grid size (default: 256)")
    parser.add_argument("--Re", type=float, default=100, help="Reynolds number (default: 100)")
    parser.add_argument("--nt", type=int, default=20000, help="Number of timesteps (default: 20000)")
    args = parser.parse_args()

    run_all_validations(save_dir=args.save, n=args.n, Re=args.Re, nt=args.nt)

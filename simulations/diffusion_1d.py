"""
Visualization for D1Q3 LBM simulations.

Creates animations showing density wave propagation and diffusion.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lbm import D1Q3, compute_equilibrium
from lbm.simulation import run


def animate_gaussian_advection(nx=256, nt=400, tau=0.8, u0=0.1, save_path=None):
    """
    Animate a Gaussian density pulse advecting and diffusing.

    Args:
        nx: Number of grid points
        nt: Number of timesteps
        tau: Relaxation time
        u0: Background velocity
        save_path: If provided, save animation to this path
    """
    # Initial condition: Gaussian density perturbation
    x = np.arange(nx)
    x0, sigma = nx / 4, 8.0
    rho_init = 1.0 + 0.3 * np.exp(-((x - x0)**2) / (2 * sigma**2))
    u_init = np.ones(nx) * u0

    f_init = compute_equilibrium(D1Q3, rho_init, u_init)
    result = run(D1Q3, f_init, tau, nt, save_every=2)

    # Set up figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(f'D1Q3 LBM: Gaussian Advection-Diffusion\n'
                 f'τ = {tau}, u₀ = {u0}, ν = {(tau - 0.5)/3:.4f}', fontsize=12)

    # Density plot
    line_rho, = ax1.plot([], [], 'b-', lw=2, label='ρ(x,t)')
    line_rho_init, = ax1.plot(x, rho_init, 'k--', lw=1, alpha=0.5, label='ρ(x,0)')
    ax1.set_xlim(0, nx)
    ax1.set_ylim(0.9, 1.4)
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density ρ')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Velocity plot
    line_u, = ax2.plot([], [], 'r-', lw=2, label='u(x,t)')
    ax2.axhline(y=u0, color='k', linestyle='--', lw=1, alpha=0.5, label=f'u₀ = {u0}')
    ax2.set_xlim(0, nx)
    ax2.set_ylim(u0 - 0.05, u0 + 0.05)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Velocity u')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=10)

    def init():
        line_rho.set_data([], [])
        line_u.set_data([], [])
        time_text.set_text('')
        return line_rho, line_u, time_text

    def animate(frame):
        rho = result.rho_history[frame]
        u = result.u_history[frame]
        t = frame * 2  # save_every=2

        line_rho.set_data(x, rho)
        line_u.set_data(x, u)
        time_text.set_text(f't = {t}')
        return line_rho, line_u, time_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(result.rho_history), interval=30, blit=True
    )

    plt.tight_layout()

    if save_path:
        anim.save(save_path, writer='pillow', fps=30)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim


def plot_diffusion_comparison(nx=256, nt=500, taus=[0.6, 0.8, 1.0, 1.5], save_path=None):
    """
    Compare diffusion rates for different relaxation times.

    Args:
        nx: Number of grid points
        nt: Number of timesteps
        taus: List of relaxation times to compare
        save_path: If provided, save figure to this path
    """
    x = np.arange(nx)
    x0, sigma = nx / 2, 10.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('D1Q3 LBM: Effect of Relaxation Time τ on Diffusion', fontsize=14)

    for ax, tau in zip(axes.flat, taus):
        # Kinematic viscosity: ν = (τ - 0.5) * cs² = (τ - 0.5) / 3
        nu = (tau - 0.5) / 3

        # Initial Gaussian (no flow)
        rho_init = 1.0 + 0.3 * np.exp(-((x - x0)**2) / (2 * sigma**2))
        u_init = np.zeros(nx)

        f_init = compute_equilibrium(D1Q3, rho_init, u_init)
        result = run(D1Q3, f_init, tau, nt, save_every=nt//5)

        # Plot snapshots
        colors = plt.cm.viridis(np.linspace(0, 1, len(result.rho_history)))
        for i, (rho, color) in enumerate(zip(result.rho_history, colors)):
            t = i * (nt // 5)
            ax.plot(x, rho, color=color, lw=1.5, label=f't={t}' if i < 6 else None)

        ax.set_xlim(0, nx)
        ax.set_ylim(0.95, 1.35)
        ax.set_xlabel('x')
        ax.set_ylabel('ρ')
        ax.set_title(f'τ = {tau}, ν = {nu:.4f}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_mass_momentum_conservation(nx=128, nt=500, tau=0.8, u0=0.1, save_path=None):
    """
    Plot mass and momentum conservation over time.

    Args:
        nx: Number of grid points
        nt: Number of timesteps
        tau: Relaxation time
        u0: Background velocity
        save_path: If provided, save figure to this path
    """
    x = np.arange(nx)
    x0, sigma = nx / 4, 8.0
    rho_init = 1.0 + 0.2 * np.exp(-((x - x0)**2) / (2 * sigma**2))
    u_init = np.ones(nx) * u0

    f_init = compute_equilibrium(D1Q3, rho_init, u_init)
    result = run(D1Q3, f_init, tau, nt, save_every=1)

    # Compute conserved quantities
    mass = np.array([np.sum(rho) for rho in result.rho_history])
    momentum = np.array([np.sum(rho * u) for rho, u in zip(result.rho_history, result.u_history)])

    t = np.arange(len(mass))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f'D1Q3 LBM: Conservation Properties\nτ = {tau}, u₀ = {u0}', fontsize=12)

    # Mass
    mass_error = (mass - mass[0]) / mass[0]
    ax1.semilogy(t, np.abs(mass_error) + 1e-16, 'b-', lw=1)
    ax1.set_ylabel('|ΔM/M₀|')
    ax1.set_title('Relative Mass Error')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-16, 1e-10)

    # Momentum
    momentum_error = (momentum - momentum[0]) / momentum[0]
    ax2.semilogy(t, np.abs(momentum_error) + 1e-16, 'r-', lw=1)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('|ΔP/P₀|')
    ax2.set_title('Relative Momentum Error')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-16, 1e-10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def run_all_visualizations(save_dir=None):
    """
    Run all visualizations.

    Args:
        save_dir: If provided, save all outputs to this directory
    """
    print("D1Q3 LBM Visualization")
    print("=" * 40)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n1. Running advection animation...")
        anim = animate_gaussian_advection(save_path=save_dir / "d1q3_advection.gif")

        print("\n2. Diffusion comparison for different tau...")
        plot_diffusion_comparison(save_path=save_dir / "d1q3_diffusion_comparison.png")

        print("\n3. Conservation properties...")
        plot_mass_momentum_conservation(save_path=save_dir / "d1q3_conservation.png")

        return anim
    else:
        # Use non-blocking mode for interactive display
        plt.ion()

        print("\n1. Running advection animation...")
        anim = animate_gaussian_advection()

        print("\n2. Diffusion comparison for different tau...")
        plot_diffusion_comparison()

        print("\n3. Conservation properties...")
        plot_mass_momentum_conservation()

        # Keep all windows open
        plt.ioff()
        plt.show()

        return anim


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="D1Q3 LBM Visualization")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save outputs (default: show interactively)")
    args = parser.parse_args()
    run_all_visualizations(save_dir=args.save)

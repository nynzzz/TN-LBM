"""
Visualization for D2Q9 LBM simulations.

Creates animations and plots for 2D flow fields.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lbm import D2Q9, compute_equilibrium
from lbm.simulation import run


def animate_gaussian_diffusion(nx=128, ny=128, nt=300, tau=0.8, save_path=None):
    """
    Animate a 2D Gaussian density blob diffusing.

    Args:
        nx, ny: Grid dimensions
        nt: Number of timesteps
        tau: Relaxation time
        save_path: If provided, save animation to this path
    """
    # Initial condition: centered Gaussian, no flow
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    x0, y0, sigma = nx/2, ny/2, 8.0
    rho_init = 1.0 + 0.3 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    u_init = np.zeros((2, nx, ny))

    f_init = compute_equilibrium(D2Q9, rho_init, u_init)
    result = run(D2Q9, f_init, tau, nt, save_every=3)

    nu = (tau - 0.5) / 3

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(f'D2Q9 LBM: Gaussian Diffusion\nτ = {tau}, ν = {nu:.4f}', fontsize=12)

    im = ax.imshow(result.rho_history[0].T, origin='lower', cmap='viridis',
                   vmin=0.95, vmax=1.35, extent=[0, nx, 0, ny])
    plt.colorbar(im, ax=ax, label='Density ρ')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', color='white',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    def animate(frame):
        im.set_array(result.rho_history[frame].T)
        t = frame * 3
        time_text.set_text(f't = {t}')
        return im, time_text

    anim = animation.FuncAnimation(
        fig, animate, frames=len(result.rho_history), interval=50, blit=True
    )

    plt.tight_layout()

    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim


def animate_shear_wave(nx=128, ny=64, nt=500, tau=0.7, save_path=None):
    """
    Animate a shear wave decaying due to viscosity.

    This is a classic LBM validation: a sinusoidal velocity profile
    should decay exponentially with rate determined by viscosity.

    Args:
        nx, ny: Grid dimensions
        nt: Number of timesteps
        tau: Relaxation time
        save_path: If provided, save animation to this path
    """
    # Initial condition: sinusoidal velocity in x-direction
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    rho_init = np.ones((nx, ny))
    u_init = np.zeros((2, nx, ny))

    # ux varies sinusoidally in y
    k = 2 * np.pi / ny  # wavenumber
    u0 = 0.05
    u_init[0] = u0 * np.sin(k * y)

    f_init = compute_equilibrium(D2Q9, rho_init, u_init)
    result = run(D2Q9, f_init, tau, nt, save_every=5)

    nu = (tau - 0.5) / 3

    # Set up figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'D2Q9 LBM: Shear Wave Decay\nτ = {tau}, ν = {nu:.4f}', fontsize=12)

    # Left: velocity field
    im = ax1.imshow(result.u_history[0][0].T, origin='lower', cmap='RdBu_r',
                    vmin=-u0, vmax=u0, extent=[0, nx, 0, ny], aspect='equal')
    plt.colorbar(im, ax=ax1, label='Velocity uₓ')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Velocity field uₓ(x,y)')

    # Right: velocity profile at x=nx/2
    line, = ax2.plot([], [], 'b-', lw=2, label='Simulation')
    y_coords = np.arange(ny)
    ax2.set_xlim(-u0*1.1, u0*1.1)
    ax2.set_ylim(0, ny)
    ax2.set_xlabel('uₓ')
    ax2.set_ylabel('y')
    ax2.set_title('Velocity profile at x = nx/2')
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                         verticalalignment='top', color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    def animate(frame):
        ux = result.u_history[frame][0]
        im.set_array(ux.T)
        line.set_data(ux[nx//2, :], y_coords)
        t = frame * 5
        time_text.set_text(f't = {t}')
        return im, line, time_text

    anim = animation.FuncAnimation(
        fig, animate, frames=len(result.u_history), interval=50, blit=True
    )

    plt.tight_layout()

    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim


def plot_velocity_field(rho, u, title="Velocity Field", skip=4):
    """
    Plot density as background with velocity vectors overlaid.

    Args:
        rho: Density field, shape (nx, ny)
        u: Velocity field, shape (2, nx, ny)
        title: Plot title
        skip: Plot every skip-th vector (for clarity)
    """
    nx, ny = rho.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

    fig, ax = plt.subplots(figsize=(10, 8))

    # Density as background
    im = ax.imshow(rho.T, origin='lower', cmap='Blues', alpha=0.7,
                   extent=[0, nx, 0, ny])
    plt.colorbar(im, ax=ax, label='Density ρ')

    # Velocity vectors
    speed = np.sqrt(u[0]**2 + u[1]**2)
    ax.quiver(x[::skip, ::skip], y[::skip, ::skip],
              u[0, ::skip, ::skip], u[1, ::skip, ::skip],
              speed[::skip, ::skip], cmap='Reds', scale=2, width=0.003)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_shear_wave_decay(nx=64, ny=64, nt=1000, tau=0.8):
    """
    Validate shear wave decay rate against analytical solution.

    For a sinusoidal velocity profile, the amplitude decays as:
    A(t) = A₀ * exp(-ν * k² * t)

    Args:
        nx, ny: Grid dimensions
        nt: Number of timesteps
        tau: Relaxation time
    """
    # Initial condition
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    rho_init = np.ones((nx, ny))
    u_init = np.zeros((2, nx, ny))

    k = 2 * np.pi / ny
    u0 = 0.01  # small amplitude for linear regime
    u_init[0] = u0 * np.sin(k * y)

    f_init = compute_equilibrium(D2Q9, rho_init, u_init)
    result = run(D2Q9, f_init, tau, nt, save_every=10)

    nu = (tau - 0.5) / 3

    # Measure amplitude over time
    times = np.arange(len(result.u_history)) * 10
    amplitudes = []
    for u in result.u_history:
        # Amplitude = max of velocity profile
        amp = np.max(np.abs(u[0, nx//2, :]))
        amplitudes.append(amp)
    amplitudes = np.array(amplitudes)

    # Analytical decay
    analytical = u0 * np.exp(-nu * k**2 * times)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Shear Wave Decay Validation\nτ = {tau}, ν = {nu:.4f}, k = {k:.4f}', fontsize=12)

    # Linear scale
    ax1.plot(times, amplitudes, 'b.-', label='LBM simulation', markersize=3)
    ax1.plot(times, analytical, 'r--', label='Analytical: $A_0 e^{-\\nu k^2 t}$', lw=2)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Amplitude A(t)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Linear scale')

    # Log scale
    ax2.semilogy(times, amplitudes, 'b.-', label='LBM simulation', markersize=3)
    ax2.semilogy(times, analytical, 'r--', label='Analytical', lw=2)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Amplitude A(t)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Log scale')

    # Compute relative error
    mask = analytical > 1e-10  # avoid division by zero
    rel_error = np.mean(np.abs(amplitudes[mask] - analytical[mask]) / analytical[mask])
    print(f"Mean relative error: {rel_error:.2e}")

    plt.tight_layout()
    plt.show()


def run_all_visualizations():
    """
    Run all visualizations simultaneously using non-blocking mode.
    """
    print("D2Q9 LBM Visualization")
    print("=" * 40)

    # Use non-blocking mode
    plt.ion()

    print("\n1. Gaussian diffusion animation...")
    anim1 = animate_gaussian_diffusion()

    print("\n2. Shear wave decay animation...")
    anim2 = animate_shear_wave()

    print("\n3. Shear wave decay validation...")
    plot_shear_wave_decay()

    # Keep all windows open
    plt.ioff()
    plt.show()

    return anim1, anim2


if __name__ == "__main__":
    run_all_visualizations()

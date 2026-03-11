"""
Lid-driven cavity flow simulation.

Classic benchmark problem: square cavity with moving top lid.
Validates against Ghia et al. (1982) benchmark data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lbm import (
    D2Q9, compute_equilibrium, stream, collide_bgk,
    apply_bounce_back, apply_bounce_back_moving_top
)
from lbm.collision import compute_moments


# Ghia et al. (1982) benchmark data for validation
# Source: Ghia, U., Ghia, K. N., & Shin, C. T. (1982).
# Data from: https://gist.github.com/ivan-pi/3e9326d18a366ffe6a8e5bfda6353219

# u velocity along vertical centerline (x = 0.5)
GHIA_Y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                   0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                   0.9688, 0.9766, 1.0000])

GHIA_U_RE100 = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434,
                          -0.10150, -0.15662, -0.21090, -0.20581, -0.13641,
                          0.00332, 0.23151, 0.68717, 0.73722, 0.78871,
                          0.84123, 1.00000])

GHIA_U_RE400 = np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612,
                          -0.24299, -0.32726, -0.17119, -0.11477, 0.02135,
                          0.16256, 0.29093, 0.55892, 0.61756, 0.68439,
                          0.75837, 1.00000])

GHIA_U_RE1000 = np.array([0.00000, -0.18109, -0.20196, -0.22220, -0.29730,
                           -0.38289, -0.27805, -0.10648, -0.06080, 0.05702,
                           0.18719, 0.33304, 0.46604, 0.51117, 0.57492,
                           0.65928, 1.00000])

GHIA_U_RE3200 = np.array([0.00000, -0.32407, -0.35344, -0.37827, -0.41933,
                           -0.34323, -0.24427, -0.86636, -0.04272, 0.07156,
                           0.19791, 0.34682, 0.46101, 0.46547, 0.48296,
                           0.53236, 1.00000])

GHIA_U_RE5000 = np.array([0.00000, -0.41165, -0.42901, -0.43643, -0.40435,
                           -0.33050, -0.22855, -0.07404, -0.03039, 0.08183,
                           0.20087, 0.33556, 0.46036, 0.45992, 0.46120,
                           0.48223, 1.00000])

# v velocity along horizontal centerline (y = 0.5)
GHIA_X = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                   0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                   0.9609, 0.9688, 1.0000])

GHIA_V_RE100 = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317,
                          0.16077, 0.17507, 0.17527, 0.05454, -0.24533,
                          -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
                          -0.05906, 0.00000])

GHIA_V_RE400 = np.array([0.00000, 0.18360, 0.19713, 0.20920, 0.22965,
                          0.28124, 0.30203, 0.30174, 0.05186, -0.38598,
                          -0.44993, -0.23827, -0.22847, -0.19254, -0.15663,
                          -0.12146, 0.00000])

GHIA_V_RE1000 = np.array([0.00000, 0.27485, 0.29012, 0.30353, 0.32627,
                           0.37095, 0.33075, 0.32235, 0.02526, -0.31966,
                           -0.42665, -0.51550, -0.39188, -0.33714, -0.27669,
                           -0.21388, 0.00000])

GHIA_V_RE3200 = np.array([0.00000, 0.39560, 0.40917, 0.41906, 0.42768,
                           0.37119, 0.29030, 0.28188, 0.00999, -0.31184,
                           -0.37401, -0.44307, -0.54053, -0.52357, -0.47425,
                           -0.39017, 0.00000])

GHIA_V_RE5000 = np.array([0.00000, 0.42447, 0.43329, 0.43648, 0.42951,
                           0.35368, 0.28066, 0.27280, 0.00945, -0.30018,
                           -0.36214, -0.41442, -0.52876, -0.55408, -0.55069,
                           -0.49774, 0.00000])


def get_ghia_data(Re):
    """Get Ghia benchmark data for given Reynolds number."""
    if Re == 100:
        return GHIA_Y, GHIA_U_RE100, GHIA_X, GHIA_V_RE100
    elif Re == 400:
        return GHIA_Y, GHIA_U_RE400, GHIA_X, GHIA_V_RE400
    elif Re == 1000:
        return GHIA_Y, GHIA_U_RE1000, GHIA_X, GHIA_V_RE1000
    elif Re == 3200:
        return GHIA_Y, GHIA_U_RE3200, GHIA_X, GHIA_V_RE3200
    elif Re == 5000:
        return GHIA_Y, GHIA_U_RE5000, GHIA_X, GHIA_V_RE5000
    else:
        return None, None, None, None


def create_cavity_walls(n: int):
    """
    Create boundary masks for lid-driven cavity.

    Args:
        n: Grid size (square domain n x n)

    Returns:
        walls: Boolean mask for stationary walls (left, right, bottom)
        lid: Boolean mask for moving lid (top)
    """
    walls = np.zeros((n, n), dtype=bool)
    lid = np.zeros((n, n), dtype=bool)

    # Bottom wall (y = 0)
    walls[:, 0] = True
    # Left wall (x = 0)
    walls[0, :] = True
    # Right wall (x = n-1)
    walls[-1, :] = True

    # Top lid (y = n-1)
    lid[:, -1] = True

    # Corner handling: corners belong to stationary walls, not lid
    lid[0, -1] = False
    lid[-1, -1] = False
    walls[0, -1] = True
    walls[-1, -1] = True

    return walls, lid


def run_lid_driven_cavity(
    n: int = 256,
    Re: float = 100,
    u_lid: float = 0.1,
    nt: int = 50000,
    save_every: int = 500,
    convergence_threshold: float = 1e-6,
    verbose: bool = True
):
    """
    Run lid-driven cavity simulation.

    Args:
        n: Grid size (square domain n x n)
        Re: Reynolds number
        u_lid: Lid velocity in lattice units
        nt: Maximum number of timesteps
        save_every: Save results every N steps
        convergence_threshold: Stop when velocity change < threshold
        verbose: Print progress

    Returns:
        Dictionary with results and parameters
    """
    # Compute physical parameters
    # For lid-driven cavity, characteristic length L = n (grid points)
    # Re = u_lid * L / nu  =>  nu = u_lid * L / Re
    L = n
    nu = u_lid * L / Re
    tau = 3 * nu + 0.5

    # For stability with high Re, we may need to reduce u_lid
    # Ma = u_lid / cs = u_lid * sqrt(3) should be < 0.3 for incompressible flow
    Ma = u_lid * np.sqrt(3)
    if Ma > 0.3:
        print(f"  Warning: Ma = {Ma:.3f} > 0.3, compressibility effects may occur")

    if verbose:
        print(f"  Grid: {n} x {n}")
        print(f"  Re = {Re}, u_lid = {u_lid}")
        print(f"  nu = {nu:.6f}, tau = {tau:.4f}")

    if tau <= 0.5:
        raise ValueError(f"tau = {tau:.4f} <= 0.5: unstable! Reduce Re or increase u_lid.")
    if tau > 2.0 and verbose:
        print(f"  Warning: tau = {tau:.4f} > 2.0 may be inaccurate")

    # Create boundary masks
    walls, lid = create_cavity_walls(n)

    # Initialize at rest
    rho = np.ones((n, n))
    u = np.zeros((2, n, n))
    f = compute_equilibrium(D2Q9, rho, u)

    # Lid velocity
    u_wall = np.array([u_lid, 0.0])

    # Storage
    u_history = []
    rho_history = []

    # For convergence check - compare CONSECUTIVE timesteps
    rho_prev, u_prev = compute_moments(D2Q9, f)

    converged = False
    final_t = nt

    for t in range(nt):
        # Collision
        f = collide_bgk(D2Q9, f, tau)

        # Streaming
        f = stream(D2Q9, f)

        # Boundary conditions
        # 1. Stationary walls (left, right, bottom)
        f = apply_bounce_back(D2Q9, f, walls)
        # 2. Moving lid (top)
        f = apply_bounce_back_moving_top(D2Q9, f, u_wall)

        # Compute moments for convergence check (every timestep)
        rho, u = compute_moments(D2Q9, f)

        # Check convergence (L2 norm of velocity change between CONSECUTIVE steps)
        du = np.sqrt(np.sum((u - u_prev)**2) / np.sum(u_prev**2 + 1e-10))

        if du < convergence_threshold:
            if verbose:
                print(f"  Converged at t = {t} (du = {du:.2e})")
            converged = True
            final_t = t
            # Save final state
            u_clean = u.copy()
            u_clean[0, walls | lid] = 0
            u_clean[1, walls | lid] = 0
            u_clean[0, lid] = u_lid
            u_history.append(u_clean)
            rho_history.append(rho.copy())
            break

        # Save for history (less frequently)
        if t % save_every == 0:
            u_clean = u.copy()
            u_clean[0, walls | lid] = 0
            u_clean[1, walls | lid] = 0
            u_clean[0, lid] = u_lid
            u_history.append(u_clean)
            rho_history.append(rho.copy())

            if verbose and t % (save_every * 10) == 0:
                u_max = np.max(np.sqrt(u[0]**2 + u[1]**2))
                print(f"  t = {t}/{nt}, du = {du:.2e}, u_max = {u_max:.4f}")

        # Update previous for next iteration
        u_prev = u.copy()

        # Stability check
        if np.any(np.isnan(rho)) or np.max(np.abs(u)) > 0.5:
            print(f"  UNSTABLE at t={t}!")
            break

    if verbose and not converged:
        print(f"  Finished {nt} timesteps (not fully converged)")

    return {
        'u': np.array(u_history),
        'rho': np.array(rho_history),
        'walls': walls,
        'lid': lid,
        'converged': converged,
        'final_t': final_t,
        'params': {
            'n': n, 'Re': Re, 'tau': tau, 'nu': nu,
            'u_lid': u_lid, 'save_every': save_every
        }
    }


def plot_streamlines(result, save_path=None):
    """Plot streamlines of the final flow field."""
    params = result['params']
    n = params['n']

    u = result['u'][-1]
    speed = np.sqrt(u[0]**2 + u[1]**2)

    fig, ax = plt.subplots(figsize=(8, 8))

    x = np.arange(n)
    y = np.arange(n)

    # Speed as background
    im = ax.imshow(speed.T, origin='lower', cmap='viridis',
                   extent=[0, 1, 0, 1], aspect='equal')
    plt.colorbar(im, ax=ax, label='Speed |u|/U')

    # Streamlines
    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing='ij')
    ax.streamplot(X.T, Y.T, u[0].T, u[1].T, color='white', density=2, linewidth=0.5)

    ax.set_xlabel('x/L')
    ax.set_ylabel('y/L')
    ax.set_title(f"Lid-Driven Cavity - Re = {params['Re']}")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_velocity_field(result, skip=8, save_path=None):
    """Plot velocity vectors over speed magnitude."""
    params = result['params']
    n = params['n']

    u = result['u'][-1]
    speed = np.sqrt(u[0]**2 + u[1]**2)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Speed as background
    im = ax.imshow(speed.T, origin='lower', cmap='viridis',
                   extent=[0, 1, 0, 1], aspect='equal')
    plt.colorbar(im, ax=ax, label='Speed |u|/U')

    # Velocity vectors
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x[::skip], y[::skip], indexing='ij')
    ax.quiver(X, Y, u[0, ::skip, ::skip], u[1, ::skip, ::skip],
              color='white', scale=2, width=0.003, alpha=0.7)

    ax.set_xlabel('x/L')
    ax.set_ylabel('y/L')
    ax.set_title(f"Velocity Field - Re = {params['Re']}")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_centerline_validation(result, save_path=None):
    """
    Compare velocity profiles along centerlines with Ghia et al. benchmark.
    """
    params = result['params']
    n = params['n']
    Re = params['Re']
    u_lid = params['u_lid']

    u = result['u'][-1]

    # Normalized coordinates
    y_sim = np.linspace(0, 1, n)
    x_sim = np.linspace(0, 1, n)

    # Velocity along vertical centerline (x = 0.5)
    mid_x = n // 2
    u_centerline = u[0, mid_x, :] / u_lid  # Normalize by lid velocity

    # Velocity along horizontal centerline (y = 0.5)
    mid_y = n // 2
    v_centerline = u[1, :, mid_y] / u_lid  # Normalize by lid velocity

    # Get benchmark data
    ghia_y, ghia_u, ghia_x, ghia_v = get_ghia_data(Re)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Lid-Driven Cavity Validation - Re = {Re}", fontsize=14)

    # u vs y at x = 0.5
    ax1.plot(u_centerline, y_sim, 'b-', lw=2, label='LBM simulation')
    if ghia_y is not None:
        ax1.plot(ghia_u, ghia_y, 'ro', markersize=6, label='Ghia et al. (1982)')
    ax1.set_xlabel('u / U')
    ax1.set_ylabel('y / L')
    ax1.set_title('u-velocity along vertical centerline (x = 0.5)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 1.1)

    # v vs x at y = 0.5
    ax2.plot(x_sim, v_centerline, 'b-', lw=2, label='LBM simulation')
    if ghia_x is not None:
        ax2.plot(ghia_x, ghia_v, 'ro', markersize=6, label='Ghia et al. (1982)')
    ax2.set_xlabel('x / L')
    ax2.set_ylabel('v / U')
    ax2.set_title('v-velocity along horizontal centerline (y = 0.5)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.6, 0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def animate_development(result, skip=16, save_path=None):
    """Animate the flow development over time."""
    params = result['params']
    n = params['n']
    save_every = params['save_every']

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(f"Lid-Driven Cavity Development - Re = {params['Re']}", fontsize=12)

    u = result['u'][0]
    speed = np.sqrt(u[0]**2 + u[1]**2)
    vmax = np.max(np.sqrt(result['u'][-1][0]**2 + result['u'][-1][1]**2)) * 1.1

    im = ax.imshow(speed.T, origin='lower', cmap='viridis',
                   extent=[0, 1, 0, 1], aspect='equal', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Speed |u|/U')

    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x[::skip], y[::skip], indexing='ij')
    Q = ax.quiver(X, Y, u[0, ::skip, ::skip], u[1, ::skip, ::skip],
                  color='white', scale=2, width=0.003, alpha=0.7)

    ax.set_xlabel('x/L')
    ax.set_ylabel('y/L')

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', color='white',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    def animate(frame):
        u = result['u'][frame]
        speed = np.sqrt(u[0]**2 + u[1]**2)
        im.set_array(speed.T)
        Q.set_UVC(u[0, ::skip, ::skip], u[1, ::skip, ::skip])
        time_text.set_text(f't = {frame * save_every}')
        return im, Q, time_text

    anim = animation.FuncAnimation(
        fig, animate, frames=len(result['u']), interval=100, blit=True
    )

    plt.tight_layout()

    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim


def run_all_visualizations(save_dir=None, Re=100):
    """
    Run lid-driven cavity simulation and create visualizations.

    Args:
        save_dir: If provided, save all outputs to this directory
        Re: Reynolds number to simulate
    """
    print("=" * 60)
    print(f"Lid-Driven Cavity Simulation - Re = {Re}")
    print("=" * 60)

    result = run_lid_driven_cavity(
        n=256,
        Re=Re,
        u_lid=0.1,
        nt=150000,
        save_every=1000,
        verbose=True
    )

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\nSaving visualizations...")
        plot_streamlines(result, save_path=save_dir / "cavity_streamlines.png")
        plot_velocity_field(result, save_path=save_dir / "cavity_velocity_field.png")
        plot_centerline_validation(result, save_path=save_dir / "cavity_validation.png")
        anim = animate_development(result, save_path=save_dir / "cavity_development.gif")

        return result, anim
    else:
        print("\nShowing visualizations...")
        plt.ion()
        plot_streamlines(result)
        plot_velocity_field(result)
        plot_centerline_validation(result)
        anim = animate_development(result)
        plt.ioff()
        plt.show()

        return result, anim


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lid-Driven Cavity Simulation")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save outputs (default: show interactively)")
    parser.add_argument("--Re", type=float, default=100,
                        help="Reynolds number (default: 100)")
    args = parser.parse_args()
    run_all_visualizations(save_dir=args.save, Re=args.Re)

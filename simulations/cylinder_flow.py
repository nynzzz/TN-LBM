"""
Flow around a cylinder using inlet/outlet boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lbm import (
    D2Q9, compute_equilibrium, stream, collide_bgk,
    apply_bounce_back, equilibrium_inlet_left, extrapolation_outlet_right,
    create_cylinder_mask, create_channel_walls
)
from lbm.collision import compute_moments


def run_cylinder_flow(
    n: int = None,
    nx: int = 420,
    ny: int = 180,
    cylinder_x: float = None,
    cylinder_y: float = None,
    cylinder_r: float = None,
    u_inlet: float = 0.04,
    Re: float = 100,
    nt: int = 30000,
    save_every: int = 100,
    verbose: bool = True
):
    """
    Run flow around a cylinder simulation.

    Uses inlet/outlet with bounce-back on top/bottom walls and cylinder surface.

    Args:
        n: If specified, use square domain n x n (for TN-LBM compatibility).
           Should be a power of 2 (e.g., 128, 256, 512).
        nx, ny: Grid dimensions (used if n is None)
        cylinder_x, cylinder_y: Cylinder center position
        cylinder_r: Cylinder radius
        u_inlet: Inlet velocity
        Re: Reynolds number (based on cylinder diameter)
        nt: Number of timesteps
        save_every: Save every N steps
        verbose: Print progress
    """
    # Square domain mode (for TN-LBM)
    if n is not None:
        nx = n
        ny = n
        # For square domain, use smaller cylinder for reasonable blockage
        # D/N ~ 1/8 gives ~12.5% blockage ratio
        if cylinder_r is None:
            cylinder_r = n / 16  # D = N/8
        if cylinder_x is None:
            cylinder_x = n / 4  # 2D from inlet (quarter of domain)
        if cylinder_y is None:
            cylinder_y = n / 2  # centered
    else:
        # Rectangular domain (original behavior)
        if cylinder_r is None:
            cylinder_r = ny / 10
        if cylinder_x is None:
            cylinder_x = 5 * (2 * cylinder_r)  # 5 diameters from inlet
        if cylinder_y is None:
            cylinder_y = ny / 2  # centered

    D = 2 * cylinder_r
    nu = u_inlet * D / Re
    tau = 3 * nu + 0.5

    if verbose:
        print(f"  Grid: {nx} x {ny}")
        print(f"  Cylinder: center=({cylinder_x:.1f}, {cylinder_y:.1f}), r={cylinder_r:.1f}, D={D:.1f}")
        print(f"  Domain: {cylinder_x/D:.1f}D upstream, {(nx - cylinder_x)/D:.1f}D downstream")
        print(f"  Re = {Re}, u_inlet = {u_inlet}, tau = {tau:.4f}, nu = {nu:.6f}")

    if tau <= 0.5:
        raise ValueError(f"tau = {tau:.4f} <= 0.5: unstable! Reduce Re or increase u_inlet.")
    if tau > 2.0:
        print(f"  Warning: tau = {tau:.4f} > 2.0 may be inaccurate")

    # Create geometry masks
    cylinder = create_cylinder_mask(nx, ny, cylinder_x, cylinder_y, cylinder_r)
    walls = create_channel_walls(nx, ny)
    solid = cylinder | walls

    # Initialize with uniform flow
    rho = np.ones((nx, ny))
    u = np.zeros((2, nx, ny))
    u[0, :, :] = u_inlet
    u[0, solid] = 0
    u[1, solid] = 0

    f = compute_equilibrium(D2Q9, rho, u)

    # Inlet velocity profile with small asymmetry to trigger instability
    y = np.arange(ny)
    u_inlet_profile = np.zeros((2, ny))
    perturbation = 0.01 * u_inlet * (y - ny/2) / (ny/2)
    u_inlet_profile[0, :] = u_inlet + perturbation

    # Storage
    rho_history = []
    u_history = []
    vorticity_history = []

    for t in range(nt):
        # Collision
        f = collide_bgk(D2Q9, f, tau)

        # Streaming (periodic - BCs will override at boundaries)
        f = stream(D2Q9, f)

        # Boundary conditions (order matters!)
        # 1. Bounce-back on walls and cylinder
        f = apply_bounce_back(D2Q9, f, solid)
        # 2. Inlet: set equilibrium with prescribed velocity
        f = equilibrium_inlet_left(D2Q9, f, u_inlet_profile, rho_inlet=1.0)
        # 3. Outlet: extrapolate from interior (simple and stable)
        f = extrapolation_outlet_right(f)

        if t % save_every == 0:
            rho, u = compute_moments(D2Q9, f)

            # Detailed stability check
            has_nan = np.any(np.isnan(rho))
            max_u = np.max(np.abs(u))
            min_rho = np.min(rho)
            max_rho = np.max(rho)

            if has_nan or max_u > 0.5 or min_rho < 0 or max_rho > 2:
                print(f"  UNSTABLE at t={t}!")
                print(f"    has_nan={has_nan}, max_u={max_u:.4f}")
                print(f"    rho: min={min_rho:.4f}, max={max_rho:.4f}")
                # Find where the problem is
                if has_nan:
                    nan_locs = np.where(np.isnan(rho))
                    print(f"    NaN locations: x={nan_locs[0][:5]}, y={nan_locs[1][:5]}")
                if max_u > 0.5:
                    max_loc = np.unravel_index(np.argmax(np.abs(u)), u.shape)
                    print(f"    Max velocity at: {max_loc}")
                break

            u[0, solid] = 0
            u[1, solid] = 0

            rho_history.append(rho.copy())
            u_history.append(u.copy())

            omega = np.zeros_like(rho)
            omega[1:-1, 1:-1] = (
                (u[1, 2:, 1:-1] - u[1, :-2, 1:-1]) / 2 -
                (u[0, 1:-1, 2:] - u[0, 1:-1, :-2]) / 2
            )
            vorticity_history.append(omega.copy())

            if verbose and t % (save_every * 10) == 0:
                u_mean = np.mean(u[0, ~solid])
                print(f"  t = {t}/{nt}, u_mean = {u_mean:.4f}")

    if verbose:
        print("  Done!")

    return {
        'rho': np.array(rho_history),
        'u': np.array(u_history),
        'vorticity': np.array(vorticity_history),
        'solid': solid,
        'cylinder': cylinder,
        'params': {
            'n': n, 'nx': nx, 'ny': ny, 'Re': Re, 'tau': tau, 'nu': nu,
            'u_inlet': u_inlet, 'cylinder_x': cylinder_x,
            'cylinder_y': cylinder_y, 'cylinder_r': cylinder_r,
            'save_every': save_every
        }
    }


def animate_cylinder_flow(result, save_path=None):
    """
    Animate the flow field (vorticity).

    Args:
        result: Output from run_cylinder_flow
        save_path: If provided, save animation to this file
    """
    params = result['params']
    nx, ny = params['nx'], params['ny']
    cx, cy, r = params['cylinder_x'], params['cylinder_y'], params['cylinder_r']

    # Dynamic figure size based on aspect ratio
    aspect = nx / ny
    fig_width = min(14, 8 * aspect)
    fig_height = max(4, 8 / aspect)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.suptitle(f"Flow Around Cylinder - Re = {params['Re']}", fontsize=12)

    # Vorticity colormap
    vmax = np.percentile(np.abs(result['vorticity']), 99)
    im = ax.imshow(result['vorticity'][0].T, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, extent=[0, nx, 0, ny], aspect='equal')
    plt.colorbar(im, ax=ax, label='Vorticity ω')

    # Draw cylinder
    circle = Circle((cx, cy), r, fill=True, color='gray', ec='black', lw=2)
    ax.add_patch(circle)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                        color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    def animate(frame):
        im.set_array(result['vorticity'][frame].T)
        time_text.set_text(f"t = {frame * 100}")
        return im, time_text

    anim = animation.FuncAnimation(
        fig, animate, frames=len(result['vorticity']), interval=50, blit=True
    )

    plt.tight_layout()

    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim


def animate_velocity_field(result, skip=8, save_path=None):
    """
    Animate velocity field (speed magnitude).

    Args:
        result: Output from run_cylinder_flow
        skip: Plot every skip-th vector
        save_path: If provided, save animation to this file
    """
    params = result['params']
    nx, ny = params['nx'], params['ny']
    cx, cy, r = params['cylinder_x'], params['cylinder_y'], params['cylinder_r']
    save_every = params.get('save_every', 100)

    # Dynamic figure size based on aspect ratio
    aspect = nx / ny
    fig_width = min(14, 8 * aspect)
    fig_height = max(4, 8 / aspect)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.suptitle(f"Velocity Field - Re = {params['Re']}", fontsize=12)

    # Initial speed
    u = result['u'][0]
    speed = np.sqrt(u[0]**2 + u[1]**2)
    # Compute vmax from ALL frames for truly static colorbar
    all_speeds = np.sqrt(result['u'][:, 0]**2 + result['u'][:, 1]**2)
    vmax = np.percentile(all_speeds, 99) * 1.1

    im = ax.imshow(speed.T, origin='lower', cmap='viridis',
                   extent=[0, nx, 0, ny], aspect='equal', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Speed |u|')

    # Velocity vectors
    x, y = np.meshgrid(np.arange(0, nx, skip), np.arange(0, ny, skip), indexing='ij')
    Q = ax.quiver(x, y, u[0, ::skip, ::skip], u[1, ::skip, ::skip],
                  color='white', scale=2, width=0.002, alpha=0.8)

    # Cylinder
    circle = Circle((cx, cy), r, fill=True, color='gray', ec='black', lw=2)
    ax.add_patch(circle)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                        color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    def animate(frame):
        u = result['u'][frame]
        speed = np.sqrt(u[0]**2 + u[1]**2)
        im.set_array(speed.T)
        Q.set_UVC(u[0, ::skip, ::skip], u[1, ::skip, ::skip])
        time_text.set_text(f't = {frame * save_every}')
        return im, Q, time_text

    anim = animation.FuncAnimation(
        fig, animate, frames=len(result['u']), interval=50, blit=True
    )

    plt.tight_layout()

    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim


def plot_velocity_field(result, time_idx=-1, skip=8, save_path=None):
    """
    Plot velocity vectors over speed magnitude (static).

    Args:
        result: Output from run_cylinder_flow
        time_idx: Which timestep to plot (-1 for last)
        skip: Plot every skip-th vector
        save_path: If provided, save figure to this path
    """
    params = result['params']
    nx, ny = params['nx'], params['ny']
    cx, cy, r = params['cylinder_x'], params['cylinder_y'], params['cylinder_r']

    u = result['u'][time_idx]
    speed = np.sqrt(u[0]**2 + u[1]**2)

    # Dynamic figure size based on aspect ratio
    aspect = nx / ny
    fig_width = min(14, 8 * aspect)
    fig_height = max(4, 8 / aspect)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Speed as background
    im = ax.imshow(speed.T, origin='lower', cmap='viridis',
                   extent=[0, nx, 0, ny], aspect='equal')
    plt.colorbar(im, ax=ax, label='Speed |u|')

    # Velocity vectors (same style as animation)
    x, y = np.meshgrid(np.arange(0, nx, skip), np.arange(0, ny, skip), indexing='ij')
    ax.quiver(x, y, u[0, ::skip, ::skip], u[1, ::skip, ::skip],
              color='white', scale=8, width=0.001, alpha=0.7)

    # Cylinder
    circle = Circle((cx, cy), r, fill=True, color='gray', ec='black', lw=2)
    ax.add_patch(circle)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Velocity Field - Re = {params['Re']}")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_streamlines(result, time_idx=-1, save_path=None):
    """
    Plot streamlines.

    Args:
        result: Output from run_cylinder_flow
        time_idx: Which timestep to plot
        save_path: If provided, save figure to this path
    """
    params = result['params']
    nx, ny = params['nx'], params['ny']
    cx, cy, r = params['cylinder_x'], params['cylinder_y'], params['cylinder_r']

    u = result['u'][time_idx]
    speed = np.sqrt(u[0]**2 + u[1]**2)

    # Dynamic figure size based on aspect ratio
    aspect = nx / ny
    fig_width = min(14, 8 * aspect)
    fig_height = max(4, 8 / aspect)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(nx)
    y = np.arange(ny)

    # Speed as background
    ax.imshow(speed.T, origin='lower', cmap='Blues', alpha=0.5,
              extent=[0, nx, 0, ny], aspect='equal')

    # Streamlines
    ax.streamplot(x, y, u[0].T, u[1].T, color=speed.T, cmap='viridis',
                  density=2, linewidth=0.8)

    # Cylinder
    circle = Circle((cx, cy), r, fill=True, color='gray', ec='black', lw=2)
    ax.add_patch(circle)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Streamlines - Re = {params['Re']}")
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def animate_all(result, skip=8, save_path=None):
    """
    Animate velocity field and vorticity side by side in one figure.

    Args:
        result: Output from run_cylinder_flow
        skip: Plot every skip-th vector for velocity
        save_path: If provided, save animation to this file
    """
    params = result['params']
    nx, ny = params['nx'], params['ny']
    cx, cy, r = params['cylinder_x'], params['cylinder_y'], params['cylinder_r']
    save_every = params.get('save_every', 100)

    # Dynamic figure size based on aspect ratio
    aspect = nx / ny
    fig_width = min(14, 10 * aspect)
    fig_height = max(8, 10 / aspect)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    fig.suptitle(f"Cylinder Flow - Re = {params['Re']}", fontsize=14)

    # --- Top: Velocity field ---
    u = result['u'][0]
    speed = np.sqrt(u[0]**2 + u[1]**2)
    # Compute vmax from ALL frames for truly static colorbar
    all_speeds = np.sqrt(result['u'][:, 0]**2 + result['u'][:, 1]**2)
    vmax_speed = np.percentile(all_speeds, 99) * 1.1

    im1 = ax1.imshow(speed.T, origin='lower', cmap='viridis',
                     extent=[0, nx, 0, ny], aspect='equal', vmin=0, vmax=vmax_speed)
    plt.colorbar(im1, ax=ax1, label='Speed |u|')

    x, y = np.meshgrid(np.arange(0, nx, skip), np.arange(0, ny, skip), indexing='ij')
    Q = ax1.quiver(x, y, u[0, ::skip, ::skip], u[1, ::skip, ::skip],
                   color='white', scale=8, width=0.001, alpha=0.7)

    circle1 = Circle((cx, cy), r, fill=True, color='gray', ec='black', lw=2)
    ax1.add_patch(circle1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Velocity Field')

    # Time text on ax1 (not fig)
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12,
                         verticalalignment='top', color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # --- Bottom: Vorticity ---
    vmax_vort = np.percentile(np.abs(result['vorticity']), 99)
    im2 = ax2.imshow(result['vorticity'][0].T, origin='lower', cmap='RdBu_r',
                     vmin=-vmax_vort, vmax=vmax_vort, extent=[0, nx, 0, ny], aspect='equal')
    plt.colorbar(im2, ax=ax2, label='Vorticity ω')

    circle2 = Circle((cx, cy), r, fill=True, color='gray', ec='black', lw=2)
    ax2.add_patch(circle2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Vorticity')

    def animate(frame):
        u = result['u'][frame]
        speed = np.sqrt(u[0]**2 + u[1]**2)
        im1.set_array(speed.T)
        Q.set_UVC(u[0, ::skip, ::skip], u[1, ::skip, ::skip])

        im2.set_array(result['vorticity'][frame].T)

        time_text.set_text(f't = {frame * save_every}')
        return im1, Q, im2, time_text

    anim = animation.FuncAnimation(
        fig, animate, frames=len(result['u']), interval=50, blit=True
    )

    plt.tight_layout()

    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim


def run_all_visualizations(save_dir=None, n=None, Re=100, nt=30000):
    """
    Run cylinder flow simulation and create visualizations.

    Args:
        save_dir: If provided, save all outputs to this directory
        n: If specified, use square n×n domain (for TN-LBM compatibility)
        Re: Reynolds number
        nt: Number of timesteps
    """
    print("=" * 60)
    print("Cylinder Flow Simulation")
    print("=" * 60)

    # Adjust save_every based on nt
    save_every = max(100, nt // 150)

    if n is not None:
        # Square domain mode (TN-compatible)
        print(f"  Mode: Square domain ({n}×{n}) for TN-LBM")
        result = run_cylinder_flow(
            n=n,
            Re=Re,
            u_inlet=0.1,
            nt=nt,
            save_every=save_every
        )
    else:
        # Rectangular domain mode (original)
        print("  Mode: Rectangular domain (600×180)")
        result = run_cylinder_flow(
            nx=600,
            ny=180,
            Re=Re,
            u_inlet=0.1,
            nt=nt,
            save_every=save_every
        )

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\nSaving visualizations...")
        anim = animate_all(result, save_path=save_dir / "cylinder_flow.gif")
        plot_velocity_field(result, save_path=save_dir / "cylinder_velocity_field.png")
        plot_streamlines(result, save_path=save_dir / "cylinder_streamlines.png")

        return result, anim
    else:
        print("\nAnimating results...")
        anim = animate_all(result)
        return result, anim


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cylinder Flow Simulation")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save outputs (default: show interactively)")
    parser.add_argument("--n", type=int, default=None,
                        help="Square domain size (e.g., 256 for 256×256). If not specified, uses rectangular 600×180")
    parser.add_argument("--Re", type=float, default=100,
                        help="Reynolds number (default: 100)")
    parser.add_argument("--nt", type=int, default=30000,
                        help="Number of timesteps (default: 30000)")
    args = parser.parse_args()
    run_all_visualizations(save_dir=args.save, n=args.n, Re=args.Re, nt=args.nt)

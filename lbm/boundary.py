"""
Boundary conditions for LBM.

Implements:
  - Bounce-back (no-slip walls)
  - Equilibrium inlet (velocity BC)
  - Extrapolation outlet (open boundary)
  - Periodic (already in streaming.py)
"""

import numpy as np
from .lattice import Lattice


def apply_bounce_back(lattice: Lattice, f: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """
    Apply bounce-back boundary condition for solid walls.

    Uses "on-node" bounce-back: at solid nodes, populations are reflected.
    After streaming, f_i at a solid node came from the fluid in direction -c_i.
    We bounce it back so it will stream back to that fluid node.

    Args:
        lattice: Lattice definition
        f: Distribution functions after streaming, shape (q, nx, ny)
        solid: Boolean mask of solid nodes, shape (nx, ny)

    Returns:
        f_new: Distributions with bounce-back applied
    """
    f_new = f.copy()

    for i in range(lattice.q):
        opp = lattice.opposite[i]
        f_new[i, solid] = f[opp, solid]

    return f_new


def apply_bounce_back_moving(lattice: Lattice, f: np.ndarray, solid: np.ndarray,
                              wall_velocity: np.ndarray, rho_wall: float = 1.0) -> np.ndarray:
    """
    Apply bounce-back with moving wall (e.g., lid-driven cavity top wall).

    For a moving wall, the bounce-back includes a momentum correction:
    f_i = f_opp + 2 * w_i * rho * (c_i · u_wall) / cs²

    Args:
        lattice: Lattice definition
        f: Distribution functions after streaming
        solid: Boolean mask of wall nodes
        wall_velocity: Wall velocity, shape (2,) for 2D
        rho_wall: Density at wall (usually 1.0)

    Returns:
        f_new: Distributions with moving bounce-back applied
    """
    if lattice.d != 2:
        raise ValueError("Moving bounce-back only implemented for 2D")

    f_new = f.copy()

    for i in range(lattice.q):
        opp = lattice.opposite[i]
        # c_i · u_wall
        cu = lattice.c[i, 0] * wall_velocity[0] + lattice.c[i, 1] * wall_velocity[1]
        # Momentum correction term
        correction = 2 * lattice.w[i] * rho_wall * cu / lattice.cs2
        f_new[i, solid] = f[opp, solid] - correction

    return f_new


def extrapolation_outlet_right(f: np.ndarray) -> np.ndarray:
    """
    Extrapolation outlet boundary condition at right boundary (x=nx-1).

    Copies all populations from the second-to-last column to the last column.
    Simple and stable for fully-developed flow.

    Args:
        f: Distribution functions, shape (q, nx, ny)

    Returns:
        f_new: Distributions with outlet BC applied
    """
    f_new = f.copy()
    f_new[:, -1, :] = f[:, -2, :]
    return f_new


def equilibrium_inlet_left(lattice: Lattice, f: np.ndarray,
                           u_inlet: np.ndarray, rho_inlet: float = 1.0) -> np.ndarray:
    """
    Equilibrium inlet boundary condition at left boundary (x=0).

    Sets all populations at inlet to equilibrium with prescribed velocity and density.

    Args:
        lattice: Lattice definition
        f: Distribution functions, shape (q, nx, ny)
        u_inlet: Inlet velocity, shape (2, ny) or (2,) for uniform
        rho_inlet: Inlet density

    Returns:
        f_new: Distributions with inlet BC applied
    """
    from .equilibrium import compute_equilibrium

    f_new = f.copy()
    ny = f.shape[2]

    # Handle uniform vs varying inlet velocity
    if u_inlet.ndim == 1:
        rho_bc = np.ones(ny) * rho_inlet
        u_bc = np.zeros((2, ny))
        u_bc[0, :] = u_inlet[0]
        u_bc[1, :] = u_inlet[1]
    else:
        rho_bc = np.ones(ny) * rho_inlet
        u_bc = u_inlet

    # Compute equilibrium for 1D slice (need to reshape for compute_equilibrium)
    # compute_equilibrium expects rho: (nx, ny), u: (2, nx, ny)
    # We'll compute it manually for efficiency
    for i in range(lattice.q):
        cu = lattice.c[i, 0] * u_bc[0, :] + lattice.c[i, 1] * u_bc[1, :]
        usq = u_bc[0, :]**2 + u_bc[1, :]**2
        f_new[i, 0, :] = lattice.w[i] * rho_bc * (
            1.0 + cu / lattice.cs2 +
            (cu**2) / (2 * lattice.cs2**2) -
            usq / (2 * lattice.cs2)
        )

    return f_new


def create_cylinder_mask(nx: int, ny: int, cx: float, cy: float, r: float) -> np.ndarray:
    """
    Create a boolean mask for a circular cylinder.

    Args:
        nx, ny: Grid dimensions
        cx, cy: Cylinder center coordinates
        r: Cylinder radius

    Returns:
        solid: Boolean mask, True inside cylinder
    """
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    return (x - cx)**2 + (y - cy)**2 <= r**2


def create_channel_walls(nx: int, ny: int) -> np.ndarray:
    """
    Create a boolean mask for top and bottom channel walls.

    Args:
        nx, ny: Grid dimensions

    Returns:
        solid: Boolean mask, True at y=0 and y=ny-1
    """
    solid = np.zeros((nx, ny), dtype=bool)
    solid[:, 0] = True
    solid[:, -1] = True
    return solid

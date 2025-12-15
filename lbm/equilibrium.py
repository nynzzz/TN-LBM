"""
Equilibrium distribution functions for LBM.

The equilibrium distribution is:
  f_i^eq = w_i * rho * (1 + (c_i · u)/cs2 + (c_i · u)^2/(2*cs2^2) - u^2/(2*cs2))
"""

import numpy as np
from .lattice import Lattice


def compute_equilibrium(lattice: Lattice, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Compute equilibrium distribution for a given lattice.

    Args:
        lattice: Lattice definition (D1Q3 or D2Q9)
        rho: Density field
             - D1Q3: shape (nx,)
             - D2Q9: shape (nx, ny)
        u: Velocity field
           - D1Q3: shape (nx,)
           - D2Q9: shape (2, nx, ny) for (ux, uy)

    Returns:
        f_eq: Equilibrium distributions
              - D1Q3: shape (q, nx)
              - D2Q9: shape (q, nx, ny)
    """
    if lattice.d == 1:
        return _equilibrium_1d(lattice, rho, u)
    elif lattice.d == 2:
        return _equilibrium_2d(lattice, rho, u)
    else:
        raise ValueError(f"Unsupported lattice dimension: {lattice.d}")


def _equilibrium_1d(lattice: Lattice, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Equilibrium for 1D lattices."""
    q = lattice.q
    nx = len(rho)
    f_eq = np.zeros((q, nx))

    u_sq = u**2

    for i in range(q):
        cu = lattice.c[i] * u
        f_eq[i] = lattice.w[i] * rho * (
            1.0
            + cu / lattice.cs2
            + cu**2 / (2.0 * lattice.cs2**2)
            - u_sq / (2.0 * lattice.cs2)
        )

    return f_eq


def _equilibrium_2d(lattice: Lattice, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Equilibrium for 2D lattices."""
    q = lattice.q
    nx, ny = rho.shape
    f_eq = np.zeros((q, nx, ny))

    # u has shape (2, nx, ny): u[0] = ux, u[1] = uy
    u_sq = u[0]**2 + u[1]**2

    for i in range(q):
        # c_i · u = cx*ux + cy*uy
        cu = lattice.c[i, 0] * u[0] + lattice.c[i, 1] * u[1]
        f_eq[i] = lattice.w[i] * rho * (
            1.0
            + cu / lattice.cs2
            + cu**2 / (2.0 * lattice.cs2**2)
            - u_sq / (2.0 * lattice.cs2)
        )

    return f_eq

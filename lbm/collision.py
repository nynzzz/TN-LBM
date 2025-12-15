"""
Collision operators for LBM.

Currently implements:
  - BGK (Bhatnagar-Gross-Krook) single relaxation time
"""

import numpy as np
from .lattice import Lattice
from .equilibrium import compute_equilibrium


def compute_moments_1d(lattice: Lattice, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute macroscopic moments from distribution functions (1D).

    Args:
        lattice: Lattice definition
        f: Distribution functions, shape (q, nx)

    Returns:
        rho: Density, shape (nx,)
        u: Velocity, shape (nx,)
    """
    rho = np.sum(f, axis=0)
    u = np.sum(lattice.c[:, np.newaxis] * f, axis=0) / rho
    return rho, u


def compute_moments_2d(lattice: Lattice, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute macroscopic moments from distribution functions (2D).

    Args:
        lattice: Lattice definition
        f: Distribution functions, shape (q, nx, ny)

    Returns:
        rho: Density, shape (nx, ny)
        u: Velocity, shape (2, nx, ny)
    """
    rho = np.sum(f, axis=0)
    u = np.zeros((2, *rho.shape))
    for i in range(lattice.q):
        u[0] += lattice.c[i, 0] * f[i]
        u[1] += lattice.c[i, 1] * f[i]
    u /= rho
    return rho, u


def compute_moments(lattice: Lattice, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute macroscopic moments from distribution functions.

    Args:
        lattice: Lattice definition
        f: Distribution functions

    Returns:
        rho: Density field
        u: Velocity field
    """
    if lattice.d == 1:
        return compute_moments_1d(lattice, f)
    elif lattice.d == 2:
        return compute_moments_2d(lattice, f)
    else:
        raise ValueError(f"Unsupported lattice dimension: {lattice.d}")


def collide_bgk(lattice: Lattice, f: np.ndarray, tau: float) -> np.ndarray:
    """
    BGK collision operator.

    f_i* = f_i - (f_i - f_i^eq) / tau

    Args:
        lattice: Lattice definition
        f: Distribution functions
        tau: Relaxation time (must be > 0.5 for stability)

    Returns:
        f_post: Post-collision distributions
    """
    rho, u = compute_moments(lattice, f)
    f_eq = compute_equilibrium(lattice, rho, u)
    return f - (f - f_eq) / tau

"""
Streaming operators for LBM.

Streaming moves populations to neighboring nodes according to their velocities:
  f_i(x + c_i, t+1) = f_i*(x, t)
"""

import numpy as np
from .lattice import Lattice


def stream(lattice: Lattice, f: np.ndarray) -> np.ndarray:
    """
    Streaming step with periodic boundary conditions.

    Args:
        lattice: Lattice definition
        f: Post-collision distributions

    Returns:
        f_streamed: Distributions after streaming
    """
    if lattice.d == 1:
        return _stream_1d(lattice, f)
    elif lattice.d == 2:
        return _stream_2d(lattice, f)
    else:
        raise ValueError(f"Unsupported lattice dimension: {lattice.d}")


def _stream_1d(lattice: Lattice, f: np.ndarray) -> np.ndarray:
    """
    Streaming for 1D lattices (periodic BC).

    Args:
        f: shape (q, nx)

    Returns:
        f_new: shape (q, nx)
    """
    f_new = np.zeros_like(f)
    for i in range(lattice.q):
        # Roll by c[i] positions (positive = shift right)
        f_new[i] = np.roll(f[i], lattice.c[i])
    return f_new


def _stream_2d(lattice: Lattice, f: np.ndarray) -> np.ndarray:
    """
    Streaming for 2D lattices (periodic BC).

    Args:
        f: shape (q, nx, ny)

    Returns:
        f_new: shape (q, nx, ny)
    """
    f_new = np.zeros_like(f)
    for i in range(lattice.q):
        # Roll along x (axis=1) by c[i,0], then along y (axis=2) by c[i,1]
        f_new[i] = np.roll(np.roll(f[i], lattice.c[i, 0], axis=0), lattice.c[i, 1], axis=1)
    return f_new



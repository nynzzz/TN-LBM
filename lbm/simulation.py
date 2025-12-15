"""
Simulation orchestration for LBM.

Provides a high-level interface to run LBM simulations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from .lattice import Lattice
from .equilibrium import compute_equilibrium
from .collision import collide_bgk, compute_moments
from .streaming import stream


@dataclass
class SimulationResult:
    """Container for simulation results."""
    rho_history: list[np.ndarray] = field(default_factory=list)
    u_history: list[np.ndarray] = field(default_factory=list)
    f_final: np.ndarray = None
    params: dict = field(default_factory=dict)


def step(lattice: Lattice, f: np.ndarray, tau: float) -> np.ndarray:
    """
    Single LBM timestep: collision + streaming.

    Args:
        lattice: Lattice definition
        f: Distribution functions
        tau: Relaxation time

    Returns:
        f_new: Updated distributions
    """
    f_post = collide_bgk(lattice, f, tau)
    return stream(lattice, f_post)


def run(
    lattice: Lattice,
    f_init: np.ndarray,
    tau: float,
    nt: int,
    save_every: int = 1,
    callback: Callable[[int, np.ndarray], None] = None,
) -> SimulationResult:
    """
    Run LBM simulation.

    Args:
        lattice: Lattice definition
        f_init: Initial distribution functions
        tau: Relaxation time
        nt: Number of timesteps
        save_every: Save results every N steps (0 = only final)
        callback: Optional function called each step with (t, f)

    Returns:
        SimulationResult with history and final state
    """
    result = SimulationResult()
    result.params = {
        'lattice': lattice.name,
        'tau': tau,
        'nt': nt,
    }

    f = f_init.copy()

    # Save initial state
    if save_every > 0:
        rho, u = compute_moments(lattice, f)
        result.rho_history.append(rho.copy())
        result.u_history.append(u.copy())

    # Time evolution
    for t in range(1, nt + 1):
        f = step(lattice, f, tau)

        if callback:
            callback(t, f)

        if save_every > 0 and t % save_every == 0:
            rho, u = compute_moments(lattice, f)
            result.rho_history.append(rho.copy())
            result.u_history.append(u.copy())

    result.f_final = f

    # Convert to arrays
    if result.rho_history:
        result.rho_history = np.array(result.rho_history)
        result.u_history = np.array(result.u_history)

    return result

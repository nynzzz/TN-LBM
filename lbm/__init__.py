"""
Lattice Boltzmann Method (LBM) Package

A modular implementation of LBM for 1D (D1Q3) and 2D (D2Q9) simulations.
"""

from .lattice import D1Q3, D2Q9
from .equilibrium import compute_equilibrium
from .collision import collide_bgk
from .streaming import stream
from .boundary import (
    apply_bounce_back,
    apply_bounce_back_moving,
    equilibrium_inlet_left,
    extrapolation_outlet_right,
    create_cylinder_mask,
    create_channel_walls,
)

__all__ = [
    "D1Q3",
    "D2Q9",
    "compute_equilibrium",
    "collide_bgk",
    "stream",
    "apply_bounce_back",
    "apply_bounce_back_moving",
    "equilibrium_inlet_left",
    "extrapolation_outlet_right",
    "create_cylinder_mask",
    "create_channel_walls",
]

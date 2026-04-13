"""
MPS-native LBM operations (Stage 3).

This module contains operations for running LBM entirely in MPS space:
- Arithmetic: add, scale, Hadamard product
- Streaming: shift operators as MPO
- Boundary conditions: masks and bounce-back
- Collision: moments and equilibrium computation

Goal: True O(log N) scaling without decompression.
"""

from .arithmetic import (
    mps_add,
    mps_subtract,
    mps_scale,
    mps_hadamard,
    mps_inner,
    mps_norm,
    mps_sum_value,
)

from .mpo import (
    build_shift_mpo,
    apply_shift_mps,
    shift_1d_dense,
)

from .streaming import (
    stream_population_2d,
    stream_all_populations,
    stream_dense_2d,
)

from .boundary import (
    mask_to_mps,
    precompute_cavity_bc,
    apply_mps_boundary,
)

from .collision import (
    build_ones_mps,
    compute_moments_mps,
    compute_inverse_density_mps,
    compute_equilibrium_mps,
    collide_bgk_mps,
)

__all__ = [
    # Arithmetic
    'mps_add',
    'mps_subtract',
    'mps_scale',
    'mps_hadamard',
    'mps_inner',
    'mps_norm',
    'mps_sum_value',
    # Shift/Streaming
    'build_shift_mpo',
    'apply_shift_mps',
    'shift_1d_dense',
    'stream_population_2d',
    'stream_all_populations',
    'stream_dense_2d',
    # Boundary conditions
    'mask_to_mps',
    'precompute_cavity_bc',
    'apply_mps_boundary',
    # Collision
    'build_ones_mps',
    'compute_moments_mps',
    'compute_inverse_density_mps',
    'compute_equilibrium_mps',
    'collide_bgk_mps',
]

"""
Tensor Network LBM compression utilities.
"""

from .mapping import (
    flatten_2d,
    unflatten_2d,
    MAPPINGS,
    D2Q9_POP_MAPPINGS,
)
from .compression import (
    field_to_qtt,
    qtt_to_field,
    mps_memory,
    compression_stats,
    compress_populations,
    decompress_populations,
)

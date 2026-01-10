"""
Spatial mappings for 2D -> 1D linearization.

Different mappings affect QTT compression quality by changing
how spatial correlations map to tensor network structure.
"""

import numpy as np
from hilbert import decode


# =============================================================================
# Basic mappings: snake (row-major) and col-major
# =============================================================================

def snake_flatten(field_2d):
    """
    Row-major (C-order) flattening.

    Simple but non-local: vertical neighbors map to indices N apart.

    Args:
        field_2d: array of shape (nx, ny)

    Returns:
        flat: array of shape (nx * ny,)
        metadata: dict for reconstruction
    """
    shape = field_2d.shape
    flat = field_2d.flatten(order='C')
    return flat, {'shape': shape, 'mapping': 'snake'}


def snake_unflatten(flat, metadata):
    """Inverse of snake_flatten."""
    return flat.reshape(metadata['shape'], order='C')


def colmajor_flatten(field_2d):
    """
    Column-major (Fortran-order) flattening.

    Vertical neighbors are adjacent in 1D. Good for populations
    with primarily vertical velocity (pop 2: North, pop 4: South).

    Args:
        field_2d: array of shape (nx, ny)

    Returns:
        flat: array of shape (nx * ny,)
        metadata: dict for reconstruction
    """
    shape = field_2d.shape
    flat = field_2d.flatten(order='F')
    return flat, {'shape': shape, 'mapping': 'colmajor'}


def colmajor_unflatten(flat, metadata):
    """Inverse of colmajor_flatten."""
    return flat.reshape(metadata['shape'], order='F')


# =============================================================================
# Diagonal mappings
# =============================================================================

def _diagonal_indices_topleft(n):
    """
    Generate indices for diagonal traversal from top-left corner.

    Traverses diagonals going from top-left to bottom-right.
    For a 4x4 grid:
        0  1  3  6
        2  4  7 10
        5  8 11 13
        9 12 14 15

    Good for populations moving NE (pop 5) or SW (pop 7).
    """
    indices = []
    # upper-left triangle including main diagonal
    for diag in range(n):
        for i in range(diag + 1):
            indices.append((i, diag - i))
    # lower-right triangle
    for diag in range(1, n):
        for i in range(diag, n):
            indices.append((i, n - 1 - (i - diag)))
    return np.array(indices)


def _diagonal_indices_topright(n):
    """
    Generate indices for diagonal traversal from top-right corner.

    Traverses diagonals going from top-right to bottom-left.
    For a 4x4 grid:
        6  3  1  0
       10  7  4  2
       13 11  8  5
       15 14 12  9

    Good for populations moving NW (pop 6) or SE (pop 8).
    """
    indices = []
    # upper-right triangle including anti-diagonal
    for diag in range(n):
        for i in range(diag + 1):
            indices.append((i, n - 1 - diag + i))
    # lower-left triangle
    for diag in range(1, n):
        for i in range(diag, n):
            indices.append((i, i - diag))
    return np.array(indices)


def diagonal_topleft_flatten(field_2d):
    """
    Flatten 2D field using diagonal traversal from top-left.

    Points along NE-SW diagonals are adjacent in 1D.
    Good for populations 5 (NE) and 7 (SW).

    Args:
        field_2d: array of shape (n, n), must be square

    Returns:
        flat: array of shape (n * n,)
        metadata: dict for reconstruction
    """
    n = field_2d.shape[0]
    if field_2d.shape[1] != n:
        raise ValueError(f"Field must be square, got {field_2d.shape}")

    coords = _diagonal_indices_topleft(n)
    flat = field_2d[coords[:, 0], coords[:, 1]]

    return flat, {'shape': field_2d.shape, 'mapping': 'diagonal_tl', 'coords': coords}


def diagonal_topleft_unflatten(flat, metadata):
    """Inverse of diagonal_topleft_flatten."""
    shape = metadata['shape']
    coords = metadata['coords']

    field_2d = np.zeros(shape, dtype=flat.dtype)
    field_2d[coords[:, 0], coords[:, 1]] = flat

    return field_2d


def diagonal_topright_flatten(field_2d):
    """
    Flatten 2D field using diagonal traversal from top-right.

    Points along NW-SE diagonals are adjacent in 1D.
    Good for populations 6 (NW) and 8 (SE).

    Args:
        field_2d: array of shape (n, n), must be square

    Returns:
        flat: array of shape (n * n,)
        metadata: dict for reconstruction
    """
    n = field_2d.shape[0]
    if field_2d.shape[1] != n:
        raise ValueError(f"Field must be square, got {field_2d.shape}")

    coords = _diagonal_indices_topright(n)
    flat = field_2d[coords[:, 0], coords[:, 1]]

    return flat, {'shape': field_2d.shape, 'mapping': 'diagonal_tr', 'coords': coords}


def diagonal_topright_unflatten(flat, metadata):
    """Inverse of diagonal_topright_flatten."""
    shape = metadata['shape']
    coords = metadata['coords']

    field_2d = np.zeros(shape, dtype=flat.dtype)
    field_2d[coords[:, 0], coords[:, 1]] = flat

    return field_2d


# =============================================================================
# Hilbert curve mapping
# =============================================================================

def hilbert_flatten(field_2d):
    """
    Flatten 2D field using Hilbert curve ordering.

    Uses numpy-hilbert-curve library (Skilling 2004 algorithm).
    Preserves locality: nearby 2D points stay nearby in 1D.

    Args:
        field_2d: array of shape (n, n), n must be power of 2

    Returns:
        flat: array of shape (n * n,)
        metadata: dict for reconstruction
    """
    n = field_2d.shape[0]
    if field_2d.shape[1] != n:
        raise ValueError(f"Field must be square, got {field_2d.shape}")
    if n & (n - 1) != 0:
        raise ValueError(f"n must be power of 2, got {n}")

    bits = int(np.log2(n))
    total = n * n

    # decode gives (x, y) coords for each Hilbert index
    indices = np.arange(total, dtype=np.uint64)
    coords = decode(indices, 2, bits)  # shape (total, 2)

    flat = field_2d[coords[:, 0], coords[:, 1]]

    return flat, {'shape': field_2d.shape, 'mapping': 'hilbert', 'coords': coords}


def hilbert_unflatten(flat, metadata):
    """Inverse of hilbert_flatten."""
    shape = metadata['shape']
    coords = metadata['coords']

    field_2d = np.zeros(shape, dtype=flat.dtype)
    field_2d[coords[:, 0], coords[:, 1]] = flat

    return field_2d


# =============================================================================
# Unified interface
# =============================================================================

MAPPINGS = {
    'snake': (snake_flatten, snake_unflatten),
    'colmajor': (colmajor_flatten, colmajor_unflatten),
    'diagonal_tl': (diagonal_topleft_flatten, diagonal_topleft_unflatten),
    'diagonal_tr': (diagonal_topright_flatten, diagonal_topright_unflatten),
    'hilbert': (hilbert_flatten, hilbert_unflatten),
}

# D2Q9 velocity directions:
#   6  2  5
#    \ | /
#   3--0--1
#    / | \
#   7  4  8
#
# Population-specific mapping based on velocity direction:
# - Pop 0: rest particle, use snake (arbitrary)
# - Pop 1, 3: East/West (horizontal) -> snake (row-major)
# - Pop 2, 4: North/South (vertical) -> colmajor
# - Pop 5, 7: NE/SW diagonal -> diagonal_tl
# - Pop 6, 8: NW/SE diagonal -> diagonal_tr

D2Q9_POP_MAPPINGS = {
    0: 'snake',       # rest
    1: 'snake',       # East
    2: 'colmajor',    # North
    3: 'snake',       # West
    4: 'colmajor',    # South
    5: 'diagonal_tl', # NE
    6: 'diagonal_tr', # NW
    7: 'diagonal_tl', # SW
    8: 'diagonal_tr', # SE
}


def flatten_2d(field_2d, mapping='snake'):
    """
    Flatten 2D field to 1D using specified mapping.

    Args:
        field_2d: 2D array
        mapping: 'snake' or 'hilbert'

    Returns:
        flat: 1D array
        metadata: dict for reconstruction
    """
    if mapping not in MAPPINGS:
        raise ValueError(f"Unknown mapping: {mapping}. Use one of {list(MAPPINGS.keys())}")

    flatten_fn, _ = MAPPINGS[mapping]
    return flatten_fn(field_2d)


def unflatten_2d(flat, metadata):
    """
    Unflatten 1D array back to 2D.

    Args:
        flat: 1D array
        metadata: dict from flatten_2d

    Returns:
        field_2d: 2D array
    """
    mapping = metadata['mapping']
    _, unflatten_fn = MAPPINGS[mapping]
    return unflatten_fn(flat, metadata)

"""
QTT compression utilities for LBM fields.
"""

import numpy as np
import quimb.tensor as qtn

from .mapping import flatten_2d, unflatten_2d, D2Q9_POP_MAPPINGS


def field_to_qtt(field, max_bond=None, mapping='snake'):
    """
    Convert 1D or 2D field to QTT (MPS) format.

    Args:
        field: numpy array, shape (N,) for 1D or (nx, ny) for 2D
        max_bond: optional max bond dimension for truncation
        mapping: 2D linearization: 'snake', 'hilbert', or 'interleaved'

    Returns:
        mps: quimb MatrixProductState
        metadata: dict with original_shape, padded_size, L, mapping info

    Note on interleaved mapping:
        For 2D grids, interleaved mapping produces an MPS with 2L sites
        (where L = bits per coordinate). This enables O(log N) streaming.
        The 'L' in metadata refers to bits per coordinate for interleaved,
        while for other mappings it's total bits.
    """
    original_shape = field.shape

    # handle 1D vs 2D
    if field.ndim == 1:
        flat = field
        map_meta = {'mapping': 'none'}
    else:
        flat, map_meta = flatten_2d(field, mapping=mapping)

    N = len(flat)

    # For interleaved mapping, L is bits per coordinate
    # MPS will have 2L sites
    if mapping == 'interleaved':
        # Grid is N x N, so N^2 total points
        # L = log2(N) where N is grid dimension
        L_coord = map_meta['L']  # bits per coordinate (from mapping)
        total_sites = 2 * L_coord
        padded_size = 2**(2 * L_coord)  # N^2

        # pad if necessary
        if N < padded_size:
            flat = np.pad(flat, (0, padded_size - N), mode='constant', constant_values=0)

        # reshape to (2, 2, ..., 2) with 2L dimensions
        tensor = flat.reshape([2] * total_sites)

        # convert to MPS
        if max_bond is not None:
            mps = qtn.MatrixProductState.from_dense(tensor, dims=[2]*total_sites, max_bond=max_bond)
        else:
            mps = qtn.MatrixProductState.from_dense(tensor, dims=[2]*total_sites)

        metadata = {
            'original_shape': original_shape,
            'padded_size': padded_size,
            'L': L_coord,  # bits per coordinate for interleaved
            'total_sites': total_sites,
            'original_size': N,
            'mapping': 'interleaved',
            'map_meta': map_meta
        }
    else:
        # Standard case: snake, hilbert, etc.
        # find L such that 2^L >= N
        L = int(np.ceil(np.log2(N)))
        padded_size = 2**L

        # pad if necessary
        if N < padded_size:
            flat = np.pad(flat, (0, padded_size - N), mode='constant', constant_values=0)

        # reshape to (2, 2, ..., 2) with L dimensions
        tensor = flat.reshape([2] * L)

        # convert to MPS
        if max_bond is not None:
            mps = qtn.MatrixProductState.from_dense(tensor, dims=[2]*L, max_bond=max_bond)
        else:
            mps = qtn.MatrixProductState.from_dense(tensor, dims=[2]*L)

        metadata = {
            'original_shape': original_shape,
            'padded_size': padded_size,
            'L': L,
            'total_sites': L,
            'original_size': N,
            'mapping': mapping if field.ndim > 1 else 'none',
            'map_meta': map_meta
        }

    return mps, metadata


def qtt_to_field(mps, metadata):
    """
    Reconstruct field from MPS.

    Args:
        mps: quimb MatrixProductState
        metadata: dict from field_to_qtt

    Returns:
        field: numpy array with original shape
    """
    tensor = mps.to_dense()
    flat = tensor.flatten()

    # remove padding
    original_size = np.prod(metadata['original_shape'])
    flat = flat[:original_size]

    # unflatten based on mapping
    map_meta = metadata['map_meta']
    mapping = metadata.get('mapping', map_meta.get('mapping', 'none'))

    if mapping == 'none':
        # 1D case
        return flat
    else:
        # 2D case - use inverse mapping
        return unflatten_2d(flat, map_meta)


def mps_memory(mps):
    """
    Compute memory usage of MPS in bytes.

    Args:
        mps: quimb MatrixProductState

    Returns:
        bytes: total memory for all cores
    """
    total = 0
    for tensor in mps:
        total += tensor.data.nbytes
    return total


def compression_stats(original, mps, metadata):
    """
    Compute compression statistics.

    Args:
        original: original numpy array
        mps: compressed MPS
        metadata: dict from field_to_qtt

    Returns:
        dict with:
            - compression_ratio: original_size / mps_size
            - relative_error: ||orig - recon||_2 / ||orig||_2
            - max_bond: maximum bond dimension
            - bond_dims: list of bond dimensions
            - mps_bytes: MPS memory in bytes
            - original_bytes: original array memory
    """
    reconstructed = qtt_to_field(mps, metadata)

    orig_norm = np.linalg.norm(original)
    if orig_norm > 0:
        rel_error = np.linalg.norm(original - reconstructed) / orig_norm
    else:
        rel_error = 0.0

    mps_bytes = mps_memory(mps)
    original_bytes = original.nbytes

    total_sites = metadata.get('total_sites', metadata['L'])
    bond_dims = [mps.bond_size(i, i+1) for i in range(total_sites-1)]

    return {
        'compression_ratio': original_bytes / mps_bytes,
        'relative_error': rel_error,
        'max_bond': mps.max_bond(),
        'bond_dims': bond_dims,
        'mps_bytes': mps_bytes,
        'original_bytes': original_bytes
    }


def compress_populations(f, max_bond=None, mapping='snake'):
    """
    Compress all LBM populations.

    Args:
        f: distribution array, shape (q, nx) for 1D or (q, nx, ny) for 2D
        max_bond: optional max bond dimension
        mapping: for 2D fields, 'snake', 'hilbert', or 'perpop' (population-specific)

    Returns:
        mps_list: list of MPS, one per population
        metadata_list: list of metadata dicts (one per population if 'perpop', else shared)
    """
    q = f.shape[0]
    mps_list = []
    metadata_list = []

    for i in range(q):
        pop = f[i]
        if mapping == 'perpop':
            pop_mapping = D2Q9_POP_MAPPINGS.get(i, 'snake')
        else:
            pop_mapping = mapping
        mps, meta = field_to_qtt(pop, max_bond=max_bond, mapping=pop_mapping)
        mps_list.append(mps)
        metadata_list.append(meta)

    # for backward compatibility, return single metadata if all same
    if mapping != 'perpop':
        return mps_list, metadata_list[0]
    else:
        return mps_list, metadata_list


def decompress_populations(mps_list, metadata):
    """
    Decompress all LBM populations.

    Args:
        mps_list: list of MPS
        metadata: from compress_populations (single dict or list of dicts for 'perpop')

    Returns:
        f: distribution array, shape (q, ...)
    """
    if isinstance(metadata, list):
        # per-population metadata (from 'perpop' mapping)
        populations = [qtt_to_field(mps, meta) for mps, meta in zip(mps_list, metadata)]
    else:
        # shared metadata
        populations = [qtt_to_field(mps, metadata) for mps in mps_list]
    return np.stack(populations, axis=0)

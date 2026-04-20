"""
Streaming operations for MPS-native LBM.

Implements streaming (advection) for D2Q9 lattice in MPS format.

MPO-based streaming (TRUE O(log N)) for:
- interleaved: x/y bits alternate on MPS sites
- snake: y-bits on sites 0..L-1, x-bits on sites L..2L-1

Both achieve chi=2 shift MPOs for all D2Q9 directions.
Dense reference implementation kept for validation only.
"""

import numpy as np
import quimb.tensor as qtn


# =============================================================================
# Internal: MPO construction for interleaved 2D streaming
# =============================================================================

def _build_shift_mpo_2d_interleaved(L, cx, cy, cyclic=False):
    """
    Build 2D shift MPO for interleaved mapping.

    With interleaved mapping (2L sites total) and numpy's reshape convention:
    - Even sites (0, 2, 4, ...) hold y-bits (from MSB to LSB)
    - Odd sites (1, 3, 5, ...) hold x-bits (from MSB to LSB)

    Note: This is because numpy reshape uses row-major (C) order where
    the first index varies slowest. Combined with little-endian bit
    interleaving (bit 2k = x_k, bit 2k+1 = y_k), the MPS sites end up
    with y-bits on even positions and x-bits on odd positions.

    A 2D shift (cx, cy) decomposes into independent x and y shifts.
    Bond dimension: chi = 2 (single direction), chi = 4 (diagonal)

    Args:
        L: bits per coordinate (grid is 2^L × 2^L)
        cx: x-shift direction (-1, 0, or 1)
        cy: y-shift direction (-1, 0, or 1)

    Returns:
        mpo: quimb MatrixProductOperator with 2L sites
    """
    I = np.eye(2)
    J = np.array([[0, 1], [0, 0]])   # J|1> = |0>
    Jp = np.array([[0, 0], [1, 0]])  # J'|0> = |1>
    P = np.array([[0, 1], [1, 0]])   # P|0> = |1>, P|1> = |0> (cyclic wrap)

    # Direction-dependent operators
    Jx = Jp.copy() if cx == 1 else J.copy()
    Jpx = J.copy() if cx == 1 else Jp.copy()
    Jy = Jp.copy() if cy == 1 else J.copy()
    Jpy = J.copy() if cy == 1 else Jp.copy()

    # For cyclic: first core of each shift uses P instead of J
    Jx_first = P if cyclic else Jx
    Jy_first = P if cyclic else Jy

    arrays = []

    # IMPORTANT: Due to numpy reshape convention:
    # - x-bits are at ODD sites (1, 3, 5, ...)
    # - y-bits are at EVEN sites (0, 2, 4, ...)

    # Case 1: Only x-shift (cy = 0)
    # x-bits are at ODD sites (1, 3, 5, ...)
    # y-bits are at EVEN sites (0, 2, 4, ...) - just pass through
    if cy == 0:
        for site in range(2 * L):
            k = site // 2
            is_x_site = (site % 2 == 1)  # ODD sites are x-bits

            if site == 0:
                # First site (y-bit): identity, pass through with bond dim 2
                core = np.zeros((2, 2, 2))  # (r, u, d)
                core[0] = I  # bond 0 -> identity
                core[1] = I  # bond 1 -> identity (need both for x-shift to start)
            elif site == 2 * L - 1:
                # Last site (x-bit): terminate x-shift
                core = np.zeros((2, 2, 2))  # (l, u, d)
                core[0] = Jx   # bond 0 -> apply J (decrement)
                core[1] = Jpx  # bond 1 -> apply J' (increment)
            elif not is_x_site:
                # Middle y-site: pass through bond unchanged
                core = np.zeros((2, 2, 2, 2))  # (l, r, u, d)
                core[0, 0] = I
                core[1, 1] = I
            elif site == 1:
                # First x-site (site 1): start x-shift
                core = np.zeros((2, 2, 2, 2))  # (l, r, u, d)
                core[0, 0] = I        # stay on path 0
                core[0, 1] = Jx_first # start shift (P for cyclic, Jx for non-cyclic)
                core[1, 1] = 0        # from path 1, stay on 1
            else:
                # Middle x-site: x-shift logic
                core = np.zeros((2, 2, 2, 2))  # (l, r, u, d)
                core[0, 0] = I
                core[0, 1] = Jx
                core[1, 1] = Jpx
            arrays.append(core)

        return qtn.MatrixProductOperator(arrays, shape='lrud')

    # Case 2: Only y-shift (cx = 0) - apply to EVEN sites
    if cx == 0:
        for site in range(2 * L):
            k = site // 2
            is_y_site = (site % 2 == 0)  # EVEN sites are y-bits

            if site == 0:
                # First site (y-bit): start y-shift
                core = np.zeros((2, 2, 2))  # (r, u, d)
                core[0] = I
                core[1] = Jy_first
            elif site == 2 * L - 1:
                # Last site (x-bit): identity with bond dim 1
                core = np.zeros((1, 2, 2))  # (l, u, d)
                core[0] = I
            elif not is_y_site:
                # x-bit site: pass through bond unchanged
                core = np.zeros((2, 2, 2, 2))  # (l, r, u, d)
                core[0, 0] = I
                core[1, 1] = I
            elif k == L - 1:
                # Last y-site (site 2L-2): terminate y-shift, output bond dim 1
                core = np.zeros((2, 1, 2, 2))  # (l, r, u, d)
                core[0, 0] = Jy
                core[1, 0] = Jpy
            else:
                # Middle y-site: y-shift logic
                core = np.zeros((2, 2, 2, 2))  # (l, r, u, d)
                core[0, 0] = I
                core[0, 1] = Jy
                core[1, 1] = Jpy
            arrays.append(core)

        return qtn.MatrixProductOperator(arrays, shape='lrud')

    # Case 3: Both x and y shift (diagonal movement)
    # Return tuple of (x_mpo, y_mpo) to be applied sequentially.
    # Each has chi = 2, achieving optimal bond dimension.
    # The caller will apply them one after another.

    # Build x-shift MPO (acts on odd sites, identity on even)
    x_arrays = []
    for site in range(2 * L):
        k = site // 2
        is_x_site = (site % 2 == 1)

        if site == 0:
            core = np.zeros((2, 2, 2))
            core[0] = I
            core[1] = I
        elif site == 2 * L - 1:
            core = np.zeros((2, 2, 2))
            core[0] = Jx
            core[1] = Jpx
        elif not is_x_site:
            core = np.zeros((2, 2, 2, 2))
            core[0, 0] = I
            core[1, 1] = I
        elif site == 1:
            core = np.zeros((2, 2, 2, 2))
            core[0, 0] = I
            core[0, 1] = Jx_first
            core[1, 1] = 0
        else:
            core = np.zeros((2, 2, 2, 2))
            core[0, 0] = I
            core[0, 1] = Jx
            core[1, 1] = Jpx
        x_arrays.append(core)

    # Build y-shift MPO (acts on even sites, identity on odd)
    y_arrays = []
    for site in range(2 * L):
        k = site // 2
        is_y_site = (site % 2 == 0)

        if site == 0:
            core = np.zeros((2, 2, 2))
            core[0] = I
            core[1] = Jy_first
        elif site == 2 * L - 1:
            core = np.zeros((1, 2, 2))
            core[0] = I
        elif not is_y_site:
            core = np.zeros((2, 2, 2, 2))
            core[0, 0] = I
            core[1, 1] = I
        elif k == L - 1:
            core = np.zeros((2, 1, 2, 2))
            core[0, 0] = Jy
            core[1, 0] = Jpy
        else:
            core = np.zeros((2, 2, 2, 2))
            core[0, 0] = I
            core[0, 1] = Jy
            core[1, 1] = Jpy
        y_arrays.append(core)

    x_mpo = qtn.MatrixProductOperator(x_arrays, shape='lrud')
    y_mpo = qtn.MatrixProductOperator(y_arrays, shape='lrud')

    # Return tuple for sequential application
    return (x_mpo, y_mpo)


# =============================================================================
# Internal: MPO construction for snake (row-major) 2D streaming
# =============================================================================

def _build_shift_mpo_2d_snake(L, cx, cy, cyclic=False):
    """
    Build 2D shift MPO for snake (row-major) mapping.

    With snake mapping I = y*N + x (N = 2^L), after reshape to [2]*(2L):
    - Sites 0..L-1 = y-bits (MSB to LSB)
    - Sites L..2L-1 = x-bits (MSB to LSB)

    Since x and y bits occupy disjoint contiguous blocks, shifts in x and y
    are independent. This gives chi=2 for ALL directions, including diagonals
    as a single MPO (no need for sequential application).

    Args:
        L: bits per coordinate (grid is 2^L x 2^L)
        cx: x-shift direction (-1, 0, or 1)
        cy: y-shift direction (-1, 0, or 1)
        cyclic: if True, use periodic (cyclic) shift (wraps overflow via P matrix)

    Returns:
        mpo: quimb MatrixProductOperator with 2L sites
    """
    Id = np.eye(2)
    J = np.array([[0, 1], [0, 0]])   # J|1> = |0>
    Jp = np.array([[0, 0], [1, 0]])  # J'|0> = |1>
    P = np.array([[0, 1], [1, 0]])   # P|0> = |1>, P|1> = |0> (cyclic wrap)

    # Direction-dependent operators (same convention as interleaved)
    # cx = +1 means pull from x-1 → increment index
    Jx = Jp.copy() if cx == 1 else J.copy()
    Jpx = J.copy() if cx == 1 else Jp.copy()
    Jy = Jp.copy() if cy == 1 else J.copy()
    Jpy = J.copy() if cy == 1 else Jp.copy()

    # For cyclic shift, first core of each sub-chain uses P instead of J
    # (Gross et al.: S_cyclic = [I,P] x [I,J; 0,J'] x ... x [J; J'])
    Jx_first = P if cyclic else Jx
    Jy_first = P if cyclic else Jy

    total_sites = 2 * L
    arrays = []

    # Helper: build standard shift cores for a block of sites
    def _shift_first_core(Js):
        """First core of a shift sub-chain (bond_l=1, bond_r=2)."""
        core = np.zeros((1, 2, 2, 2))  # (l, r, u, d)
        core[0, 0] = Id
        core[0, 1] = Js
        return core

    def _shift_middle_core(Js, Jps):
        """Middle core of a shift sub-chain (bond_l=2, bond_r=2)."""
        core = np.zeros((2, 2, 2, 2))
        core[0, 0] = Id
        core[0, 1] = Js
        core[1, 1] = Jps
        return core

    def _shift_last_core(Js, Jps, has_right_bond=False, right_bond=1):
        """Last core of a shift sub-chain."""
        if has_right_bond:
            core = np.zeros((2, right_bond, 2, 2))  # (l, r, u, d)
            core[0, 0] = Js
            core[1, 0] = Jps
        else:
            core = np.zeros((2, 2, 2))  # (l, u, d)
            core[0] = Js
            core[1] = Jps
        return core

    def _shift_single_core(Js, has_left_bond=True, has_right_bond=False, right_bond=1):
        """Single-site shift (L=1 for that coordinate)."""
        if has_left_bond and has_right_bond:
            core = np.zeros((1, right_bond, 2, 2))
            core[0, 0] = Js
        elif has_left_bond:
            core = np.zeros((1, 2, 2))
            core[0] = Js
        elif has_right_bond:
            core = np.zeros((right_bond, 2, 2))
            core[0] = Js
        else:
            core = Js.reshape(2, 2)  # shouldn't happen for 2L>=2
        return core

    def _identity_core(bond_l, bond_r):
        """Identity passthrough core."""
        if bond_l is None:
            core = np.zeros((bond_r, 2, 2))
            core[0] = Id
        elif bond_r is None:
            core = np.zeros((bond_l, 2, 2))
            core[0] = Id
        else:
            core = np.zeros((bond_l, bond_r, 2, 2))
            core[0, 0] = Id
        return core

    # ---- Case 1: x-shift only (cy=0) ----
    # y-block: identity (bond=1), x-block: shift (bond=2)
    if cy == 0:
        # Y-block (sites 0..L-1): identity passthrough
        for site in range(L):
            if site == 0 and L > 1:
                arrays.append(_identity_core(None, 1))  # (r=1, u, d)
            elif site == 0 and L == 1:
                arrays.append(_identity_core(None, 1))  # (r=1, u, d)
            elif site < L - 1:
                arrays.append(_identity_core(1, 1))  # (1, 1, u, d)
            else:
                # Last y-site: identity, but bond expands to 2 for x-block
                # Just pass through — site L will handle the expansion
                arrays.append(_identity_core(1, 1))

        # X-block (sites L..2L-1): standard shift
        x_len = L  # number of x-bits
        for i in range(x_len):
            site = L + i
            if x_len == 1:
                # Single x-bit: use Jx_first for cyclic support
                arrays.append(_shift_single_core(Jx_first, has_left_bond=True, has_right_bond=False))
            elif i == 0:
                arrays.append(_shift_first_core(Jx_first))
            elif i == x_len - 1:
                arrays.append(_shift_last_core(Jx, Jpx))
            else:
                arrays.append(_shift_middle_core(Jx, Jpx))

        return qtn.MatrixProductOperator(arrays, shape='lrud')

    # ---- Case 2: y-shift only (cx=0) ----
    # y-block: shift (bond=2), x-block: identity (bond=1)
    if cx == 0:
        # Y-block (sites 0..L-1): standard shift
        y_len = L
        for i in range(y_len):
            if y_len == 1:
                arrays.append(_shift_single_core(Jy_first, has_left_bond=False, has_right_bond=True, right_bond=1))
            elif i == 0:
                # First y-site: standard first core (no left bond)
                core = np.zeros((2, 2, 2))  # (r, u, d)
                core[0] = Id
                core[1] = Jy_first
                arrays.append(core)
            elif i == y_len - 1:
                # Last y-site: terminate shift, contract bond to 1
                arrays.append(_shift_last_core(Jy, Jpy, has_right_bond=True, right_bond=1))
            else:
                arrays.append(_shift_middle_core(Jy, Jpy))

        # X-block (sites L..2L-1): identity passthrough
        for i in range(L):
            site = L + i
            if site == total_sites - 1:
                arrays.append(_identity_core(1, None))  # (l=1, u, d)
            else:
                arrays.append(_identity_core(1, 1))

        return qtn.MatrixProductOperator(arrays, shape='lrud')

    # ---- Case 3: Diagonal (cx!=0, cy!=0) ----
    # Single MPO! y-shift on sites 0..L-1, x-shift on sites L..2L-1
    # Junction bond between y-block and x-block = 1

    # Y-block: shift
    y_len = L
    for i in range(y_len):
        if y_len == 1:
            arrays.append(_shift_single_core(Jy_first, has_left_bond=False, has_right_bond=True, right_bond=1))
        elif i == 0:
            core = np.zeros((2, 2, 2))  # (r, u, d)
            core[0] = Id
            core[1] = Jy_first
            arrays.append(core)
        elif i == y_len - 1:
            arrays.append(_shift_last_core(Jy, Jpy, has_right_bond=True, right_bond=1))
        else:
            arrays.append(_shift_middle_core(Jy, Jpy))

    # X-block: shift
    x_len = L
    for i in range(x_len):
        if x_len == 1:
            arrays.append(_shift_single_core(Jx_first, has_left_bond=True, has_right_bond=False))
        elif i == 0:
            arrays.append(_shift_first_core(Jx_first))
        elif i == x_len - 1:
            arrays.append(_shift_last_core(Jx, Jpx))
        else:
            arrays.append(_shift_middle_core(Jx, Jpx))

    return qtn.MatrixProductOperator(arrays, shape='lrud')


# =============================================================================
# Public API: Streaming functions
# =============================================================================

def stream_population_2d_interleaved(mps, L, cx, cy, max_bond=None, cyclic=False, cutoff=1e-10):
    """
    Stream a 2D population using interleaved mapping with MPO.

    Args:
        mps: MPS representation (2L sites, interleaved format)
        L: bits per coordinate (grid is 2^L x 2^L)
        cx, cy: velocity components (-1, 0, or 1)
        max_bond: optional bond dimension truncation
        cyclic: if True, use periodic shift (wraps like np.roll)

    Returns:
        streamed_mps: MPS after streaming
    """
    if cx == 0 and cy == 0:
        return mps.copy()

    # Swap cx/cy: numpy C-order flatten gives I where axis=0 (x) is MSB.
    # The interleaved MPO builder's internal convention has the opposite assignment,
    # so we swap to match the actual data layout.
    result = _build_shift_mpo_2d_interleaved(L, cy, cx, cyclic=cyclic)

    # For diagonal shifts, result is tuple (x_mpo, y_mpo)
    # Apply sequentially for optimal chi = 2
    if isinstance(result, tuple):
        x_mpo, y_mpo = result
        # Apply y-shift first, then x-shift
        new_mps = mps.gate_with_mpo(y_mpo, max_bond=max_bond, cutoff=cutoff)
        new_mps = new_mps.gate_with_mpo(x_mpo, max_bond=max_bond, cutoff=cutoff)
    else:
        # Axis-aligned shift: single MPO
        new_mps = mps.gate_with_mpo(result, max_bond=max_bond, cutoff=cutoff)

    if max_bond is not None:
        new_mps.compress(max_bond=max_bond, cutoff=cutoff)

    return new_mps


def stream_population_2d_snake(mps, L, cx, cy, max_bond=None, cyclic=False, cutoff=1e-10):
    """
    Stream a 2D population using snake mapping with MPO.

    TRUE O(log N) complexity! Same as interleaved.

    Note: numpy C-order flatten gives I = x*N + y, so sites 0..L-1 are x-bits
    and sites L..2L-1 are y-bits. The MPO builder expects (cx_mpo, cy_mpo) where
    cx_mpo acts on sites 0..L-1 (x-bits) and cy_mpo on sites L..2L-1 (y-bits).
    Since np.roll uses cx on axis=0 (x) and cy on axis=1 (y), we swap cx/cy
    to match the MPO's internal convention.

    Args:
        mps: MPS representation (2L sites, snake format)
        L: bits per coordinate (grid is 2^L x 2^L)
        cx, cy: velocity components (-1, 0, or 1) in LBM convention
        max_bond: optional bond dimension truncation
        cyclic: if True, use periodic shift (wraps boundaries like np.roll)

    Returns:
        streamed_mps: MPS after streaming
    """
    if cx == 0 and cy == 0:
        return mps.copy()

    # Swap cx/cy: numpy C-order gives I = x*N + y, so MSBs (sites 0..L-1) are x,
    # LSBs (sites L..2L-1) are y. The MPO builder's "cx" acts on LSBs and "cy" on MSBs
    # (it assumes I = y*N + x), so we swap to correct for the actual C-order layout.
    mpo = _build_shift_mpo_2d_snake(L, cy, cx, cyclic=cyclic)
    new_mps = mps.gate_with_mpo(mpo, max_bond=max_bond, cutoff=cutoff)

    if max_bond is not None:
        new_mps.compress(max_bond=max_bond, cutoff=cutoff)

    return new_mps


def stream_population_2d(mps, metadata, cx, cy, max_bond=None, cyclic=False, cutoff=1e-10):
    """
    Stream a single 2D population in MPS format.

    Dispatches to appropriate MPO-based method:
    - interleaved: x/y bits alternate, chi=2
    - snake: y-bits then x-bits in contiguous blocks, chi=2

    Args:
        mps: MPS representation of the 2D population field
        metadata: metadata from field_to_qtt
        cx, cy: velocity components (-1, 0, or 1)
        max_bond: optional bond dimension truncation
        cyclic: if True, use periodic shift (wraps like np.roll, needed for LBM)

    Returns:
        streamed_mps: MPS after streaming
    """
    if cx == 0 and cy == 0:
        return mps.copy()

    mapping = metadata.get('mapping', 'snake')

    if mapping == 'interleaved':
        L = metadata['L']
        return stream_population_2d_interleaved(mps, L, cx, cy, max_bond, cyclic=cyclic, cutoff=cutoff)

    if mapping == 'snake':
        L_coord = metadata['L'] // 2  # total bits / 2 = bits per coordinate
        return stream_population_2d_snake(mps, L_coord, cx, cy, max_bond, cyclic=cyclic, cutoff=cutoff)

    # Fallback: dense reference for unsupported mappings (e.g. hilbert)
    return _stream_population_2d_reference(mps, metadata, cx, cy, max_bond)


def _stream_population_2d_reference(mps, metadata, cx, cy, max_bond=None):
    """
    Stream using dense conversion — O(N^2) REFERENCE implementation.

    Decompresses to dense, applies shift with numpy, recompresses.
    Only use for validation/comparison. NOT for production.
    """
    original_shape = metadata['original_shape']
    if len(original_shape) != 2:
        raise ValueError("Expected 2D field")

    nx, ny = original_shape
    N = metadata['padded_size']
    L = metadata['L']

    # Get dense field
    dense = mps.to_dense().flatten()
    if len(dense) < N:
        dense = np.pad(dense, (0, N - len(dense)))

    # Unflatten to 2D
    field_2d = dense[:nx * ny].reshape(ny, nx)

    # Apply 2D shift
    shifted_2d = np.zeros_like(field_2d)

    for y in range(ny):
        for x in range(nx):
            src_x = x - cx
            src_y = y - cy

            if 0 <= src_x < nx and 0 <= src_y < ny:
                shifted_2d[y, x] = field_2d[src_y, src_x]
            else:
                shifted_2d[y, x] = 0.0

    # Convert back to MPS
    shifted_flat = shifted_2d.flatten()
    if len(shifted_flat) < N:
        shifted_flat = np.pad(shifted_flat, (0, N - len(shifted_flat)))

    shifted_tensor = shifted_flat.reshape([2] * L)
    new_mps = qtn.MatrixProductState.from_dense(shifted_tensor, dims=[2] * L)

    if max_bond is not None:
        new_mps.compress(max_bond=max_bond)

    return new_mps


def stream_all_populations(mps_list, metadata, lattice, max_bond=None, cyclic=False):
    """
    Stream all D2Q9 populations.

    Args:
        mps_list: list of 9 MPS (one per population)
        metadata: metadata from compress_populations
        lattice: D2Q9 lattice definition (from lbm module)
        max_bond: optional bond dimension truncation
        cyclic: if True, use periodic shift (needed for LBM with bounce-back)

    Returns:
        streamed_list: list of 9 streamed MPS
    """
    streamed_list = []

    for i in range(9):
        cx, cy = lattice.c[i]
        streamed = stream_population_2d(
            mps_list[i], metadata, int(cx), int(cy),
            max_bond=max_bond, cyclic=cyclic
        )
        streamed_list.append(streamed)

    return streamed_list


def stream_dense_2d(f, lattice):
    """
    Dense 2D streaming for reference/testing.

    Args:
        f: distribution array (q, ny, nx)
        lattice: D2Q9 lattice

    Returns:
        f_streamed: streamed distribution array
    """
    q, ny, nx = f.shape
    f_new = np.zeros_like(f)

    for i in range(q):
        cx, cy = lattice.c[i]
        cx, cy = int(cx), int(cy)

        for y in range(ny):
            for x in range(nx):
                src_x = x - cx
                src_y = y - cy

                if 0 <= src_x < nx and 0 <= src_y < ny:
                    f_new[i, y, x] = f[i, src_y, src_x]
                else:
                    f_new[i, y, x] = 0.0

    return f_new

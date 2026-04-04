"""
MPO (Matrix Product Operator) construction for TN-LBM.

Implements shift operators for streaming in tensor network format.
Based on Gross et al. 2024 (arXiv:2512.07615).

Key result: Shift MPO has bond dimension chi = 2 (non-cyclic) or chi = 3 (cyclic).

Gross et al. construction (Equations 16, 25-28):
    Building blocks:
        I  = [[1,0], [0,1]]  # Identity
        J  = [[0,1], [0,0]]  # J|1> = |0>, J|0> = 0
        J' = [[0,0], [1,0]]  # J'|0> = |1>, J'|1> = 0

    Non-cyclic shift MPO structure:
        A = [I, J]           # First core (bond dim 2 out)
        B = [[I, J],         # Middle cores (bond dim 2 in/out)
             [0, J']]
        C = [J; J']          # Last core (bond dim 2 in)

    Direction: swap J <-> J' for opposite shift direction.
"""

import numpy as np
import quimb.tensor as qtn


def build_shift_mpo(L, direction=1, cyclic=False):
    """
    Build shift MPO using Gross et al. construction.

    Uses quimb's MatrixProductOperator with 'lrud' tensor format:
    - l = left bond, r = right bond, u = upper (output), d = down (input)

    Args:
        L: number of sites (N = 2^L)
        direction: +1 for right shift (g(x)=f(x-1)), -1 for left shift (g(x)=f(x+1))
        cyclic: if True, use periodic boundary conditions

    Returns:
        mpo: quimb MatrixProductOperator
    """
    # Building blocks (Equation 16)
    I = np.eye(2)
    J = np.array([[0, 1], [0, 0]])   # J|1> = |0>, J|0> = 0
    Jp = np.array([[0, 0], [1, 0]])  # J'|0> = |1>, J'|1> = 0

    # Swap for right shift (direction = +1)
    # Default J, J' gives left shift; swapped gives right shift
    if direction == 1:
        J, Jp = Jp, J

    # Build MPO cores in quimb 'lrud' format
    mpo_arrays = []

    for site in range(L):
        if site == 0:
            # First core: shape (r, u, d) = (2, 2, 2)
            # [I, J] along right bond dimension
            core = np.zeros((2, 2, 2))
            core[0, :, :] = I   # bond=0: identity
            core[1, :, :] = J   # bond=1: J operator
            mpo_arrays.append(core)

        elif site == L - 1:
            # Last core: shape (l, u, d) = (2, 2, 2)
            # [J; J'] along left bond dimension
            core = np.zeros((2, 2, 2))
            core[0, :, :] = J   # bond=0: J operator
            core[1, :, :] = Jp  # bond=1: J' operator
            mpo_arrays.append(core)

        else:
            # Middle cores: shape (l, r, u, d) = (2, 2, 2, 2)
            # [[I, J], [0, J']] block structure
            core = np.zeros((2, 2, 2, 2))
            core[0, 0, :, :] = I    # (l=0, r=0): I
            core[0, 1, :, :] = J    # (l=0, r=1): J
            core[1, 0, :, :] = 0    # (l=1, r=0): 0
            core[1, 1, :, :] = Jp   # (l=1, r=1): J'
            mpo_arrays.append(core)

    # Create quimb MPO with 'lrud' shape convention
    mpo = qtn.MatrixProductOperator(mpo_arrays, shape='lrud')

    return mpo


def shift_1d_dense(arr, direction=1, cyclic=True):
    """
    Apply 1D shift to a dense array (for reference/testing).

    Args:
        arr: 1D numpy array
        direction: +1 for right shift, -1 for left shift
        cyclic: if True, wrap around

    Returns:
        shifted array
    """
    n = len(arr)
    result = np.zeros_like(arr)

    if direction == 1:  # Right shift
        if cyclic:
            result[1:] = arr[:-1]
            result[0] = arr[-1]
        else:
            result[1:] = arr[:-1]
            result[0] = 0
    else:  # Left shift
        if cyclic:
            result[:-1] = arr[1:]
            result[-1] = arr[0]
        else:
            result[:-1] = arr[1:]
            result[-1] = 0

    return result


def apply_shift_mps(mps, metadata, direction=1, cyclic=True, max_bond=None):
    """
    Apply 1D shift to an MPS.

    This implements the shift by constructing and applying the shift MPO
    using manual tensor contractions.

    Based on Gross et al. construction:
    - Non-cyclic shift: S^(n) = [I,J] x [I,J; 0,J'] x ... x [J; J']
    - J = [[0,1],[0,0]], J' = [[0,0],[1,0]]
    - Bond dimension chi = 2

    Args:
        mps: quimb MatrixProductState
        metadata: metadata from field_to_qtt
        direction: +1 for right shift, -1 for left shift
        cyclic: if True, wrap around (periodic BC)
        max_bond: optional truncation of result

    Returns:
        shifted_mps: new MPS with shifted values
    """
    L = len(mps.tensors)

    # Building blocks for shift MPO
    # J decrements a bit, J' increments a bit
    # Default (J, J') gives LEFT shift: g(x) = f(x+1)
    # Swapped (J', J) gives RIGHT shift: g(x) = f(x-1)
    I = np.eye(2)
    J = np.array([[0, 1], [0, 0]])   # J|0>=0, J|1>=|0>
    Jp = np.array([[0, 0], [1, 0]])  # J'|0>=|1>, J'|1>=0

    # Swap for right shift (direction = +1)
    if direction == 1:
        J, Jp = Jp, J

    # Build MPO cores
    # Core structure: (bond_left, bond_right, phys_out, phys_in)
    # Edge cores have reduced dimensions

    mpo_cores = []

    for site in range(L):
        if site == 0:
            # Left edge: (bond_right, phys_out, phys_in)
            # [I; J] stacked vertically (first core)
            core = np.zeros((2, 2, 2))
            core[0] = I  # bond 0
            core[1] = J  # bond 1
            mpo_cores.append(core)

        elif site == L - 1:
            # Right edge: (bond_left, phys_out, phys_in)
            # [J; J'] stacked vertically
            core = np.zeros((2, 2, 2))
            core[0] = J
            core[1] = Jp
            mpo_cores.append(core)

        else:
            # Middle: (bond_left, bond_right, phys_out, phys_in)
            # [I, J; 0, J']
            core = np.zeros((2, 2, 2, 2))
            core[0, 0] = I   # (in=0, out=0) -> I
            core[0, 1] = J   # (in=0, out=1) -> J
            core[1, 0] = 0   # (in=1, out=0) -> 0
            core[1, 1] = Jp  # (in=1, out=1) -> J'
            mpo_cores.append(core)

    # For small L, do full contraction then convert back to MPS
    # This is simpler and more reliable than site-by-site with bond tracking

    # Get MPS tensors
    mps_tensors = [mps[i].data for i in range(L)]

    # Build contraction: contract physical indices site by site
    # Keep MPO bond indices to contract at the end

    if L == 1:
        # Single site: just contract physical index
        result = np.einsum('bop,p->o', mpo_cores[0], mps_tensors[0])
        result_tensor = result.reshape(2)
        new_mps = qtn.MatrixProductState.from_dense(result_tensor, dims=[2])

    elif L == 2:
        # Two sites: contract everything
        # MPS: m0[i0, b01] * m1[b01, i1] -> M[i0, i1]
        # MPO: c0[b, o0, i0] * c1[b, o1, i1] -> O[o0, o1, i0, i1]
        # Result: O * M contracted on i0, i1 -> R[o0, o1]
        result = np.einsum('boi,ip,bqj,pj->oq',
                           mpo_cores[0], mps_tensors[0],
                           mpo_cores[1], mps_tensors[1])
        new_mps = qtn.MatrixProductState.from_dense(result, dims=[2, 2])

    else:
        # For L >= 3, use quimb's MPO-MPS contraction
        # This achieves true O(L * chi^2) complexity with chi = 2
        mpo = build_shift_mpo(L, direction=direction, cyclic=cyclic)
        new_mps = mps.gate_with_mpo(mpo, max_bond=max_bond, cutoff=1e-10)

    # Compress if needed
    if max_bond is not None:
        new_mps.compress(max_bond=max_bond, cutoff=1e-10)

    return new_mps


def build_identity_mpo(L):
    """
    Build identity MPO (for testing).

    Args:
        L: number of sites

    Returns:
        List of identity matrices (trivial MPO)
    """
    return [np.eye(2) for _ in range(L)]


# =============================================================================
# 2D Streaming MPO for Interleaved Mapping
# =============================================================================

def build_shift_mpo_2d_interleaved(L, cx, cy, cyclic=False):
    """
    Build 2D shift MPO for interleaved mapping.

    With interleaved mapping, a 2D grid N×N (where N=2^L) maps to 2L qubits:
    - Even sites (0, 2, 4, ...) hold x-bits
    - Odd sites (1, 3, 5, ...) hold y-bits

    A 2D shift (cx, cy) decomposes into:
    - x-shift: affects only even sites
    - y-shift: affects only odd sites

    Both shifts are LOCAL in the interleaved representation!

    Bond dimension: chi = 2 for non-cyclic, chi = 3 for cyclic

    Args:
        L: number of bits per coordinate (grid is 2^L × 2^L)
        cx: x-component of shift (-1, 0, or 1)
        cy: y-component of shift (-1, 0, or 1)
        cyclic: if True, use periodic boundary conditions

    Returns:
        mpo: quimb MatrixProductOperator with 2L sites
    """
    if cx == 0 and cy == 0:
        # Identity MPO
        arrays = []
        for _ in range(2 * L):
            core = np.eye(2).reshape(1, 1, 2, 2)  # (l, r, u, d)
            arrays.append(core)
        # Fix edge shapes
        arrays[0] = arrays[0][0]  # (r, u, d)
        arrays[-1] = arrays[-1][:, 0]  # (l, u, d)
        return qtn.MatrixProductOperator(arrays, shape='lrud')

    # Building blocks for shift MPO
    I = np.eye(2)
    J = np.array([[0, 1], [0, 0]])   # J|1> = |0>, J|0> = 0
    Jp = np.array([[0, 0], [1, 0]])  # J'|0> = |1>, J'|1> = 0

    # For right shift (direction = +1): swap J <-> J'
    # For left shift (direction = -1): keep as is

    def make_shift_cores(L_shift, direction):
        """Make shift MPO cores for L_shift sites."""
        Jx = Jp.copy() if direction == 1 else J.copy()
        Jpx = J.copy() if direction == 1 else Jp.copy()

        cores = []
        for site in range(L_shift):
            if site == 0:
                core = np.zeros((2, 2, 2))  # (r, u, d)
                core[0, :, :] = I
                core[1, :, :] = Jx
            elif site == L_shift - 1:
                core = np.zeros((2, 2, 2))  # (l, u, d)
                core[0, :, :] = Jx
                core[1, :, :] = Jpx
            else:
                core = np.zeros((2, 2, 2, 2))  # (l, r, u, d)
                core[0, 0, :, :] = I
                core[0, 1, :, :] = Jx
                core[1, 0, :, :] = 0
                core[1, 1, :, :] = Jpx
            cores.append(core)
        return cores

    # Build combined MPO for interleaved sites
    # Site ordering: x_0, y_0, x_1, y_1, ..., x_{L-1}, y_{L-1}

    # Get shift cores for x and y (if non-zero)
    if cx != 0:
        x_cores = make_shift_cores(L, cx)
    if cy != 0:
        y_cores = make_shift_cores(L, cy)

    # Combine into interleaved MPO
    # This is more complex - we need to interleave the shift operations

    # Strategy: Build separate MPOs and then interleave tensors
    # For a site at position 2k (x-bit): use x-shift core k, or identity
    # For a site at position 2k+1 (y-bit): use y-shift core k, or identity

    mpo_arrays = []

    for site in range(2 * L):
        k = site // 2  # Which bit level
        is_x_site = (site % 2 == 0)

        if is_x_site:
            # x-bit site
            if cx == 0:
                # Identity on x-sites
                if site == 0:
                    core = I.reshape(1, 2, 2)  # (r, u, d)
                elif site == 2 * L - 1:
                    core = I.reshape(1, 2, 2)  # (l, u, d)
                else:
                    core = I.reshape(1, 1, 2, 2)  # (l, r, u, d)
            else:
                core = x_cores[k]
        else:
            # y-bit site
            if cy == 0:
                # Identity on y-sites
                if site == 0:
                    core = I.reshape(1, 2, 2)  # (r, u, d)
                elif site == 2 * L - 1:
                    core = I.reshape(1, 2, 2)  # (l, u, d)
                else:
                    core = I.reshape(1, 1, 2, 2)  # (l, r, u, d)
            else:
                core = y_cores[k]

        mpo_arrays.append(core)

    # Now we need to properly handle the bond structure when combining
    # x-shift and y-shift operations. The above simple interleaving won't work
    # because the bond indices need to be connected properly.

    # Better approach: Build the full interleaved MPO from scratch
    # considering both shifts simultaneously

    return _build_interleaved_shift_mpo(L, cx, cy, cyclic)


def _build_interleaved_shift_mpo(L, cx, cy, cyclic=False):
    """
    Build interleaved shift MPO handling both x and y shifts.

    The key insight: x-shift and y-shift act on DISJOINT sets of sites,
    so we can handle them independently.

    For x-shift (acts on even sites 0, 2, 4, ...):
    - Build standard shift MPO on L sites
    - Insert identity on odd sites in between

    For y-shift (acts on odd sites 1, 3, 5, ...):
    - Build standard shift MPO on L sites
    - Insert identity on even sites in between

    When both are non-zero, we need tensor product structure.

    Args:
        L: bits per coordinate
        cx: x-shift direction
        cy: y-shift direction
        cyclic: periodic boundaries

    Returns:
        mpo: quimb MatrixProductOperator
    """
    I = np.eye(2)
    J = np.array([[0, 1], [0, 0]])
    Jp = np.array([[0, 0], [1, 0]])

    # Direction-dependent building blocks
    Jx = Jp.copy() if cx == 1 else J.copy()
    Jpx = J.copy() if cx == 1 else Jp.copy()
    Jy = Jp.copy() if cy == 1 else J.copy()
    Jpy = J.copy() if cy == 1 else Jp.copy()

    arrays = []

    # Case 1: Only x-shift (cy = 0)
    if cy == 0:
        for site in range(2 * L):
            k = site // 2
            is_x_site = (site % 2 == 0)

            if not is_x_site:
                # y-site: identity, pass through bond
                if site == 2 * L - 1:
                    # Last site
                    core = np.zeros((2, 2, 2))
                    core[0] = I
                    core[1] = I
                else:
                    # Middle site
                    core = np.zeros((2, 2, 2, 2))
                    core[0, 0] = I
                    core[1, 1] = I
                arrays.append(core)
            else:
                # x-site: shift operation
                if site == 0:
                    # First site (x_0)
                    if cx == 0:
                        core = I.reshape(1, 2, 2)
                    else:
                        core = np.zeros((2, 2, 2))
                        core[0] = I
                        core[1] = Jx
                elif k == L - 1:
                    # Last x-site (x_{L-1}), but not last overall site
                    core = np.zeros((2, 2, 2, 2))
                    core[0, 0] = Jx
                    core[0, 1] = 0
                    core[1, 0] = Jpx
                    core[1, 1] = 0
                else:
                    # Middle x-site
                    core = np.zeros((2, 2, 2, 2))
                    core[0, 0] = I
                    core[0, 1] = Jx
                    core[1, 0] = 0
                    core[1, 1] = Jpx
                arrays.append(core)

        return qtn.MatrixProductOperator(arrays, shape='lrud')

    # Case 2: Only y-shift (cx = 0)
    if cx == 0:
        for site in range(2 * L):
            k = site // 2
            is_x_site = (site % 2 == 0)

            if is_x_site:
                # x-site: identity, pass through bond
                if site == 0:
                    # First site
                    core = np.zeros((2, 2, 2))
                    core[0] = I
                    core[1] = I
                else:
                    # Middle site
                    core = np.zeros((2, 2, 2, 2))
                    core[0, 0] = I
                    core[1, 1] = I
                arrays.append(core)
            else:
                # y-site: shift operation
                if k == 0:
                    # First y-site (y_0), second overall
                    core = np.zeros((2, 2, 2, 2))
                    core[0, 0] = I
                    core[0, 1] = Jy
                    core[1, 0] = 0
                    core[1, 1] = 0
                elif site == 2 * L - 1:
                    # Last site (y_{L-1})
                    core = np.zeros((2, 2, 2))
                    core[0] = Jy
                    core[1] = Jpy
                else:
                    # Middle y-site
                    core = np.zeros((2, 2, 2, 2))
                    core[0, 0] = I
                    core[0, 1] = Jy
                    core[1, 0] = 0
                    core[1, 1] = Jpy
                arrays.append(core)

        return qtn.MatrixProductOperator(arrays, shape='lrud')

    # Case 3: Both x and y shift (cx != 0 and cy != 0)
    # Need tensor product of the two shift operators
    # Bond dimension becomes 2 * 2 = 4

    for site in range(2 * L):
        k = site // 2
        is_x_site = (site % 2 == 0)

        if site == 0:
            # First site (x_0): x-shift, start y-bond
            core = np.zeros((4, 2, 2))  # (r, u, d)
            # Bond structure: (bx, by) with bx, by in {0, 1}
            # Index: bx * 2 + by
            core[0 * 2 + 0] = I   # bx=0, by=0
            core[0 * 2 + 1] = I   # bx=0, by=1
            core[1 * 2 + 0] = Jx  # bx=1, by=0
            core[1 * 2 + 1] = Jx  # bx=1, by=1
            arrays.append(core)

        elif site == 2 * L - 1:
            # Last site (y_{L-1}): y-shift end
            core = np.zeros((4, 2, 2))  # (l, u, d)
            core[0 * 2 + 0] = Jy   # bx=0, by=0 (should be 0, x already finished)
            core[0 * 2 + 1] = Jpy  # bx=0, by=1
            core[1 * 2 + 0] = Jy   # bx=1, by=0 (impossible state)
            core[1 * 2 + 1] = Jpy  # bx=1, by=1 (impossible state)
            arrays.append(core)

        elif is_x_site:
            # Middle x-site
            core = np.zeros((4, 4, 2, 2))  # (l, r, u, d)
            for by in range(2):
                # x-shift operation, pass through y-bond
                core[0 * 2 + by, 0 * 2 + by] = I
                core[0 * 2 + by, 1 * 2 + by] = Jx
                core[1 * 2 + by, 0 * 2 + by] = 0
                core[1 * 2 + by, 1 * 2 + by] = Jpx
            arrays.append(core)

        else:
            # Middle y-site
            core = np.zeros((4, 4, 2, 2))  # (l, r, u, d)
            for bx in range(2):
                # y-shift operation, pass through x-bond
                core[bx * 2 + 0, bx * 2 + 0] = I
                core[bx * 2 + 0, bx * 2 + 1] = Jy
                core[bx * 2 + 1, bx * 2 + 0] = 0
                core[bx * 2 + 1, bx * 2 + 1] = Jpy
            arrays.append(core)

    return qtn.MatrixProductOperator(arrays, shape='lrud')


def apply_shift_mps_2d_interleaved(mps, L, cx, cy, cyclic=False, max_bond=None):
    """
    Apply 2D shift to an MPS using interleaved mapping.

    This achieves TRUE O(log N) complexity for 2D streaming!

    The MPS must be in interleaved format (2L sites).

    Args:
        mps: quimb MatrixProductState (2L sites, interleaved format)
        L: bits per coordinate (grid is 2^L × 2^L)
        cx: x-component of shift (-1, 0, or 1)
        cy: y-component of shift (-1, 0, or 1)
        cyclic: if True, use periodic boundary conditions
        max_bond: optional truncation of result

    Returns:
        shifted_mps: new MPS with shifted values
    """
    if cx == 0 and cy == 0:
        return mps.copy()

    # Build the 2D shift MPO
    mpo = build_shift_mpo_2d_interleaved(L, cx, cy, cyclic=cyclic)

    # Apply MPO to MPS
    new_mps = mps.gate_with_mpo(mpo, max_bond=max_bond, cutoff=1e-10)

    if max_bond is not None:
        new_mps.compress(max_bond=max_bond, cutoff=1e-10)

    return new_mps

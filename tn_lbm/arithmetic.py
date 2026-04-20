"""
MPS arithmetic operations for TN-LBM.

Provides basic operations on Matrix Product States:
- Addition (direct sum then truncate)
- Scalar multiplication
- Hadamard (elementwise) product
"""

import numpy as np
import quimb.tensor as qtn


def mps_add(mps_a, mps_b, max_bond=None, cutoff=1e-10):
    """
    Add two MPS: |C> = |A> + |B>.

    The result has bond dimension up to chi_A + chi_B before truncation.

    Args:
        mps_a: first MPS
        mps_b: second MPS (must have same physical dimensions)
        max_bond: optional truncation of bond dimension
        cutoff: SVD cutoff for small singular values

    Returns:
        mps_c: sum MPS
    """
    # Use quimb's built-in addition which handles the direct sum
    mps_c = mps_a + mps_b

    # Compress if max_bond specified
    if max_bond is not None:
        mps_c.compress(max_bond=max_bond, cutoff=cutoff)

    return mps_c


def mps_scale(mps, scalar):
    """
    Multiply MPS by a scalar: |B> = c * |A>.

    Modifies the first tensor of the MPS by the scalar factor.

    Args:
        mps: input MPS
        scalar: scalar multiplier

    Returns:
        mps_scaled: new MPS with scaled values
    """
    # Copy to avoid modifying original
    mps_scaled = mps.copy()

    # Scale the first tensor
    # Access the first tensor and multiply its data
    first_tensor = mps_scaled[0]
    first_tensor.modify(data=first_tensor.data * scalar)

    return mps_scaled


def mps_hadamard(mps_a, mps_b, max_bond=None, cutoff=1e-10):
    """
    Elementwise (Hadamard) product of two MPS: C[i] = A[i] * B[i].

    Naive implementation:
    1. Contract corresponding physical indices -> chi_A * chi_B intermediate
    2. SVD truncate to max_bond

    This is O(L * chi^4) where L = number of sites.
    Robust to different tensor index orderings from quimb operations.

    Args:
        mps_a: first MPS
        mps_b: second MPS (must have same physical dimensions)
        max_bond: truncation bond dimension (required to avoid blowup)
        cutoff: SVD cutoff for small singular values

    Returns:
        mps_c: Hadamard product MPS
    """
    L = len(mps_a.tensors)
    site_ind_id = mps_a.site_ind_id  # e.g. 'k{}'

    new_tensors = []

    for i in range(L):
        A_t = mps_a[i]
        B_t = mps_b[i]

        # Find physical index by name (e.g. 'k0', 'k1', ...)
        phys_ind = site_ind_id.format(i)

        # Reorder A tensor to put physical index in standard position:
        # edge (2D): (phys, bond) for left, (bond, phys) for right
        # middle (3D): (bond_left, phys, bond_right)
        A_data = _reorder_tensor(A_t, phys_ind, i, L)
        B_data = _reorder_tensor(B_t, phys_ind, i, L)

        A_shape = A_data.shape
        B_shape = B_data.shape

        if i == 0:
            # Left edge: (phys, bond_right)
            C_data = np.einsum('pr,ps->prs', A_data, B_data)
            C_data = C_data.reshape(A_shape[0], A_shape[1] * B_shape[1])

        elif i == L - 1:
            # Right edge: (bond_left, phys)
            C_data = np.einsum('lp,mp->lmp', A_data, B_data)
            C_data = C_data.reshape(A_shape[0] * B_shape[0], A_shape[1])

        else:
            # Middle site: (bond_left, phys, bond_right)
            C_data = np.einsum('lpr,mps->lmprs', A_data, B_data)
            new_left = A_shape[0] * B_shape[0]
            new_phys = A_shape[1]
            new_right = A_shape[2] * B_shape[2]
            C_data = C_data.reshape(new_left, new_phys, new_right)

        new_tensors.append(C_data)

    mps_c = qtn.MatrixProductState(new_tensors, shape='lpr')

    if max_bond is not None:
        mps_c.compress(max_bond=max_bond, cutoff=cutoff)

    return mps_c


def _reorder_tensor(tensor, phys_ind, site_idx, _num_sites):
    """
    Reorder a quimb tensor to standard lpr format.

    Returns numpy array with shape:
    - Left edge (site 0): (phys, bond_right)
    - Middle: (bond_left, phys, bond_right)
    - Right edge (site L-1): (bond_left, phys)
    """
    inds = tensor.inds
    data = tensor.data
    phys_pos = inds.index(phys_ind)

    if len(inds) == 2:
        # Edge tensor: put phys first for left edge, last for right edge
        if site_idx == 0:
            # Want (phys, bond)
            if phys_pos != 0:
                data = data.T
        else:
            # Want (bond, phys)
            if phys_pos != 1:
                data = data.T
    elif len(inds) == 3:
        # Middle tensor: want (bond_left, phys, bond_right)
        if phys_pos != 1:
            # Move physical index to position 1
            data = np.moveaxis(data, phys_pos, 1)

    return data


def mps_subtract(mps_a, mps_b, max_bond=None, cutoff=1e-10):
    """
    Subtract two MPS: |C> = |A> - |B>.

    Args:
        mps_a: first MPS
        mps_b: second MPS
        max_bond: optional truncation
        cutoff: SVD cutoff

    Returns:
        mps_c: difference MPS
    """
    # Scale B by -1 and add
    mps_neg_b = mps_scale(mps_b, -1.0)
    return mps_add(mps_a, mps_neg_b, max_bond=max_bond, cutoff=cutoff)


def mps_inner(mps_a, mps_b):
    """
    Compute inner product <A|B> of two MPS.

    Args:
        mps_a: first MPS (will be conjugated)
        mps_b: second MPS

    Returns:
        scalar: inner product value
    """
    return mps_a.H @ mps_b


def mps_norm(mps):
    """
    Compute the 2-norm of an MPS: sqrt(<A|A>).

    Args:
        mps: input MPS

    Returns:
        scalar: norm value
    """
    return np.sqrt(np.abs(mps_inner(mps, mps)))


def mps_sum_value(mps):
    """
    Compute sum of all elements in MPS (contract with all-ones vector).

    This is useful for computing mean density: rho_0 = sum(rho) / N.

    Args:
        mps: input MPS

    Returns:
        scalar: sum of all elements
    """
    # Contract each site's physical index with [1,1] (sum over bits),
    # then contract bond indices left-to-right.
    # Robust to different tensor index orderings from quimb operations.
    L = len(mps.tensors)
    site_ind_id = mps.site_ind_id

    result = None
    for i in range(L):
        tensor = mps[i]
        phys_ind = site_ind_id.format(i)
        data = _reorder_tensor(tensor, phys_ind, i, L)

        # Sum over physical index (now at standard position after reorder)
        if i == 0:
            contracted = data.sum(axis=0)       # (phys, right) -> (right,)
        elif i == L - 1:
            contracted = data.sum(axis=1)       # (left, phys) -> (left,)
        else:
            contracted = data.sum(axis=1)       # (left, phys, right) -> (left, right)

        if result is None:
            result = contracted
        else:
            if i == L - 1:
                result = np.dot(result, contracted)
            else:
                result = np.einsum('l,lr->r', result, contracted)

    return float(result)

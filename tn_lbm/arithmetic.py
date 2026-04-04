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

    Args:
        mps_a: first MPS
        mps_b: second MPS (must have same physical dimensions)
        max_bond: truncation bond dimension (required to avoid blowup)
        cutoff: SVD cutoff for small singular values

    Returns:
        mps_c: Hadamard product MPS
    """
    L = len(mps_a.tensors)

    # Build the product MPS site by site
    # quimb MPS tensor shapes:
    # - Site 0 (left edge): (phys, bond_right)
    # - Site i (middle): (bond_left, phys, bond_right)
    # - Site L-1 (right edge): (bond_left, phys)

    new_tensors = []

    for i in range(L):
        A_tensor = mps_a[i].data
        B_tensor = mps_b[i].data

        A_shape = A_tensor.shape
        B_shape = B_tensor.shape

        if i == 0:
            # Left edge: (phys, bond_right)
            # C[p, ra*rb] = A[p, ra] * B[p, rb]
            if len(A_shape) == 2:
                C_data = np.einsum('pr,ps->prs', A_tensor, B_tensor)
                C_data = C_data.reshape(A_shape[0], A_shape[1] * B_shape[1])
            else:
                raise ValueError(f"Unexpected left edge shape: {A_shape}")

        elif i == L - 1:
            # Right edge: (bond_left, phys)
            # C[la*lb, p] = A[la, p] * B[lb, p]
            if len(A_shape) == 2:
                C_data = np.einsum('lp,mp->lmp', A_tensor, B_tensor)
                C_data = C_data.reshape(A_shape[0] * B_shape[0], A_shape[1])
            else:
                raise ValueError(f"Unexpected right edge shape: {A_shape}")

        else:
            # Middle site: (bond_left, phys, bond_right)
            # C[la*lb, p, ra*rb] = A[la, p, ra] * B[lb, p, rb]
            if len(A_shape) == 3:
                C_data = np.einsum('lpr,mps->lmprs', A_tensor, B_tensor)
                new_left = A_shape[0] * B_shape[0]
                new_phys = A_shape[1]
                new_right = A_shape[2] * B_shape[2]
                C_data = C_data.reshape(new_left, new_phys, new_right)
            else:
                raise ValueError(f"Unexpected middle shape: {A_shape}")

        new_tensors.append(C_data)

    # Build MPS from tensors using quimb constructor
    # Shape 'lpr' = (left_bond, physical, right_bond)
    mps_c = qtn.MatrixProductState(new_tensors, shape='lpr')

    # Compress to target bond dimension
    if max_bond is not None:
        mps_c.compress(max_bond=max_bond, cutoff=cutoff)

    return mps_c


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
    # Simple approach: convert to dense and sum
    # For large MPS this defeats the purpose, but for small test cases it works
    # TODO: Implement efficient contraction with ones vector

    # For now, use a manual contraction approach
    # Contract each site with the [1,1] vector on the physical index
    L = len(mps.tensors)

    # Start from left
    result = None
    for i in range(L):
        tensor_data = mps[i].data

        # Contract physical index with [1, 1]
        if i == 0:
            # Shape: (phys, right) -> sum over phys -> (right,)
            contracted = tensor_data.sum(axis=0)
        elif i == L - 1:
            # Shape: (left, phys) -> sum over phys -> (left,)
            contracted = tensor_data.sum(axis=1)
        else:
            # Shape: (left, phys, right) -> sum over phys -> (left, right)
            contracted = tensor_data.sum(axis=1)

        # Combine with running result
        if result is None:
            result = contracted
        else:
            if i == L - 1:
                # Final contraction: (left,) @ (left,) -> scalar
                result = np.dot(result, contracted)
            else:
                # (left_prev,) @ (left, right) -> (right,) via einsum
                result = np.einsum('l,lr->r', result, contracted)

    return float(result)

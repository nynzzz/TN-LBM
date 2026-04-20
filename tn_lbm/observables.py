"""
MPS-native observables for TN-LBM.

Extract physical quantities directly from MPS without decompression:
- Point evaluation: scalar value at a specific grid point
- Coarse field: lower-resolution field by contracting LSB sites
- Drag/lift: surface forces via momentum exchange method
- Convergence check: L2 norm of population change
"""

import numpy as np
import quimb.tensor as qtn

from .arithmetic import mps_add, mps_scale, mps_hadamard, mps_subtract, mps_norm, mps_sum_value


def mps_evaluate_at_index(mps, flat_index, num_sites):
    """
    Extract a single scalar value from MPS at a given flattened index.

    Contracts the MPS by selecting bit q_k at each site k,
    where (q_0, ..., q_{L-1}) is the binary representation of flat_index.

    Cost: O(L * chi^2) per point.

    Args:
        mps: quimb MatrixProductState
        flat_index: integer index into the flattened field
        num_sites: number of MPS sites (2L for 2D)

    Returns:
        scalar: float value at that index
    """
    # Convert flat index to binary bits (MSB first, matching QTT convention)
    bits = [(flat_index >> (num_sites - 1 - k)) & 1 for k in range(num_sites)]

    # Contract site-by-site
    site_ind_id = mps.site_ind_id  # e.g. 'k{}'
    result = None

    for k in range(num_sites):
        tensor = mps[k]
        phys_ind = site_ind_id.format(k)

        # Select the bit value at this site
        # Get the tensor data and find which axis is the physical index
        inds = tensor.inds
        phys_pos = inds.index(phys_ind)
        data = tensor.data

        # Select along the physical dimension
        sliced = np.take(data, bits[k], axis=phys_pos)

        # Contract with running result
        if result is None:
            result = sliced
        else:
            result = result @ sliced

    # Result should be a scalar
    return float(result.squeeze())


def mps_evaluate_at_point(mps, x, y, n, metadata):
    """
    Extract scalar value at grid point (x, y) via MPS contraction.

    Handles the mapping from (x, y) to flat index automatically.
    O(L * chi^2) per point. No decompression.

    Args:
        mps: MPS of a 2D field
        x, y: grid coordinates (0-indexed)
        n: grid side length
        metadata: from field_to_qtt (needs 'mapping', 'total_sites')

    Returns:
        scalar: float value at (x, y)
    """
    mapping = metadata.get('mapping', 'snake')
    num_sites = metadata.get('total_sites', metadata['L'])

    if mapping == 'snake':
        flat_index = x * n + y  # numpy C-order: I = x*N + y
    elif mapping == 'interleaved':
        from tn.mapping import _interleave_bits
        L = metadata['L']  # bits per coordinate
        # interleaved_flatten uses field_2d[y, x] with _interleave_bits(x_col, y_row, L)
        # Our LBM convention: axis-0 = x, axis-1 = y. So swap: y_row = x, x_col = y
        flat_index = _interleave_bits(y, x, L)
    else:
        flat_index = x * n + y  # default to snake

    return mps_evaluate_at_index(mps, flat_index, num_sites)


def mps_coarse_field(mps, metadata, coarse_level=1):
    """
    Extract a coarse-grained field by contracting LSB sites with [1,1] vectors.

    Each coarse level halves each spatial dimension. The coarse value at each
    cell is the SUM of the fine cells it contains.

    For snake mapping (2L sites: x-bits then y-bits in QTT order):
    - Contract the last `coarse_level` x-LSB sites (end of chain)
    - Contract the last `coarse_level` y-LSB sites (middle of chain)

    Args:
        mps: MPS of a 2D field
        metadata: from field_to_qtt
        coarse_level: number of levels to coarsen (k halves each dim k times)

    Returns:
        coarse_field: dense 2D array of shape (n/2^k, n/2^k)
    """
    mapping = metadata.get('mapping', 'snake')
    num_sites = metadata.get('total_sites', metadata['L'])

    if mapping == 'snake':
        L_coord = num_sites // 2
    elif mapping == 'interleaved':
        L_coord = num_sites // 2
    else:
        raise ValueError(f"Unsupported mapping: {mapping}")

    n_fine = 2 ** L_coord
    n_coarse = n_fine // (2 ** coarse_level)

    if coarse_level >= L_coord:
        raise ValueError(f"coarse_level={coarse_level} too large for L={L_coord}")

    # MPS-native coarsening: contract LSB sites with [1,1] sum vectors.
    # This marginalizes the fine-scale bits, producing a shorter MPS
    # representing the coarse field. No decompression needed.
    #
    # For snake: sites are [x_MSB...x_LSB | y_MSB...y_LSB]
    #   x-block LSBs are at positions L-1, L-2, ... (end of x-block)
    #   y-block LSBs are at positions 2L-1, 2L-2, ... (end of chain)
    # We contract from the RIGHT end first (y-LSBs), then x-LSBs.

    site_ind_id = mps.site_ind_id
    sum_vec = np.array([1.0, 1.0])

    # Work on raw tensor arrays for simplicity
    # Extract all tensor data in lpr order
    tensors = []
    for k in range(num_sites):
        t = mps[k]
        phys_ind = site_ind_id.format(k)
        data = t.data
        # Reorder to (left, phys, right) — handle edge cases
        phys_pos = t.inds.index(phys_ind)
        if len(t.inds) == 2:
            if k == 0:
                # Left edge: ensure (phys, right)
                if phys_pos != 0:
                    data = data.T
            else:
                # Right edge: ensure (left, phys)
                if phys_pos != 1:
                    data = data.T
        elif len(t.inds) == 3:
            if phys_pos != 1:
                data = np.moveaxis(data, phys_pos, 1)
        tensors.append(data)

    # Contract y-block LSBs (rightmost sites): sites 2L-1, 2L-2, ...
    for _ in range(coarse_level):
        last = tensors.pop()  # right edge: shape (left_bond, phys)
        # Contract phys with sum_vec: result shape (left_bond,)
        contracted = last @ sum_vec  # (left_bond,)
        # Absorb into new last tensor
        if tensors:
            prev = tensors[-1]
            if len(prev.shape) == 3:
                # Middle tensor (left, phys, right) @ contracted(right) → (left, phys)
                tensors[-1] = np.einsum('lpr,r->lp', prev, contracted)
            elif len(prev.shape) == 2 and len(tensors) == 1:
                # Left edge (phys, right) @ contracted(right) → (phys,)
                tensors[-1] = prev @ contracted

    # Contract x-block LSBs: these are at positions L-1, L-2, ...
    # In the current tensor list, they're at index L_coord-1, L_coord-2, ...
    # after we removed coarse_level y-sites from the end.
    # The x-block LSB is now at position L_coord - 1 (0-indexed), which is
    # in the middle of the chain. We need to contract it and merge with neighbors.
    for _ in range(coarse_level):
        x_lsb_pos = L_coord - 1  # last x-bit position
        if x_lsb_pos <= 0 or x_lsb_pos >= len(tensors) - 1:
            break

        lsb = tensors[x_lsb_pos]  # shape (left, phys, right)
        # Contract phys with sum_vec: (left, right)
        contracted = np.einsum('lpr,p->lr', lsb, sum_vec)
        # Absorb into left neighbor
        left_neighbor = tensors[x_lsb_pos - 1]
        if len(left_neighbor.shape) == 3:
            # (l, p, r) @ (r, r2) → (l, p, r2)
            tensors[x_lsb_pos - 1] = np.einsum('lpr,rs->lps', left_neighbor, contracted)
        elif len(left_neighbor.shape) == 2:
            # Left edge (p, r) @ (r, r2) → (p, r2)
            tensors[x_lsb_pos - 1] = left_neighbor @ contracted

        tensors.pop(x_lsb_pos)
        L_coord -= 1

    # Build coarse MPS and decompress
    coarse_mps = qtn.MatrixProductState(tensors, shape='lpr')
    coarse_dense = coarse_mps.to_dense().flatten()[:n_coarse * n_coarse]
    return coarse_dense.reshape(n_coarse, n_coarse)


def compute_drag_lift_mps(mps_list, lattice, boundary_mask_mps,
                           max_bond=None, cutoff=1e-10):
    """
    Compute drag and lift forces using momentum exchange method in MPS space.

    F_alpha = sum_i c_{i,alpha} * (f_i + f_ibar) at boundary nodes

    In MPS: for each population, mask the sum f_i + f_ibar to boundary nodes
    and contract to get the total.

    Args:
        mps_list: list of 9 population MPS
        lattice: D2Q9 lattice definition
        boundary_mask_mps: MPS of boundary mask (1 at boundary, 0 elsewhere)
        max_bond: bond dimension limit for Hadamard products
        cutoff: SVD cutoff

    Returns:
        (drag, lift): tuple of float scalars
    """
    drag = 0.0
    lift = 0.0

    for i in range(lattice.q):
        cx = float(lattice.c[i, 0])
        cy = float(lattice.c[i, 1])

        if cx == 0 and cy == 0:
            continue

        opp = lattice.opposite[i]

        # f_i + f_ibar at boundary
        f_sum = mps_add(mps_list[i], mps_list[opp], max_bond=max_bond, cutoff=cutoff)
        f_masked = mps_hadamard(f_sum, boundary_mask_mps, max_bond=max_bond, cutoff=cutoff)
        total = mps_sum_value(f_masked)

        drag += cx * total
        lift += cy * total

    return drag, lift


def check_convergence_mps(mps_list_new, mps_list_old, tol=1e-6,
                          mode='velocity', lattice=None,
                          moments_new=None, moments_old=None):
    """
    Check convergence via relative L2 norm. No decompression needed.

    Two modes:
    - 'populations': ||f_new - f_old||_2 / ||f_new||_2  (over all 9 populations)
    - 'velocity': ||rhou_new - rhou_old||_2 / ||rhou_new||_2  (momentum, approx velocity)
      For near-incompressible flows (rho ~ 1), momentum ~ velocity.
      Matches the standard LBM convergence criterion.

    For 'velocity' mode, pass moments from collide_bgk_mps(return_moments=True)
    to avoid recomputing them (zero extra cost).

    Args:
        mps_list_new: list of 9 current population MPS
        mps_list_old: list of 9 previous population MPS
        tol: convergence tolerance (default 1e-6, standard for LBM)
        mode: 'velocity' (default, standard) or 'populations'
        lattice: D2Q9 lattice definition (required for mode='velocity' without moments)
        moments_new: from collide_bgk_mps(return_moments=True).
                     (rho, rhou_x, rhou_y, u_x, u_y) uses actual velocity,
                     (rho, rhou_x, rhou_y) falls back to momentum approximation.
        moments_old: same for previous step

    Returns:
        (converged, relative_change): tuple of (bool, float)
    """
    if mode == 'populations':
        num_sq = 0.0
        den_sq = 0.0
        for i in range(len(mps_list_new)):
            diff = mps_subtract(mps_list_new[i], mps_list_old[i])
            num_sq += mps_norm(diff) ** 2
            den_sq += mps_norm(mps_list_new[i]) ** 2
        relative_change = np.sqrt(num_sq / (den_sq + 1e-30))
        return relative_change < tol, relative_change

    elif mode == 'velocity':
        # If moments provided (from collide_bgk_mps), use them directly — zero cost
        if moments_new is not None and moments_old is not None:
            # moments = (rho, rhou_x, rhou_y, u_x, u_y) if velocity available
            # or (rho, rhou_x, rhou_y) for momentum-only
            if len(moments_new) >= 5:
                # Use actual velocity (exact match with vanilla criterion)
                _, _, _, u_x_new, u_y_new = moments_new
                _, _, _, u_x_old, u_y_old = moments_old
            else:
                # Fall back to momentum (rho*u, approximate for rho ~ 1)
                u_x_new = moments_new[1]  # rhou_x
                u_y_new = moments_new[2]  # rhou_y
                u_x_old = moments_old[1]
                u_y_old = moments_old[2]

            num_sq = 0.0
            den_sq = 0.0
            for v_new, v_old in [(u_x_new, u_x_old), (u_y_new, u_y_old)]:
                diff = mps_subtract(v_new, v_old)
                num_sq += mps_norm(diff) ** 2
                den_sq += mps_norm(v_new) ** 2

            relative_change = np.sqrt(num_sq / (den_sq + 1e-30))
            return relative_change < tol, relative_change

        # Otherwise compute momentum from populations (more expensive)
        if lattice is None:
            raise ValueError("Need either moments or lattice for mode='velocity'")

        from .collision import compute_moments_mps
        _, rhou_x_new, rhou_y_new = compute_moments_mps(mps_list_new, lattice)
        _, rhou_x_old, rhou_y_old = compute_moments_mps(mps_list_old, lattice)

        num_sq = 0.0
        den_sq = 0.0
        for rhou_new, rhou_old in [(rhou_x_new, rhou_x_old),
                                    (rhou_y_new, rhou_y_old)]:
            diff = mps_subtract(rhou_new, rhou_old)
            num_sq += mps_norm(diff) ** 2
            den_sq += mps_norm(rhou_new) ** 2

        relative_change = np.sqrt(num_sq / (den_sq + 1e-30))
        return relative_change < tol, relative_change

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'velocity' or 'populations'.")

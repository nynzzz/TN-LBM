"""
MPS-native boundary conditions for TN-LBM.

Implements mask-based bounce-back following Gross et al. (arXiv:2512.07615)
Section 3.2.1, Eq 36 (non-periodic boundaries).

For each population i, the combined streaming + BC step is:
  f_i^new = m_B_i * b(f_ibar^pre) + S_hat(a_i) * f_i^pre

Where:
- m_B_i = per-direction boundary mask (1 where non-cyclic shift gives 0 for pop i)
- b() = boundary function (f_ibar for no-slip, f_ibar + corr for velocity BC)
- S_hat = non-cyclic shift MPO

The per-direction mask avoids double-counting: m_B_i is 1 ONLY at cells where
S_hat would give 0, so the two terms never overlap.
"""

import numpy as np

from .arithmetic import mps_add, mps_scale, mps_hadamard
from .streaming import stream_population_2d
from tn.compression import field_to_qtt


def mask_to_mps(mask_2d, mapping='snake', max_bond=None):
    """
    Convert a 2D boolean mask to MPS format.

    Args:
        mask_2d: boolean array of shape (n, n)
        mapping: spatial mapping, must match population MPS
        max_bond: optional bond dimension limit

    Returns:
        mps, metadata
    """
    return field_to_qtt(mask_2d.astype(np.float64), max_bond=max_bond, mapping=mapping)


def _build_direction_boundary_mask(n, cx, cy):
    """
    Build the boundary mask for a specific D2Q9 direction in a fully-walled cavity.

    m_B_i(x, y) = 1 where non-cyclic streaming of pop i gives 0
    (i.e., where pop i would come from outside the domain).

    Pop i streams from (x - cx, y - cy). This is outside if:
    - cx > 0 and x - cx < 0  (i.e., x = 0 for cx = +1)
    - cx < 0 and x - cx >= n (i.e., x = n-1 for cx = -1)
    - cy > 0 and y - cy < 0  (i.e., y = 0 for cy = +1)
    - cy < 0 and y - cy >= n (i.e., y = n-1 for cy = -1)

    Args:
        n: grid size
        cx, cy: velocity components

    Returns:
        mask: boolean array (n, n), True where pop comes from outside
    """
    mask = np.zeros((n, n), dtype=bool)

    if cx == 1:
        mask[0, :] = True       # x=0: comes from x=-1
    elif cx == -1:
        mask[n-1, :] = True     # x=n-1: comes from x=n

    if cy == 1:
        mask[:, 0] = True       # y=0: comes from y=-1
    elif cy == -1:
        mask[:, n-1] = True     # y=n-1: comes from y=n

    return mask


def precompute_cavity_bc(n, lattice, u_wall, rho_wall=1.0, mapping='snake'):
    """
    Precompute everything needed for cavity boundary conditions (Gross Eq 36).

    Creates per-direction boundary masks and velocity corrections.

    Args:
        n: grid size (n x n square cavity)
        lattice: D2Q9 lattice definition
        u_wall: wall velocity array, shape (2,) — typically (u_lid, 0)
        rho_wall: wall density (typically 1.0)
        mapping: spatial mapping ('snake' or 'interleaved')

    Returns:
        dict with:
            'masks': list of 9 (mps, metadata) for per-direction boundary masks
            'corr_mps': list of 9 correction MPS (None where correction=0)
            'mapping': mapping used
    """
    from simulations.lid_driven_cavity import create_cavity_walls
    _, lid = create_cavity_walls(n)

    masks = []
    corr_mps_list = []

    for i in range(lattice.q):
        cx = int(lattice.c[i, 0])
        cy = int(lattice.c[i, 1])

        if cx == 0 and cy == 0:
            # Rest population: no mask needed
            masks.append(None)
            corr_mps_list.append(None)
            continue

        # Per-direction boundary mask
        mask_2d = _build_direction_boundary_mask(n, cx, cy)
        mask_mps, metadata = mask_to_mps(mask_2d, mapping=mapping)
        masks.append((mask_mps, metadata))

        # Velocity correction at the lid
        # b_velocity = f_ibar + corr_i (Eq 38: corr = -(2 w_i rho_b / cs2) * c_i . u_b)
        # Only non-zero where this direction's mask overlaps with the lid
        lid_overlap = mask_2d & lid
        if np.any(lid_overlap):
            c_dot_u = lattice.c[i, 0] * u_wall[0] + lattice.c[i, 1] * u_wall[1]
            if abs(c_dot_u) > 1e-15:
                corr_scalar = (2.0 * lattice.w[i] * rho_wall / lattice.cs2) * c_dot_u
                lid_mask_mps, _ = mask_to_mps(lid_overlap, mapping=mapping)
                corr_mps_list.append(mps_scale(lid_mask_mps, corr_scalar))
            else:
                corr_mps_list.append(None)
        else:
            corr_mps_list.append(None)

    return {
        'masks': masks,
        'corr_mps': corr_mps_list,
        'mapping': mapping,
    }


def apply_mps_boundary(pre_streaming_list, lattice, bc_data, max_bond=None, cutoff=1e-10):
    """
    Combined streaming + boundary condition step (Gross et al. Eq 36).

    For each population i:
      f_i^new = m_B_i * f_ibar^pre + S_hat(a_i) * f_i^pre [+ corr_i at lid]

    m_B_i is the per-direction mask (1 only where non-cyclic shift gives 0).
    This avoids double-counting: mask and shift never overlap.

    Takes PRE-streaming (post-collision) populations.
    Returns POST-streaming, POST-BC populations.

    Args:
        pre_streaming_list: list of 9 MPS (post-collision, pre-streaming)
        lattice: D2Q9 lattice definition
        bc_data: output of precompute_cavity_bc
        max_bond: bond dimension limit
        cutoff: SVD cutoff

    Returns:
        new_list: list of 9 MPS (post-streaming, post-BC)
    """
    masks = bc_data['masks']
    corr_mps = bc_data['corr_mps']
    metadata = bc_data['masks'][1][1]  # metadata from first non-rest mask

    new_list = []

    for i in range(lattice.q):
        cx = int(lattice.c[i, 0])
        cy = int(lattice.c[i, 1])

        # Population 0 (rest): no streaming, no BC
        if cx == 0 and cy == 0:
            new_list.append(pre_streaming_list[i].copy())
            continue

        opp = lattice.opposite[i]
        mask_mps, mask_meta = masks[i]

        # Term 1: bounce-back at boundary (m_B_i * f_ibar^pre)
        bc_value = mps_hadamard(pre_streaming_list[opp], mask_mps,
                                max_bond=max_bond, cutoff=cutoff)

        # Add velocity correction at lid (if any)
        if corr_mps[i] is not None:
            bc_value = mps_add(bc_value, corr_mps[i],
                               max_bond=max_bond, cutoff=cutoff)

        # Term 2: non-cyclic streaming
        streamed = stream_population_2d(pre_streaming_list[i], metadata,
                                         cx, cy, max_bond=max_bond,
                                         cyclic=False)

        # Combine: no overlap between mask and shift
        result = mps_add(bc_value, streamed,
                          max_bond=max_bond, cutoff=cutoff)

        new_list.append(result)

    return new_list

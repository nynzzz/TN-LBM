"""
MPS-native boundary conditions for TN-LBM.

Implements boundary conditions from Gross et al. (arXiv:2512.07615) Section 3.2.1:

Eq 36 (non-periodic domain boundaries):
  f_i^new = m_B_i * b(f_i^pre) + S_hat(a_i) * f_i^pre
  Where b() depends on BC type:
    - No-slip (Eq 37): b = f_ibar
    - Velocity inlet (Eq 38): b = f_ibar + velocity correction
    - Pressure outlet (Eq 39): b = -f_i + equilibrium correction

Eq 40 (immersed objects):
  f_i^new = (1 - m_O) * f_i + S(a_i) * (m_O * f_ibar)
  Applied AFTER streaming. Cyclic shift reverses the streaming for solid cells.
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
                                         cyclic=False, cutoff=cutoff)

        # Combine: no overlap between mask and shift
        result = mps_add(bc_value, streamed,
                          max_bond=max_bond, cutoff=cutoff)

        new_list.append(result)

    return new_list


def precompute_cylinder_bc(n, lattice, u_inlet, rho_inlet=1.0,
                            cylinder_x=None, cylinder_y=None, cylinder_r=None,
                            mapping='snake'):
    """
    Precompute everything needed for cylinder flow boundary conditions.

    Uses Gross et al. Eq 36 with per-region b() functions:
    - Top/bottom walls (Eq 37): b = f_ibar
    - Inlet x=0 (Eq 38): b = f_ibar + velocity correction
    - Outlet x=n-1 (Eq 39): b = -f_i + f_eq(rho_0, u_b)
    Then Eq 40 for immersed cylinder.

    Args:
        n: grid size (n x n square domain)
        lattice: D2Q9 lattice definition
        u_inlet: inlet velocity (scalar, x-direction)
        rho_inlet: inlet density (default 1.0)
        cylinder_x, cylinder_y: cylinder center (default n/4, n/2)
        cylinder_r: cylinder radius (default n/16)
        mapping: spatial mapping

    Returns:
        dict with all precomputed BC data
    """
    from lbm.boundary import create_cylinder_mask, create_channel_walls
    from .arithmetic import mps_subtract
    from .collision import build_ones_mps

    if cylinder_r is None: cylinder_r = n / 16
    if cylinder_x is None: cylinder_x = n / 4
    if cylinder_y is None: cylinder_y = n / 2

    channel_walls = create_channel_walls(n, n)
    cylinder = create_cylinder_mask(n, n, cylinder_x, cylinder_y, cylinder_r)

    cs2 = lattice.cs2
    rho_0 = rho_inlet
    u_b = np.array([u_inlet, 0.0])  # outlet velocity approximation

    # --- Per-direction boundary data (Eq 36 with split regions) ---
    # For each non-rest direction, precompute:
    #   bounce_masks[i]: wall + inlet mask (both use f_ibar)
    #   inlet_corrs[i]: velocity correction MPS at inlet
    #   outlet_masks[i]: outlet region mask
    #   outlet_eqs[i]: f_eq * outlet_mask (precomputed)

    bounce_masks = []   # (mps, metadata) or None
    inlet_corrs = []    # mps or None
    outlet_masks = []   # (mps, metadata) or None
    outlet_eqs = []     # mps or None

    ref_metadata = None

    for i in range(lattice.q):
        cx = int(lattice.c[i, 0])
        cy = int(lattice.c[i, 1])

        if cx == 0 and cy == 0:
            bounce_masks.append(None)
            inlet_corrs.append(None)
            outlet_masks.append(None)
            outlet_eqs.append(None)
            continue

        # Full boundary mask for this direction
        full_mask = _build_direction_boundary_mask(n, cx, cy)

        # Split into regions (mutually exclusive, wall priority)
        wall_region = np.zeros((n, n), dtype=bool)
        inlet_region = np.zeros((n, n), dtype=bool)
        outlet_region = np.zeros((n, n), dtype=bool)

        if cx == 1:   inlet_region[0, :] = True
        if cx == -1:  outlet_region[n-1, :] = True
        if cy == 1:   wall_region[:, 0] = True
        if cy == -1:  wall_region[:, n-1] = True

        wall_region &= full_mask
        inlet_region &= full_mask
        outlet_region &= full_mask

        # Walls take priority at corners
        inlet_region &= ~wall_region
        outlet_region &= ~wall_region

        # Bounce mask: wall + inlet (both use f_ibar)
        bounce_2d = wall_region | inlet_region
        if np.any(bounce_2d):
            bm_mps, bm_meta = mask_to_mps(bounce_2d, mapping=mapping)
            bounce_masks.append((bm_mps, bm_meta))
            if ref_metadata is None:
                ref_metadata = bm_meta
        else:
            bounce_masks.append(None)

        # Inlet velocity correction (Eq 38)
        if np.any(inlet_region):
            c_dot_u = lattice.c[i, 0] * u_inlet  # u_inlet is x-only
            if abs(c_dot_u) > 1e-15:
                corr_scalar = (2.0 * lattice.w[i] * rho_inlet / cs2) * c_dot_u
                inlet_mask_mps, _ = mask_to_mps(inlet_region, mapping=mapping)
                inlet_corrs.append(mps_scale(inlet_mask_mps, corr_scalar))
            else:
                inlet_corrs.append(None)
        else:
            inlet_corrs.append(None)

        # Outlet mask and equilibrium (Eq 39)
        if np.any(outlet_region):
            om_mps, om_meta = mask_to_mps(outlet_region, mapping=mapping)
            outlet_masks.append((om_mps, om_meta))
            if ref_metadata is None:
                ref_metadata = om_meta

            # f_eq_out = 2 * w_i * rho_0 * (1 + cu^2/(2*cs4) - u^2/(2*cs2))
            cu_out = lattice.c[i, 0] * u_b[0] + lattice.c[i, 1] * u_b[1]
            usq_out = u_b[0]**2 + u_b[1]**2
            f_eq_out = 2.0 * lattice.w[i] * rho_0 * (
                1.0 + cu_out**2 / (2.0 * cs2**2) - usq_out / (2.0 * cs2)
            )
            outlet_eqs.append(mps_scale(om_mps, f_eq_out))
        else:
            outlet_masks.append(None)
            outlet_eqs.append(None)

    # --- Cylinder mask (Eq 40) ---
    cylinder_mps, cyl_metadata = mask_to_mps(cylinder, mapping=mapping)
    ones = build_ones_mps(len(cylinder_mps.tensors))
    fluid_mask_mps = mps_subtract(ones, cylinder_mps)

    if ref_metadata is None:
        ref_metadata = cyl_metadata

    return {
        # Eq 36: per-direction boundary data
        'bounce_masks': bounce_masks,
        'inlet_corrs': inlet_corrs,
        'outlet_masks': outlet_masks,
        'outlet_eqs': outlet_eqs,
        'mapping': mapping,
        'metadata': ref_metadata,
        # Eq 40: cylinder
        'cylinder_mps': cylinder_mps,
        'fluid_mask_mps': fluid_mask_mps,
        # Geometry
        'cylinder_2d': cylinder,
        'channel_walls_2d': channel_walls,
        'n': n,
    }


def apply_mps_cylinder_bc(pre_streaming_list, lattice, bc_data,
                           max_bond=None, cutoff=1e-10):
    """
    Combined streaming + boundary conditions for cylinder flow.

    Two steps:
    1. Eq 36: non-cyclic streaming + per-region b() at domain edges
       - Walls (Eq 37): b = f_ibar
       - Inlet (Eq 38): b = f_ibar + velocity correction
       - Outlet (Eq 39): b = -f_i + f_eq(rho_0, u_b)
    2. Eq 40: immersed cylinder correction

    Takes PRE-streaming (post-collision) populations.
    Returns POST-streaming, POST-BC populations.

    Args:
        pre_streaming_list: list of 9 MPS (post-collision)
        lattice: D2Q9 lattice definition
        bc_data: output of precompute_cylinder_bc
        max_bond: bond dimension limit
        cutoff: SVD cutoff

    Returns:
        new_list: list of 9 MPS (post-streaming, post-BC)
    """
    bounce_masks = bc_data['bounce_masks']
    inlet_corrs = bc_data['inlet_corrs']
    outlet_masks = bc_data['outlet_masks']
    outlet_eqs = bc_data['outlet_eqs']
    metadata = bc_data['metadata']
    cylinder_mps = bc_data['cylinder_mps']
    fluid_mask_mps = bc_data['fluid_mask_mps']

    # --- Step 1: Domain boundaries (Eq 36 with per-region b()) ---
    streamed_list = []

    for i in range(lattice.q):
        cx = int(lattice.c[i, 0])
        cy = int(lattice.c[i, 1])

        if cx == 0 and cy == 0:
            streamed_list.append(pre_streaming_list[i].copy())
            continue

        opp = lattice.opposite[i]

        # Start with zero bc contribution
        bc_value = None

        # Walls + inlet bounce-back: mask * f_ibar (Eq 37/38)
        if bounce_masks[i] is not None:
            bm_mps, _ = bounce_masks[i]
            bc_value = mps_hadamard(pre_streaming_list[opp], bm_mps,
                                    max_bond=max_bond, cutoff=cutoff)

        # Inlet velocity correction (Eq 38)
        if inlet_corrs[i] is not None:
            if bc_value is not None:
                bc_value = mps_add(bc_value, inlet_corrs[i],
                                   max_bond=max_bond, cutoff=cutoff)
            else:
                bc_value = inlet_corrs[i]

        # Outlet (Eq 39): -f_i + f_eq at outlet nodes
        if outlet_masks[i] is not None:
            om_mps, _ = outlet_masks[i]
            # -f_i at outlet
            neg_f_outlet = mps_hadamard(pre_streaming_list[i], om_mps,
                                         max_bond=max_bond, cutoff=cutoff)
            neg_f_outlet = mps_scale(neg_f_outlet, -1.0)
            # -f_i + f_eq at outlet
            outlet_bc = mps_add(neg_f_outlet, outlet_eqs[i],
                                 max_bond=max_bond, cutoff=cutoff)
            if bc_value is not None:
                bc_value = mps_add(bc_value, outlet_bc,
                                   max_bond=max_bond, cutoff=cutoff)
            else:
                bc_value = outlet_bc

        # Non-cyclic streaming
        streamed = stream_population_2d(pre_streaming_list[i], metadata,
                                         cx, cy, max_bond=max_bond,
                                         cyclic=False, cutoff=cutoff)

        # Combine: bc + streamed (no overlap between mask and shift)
        if bc_value is not None:
            result = mps_add(bc_value, streamed,
                              max_bond=max_bond, cutoff=cutoff)
        else:
            result = streamed
        streamed_list.append(result)

    # --- Step 2: Immersed cylinder (Eq 40) ---
    new_list = []

    for i in range(lattice.q):
        cx = int(lattice.c[i, 0])
        cy = int(lattice.c[i, 1])

        # (1 - m_O) * f_i — keep fluid, zero cylinder
        fluid_part = mps_hadamard(streamed_list[i], fluid_mask_mps,
                                   max_bond=max_bond, cutoff=cutoff)

        if cx == 0 and cy == 0:
            new_list.append(fluid_part)
            continue

        opp = lattice.opposite[i]

        # S(a_i) * (m_O * f_ibar) — bounce-back at cylinder, then shift
        cyl_bounce = mps_hadamard(streamed_list[opp], cylinder_mps,
                                   max_bond=max_bond, cutoff=cutoff)
        shifted_bounce = stream_population_2d(cyl_bounce, metadata,
                                               cx, cy, max_bond=max_bond,
                                               cyclic=True, cutoff=cutoff)

        result = mps_add(fluid_part, shifted_bounce,
                          max_bond=max_bond, cutoff=cutoff)
        new_list.append(result)

    return new_list

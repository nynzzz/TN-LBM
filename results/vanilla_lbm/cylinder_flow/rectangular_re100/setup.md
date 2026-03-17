# Cylinder Flow Simulation

## Lattice
- **Type**: D2Q9 (2D, 9 velocities)
- **Speed of sound**: cs² = 1/3

## Setup
- **Grid**: nx × ny = 600 × 180
- **Cylinder**:
  - Radius: r = ny/10 = 18
  - Diameter: D = 36
  - Center: (5D, ny/2) = (180, 90)
  - Domain: 5D upstream, ~11.7D downstream

## Physical Parameters
- **Reynolds number**: Re = 100
- **Inlet velocity**: u_inlet = 0.1
- **Kinematic viscosity**: ν = u_inlet × D / Re = 0.036
- **Relaxation time**: τ = 3ν + 0.5 = 0.608

## Boundary Conditions
1. **Inlet (left)**: Equilibrium BC with prescribed velocity profile
   - Small perturbation to break symmetry: u_y = 0.01 × u_inlet × (y - ny/2) / (ny/2)
2. **Outlet (right)**: Extrapolation (copy from second-to-last column)
3. **Top/Bottom walls**: Bounce-back (no-slip)
4. **Cylinder surface**: Bounce-back (no-slip)

## Simulation
- **Timesteps**: nt = 20,000
- **Save interval**: every 200 steps (100 frames total)

## Output Files
- `cylinder_flow.gif`: Combined animation showing:
  - Top panel: Velocity field (speed magnitude + vectors)
  - Bottom panel: Vorticity field
- `cylinder_velocity_field.png`: Final velocity field at t = 20,000
- `cylinder_streamlines.png`: Final streamlines at t = 20,000


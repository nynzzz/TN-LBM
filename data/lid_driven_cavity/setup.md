# Lid-Driven Cavity Experiments

## Problem Description
Square cavity with stationary walls on three sides and a moving lid on top.
Classic benchmark for incompressible flow validation.

## Lattice
- **Type**: D2Q9 (2D, 9 velocities)
- **Speed of sound**: cs² = 1/3

## Domain
- **Grid**: 256 × 256
- **Geometry**: Unit square cavity [0,1] × [0,1]

## Boundary Conditions
- **Top (lid)**: Moving wall bounce-back with u = (U, 0)
- **Left, Right, Bottom**: Stationary wall bounce-back (no-slip)
- **Corners**: Belong to stationary walls

## Physical Parameters
- **Lid velocity**: U = 0.1 (lattice units)
- **Mach number**: Ma = U√3 ≈ 0.17 (incompressible regime)
- **Reynolds number**: Re = U·L/ν where L = 256

## Experiments

### Re = 100 (`re100/`)
- Kinematic viscosity: ν = U·L/Re = 0.256
- Relaxation time: τ = 3ν + 0.5 = 1.268
- Timesteps: 150,000 (converged)
- Flow regime: Laminar, single primary vortex

### Re = 1000 (`re1000/`)
- Kinematic viscosity: ν = 0.0256
- Relaxation time: τ = 0.5768
- Timesteps: 150,000
- Flow regime: Laminar, primary vortex with secondary corner vortices

### Re = 5000 (`re5000/`)
- Kinematic viscosity: ν = 0.00512
- Relaxation time: τ = 0.5154
- Timesteps: 150,000
- Flow regime: Near-transitional, multiple vortices

## Validation
Compared against Ghia et al. (1982) benchmark data:
- u-velocity along vertical centerline (x = 0.5)
- v-velocity along horizontal centerline (y = 0.5)

Reference: Ghia, U., Ghia, K. N., & Shin, C. T. (1982).
High-Re solutions for incompressible flow using the Navier-Stokes equations
and a multigrid method. Journal of Computational Physics, 48(3), 387-411.

## Output Files (per Re)
- `cavity_streamlines.png`: Streamlines over speed magnitude
- `cavity_velocity_field.png`: Velocity vectors
- `cavity_validation.png`: Centerline profiles vs Ghia benchmark
- `cavity_development.gif`: Flow development animation

# D1Q3 LBM Diffusion Experiments

## Lattice
- **Type**: D1Q3 (1D, 3 velocities)
- **Velocities**: c = [-1, 0, 1]
- **Weights**: w = [1/6, 2/3, 1/6]
- **Speed of sound**: cs² = 1/3

## Boundary Conditions
- **Periodic** (via `np.roll` in streaming)

## Experiments

### 1. Gaussian Advection (`d1q3_advection.gif`)
- Grid: nx = 256
- Timesteps: nt = 400
- Relaxation time: τ = 0.8
- Kinematic viscosity: ν = (τ - 0.5) / 3 = 0.1
- Background velocity: u₀ = 0.1
- Initial condition: Gaussian density perturbation at x = nx/4, σ = 8

### 2. Diffusion Comparison (`d1q3_diffusion_comparison.png`)
- Grid: nx = 256
- Timesteps: nt = 500
- Relaxation times: τ = [0.6, 0.8, 1.0, 1.5]
- Corresponding viscosities: ν = [0.033, 0.1, 0.167, 0.333]
- Initial condition: Centered Gaussian (x₀ = nx/2, σ = 10), no flow (u = 0)

### 3. Conservation (`d1q3_conservation.png`)
- Grid: nx = 128
- Timesteps: nt = 500
- Relaxation time: τ = 0.8
- Background velocity: u₀ = 0.1
- Initial condition: Gaussian at x = nx/4, σ = 8
- Validates mass and momentum conservation (should be ~machine precision)

# D2Q9 LBM Diffusion Experiments

## Lattice
- **Type**: D2Q9 (2D, 9 velocities)
- **Velocities**:
  ```
  6  2  5
   \ | /
  3--0--1
   / | \
  7  4  8
  ```
- **Weights**: w₀ = 4/9, w₁₋₄ = 1/9, w₅₋₈ = 1/36
- **Speed of sound**: cs² = 1/3

## Boundary Conditions
- **Periodic** (via `np.roll` in streaming)

## Experiments

### 1. Gaussian Diffusion (`d2q9_gaussian_diffusion.gif`)
- Grid: nx × ny = 128 × 128
- Timesteps: nt = 300
- Relaxation time: τ = 0.8
- Kinematic viscosity: ν = (τ - 0.5) / 3 = 0.1
- Initial condition: 2D Gaussian blob at center (σ = 8), no flow (u = 0)
- Shows diffusion spreading

### 2. Shear Wave Decay (`d2q9_shear_wave.gif`)
- Grid: nx × ny = 128 × 64
- Timesteps: nt = 500
- Relaxation time: τ = 0.7
- Kinematic viscosity: ν = (τ - 0.5) / 3 ≈ 0.067
- Initial condition: Sinusoidal velocity profile uₓ = u₀ sin(ky), u₀ = 0.05
- Wavenumber: k = 2π/ny
- Shows viscous decay of shear wave

### 3. Shear Wave Validation (`d2q9_shear_wave_validation.png`)
- Grid: nx × ny = 64 × 64
- Timesteps: nt = 1000
- Relaxation time: τ = 0.8
- Kinematic viscosity: ν = 0.1
- Initial amplitude: u₀ = 0.01 (small for linear regime)
- Analytical solution: A(t) = A₀ exp(-νk²t)
- Validates LBM viscosity against analytical decay rate

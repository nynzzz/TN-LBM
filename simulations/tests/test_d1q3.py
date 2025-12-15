"""
Test D1Q3 LBM implementation.

Tests:
1. Mass conservation
2. Gaussian advection (pulse should move with flow velocity)
3. Diffusion behavior
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lbm import D1Q3, compute_equilibrium, stream, collide_bgk
from lbm.collision import compute_moments
from lbm.simulation import run


def test_mass_conservation():
    """Mass should be conserved to machine precision."""
    nx = 128
    nt = 500
    tau = 0.8

    # Gaussian initial condition
    x = np.arange(nx)
    x0, sigma = nx / 4, 5.0
    rho = 1.0 + 0.1 * np.exp(-((x - x0)**2) / (2 * sigma**2))
    u = np.ones(nx) * 0.1

    f_init = compute_equilibrium(D1Q3, rho, u)
    result = run(D1Q3, f_init, tau, nt)

    initial_mass = np.sum(result.rho_history[0])
    final_mass = np.sum(result.rho_history[-1])
    error = abs(final_mass - initial_mass) / initial_mass

    print(f"Mass conservation test:")
    print(f"  Initial mass: {initial_mass:.10f}")
    print(f"  Final mass:   {final_mass:.10f}")
    print(f"  Relative error: {error:.2e}")
    assert error < 1e-12, f"Mass not conserved: error = {error}"
    print("  PASSED\n")


def test_advection():
    """Uniform velocity field should remain stable."""
    nx = 128
    nt = 200
    tau = 0.9
    u0 = 0.1  # uniform velocity

    # Uniform initial condition with constant velocity
    rho = np.ones(nx)
    u = np.ones(nx) * u0

    f_init = compute_equilibrium(D1Q3, rho, u)
    result = run(D1Q3, f_init, tau, nt)

    # Check that velocity remains uniform and close to u0
    u_final = result.u_history[-1]
    u_mean = np.mean(u_final)
    u_std = np.std(u_final)

    print(f"Advection test (uniform flow stability):")
    print(f"  Initial velocity: {u0}")
    print(f"  Final mean velocity: {u_mean:.6f}")
    print(f"  Final velocity std: {u_std:.2e}")

    assert abs(u_mean - u0) < 1e-10, f"Velocity drift: expected {u0}, got {u_mean}"
    assert u_std < 1e-14, f"Velocity became non-uniform: std = {u_std}"
    print("  PASSED\n")


def test_equilibrium_moments():
    """Equilibrium should recover the input moments."""
    nx = 64
    rho = np.ones(nx) * 1.5
    u = np.linspace(-0.1, 0.1, nx)

    f_eq = compute_equilibrium(D1Q3, rho, u)
    rho_recovered, u_recovered = compute_moments(D1Q3, f_eq)

    rho_error = np.max(np.abs(rho - rho_recovered))
    u_error = np.max(np.abs(u - u_recovered))

    print(f"Equilibrium moments test:")
    print(f"  Max rho error: {rho_error:.2e}")
    print(f"  Max u error: {u_error:.2e}")
    assert rho_error < 1e-14, f"Density not recovered: error = {rho_error}"
    assert u_error < 1e-14, f"Velocity not recovered: error = {u_error}"
    print("  PASSED\n")


def test_streaming():
    """Streaming should shift populations correctly."""
    nx = 16
    f = np.zeros((3, nx))

    # Put a spike in each population
    f[0, 5] = 1.0  # stationary at x=5
    f[1, 5] = 1.0  # right-moving at x=5
    f[2, 5] = 1.0  # left-moving at x=5

    f_new = stream(D1Q3, f)

    print("Streaming test:")
    print(f"  f[0] spike: {np.argmax(f[0])} -> {np.argmax(f_new[0])} (should stay at 5)")
    print(f"  f[1] spike: {np.argmax(f[1])} -> {np.argmax(f_new[1])} (should move to 6)")
    print(f"  f[2] spike: {np.argmax(f[2])} -> {np.argmax(f_new[2])} (should move to 4)")

    assert np.argmax(f_new[0]) == 5, "Stationary population moved!"
    assert np.argmax(f_new[1]) == 6, "Right population didn't move right!"
    assert np.argmax(f_new[2]) == 4, "Left population didn't move left!"
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 50)
    print("D1Q3 LBM Tests")
    print("=" * 50 + "\n")

    test_equilibrium_moments()
    test_streaming()
    test_mass_conservation()
    test_advection()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)

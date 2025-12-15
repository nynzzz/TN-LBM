"""
Test D2Q9 LBM implementation.

Tests:
1. Equilibrium moments recovery
2. Streaming correctness
3. Mass conservation
4. Uniform flow stability
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lbm import D2Q9, compute_equilibrium, stream
from lbm.collision import compute_moments
from lbm.simulation import run


def test_equilibrium_moments():
    """Equilibrium should recover the input moments."""
    nx, ny = 32, 32
    rho = np.ones((nx, ny)) * 1.2
    u = np.zeros((2, nx, ny))
    u[0] = 0.05  # uniform ux
    u[1] = 0.03  # uniform uy

    f_eq = compute_equilibrium(D2Q9, rho, u)
    rho_recovered, u_recovered = compute_moments(D2Q9, f_eq)

    rho_error = np.max(np.abs(rho - rho_recovered))
    ux_error = np.max(np.abs(u[0] - u_recovered[0]))
    uy_error = np.max(np.abs(u[1] - u_recovered[1]))

    print("Equilibrium moments test:")
    print(f"  Max rho error: {rho_error:.2e}")
    print(f"  Max ux error:  {ux_error:.2e}")
    print(f"  Max uy error:  {uy_error:.2e}")
    assert rho_error < 1e-14, f"Density not recovered: error = {rho_error}"
    assert ux_error < 1e-14, f"ux not recovered: error = {ux_error}"
    assert uy_error < 1e-14, f"uy not recovered: error = {uy_error}"
    print("  PASSED\n")


def test_streaming():
    """Streaming should shift populations correctly in 2D."""
    nx, ny = 16, 16
    f = np.zeros((9, nx, ny))

    # Put a spike at (5, 5) for each population
    for i in range(9):
        f[i, 5, 5] = 1.0

    f_new = stream(D2Q9, f)

    # Expected positions after streaming
    # c = [[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]]
    expected = [
        (5, 5),   # 0: stationary
        (6, 5),   # 1: East (+1, 0)
        (5, 6),   # 2: North (0, +1)
        (4, 5),   # 3: West (-1, 0)
        (5, 4),   # 4: South (0, -1)
        (6, 6),   # 5: NE (+1, +1)
        (4, 6),   # 6: NW (-1, +1)
        (4, 4),   # 7: SW (-1, -1)
        (6, 4),   # 8: SE (+1, -1)
    ]

    print("Streaming test:")
    all_correct = True
    for i, (ex, ey) in enumerate(expected):
        pos = np.unravel_index(np.argmax(f_new[i]), f_new[i].shape)
        correct = (pos[0] == ex and pos[1] == ey)
        status = "✓" if correct else "✗"
        print(f"  f[{i}]: (5,5) -> {pos}, expected ({ex},{ey}) {status}")
        if not correct:
            all_correct = False

    assert all_correct, "Streaming positions incorrect!"
    print("  PASSED\n")


def test_mass_conservation():
    """Mass should be conserved to machine precision."""
    nx, ny = 64, 64
    nt = 200
    tau = 0.8

    # Gaussian density perturbation
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    x0, y0, sigma = nx/4, ny/4, 5.0
    rho = 1.0 + 0.1 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    u = np.zeros((2, nx, ny))
    u[0] = 0.05
    u[1] = 0.02

    f_init = compute_equilibrium(D2Q9, rho, u)
    result = run(D2Q9, f_init, tau, nt)

    initial_mass = np.sum(result.rho_history[0])
    final_mass = np.sum(result.rho_history[-1])
    error = abs(final_mass - initial_mass) / initial_mass

    print("Mass conservation test:")
    print(f"  Initial mass: {initial_mass:.10f}")
    print(f"  Final mass:   {final_mass:.10f}")
    print(f"  Relative error: {error:.2e}")
    assert error < 1e-12, f"Mass not conserved: error = {error}"
    print("  PASSED\n")


def test_uniform_flow():
    """Uniform velocity field should remain stable."""
    nx, ny = 64, 64
    nt = 200
    tau = 0.9
    ux0, uy0 = 0.08, 0.04

    rho = np.ones((nx, ny))
    u = np.zeros((2, nx, ny))
    u[0] = ux0
    u[1] = uy0

    f_init = compute_equilibrium(D2Q9, rho, u)
    result = run(D2Q9, f_init, tau, nt)

    u_final = result.u_history[-1]
    ux_mean = np.mean(u_final[0])
    uy_mean = np.mean(u_final[1])
    ux_std = np.std(u_final[0])
    uy_std = np.std(u_final[1])

    print("Uniform flow stability test:")
    print(f"  Initial: ux={ux0}, uy={uy0}")
    print(f"  Final mean: ux={ux_mean:.6f}, uy={uy_mean:.6f}")
    print(f"  Final std:  ux={ux_std:.2e}, uy={uy_std:.2e}")

    assert abs(ux_mean - ux0) < 1e-10, f"ux drift: expected {ux0}, got {ux_mean}"
    assert abs(uy_mean - uy0) < 1e-10, f"uy drift: expected {uy0}, got {uy_mean}"
    assert ux_std < 1e-14, f"ux became non-uniform: std = {ux_std}"
    assert uy_std < 1e-14, f"uy became non-uniform: std = {uy_std}"
    print("  PASSED\n")


def test_isotropy():
    """Check that D2Q9 lattice produces isotropic behavior."""
    nx, ny = 128, 128
    nt = 100
    tau = 0.8

    # Central Gaussian, no mean flow
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    x0, y0, sigma = nx/2, ny/2, 8.0
    rho = 1.0 + 0.2 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    u = np.zeros((2, nx, ny))

    f_init = compute_equilibrium(D2Q9, rho, u)
    result = run(D2Q9, f_init, tau, nt)

    # Check that final density is still approximately radially symmetric
    rho_final = result.rho_history[-1]

    # Sample points at same radius but different angles
    r = 20
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    values = []
    for theta in angles:
        ix = int(x0 + r * np.cos(theta))
        iy = int(y0 + r * np.sin(theta))
        values.append(rho_final[ix, iy])

    values = np.array(values)
    spread = np.max(values) - np.min(values)

    print("Isotropy test:")
    print(f"  Density at radius r={r} for different angles:")
    for theta, val in zip(angles, values):
        print(f"    θ = {np.degrees(theta):5.1f}°: ρ = {val:.6f}")
    print(f"  Max spread: {spread:.2e}")
    # D2Q9 has slight anisotropy at 45° due to lattice structure - this is expected
    assert spread < 2e-4, f"Anisotropic behavior: spread = {spread}"
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 50)
    print("D2Q9 LBM Tests")
    print("=" * 50 + "\n")

    test_equilibrium_moments()
    test_streaming()
    test_mass_conservation()
    test_uniform_flow()
    test_isotropy()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)

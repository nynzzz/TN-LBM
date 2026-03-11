"""
Find suitable T_sampling values for memory comparison experiments.

For each simulation type, run vanilla LBM and determine when to sample errors:
- Taylor-Green: Find time when flow is representative (after initial transient, before decay to noise)
- Cylinder: Find time when periodic vortex shedding is established
- Cavity: Find convergence time for steady-state solution

Usage:
    python find_t_sampling.py --sim taylor_green --N 256 --Re 100
    python find_t_sampling.py --sim cylinder --N 256 --Re 100
    python find_t_sampling.py --sim cavity --N 256 --Re 100
    python find_t_sampling.py --sim all
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def analyze_cavity(N: int, Re: int, nt: int = 100000, save_every: int = 500):
    """
    Run Lid-Driven Cavity and analyze convergence to find T_sampling.

    Strategy: Cavity reaches steady-state (converged solution).
    T_sampling = convergence time (when du/dt < threshold).
    """
    from simulations.lid_driven_cavity import run_lid_driven_cavity

    print(f"\n{'='*60}")
    print(f"Lid-Driven Cavity Analysis: N={N}, Re={Re}")
    print(f"{'='*60}")

    result = run_lid_driven_cavity(
        n=N, Re=Re, u_lid=0.1, nt=nt, save_every=save_every,
        convergence_threshold=1e-6, verbose=True
    )

    params = result['params']
    converged = result['converged']
    final_t = result['final_t']
    conv_threshold = 1e-6  # convergence threshold used

    # For cavity, T_sampling = convergence time
    # Recommended: run slightly beyond convergence for safety margin
    t_recommended = int(final_t * 1.1) if converged else final_t

    print(f"\nResults:")
    print(f"  tau = {params['tau']:.4f}")
    print(f"  Convergence threshold = {conv_threshold:.0e}")
    print(f"  Converged: {converged} at t = {final_t}")
    print(f"  Recommended T_sampling = {t_recommended}")

    return {
        'simulation': 'cavity',
        'N': N,
        'Re': Re,
        'tau': params['tau'],
        'convergence_threshold': conv_threshold,
        'converged': converged,
        't_converged': final_t,
        't_sampling_recommended': t_recommended,
    }


def analyze_taylor_green(N: int, Re: int, nt: int = 30000, save_every: int = 100):
    """
    Run Taylor-Green and analyze error evolution to find suitable T_sampling.

    Strategy: Sample at t where error is stable (not in initial transient,
    not decayed to numerical noise).
    """
    from simulations.taylor_green import run_taylor_green

    print(f"\n{'='*60}")
    print(f"Taylor-Green Analysis: N={N}, Re={Re}")
    print(f"{'='*60}")

    result = run_taylor_green(
        n=N, Re=Re, U0=0.1, nt=nt, save_every=save_every, verbose=True
    )

    times = np.array(result['times'])
    errors = np.array(result['error'])
    tau_decay = result['params']['tau_decay']

    # Compute max velocity at each time (to see decay)
    u_max = np.array([np.max(np.sqrt(u[0]**2 + u[1]**2)) for u in result['u']])

    # Find suitable sampling range:
    # - After t > 0.5 * tau_decay (initial transient passed)
    # - Before t < 4 * tau_decay (signal still strong, >2% of initial)
    t_min = 0.5 * tau_decay
    t_max = 4.0 * tau_decay

    # Find index range
    idx_min = np.searchsorted(times, t_min)
    idx_max = np.searchsorted(times, t_max)

    if idx_max <= idx_min:
        idx_max = len(times) - 1

    # Recommended T_sampling: middle of valid range (or at 2*tau_decay)
    t_recommended = min(2.0 * tau_decay, times[-1] * 0.5)
    idx_recommended = np.argmin(np.abs(times - t_recommended))

    # Get error at recommended time
    error_at_recommended = errors[idx_recommended]

    print(f"\nResults:")
    print(f"  tau_decay = {tau_decay:.0f} timesteps")
    print(f"  Valid sampling range: [{t_min:.0f}, {t_max:.0f}] timesteps")
    print(f"  Recommended T_sampling = {times[idx_recommended]:.0f} (t/tau = {times[idx_recommended]/tau_decay:.2f})")
    print(f"  Error at T_sampling = {error_at_recommended:.4e}")
    print(f"  Velocity at T_sampling = {u_max[idx_recommended]:.4f} ({u_max[idx_recommended]/u_max[0]*100:.1f}% of initial)")

    return {
        'simulation': 'taylor_green',
        'N': N,
        'Re': Re,
        'tau_decay': tau_decay,
        't_sampling_recommended': int(times[idx_recommended]),
        't_sampling_min': int(t_min),
        't_sampling_max': int(t_max),
        'error_at_t_sampling': error_at_recommended,
        'velocity_ratio_at_t_sampling': u_max[idx_recommended] / u_max[0],
        'times': times,
        'errors': errors,
        'u_max': u_max,
    }


def analyze_cylinder(N: int, Re: int, nt: int = 50000, save_every: int = 100):
    """
    Run Cylinder flow and analyze to find when periodic shedding is established.

    Strategy: Monitor velocity at a point downstream of cylinder.
    When oscillations become regular (periodic), transient has passed.
    """
    from simulations.cylinder_flow import run_cylinder_flow

    print(f"\n{'='*60}")
    print(f"Cylinder Flow Analysis: N={N}, Re={Re}")
    print(f"{'='*60}")

    result = run_cylinder_flow(
        n=N, Re=Re, u_inlet=0.1, nt=nt, save_every=save_every, verbose=True
    )

    params = result['params']
    D = 2 * params['cylinder_r']
    u_inlet = params['u_inlet']

    # Strouhal number estimate for Re~100
    St = 0.17  # typical for Re=100
    T_shed = D / (St * u_inlet)

    # Monitor velocity at probe point (downstream of cylinder, in wake)
    probe_x = int(params['cylinder_x'] + 3 * params['cylinder_r'])  # 1.5D downstream
    probe_y = int(params['cylinder_y'])  # centerline

    # Ensure probe is within bounds
    probe_x = min(probe_x, N - 1)

    # Extract v-velocity at probe (sensitive to shedding)
    v_probe = np.array([u[1, probe_x, probe_y] for u in result['u']])
    times = np.arange(0, len(v_probe)) * save_every

    # Detect when oscillations become regular
    # Use rolling std to find when amplitude stabilizes
    window = max(5, int(T_shed / save_every))  # one shedding period

    if len(v_probe) > 2 * window:
        rolling_std = np.array([
            np.std(v_probe[max(0, i-window):i+1])
            for i in range(len(v_probe))
        ])

        # Find when std stabilizes (changes less than 10% over 5 windows)
        std_change = np.abs(np.diff(rolling_std)) / (rolling_std[:-1] + 1e-10)

        # Moving average of std_change
        smooth_window = min(window, len(std_change) // 4)
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            std_change_smooth = np.convolve(std_change, kernel, mode='same')
        else:
            std_change_smooth = std_change

        # Find first time where change is consistently small
        threshold = 0.05  # 5% change
        stable_mask = std_change_smooth < threshold

        # Find first sustained stable region (at least 2 windows)
        stable_start = None
        for i in range(len(stable_mask) - window):
            if np.mean(stable_mask[i:i+window]) > 0.8:
                stable_start = i
                break

        if stable_start is not None:
            t_transient_end = times[stable_start]
        else:
            # Fallback: assume transient is ~15 shedding periods
            t_transient_end = 15 * T_shed
    else:
        t_transient_end = 15 * T_shed

    # Recommended T_sampling: after transient + 5 periods
    t_recommended = t_transient_end + 5 * T_shed
    t_recommended = min(t_recommended, times[-1])

    print(f"\nResults:")
    print(f"  Cylinder D = {D:.0f} lattice units")
    print(f"  Estimated shedding period T_shed = {T_shed:.0f} timesteps")
    print(f"  Transient ends at ~{t_transient_end:.0f} timesteps")
    print(f"  Recommended T_sampling = {t_recommended:.0f} timesteps")
    print(f"  (This is ~{t_recommended/T_shed:.1f} shedding periods)")

    return {
        'simulation': 'cylinder',
        'N': N,
        'Re': Re,
        'D': D,
        'T_shed_estimated': T_shed,
        't_transient_end': int(t_transient_end),
        't_sampling_recommended': int(t_recommended),
        'times': times,
        'v_probe': v_probe,
        'probe_location': (probe_x, probe_y),
    }


def plot_analysis(tg_result=None, cyl_result=None, cav_result=None, save_dir=None):
    """Generate analysis plots."""

    # Determine subplot layout based on available results
    n_plots = sum([tg_result is not None, cyl_result is not None, cav_result is not None])
    if n_plots == 3:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if tg_result is not None:
        tau = tg_result['tau_decay']
        times = tg_result['times']

        # Plot 1: Error evolution
        ax = axes[0, 0]
        ax.semilogy(times / tau, tg_result['errors'], 'b.-', markersize=3)
        ax.axvline(tg_result['t_sampling_recommended'] / tau, color='r',
                   linestyle='--', label=f"T_sampling = {tg_result['t_sampling_recommended']}")
        ax.axvspan(tg_result['t_sampling_min'] / tau, tg_result['t_sampling_max'] / tau,
                   alpha=0.2, color='green', label='Valid range')
        ax.set_xlabel('t / tau_decay')
        ax.set_ylabel('Relative L2 Error')
        ax.set_title(f"Taylor-Green Error (N={tg_result['N']}, Re={tg_result['Re']})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Velocity decay
        ax = axes[0, 1]
        ax.semilogy(times / tau, tg_result['u_max'], 'g.-', markersize=3, label='LBM')
        ax.semilogy(times / tau, tg_result['u_max'][0] * np.exp(-times / tau),
                   'k--', alpha=0.5, label='Analytical')
        ax.axvline(tg_result['t_sampling_recommended'] / tau, color='r', linestyle='--')
        ax.set_xlabel('t / tau_decay')
        ax.set_ylabel('Max velocity |u|')
        ax.set_title('Velocity Decay')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if cyl_result is not None:
        times = cyl_result['times']
        T_shed = cyl_result['T_shed_estimated']

        # Plot 3: Probe velocity
        ax = axes[1, 0]
        ax.plot(times / T_shed, cyl_result['v_probe'], 'b-', linewidth=0.5)
        ax.axvline(cyl_result['t_transient_end'] / T_shed, color='orange',
                   linestyle='--', label=f"Transient ends")
        ax.axvline(cyl_result['t_sampling_recommended'] / T_shed, color='r',
                   linestyle='--', label=f"T_sampling = {cyl_result['t_sampling_recommended']}")
        ax.set_xlabel('t / T_shed')
        ax.set_ylabel('v-velocity at probe')
        ax.set_title(f"Cylinder Wake (N={cyl_result['N']}, Re={cyl_result['Re']})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Zoom on periodic region
        ax = axes[1, 1]
        # Show last 10 periods
        t_start = max(0, cyl_result['t_sampling_recommended'] - 5 * T_shed)
        idx_start = np.searchsorted(times, t_start)
        ax.plot(times[idx_start:] / T_shed, cyl_result['v_probe'][idx_start:], 'b-')
        ax.axvline(cyl_result['t_sampling_recommended'] / T_shed, color='r', linestyle='--')
        ax.set_xlabel('t / T_shed')
        ax.set_ylabel('v-velocity at probe')
        ax.set_title('Periodic Shedding (zoomed)')
        ax.grid(True, alpha=0.3)

    if cav_result is not None:
        # For cavity, we show a summary text box instead of time evolution
        # (since it's a convergence problem, not time-dependent)
        if n_plots == 3:
            ax = axes[0, 2]
        else:
            ax = axes[1, 0] if tg_result is None else axes[1, 1]

        ax.axis('off')
        summary_text = (
            f"Lid-Driven Cavity (N={cav_result['N']}, Re={cav_result['Re']})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"τ (relaxation time): {cav_result['tau']:.4f}\n\n"
            f"Convergence threshold: {cav_result['convergence_threshold']:.0e}\n\n"
            f"Converged: {'Yes' if cav_result['converged'] else 'No'}\n\n"
            f"Convergence time: {cav_result['t_converged']} timesteps\n\n"
            f"Recommended T_sampling: {cav_result['t_sampling_recommended']}\n"
        )

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.set_title('Lid-Driven Cavity Summary')

        # Hide unused subplot in 2x3 layout
        if n_plots == 3:
            axes[1, 2].axis('off')
            axes[1, 2].set_visible(False)

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / 't_sampling_analysis.png'
        fig.savefig(save_path, dpi=150)
        print(f"\nPlot saved to {save_path}")

    plt.show()
    return fig


def save_results_csv(results: list, output_path: Path):
    """Save results to CSV."""
    if not results:
        return

    # Flatten results for CSV
    rows = []
    for r in results:
        row = {
            'simulation': r['simulation'],
            'N': r['N'],
            'Re': r['Re'],
            't_sampling_recommended': r['t_sampling_recommended'],
        }
        if r['simulation'] == 'taylor_green':
            row['tau_decay'] = r['tau_decay']
            row['t_sampling_min'] = r['t_sampling_min']
            row['t_sampling_max'] = r['t_sampling_max']
            row['error_at_t_sampling'] = r['error_at_t_sampling']
        elif r['simulation'] == 'cylinder':
            row['D'] = r['D']
            row['T_shed_estimated'] = r['T_shed_estimated']
            row['t_transient_end'] = r['t_transient_end']
        elif r['simulation'] == 'cavity':
            row['tau'] = r['tau']
            row['convergence_threshold'] = r['convergence_threshold']
            row['converged'] = r['converged']
            row['t_converged'] = r['t_converged']
        rows.append(row)

    # Write CSV
    fieldnames = ['simulation', 'N', 'Re', 't_sampling_recommended',
                  'tau_decay', 't_sampling_min', 't_sampling_max', 'error_at_t_sampling',
                  'D', 'T_shed_estimated', 't_transient_end',
                  'tau', 'convergence_threshold', 'converged', 't_converged']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Find T_sampling for memory comparison')
    parser.add_argument('--sim', type=str, choices=['taylor_green', 'cylinder', 'cavity', 'all'],
                        default='all', help='Simulation to analyze')
    parser.add_argument('--N', type=int, default=256, help='Grid size')
    parser.add_argument('--Re', type=float, default=100, help='Reynolds number')
    parser.add_argument('--nt', type=int, default=None, help='Number of timesteps')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    args = parser.parse_args()

    output_dir = Path(__file__).parent
    results = []
    tg_result = None
    cyl_result = None
    cav_result = None

    if args.sim in ['taylor_green', 'all']:
        nt = args.nt or 25000
        tg_result = analyze_taylor_green(args.N, int(args.Re), nt=nt)
        results.append(tg_result)

    if args.sim in ['cylinder', 'all']:
        nt = args.nt or 50000
        cyl_result = analyze_cylinder(args.N, int(args.Re), nt=nt)
        results.append(cyl_result)

    if args.sim in ['cavity', 'all']:
        nt = args.nt or 100000
        cav_result = analyze_cavity(args.N, int(args.Re), nt=nt)
        results.append(cav_result)

    # Save results
    output_path = args.output or output_dir / 't_sampling_results.csv'
    save_results_csv(results, Path(output_path))

    # Plot
    if args.plot or args.sim == 'all':
        plot_analysis(tg_result, cyl_result, cav_result, save_dir=output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Recommended T_sampling values")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['simulation']}: T_sampling = {r['t_sampling_recommended']} timesteps")


if __name__ == '__main__':
    main()

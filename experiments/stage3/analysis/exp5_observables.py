"""
Exp 5: observable extraction correctness.

For each Exp 5 job (chi=32 and chi=64 at multiple N for tg/cavity/cylinder),
load the saved MPS final state and compute each observable two ways:

  1. MPS-direct       — call the tn_lbm observable function on the compressed MPS
  2. MPS-decompressed — decompress to dense, then use numpy

Compare them to validate "no decompression needed". Cost / timing is *not*
included here because the implementation has not been optimised; reporting
timing now would be misleading. Headline claim: MPS-direct observables agree
with the decompressed reference to machine precision (mass) or to floating-
point evaluation precision (point eval, coarse field).

Outputs a markdown-formatted accuracy table (paste into the thesis) and a
plain-text version. Notebook figures cover the visual side (drag/lift,
coarse-field maps) — see refs/exp5_notes.md for which to use.

Usage:
    python -m experiments.stage3.analysis.exp5_observables
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lbm import D2Q9
from lbm.collision import compute_moments
from tn.compression import decompress_populations
from tn_lbm.observables import mps_evaluate_at_point, mps_coarse_field
from tn_lbm.arithmetic import mps_sum_value
from tn_lbm.collision import compute_moments_mps

_RESULTS_DIR = _THIS_DIR.parent / "results"
_CKPT_DIR = _THIS_DIR.parent / "checkpoints"
_BASELINE_DIR = _THIS_DIR.parent / "baselines"
_OUT_DIR = _ROOT / "results" / "stage3" / "tables"


def find_jobs():
    out = {}
    with open(_THIS_DIR.parent / 'jobs/master_jobs.csv') as f:
        for row in csv.DictReader(f):
            if row['chi'] not in ('32', '64'):
                continue
            tc = row['test_case']
            n = int(row['n'])
            chi = int(row['chi'])
            if tc in ('tg', 'cavity') and n in (32, 64, 128, 256, 512):
                pass
            elif tc == 'cylinder' and n in (128, 256):
                pass
            else:
                continue
            if abs(float(row['cutoff']) - 1e-10) > 1e-15:
                continue
            if tc != 'cylinder' and float(row['re']) != 100:
                continue
            key = (tc, n, chi)
            if key not in out:
                out[key] = (int(row['job_id']), float(row['re']), float(row['u']))
    return out


def benchmark_one(jid: int, tc: str, n: int, chi: int):
    ck = _CKPT_DIR / f'job_{jid:05d}_final.pkl'
    if not ck.exists():
        return None
    with open(ck, 'rb') as fh:
        d = pickle.load(fh)
    mps_list, metadata = d['mps_state'], d['metadata']

    f_dense = decompress_populations(mps_list, metadata)
    rho_dense, _ = compute_moments(D2Q9, f_dense)
    rho_mps, _, _ = compute_moments_mps(mps_list, D2Q9, max_bond=chi)

    # 1. Mass: exact summation
    mass_dense = float(np.sum(rho_dense))
    mass_mps = float(mps_sum_value(rho_mps))
    err_mass = abs(mass_mps - mass_dense) / (abs(mass_dense) + 1e-30)

    # 2. Point evaluation at center
    x, y = n // 2, n // 2
    val_dense = float(rho_dense[x, y])
    val_mps = float(mps_evaluate_at_point(rho_mps, x, y, n, metadata))
    err_pt = abs(val_mps - val_dense) / (abs(val_dense) + 1e-30)

    # 3. Coarse field at level 1
    k = n // 2
    coarse_dense = rho_dense.reshape(k, 2, k, 2).sum(axis=(1, 3))
    coarse_mps = np.asarray(mps_coarse_field(rho_mps, metadata, coarse_level=1))
    if coarse_mps.shape == coarse_dense.shape:
        err_coarse = float(np.linalg.norm(coarse_mps - coarse_dense)
                            / (np.linalg.norm(coarse_dense) + 1e-30))
    else:
        err_coarse = float('nan')

    return {'test_case': tc, 'n': n, 'chi': chi,
             'mass': err_mass, 'pt_eval': err_pt, 'coarse_l1': err_coarse}


def format_table_md(rows):
    """Markdown-formatted table for direct paste into the thesis."""
    lines = []
    lines.append("| test case | N | χ | mass (sum) | point eval | coarse field (level 1) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(f"| {r['test_case']} | {r['n']} | {r['chi']} | "
                     f"{r['mass']:.1e} | {r['pt_eval']:.1e} | {r['coarse_l1']:.1e} |")
    return '\n'.join(lines)


def format_table_latex(rows):
    """LaTeX tabular for thesis-direct use."""
    lines = []
    lines.append(r'\begin{tabular}{llrrrrr}')
    lines.append(r'\hline')
    lines.append(r'test case & $N$ & $\chi$ & mass (sum) & point eval & coarse field (level 1) \\')
    lines.append(r'\hline')
    for r in rows:
        lines.append(f"{r['test_case']} & {r['n']} & {r['chi']} & "
                     f"${r['mass']:.1e}$ & ${r['pt_eval']:.1e}$ & ${r['coarse_l1']:.1e}$ \\\\".replace('e-', r'\\cdot 10^{-').replace('+', '').rstrip(r'\\') + r'}$ \\')
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)


def main():
    jobs = find_jobs()
    print(f"Found {len(jobs)} Exp 5 jobs")
    rows = []
    for (tc, n, chi), (jid, _, _) in sorted(jobs.items()):
        r = benchmark_one(jid, tc, n, chi)
        if r is None:
            print(f"  {tc} N={n} chi={chi} (jid {jid}): SKIP (no _final.pkl)")
            continue
        rows.append(r)
        print(f"  {tc:<10} N={n:>3} chi={chi:>2}  "
              f"mass={r['mass']:.1e}  pt={r['pt_eval']:.1e}  coarse={r['coarse_l1']:.1e}")

    if not rows:
        return

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    md_path = _OUT_DIR / 'exp5_observable_accuracy.md'
    md_path.write_text(format_table_md(rows) + '\n')
    print(f"\nwrote markdown table: {md_path}")

    print("\n=== Accuracy table (markdown) ===\n")
    print(format_table_md(rows))


if __name__ == '__main__':
    main()

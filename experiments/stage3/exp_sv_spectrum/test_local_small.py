"""
Tiny smoke test for the SV-capture pipeline.

Runs TG N=32 Re=100 chi=16 for 500 steps with 3 snapshots, then verifies:
  - one pickle file per snapshot exists
  - each pickle has the expected keys
  - SVs have the expected shape (one array per bond)
  - decompressed populations have shape (9, N, N)
  - plot_spectrum runs and produces a PDF

Should run in well under a minute. Use this to verify the pipeline before
launching the full chi=17 / chi=12 cluster + local runs.

    python -m experiments.stage3.exp_sv_spectrum.test_local_small
"""

from __future__ import annotations

import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent.parent.parent
_DATA = _THIS.parent / "data" / "smoke_test"
_PLOTS = _THIS.parent / "plots"


def run():
    # Clean prior smoke-test output
    if _DATA.exists():
        shutil.rmtree(_DATA)
    print(f"cleaned {_DATA}")

    cmd = [
        sys.executable, "-m", "experiments.stage3.exp_sv_spectrum.run_sv_capture",
        "--chi", "16",
        "--run-name", "smoke_test",
        "--snapshot-times", "0", "100", "500",
        "--n", "32",
        "--re", "100",
    ]
    print(f"running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(_ROOT), check=True)
    print()

    # Verify outputs
    files = sorted(_DATA.glob("snap_t*.pkl"))
    assert len(files) == 3, f"expected 3 snapshot files, got {len(files)}"
    print(f"PASS: {len(files)} snapshot files exist")

    for f in files:
        snap = pickle.load(open(f, "rb"))
        for k in ("run_name", "chi", "t", "n", "re", "u", "cutoff",
                  "mps_state", "metadata", "svs_per_pop_per_bond",
                  "f_decompressed", "env"):
            assert k in snap, f"{f.name} missing key {k}"
        assert snap["chi"] == 16
        assert snap["n"] == 32
        assert len(snap["mps_state"]) == 9
        assert len(snap["svs_per_pop_per_bond"]) == 9
        for pop_svs in snap["svs_per_pop_per_bond"]:
            for sv in pop_svs:
                assert sv.ndim == 1
                assert np.all(sv >= 0)
        assert snap["f_decompressed"].shape == (9, 32, 32)
        print(f"PASS: {f.name}  t={snap['t']}  "
              f"mps_state ok, {len(snap['svs_per_pop_per_bond'][0])} bonds/pop, "
              f"decompressed shape {snap['f_decompressed'].shape}")

    # Test plot_spectrum runs
    plot_cmd = [
        sys.executable, "-m", "experiments.stage3.exp_sv_spectrum.plot_spectrum",
        "--runs", "smoke_test", "--population", "1",
    ]
    print()
    print(f"running: {' '.join(plot_cmd)}")
    subprocess.run(plot_cmd, cwd=str(_ROOT), check=True)
    plot_path = _PLOTS / "smoke_test_pop1_bond3.pdf"
    # default bond is middle (10 sites for N=32 → 9 bonds → middle=4)
    pdfs = list(_PLOTS.glob("smoke_test_pop1_bond*.pdf"))
    assert pdfs, f"no plot produced in {_PLOTS}"
    print(f"PASS: plot produced at {pdfs[0]}")
    print()
    print("all smoke-test checks passed.")


if __name__ == "__main__":
    run()

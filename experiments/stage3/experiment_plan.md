# Stage 3: MPS-Native LBM — Experiment Plan

---

## Overview

Systematic cluster experiments to answer 5 research questions about MPS-native LBM.
All experiments use **snake mapping** (justified by stage 1/2 results).
Output format: **JSON** per job (self-documenting, supports nested timeseries).
Cluster: SLURM array jobs, single-core per job, unlimited cores available.

### Research Questions

| RQ | Question | Experiments |
|----|----------|-------------|
| RQ1 | How do spatial mappings affect compression? | Stage 1/2 data |
| RQ2 | Can observables be computed without decompression? | Exp 5 |
| RQ3 | When does MPS become more memory-efficient than dense? | Exp 1, 2, 4 |
| RQ4 | What are the dominant error sources? | Exp 1, 3 |
| RQ5 | Can all LBM operations work in MPS accurately? | Exp 1, 2 |

### Three Test Cases

| Test Case | BC Type | Reference | Stopping |
|-----------|---------|-----------|----------|
| Taylor-Green | Periodic (no BC) | Analytical solution | Fixed nt = 2 × tau_decay |
| Lid-Driven Cavity | Walls (Eq 36/37/38) | Ghia et al. (1982) | Vanilla convergence (du < 1e-6) |
| Cylinder Flow | Open + immersed (Eq 36-40) | Vanilla FWBB | Fixed nt = t_transient + 5 × T_shed |

### Standard Chi Sweep Values
`[6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64]`

For large grids (N ≥ 256), extend to: `[..., 80, 96, 112, 128]`

---

## Experiment 1: Chi Sweep — Accuracy vs Compression

### Objective
Map the accuracy-compression trade-off for all 3 test cases across multiple (N, Re) combinations. Find chi_min for target error levels.

### Parameters

**Taylor-Green:**

| N | Re | tau | nt (2×tau_decay) |
|---|---|---|---|
| 64 | 100 | 0.692 | 1,622 |
| 128 | 100 | 0.884 | 3,242 |
| 256 | 100 | 1.268 | 6,484 |
| 64 | 500 | 0.538 | 8,108 |
| 128 | 500 | 0.577 | 16,216 |
| 256 | 500 | 0.654 | 32,432 |

**Cavity:**

| N | Re | tau | nt (est. convergence) |
|---|---|---|---|
| 64 | 100 | 0.692 | ~8,000 |
| 128 | 100 | 0.884 | ~25,000 |
| 256 | 100 | 1.268 | ~80,000 |
| 64 | 500 | 0.538 | ~30,000 |
| 128 | 500 | 0.577 | ~100,000 |
| 256 | 500 | 0.654 | ~250,000 |

Cavity nt determined by running vanilla first. MPS runs for same nt.

**Cylinder (TBD — implementation decision after TG+cavity results):**

| N | Re | tau | D | nt |
|---|---|---|---|---|
| 128 | 100 | 0.548 | 16 | 6,000 |
| 256 | 100 | 0.596 | 32 | 12,000 |

All use: u = 0.1, cutoff = 1e-10, mapping = snake.

### Chi sweep
16 values: `[6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64]`

### Snapshots
Every 50 steps (N=64), every 100 steps (N≥128).

### Job structure

**Phase 1: Vanilla baselines** (fast, run first)
- One job per (test_case, N, Re)
- Save final velocity field as .npy
- For cavity: record t_converged
- 6 + 6 + 2 = 14 baseline jobs

**Phase 2: MPS chi sweep** (main cluster work)
- One job per (test_case, N, Re, chi)
- Load vanilla reference, run MPS, save JSON
- TG: 6 × 16 = 96 jobs
- Cavity: 6 × 16 = 96 jobs
- Cylinder: 2 × 16 = 32 jobs (TBD)
- **Total: ~224 jobs**

### Data collected per job (JSON)

```json
{
  "experiment": "exp1_chi_sweep",
  "params": {
    "test_case": "cavity",
    "n": 64,
    "chi": 32,
    "cutoff": 1e-10,
    "mapping": "snake",
    "Re": 100,
    "tau": 0.692,
    "u": 0.1,
    "nt": 8000,
    "snapshot_every": 50
  },
  "vanilla": {
    "final_u_path": "baselines/vanilla_cavity_64_100.npy",
    "t_converged": 7200,
    "total_wall_time_s": 5.2
  },
  "mps": {
    "timeseries": [
      {
        "t": 0,
        "mean_chi": 12,
        "max_chi": 14,
        "per_pop_chi": [12, 14, 13, 14, 13, 12, 11, 12, 13],
        "l2_err_vs_vanilla": 0.0,
        "l2_err_vs_analytical": null,
        "mass": 4096.0,
        "mass_drift_rel": 0.0,
        "du": 1.0,
        "drag": null,
        "lift": null,
        "wall_time_cumulative_s": 0.0,
        "mps_memory_floats": 5184,
        "ghia_ux_centerline": null
      }
    ],
    "final": {
      "total_wall_time_s": 670,
      "wall_time_per_step_s": 0.42,
      "final_l2_error": 0.0035,
      "final_mass_drift_rel": 0.00012,
      "ghia_rms": 0.0052,
      "compression_ratio": 12.6
    }
  }
}
```

### Thesis output
- L2 error vs chi (log-log) — one subplot per test case, curves for each (N, Re)
- Compression ratio vs L2 error
- Per-population chi evolution over time
- Mass drift vs chi
- How error-vs-chi changes with N and Re
- Table: chi_min for {1%, 0.5%, 0.1%} error per (test_case, N, Re)

### Files
- `experiments/stage3/common.py`
- `experiments/stage3/exp1_chi_sweep.py`
- `experiments/stage3/exp1_baselines.py`
- `experiments/stage3/jobs/submit_exp1.sh`
- `experiments/stage3/jobs/params/exp1_params.csv`

---

## Experiment 2: Grid Scaling — Does Chi Scale as O(log N)?

### Objective
The central theoretical claim. Let chi grow naturally (max_bond=512, no forced truncation) and observe how the natural bond dimension scales with N. If chi_natural ~ O(log N), the method achieves logarithmic scaling.

### Approach
Unlike Exp 1 (fix chi, measure error), here we observe how much compression the flow naturally admits:
- Set max_bond = 512 (effectively unlimited)
- Fix cutoff (controls noise vs signal threshold)
- Run full MPS-native simulation to completion
- chi_natural = mean bond dimension at final time

### Standardized (N, Re) grid — same for TG and cavity

All pairs with tau = 0.3·N/Re + 0.5 in range (0.505, 2.0). u = 0.1 throughout.

| N | Re | tau | L (sites) | nt_tg (2×tau_d) | nt_cav (est.) |
|---|---|---|---|---|---|
| 32 | 100 | 0.596 | 10 | 810 | ~2,000 |
| 32 | 200 | 0.548 | 10 | 1,621 | ~4,000 |
| 32 | 400 | 0.524 | 10 | 3,242 | ~7,000 |
| 32 | 1000 | 0.510 | 10 | 8,105 | ~15,000 |
| 64 | 100 | 0.692 | 12 | 1,621 | ~8,000 |
| 64 | 200 | 0.596 | 12 | 3,242 | ~15,000 |
| 64 | 400 | 0.548 | 12 | 6,484 | ~25,000 |
| 64 | 1000 | 0.519 | 12 | 16,211 | ~60,000 |
| 64 | 3200 | 0.506 | 12 | 51,876 | ~150,000 |
| 128 | 100 | 0.884 | 14 | 3,242 | ~25,000 |
| 128 | 200 | 0.692 | 14 | 6,484 | ~50,000 |
| 128 | 400 | 0.596 | 14 | 12,969 | ~80,000 |
| 128 | 1000 | 0.538 | 14 | 32,422 | ~200,000 |
| 128 | 3200 | 0.512 | 14 | 103,752 | ~500,000 |
| 256 | 100 | 1.268 | 16 | 6,484 | ~80,000 |
| 256 | 200 | 0.884 | 16 | 12,969 | ~150,000 |
| 256 | 400 | 0.692 | 16 | 25,938 | ~200,000 |
| 256 | 1000 | 0.577 | 16 | 64,845 | ~500,000 |
| 256 | 3200 | 0.524 | 16 | 207,505 | ~1,500,000 |
| 512 | 200 | 1.268 | 18 | 25,938 | ~300,000 |
| 512 | 400 | 0.884 | 18 | 51,876 | ~500,000 |
| 512 | 1000 | 0.654 | 18 | 129,691 | ~1,500,000 |
| 512 | 3200 | 0.548 | 18 | 415,011 | ~5,000,000 |

**23 (N, Re) pairs.** All jobs capped at 48h.

### Cutoff values
- cutoff = 1e-10 (standard)
- cutoff = 1e-7 (aggressive — Taylor error is ~1e-6, so SVs below 1e-7 may be noise)

### Job structure
- TG: 23 × 2 cutoffs = **46 jobs**
- Cavity: 23 × 2 cutoffs = **46 jobs**
- Vanilla baselines: 46 total (one per (test_case, N, Re), shared with Exp 1)
- **Total: 92 MPS jobs + 46 baselines**

### Data per job
Same JSON as Exp 1. Key: `per_pop_chi`, `mean_chi`, `l2_err_vs_vanilla`, `wall_time_per_step_s`, `mps_memory_floats` at every snapshot.

### Thesis output
1. **THE KEY FIGURE:** chi_natural vs log2(N) — curves for each Re, subplots TG vs cavity
2. Memory crossover: MPS memory vs dense vs N
3. Chi evolution over time: does chi grow or stabilize?
4. Cutoff sensitivity: chi at 1e-10 vs 1e-7
5. Re effect on chi_natural
6. Wall time vs N
7. Table: chi_natural for all (test_case, N, Re, cutoff)

### Files
- `experiments/stage3/exp2_grid_scaling.py`
- `experiments/stage3/jobs/submit_exp2.sh`
- `experiments/stage3/jobs/params/exp2_params.csv`

---

## Experiment 3: Error Decomposition — Taylor vs Truncation vs BC

### Objective
MPS-native LBM has three error sources. Quantify each independently.

### Error sources
1. **Taylor 1/rho** — 2nd-order Taylor expansion in collision. Theoretical: O(Ma^6)
2. **SVD truncation** — bond dimension compression during Hadamard / additions
3. **Boundary conditions** — mask-based BC operations

### Isolation strategy
- **Taylor only**: TG (no BC) + full rank + cutoff=0 → pure Taylor error
- **Taylor + truncation**: TG + fixed chi/cutoff → subtract Taylor-only → truncation
- **Taylor + truncation + BC**: Cavity (same chi) → subtract TG error → BC contribution

### 3a: Taylor error scaling with Ma
TG periodic, full rank, cutoff=0. Vary u (Ma) at N=64 Re=100.

| u | Ma | tau | nt |
|---|---|---|---|
| 0.01 | 0.017 | 0.519 | 1,621 |
| 0.02 | 0.035 | 0.538 | 1,621 |
| 0.05 | 0.087 | 0.596 | 1,621 |
| 0.1 | 0.173 | 0.692 | 1,621 |
| 0.15 | 0.260 | 0.788 | 1,621 |

Also N sweep at u=0.1: N = 32, 64, 128, 256.

**Jobs:** 9. **Key output:** log(error) vs log(Ma) — slope ~6

### 3b: Truncation error (no BC)
TG, N=64, Re=100, u=0.1. Chi sweep × 3 cutoffs (1e-10, 1e-7, 0).
error_truncation = error_mps - error_taylor_only

**Jobs:** 16 × 3 = 48

### 3c: BC error
Cavity, N=64, Re=100, u=0.1. Chi sweep × 2 cutoffs (1e-10, 1e-7).
error_bc = error_cavity - error_tg (same chi)

**Jobs:** 16 × 2 = 32

### Total: 89 jobs

### Thesis output
1. Taylor error vs Ma (log-log, verify O(Ma^6))
2. Error decomposition stacked bar per chi
3. Truncation error vs chi for 3 cutoffs
4. Taylor error vs N
5. Table: dominant error source at chi = 16, 32, 64

### Files
- `experiments/stage3/exp3_error_decomposition.py`
- `experiments/stage3/jobs/submit_exp3.sh`
- `experiments/stage3/jobs/params/exp3_params.csv`

---

## Experiment 4: Memory-Equivalent Comparison (Stage 2 Upgraded)

### Objective
Given a fixed memory budget, is MPS-native at high resolution better than vanilla on a coarser grid? Upgrade of stage 2 (compress-decompress → full MPS-native).

### What we learn
- **RQ3**: Crossover chi where MPS beats coarse vanilla
- How full MPS-native compares to stage 2 compress-decompress
- Practical guidance for practitioners

### Approach
For each chi:
1. Compute MPS memory: `9 × L × chi²` with `L = 2·log₂(N)`
2. Find N_coarse = `√L × chi` = `4 × chi` (for N=256, L=16)
3. Run: fine vanilla (reference) + MPS-native + coarse vanilla
4. Compare L2 velocity error vs fine vanilla
5. Break-even at chi=64 (N_coarse=256=N_fine, CR=1.0)

### Parameters

**Grid:** N_fine = 256

**Re values:** 100, 500, 1000

**Chi sweep (16 values each, different per test case):**
- TG: `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]`
- Cavity: `[6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64]` (same as stage 2)

### Stopping criteria
- **TG**: 2×tau_decay per grid (same approach as stage 2)
- **Cavity**: vanilla convergence (du < 1e-6), MPS runs same nt

### Job structure
- TG: 3 Re × 16 chi = **48 jobs**
- Cavity: 3 Re × 16 chi = **48 jobs**
- Vanilla baselines: 6 (shared with other experiments)
- **Total: 96 jobs**

### Data per job (JSON)
```json
{
  "experiment": "exp4_memory_comparison",
  "params": {"test_case", "n_fine": 256, "chi", "Re", "cutoff": 1e-10},
  "memory": {
    "mps_floats": 36864,
    "dense_fine_floats": 589824,
    "n_coarse": 64,
    "dense_coarse_floats": 36864,
    "compression_ratio": 16.0
  },
  "errors": {
    "mps_vs_fine": 0.0035,
    "coarse_vs_fine": 0.109
  },
  "timing": {
    "fine_vanilla_wall_s": 15.0,
    "mps_wall_s": 3600,
    "coarse_vanilla_wall_s": 0.5
  }
}
```

### Comparison with Stage 2
Overlay on stage 2 CSV data (`experiments/stage2/results/`) to show: does full MPS-native do better or worse than compress-decompress?

### Thesis output
1. Error vs memory — MPS-native vs coarse vanilla (per test case, per Re)
2. Overlay with stage 2 compress-decompress (dashed)
3. Crossover chi table per (test_case, Re)
4. Discussion: accumulated Taylor error (MPS-native) vs fresh compression (stage 2)

### Files
- `experiments/stage3/exp4_memory_comparison.py`
- `experiments/stage3/jobs/submit_exp4.sh`
- `experiments/stage3/jobs/params/exp4_params.csv`

---

## Experiment 5: Observable Validation & Scaling

### Objective
Verify MPS-native observables (point eval, coarse field, drag/lift, mass, convergence) match decompressed values. Measure observable extraction cost vs N.

### Approach
- Run MPS at multiple N values
- Extract observables from MPS AND from decompressed state
- Compare accuracy and measure timing

### Thesis output
- Observable accuracy table
- Observable cost vs N (should show O(log N · chi²))

### Status: outlined, details TBD

---

## Implementation Order

1. `common.py` — shared infrastructure
2. Experiment 1 (chi sweep) — validates pattern, most data
3. Experiment 4 (memory comparison) — extends stage 2
4. Experiment 2 (grid scaling) — key thesis figure
5. Experiment 3 (error decomposition) — understanding
6. Experiment 5 (observables) — formalize notebook results

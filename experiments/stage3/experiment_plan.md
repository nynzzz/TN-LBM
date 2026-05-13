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

**Cylinder is NOT in Exp 1.** Cylinder appears only in Exp 5 — its unique
value is the drag/lift observable on the immersed body, not the chi/error
curve (which cavity already covers). Removed from Exp 1 to save ~32 jobs.

All TG/Cavity Exp 1 jobs use: u = 0.1, cutoff = 1e-10, mapping = snake.

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
- **Total: 192 jobs** (cylinder removed — only in Exp 5)

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

**Grid extended from 23 → 34 (N, Re) pairs** (May 2026): added N=16 column,
Re=50 column, Re=800 / Re=2000 mid points. Aim is denser sampling for cleaner
log/power-law fits per Re curve. All additions respect tau ≥ 0.51 stability.

| N | Re | tau | L (sites) | nt_tg (2×tau_d) | nt_cav (est.) |
|---|---|---|---|---|---|
| 16 | 50 | 0.596 | 8 | 203 | ~500 |
| 16 | 100 | 0.548 | 8 | 405 | ~1,000 |
| 16 | 200 | 0.524 | 8 | 810 | ~2,000 |
| 32 | 50 | 0.692 | 10 | 405 | ~1,000 |
| 32 | 100 | 0.596 | 10 | 810 | ~2,000 |
| 32 | 200 | 0.548 | 10 | 1,621 | ~4,000 |
| 32 | 400 | 0.524 | 10 | 3,242 | ~7,000 |
| 32 | 1000 | 0.510 | 10 | 8,105 | ~15,000 |
| 64 | 50 | 0.884 | 12 | 810 | ~2,000 |
| 64 | 100 | 0.692 | 12 | 1,621 | ~8,000 |
| 64 | 200 | 0.596 | 12 | 3,242 | ~15,000 |
| 64 | 400 | 0.548 | 12 | 6,484 | ~25,000 |
| 64 | 800 | 0.524 | 12 | 12,969 | ~50,000 |
| 64 | 1000 | 0.519 | 12 | 16,211 | ~60,000 |
| 64 | 3200 | 0.506 | 12 | 51,876 | ~150,000 |
| 128 | 50 | 1.268 | 14 | 1,621 | ~12,000 |
| 128 | 100 | 0.884 | 14 | 3,242 | ~25,000 |
| 128 | 200 | 0.692 | 14 | 6,484 | ~50,000 |
| 128 | 400 | 0.596 | 14 | 12,969 | ~80,000 |
| 128 | 800 | 0.548 | 14 | 25,938 | ~150,000 |
| 128 | 1000 | 0.538 | 14 | 32,422 | ~200,000 |
| 128 | 2000 | 0.519 | 14 | 64,844 | ~350,000 |
| 128 | 3200 | 0.512 | 14 | 103,752 | ~500,000 |
| 256 | 100 | 1.268 | 16 | 6,484 | ~80,000 |
| 256 | 200 | 0.884 | 16 | 12,969 | ~150,000 |
| 256 | 400 | 0.692 | 16 | 25,938 | ~200,000 |
| 256 | 800 | 0.596 | 16 | 51,876 | ~350,000 |
| 256 | 1000 | 0.577 | 16 | 64,845 | ~500,000 |
| 256 | 2000 | 0.538 | 16 | 129,690 | ~800,000 |
| 256 | 3200 | 0.524 | 16 | 207,505 | ~1,500,000 |
| 512 | 200 | 1.268 | 18 | 25,938 | ~300,000 |
| 512 | 400 | 0.884 | 18 | 51,876 | ~500,000 |
| 512 | 1000 | 0.654 | 18 | 129,691 | ~1,500,000 |
| 512 | 3200 | 0.548 | 18 | 415,011 | ~5,000,000 |

**34 (N, Re) pairs.** All jobs capped at 48h (use checkpointing).

### Cutoff values
- cutoff = 1e-10 (standard)
- cutoff = 1e-7 (aggressive — Taylor error is ~1e-6, so SVs below 1e-7 may be noise)

### Job structure
- TG: 34 × 2 cutoffs = **68 jobs**
- Cavity: 34 × 2 cutoffs = **68 jobs**
- Vanilla baselines: 68 total (one per (test_case, N, Re), shared with Exp 1)
- **Total: 136 MPS jobs + 68 baselines**

### Per-Re point counts (for the log/power-law fits)
- Re=50: 4 N values (16, 32, 64, 128)
- Re=100: 5 N values (16, 32, 64, 128, 256)
- Re=200: 6 N values (16, 32, 64, 128, 256, 512)
- Re=400: 5 N values (32, 64, 128, 256, 512)
- Re=800: 3 N values (64, 128, 256) — sparse
- Re=1000: 5 N values (32, 64, 128, 256, 512)
- Re=2000: 2 N values (128, 256) — sparse
- Re=3200: 4 N values (64, 128, 256, 512)

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
1. **Taylor 1/rho** — Taylor expansion in collision. Gross paper Appendix B reports fitted exponents 2.4 (1st), 4.5 (2nd), 6.5 (3rd order)
2. **SVD truncation** — bond dimension compression during Hadamard / additions
3. **Boundary conditions** — mask-based BC operations

### Isolation strategy
- **Taylor only**: TG (no BC) + full rank + cutoff=0 → pure Taylor error
- **Taylor + truncation**: TG + fixed chi/cutoff → subtract Taylor-only → truncation
- **Taylor + truncation + BC**: Cavity (same chi) → subtract TG error → BC contribution

### 3a: Taylor error scaling with Ma — replicate Gross Appendix B
TG periodic, full rank (max_bond=None), cutoff=0. Test all three Taylor orders.

**Code change required:** extend `compute_inverse_density_mps` to support `order` parameter:
- 1st order: `1/rho ≈ 1/rho_0 - delta/rho_0²`
- 2nd order (current): `... + delta²/rho_0³`
- 3rd order: `... - delta³/rho_0⁴`

**Ma sweep** (fixed N=64, Re=100, vary u):

| u | Ma | tau | nt |
|---|---|---|---|
| 0.01 | 0.017 | 0.519 | 1,621 |
| 0.02 | 0.035 | 0.538 | 1,621 |
| 0.05 | 0.087 | 0.596 | 1,621 |
| 0.1 | 0.173 | 0.692 | 1,621 |
| 0.15 | 0.260 | 0.788 | 1,621 |

Run for each Taylor order: 1st, 2nd, 3rd. Plot log(error) vs log(Ma).

**Expected slopes** (from Gross Table B.1):
- 1st order → ~2.4
- 2nd order → ~4.5
- 3rd order → ~6.5

**N sweep** (grid independence at u=0.1, all 3 orders): N = 32, 64, 128, 256.

**Jobs:** 5 (Ma) × 3 (orders) + 4 (N) × 3 (orders) = **27 jobs** (fast, full rank at small grids)

**Key output:** Reproduce Gross Fig B.10 — confirms our implementation matches their published results.

### 3b: Truncation error (no BC)
TG, N=64, Re=100, u=0.1. Chi sweep × 3 cutoffs (1e-10, 1e-7, 0). Use 2nd-order Taylor (default).
error_truncation = error_mps - error_taylor_only

**Jobs:** 16 × 3 = 48

### 3c: BC error
Cavity, N=64, Re=100, u=0.1. Chi sweep × 2 cutoffs (1e-10, 1e-7). Use 2nd-order Taylor.
error_bc = error_cavity - error_tg (same chi)

**Jobs:** 16 × 2 = 32

### Total: 107 jobs

### Thesis output
1. **Fig:** Taylor error vs Ma (log-log) for orders 1/2/3 — reproduce Gross Fig B.10
2. **Table:** Fitted exponents per order — compare with Gross Table B.1
3. **Fig:** Error decomposition stacked bar per chi (Taylor + Truncation + BC)
4. **Fig:** Truncation error vs chi for 3 cutoffs
5. **Fig:** Taylor error vs N (grid independence)
6. **Table:** Dominant error source at chi = 16, 32, 64

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
1. Compute MPS memory using **Gross Eq 46** (exact NVPS), adapted for our snake encoding:
   ```
   NVPS_MPS = Σ_{k=1..K} min(p^{k-1}, p^{K-k+1}, χ) × p × min(p^k, p^{K-k}, χ)
   ```
   Per population, then × 9 populations.
   - **D**: spatial dimensions (D=2 for 2D LBM, D=3 for 3D)
   - **K**: number of MPS sites
   - **p**: physical dimension per site
   - **Gross's interleaved encoding**: K = log₂(N), p = 2^D (each site holds D bits, one per dimension)
   - **Our snake encoding**: K = D · log₂(N), p = 2 (each site holds 1 bit)
   - Both encodings give very similar NVPS totals for a given field — the bond cap structure mirrors

2. Find N_coarse = `floor(√(NVPS_MPS / 9))` (memory-equivalent dense grid)
3. Run: fine vanilla (reference) + MPS-native + coarse vanilla
4. Compare L2 velocity error vs fine vanilla

**Why Eq 46, not the simpler formula?** Our previous formula `9 × L × χ²` was wrong — it missed the physical dim factor (`χ × p × χ` per middle tensor) AND ignored edge bond caps (where bonds are bounded by `p^k`, often smaller than χ). These errors partially cancel but don't yield exact NVPS. For absolute comparisons we use Eq 46.

**Snake-encoded NVPS table** (N=256, K=2·log₂(256)=16, p=2):

| χ at N=256 | NVPS (per pop) | × 9 | CR vs dense (589,824) |
|---|---|---|---|
| 16 | 4,776 | 42,984 | 13.72 |
| 32 | 15,016 | 135,144 | 4.36 |
| 64 | 43,688 | 393,192 | 1.50 |
| 80 | ~60,000 | ~540,000 | ~1.09 (near break-even) |
| 96 | 72,360 | 651,240 | 0.91 (MPS slightly more) |
| 128 | 109,224 | 983,016 | 0.60 (MPS more memory) |

Break-even at χ ≈ 80 (vs χ=64 with our old simplified formula).

For comparison, interleaved encoding gives slightly lower NVPS (e.g. 41,504 per pop at χ=64 vs 43,688 for snake), with break-even at χ ≈ 96. Both are exact under Eq 46, just adapted to the encoding.

### Parameters

**Grid:** N_fine = 256

**Re values:** 100, 500, 1000

**Chi sweep (16 values each, different per test case):**
- TG: `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]`
- Cavity: `[6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80]` (extended to snake break-even at ~80)

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
    "mps_floats": 373536,
    "dense_fine_floats": 589824,
    "n_coarse": 203,
    "dense_coarse_floats": 371007,
    "compression_ratio": 1.58,
    "formula": "Gross Eq 46"
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

**Note**: stage 2 used the simpler `9·L·χ²` formula. When overlaying, recompute stage 2's N_coarse using Gross Eq 46 for fair comparison, OR plot "error vs NVPS" directly (memory on x-axis instead of N_coarse).

### Thesis output
1. Error vs memory — MPS-native vs coarse vanilla (per test case, per Re)
2. Overlay with stage 2 compress-decompress (dashed) — using consistent NVPS formula
3. Crossover chi table per (test_case, Re) — using Gross Eq 46
4. Discussion: accumulated Taylor error (MPS-native) vs fresh compression (stage 2)

### Files
- `experiments/stage3/exp4_memory_comparison.py`
- `experiments/stage3/jobs/submit_exp4.sh`
- `experiments/stage3/jobs/params/exp4_params.csv`

---

## Experiment 5: Observable Validation & Scaling

### Objective
Verify MPS-native observables work correctly and scale efficiently. Two questions:
1. **Correctness**: do MPS observables match what you'd compute from decompressed field?
2. **Physical accuracy + cost**: how much error vs vanilla, and how does extraction time scale with N?

### What we learn
- **RQ2**: Can observables be computed without decompression?
- Validates the "never decompress" claim with hard numbers
- Cost scaling: O(log N · chi²) per observable vs O(N²) for decompression

### Three-way comparison
For each observable:
1. **MPS direct** — e.g. `mps_evaluate_at_point()` from compressed state
2. **MPS decompressed** — `decompress_populations()` then numpy op on dense array
3. **Vanilla** — from dense reference simulation (load from Exp 1/2 baselines)

Error decomposition:
- `MPS vs MPS-decompressed` → implementation correctness (target: ~1e-15 machine precision)
- `MPS-decompressed vs Vanilla` → MPS solution error (Taylor + truncation + BC)
- `MPS vs Vanilla` → total error

### Observables tested

| Observable | MPS function | Dense equivalent | Cost (theory) |
|---|---|---|---|
| Point evaluation | `mps_evaluate_at_point` | `field[x, y]` | O(L · chi²) |
| Mass (sum) | `mps_sum_value` | `np.sum(field)` | O(L · chi²) |
| Coarse field (lvl k) | `mps_coarse_field(k)` | reshape + sum | O(L · chi²) |
| Drag/lift | `compute_drag_lift_mps` | momentum exchange | O(L · chi² · #boundary_pops) |
| Convergence du | `check_convergence_mps` | L2 norm of u diff | O(L · chi³) |
| Inverse density | `compute_inverse_density_mps` | `1/rho` | O(L · chi³) (Taylor approx) |

### Parameters

| N | L | Test cases |
|---|---|---|
| 32 | 10 | TG, cavity |
| 64 | 12 | TG, cavity |
| 128 | 14 | TG, cavity, cylinder |
| 256 | 16 | TG, cavity, cylinder |
| 512 | 18 | TG, cavity (if feasible) |

Chi values: 32, 64 per (N, test_case).

### Sub-experiments

**5a: Point evaluation accuracy** — sample at 50 random fluid points, record (MPS, decomp, vanilla) per point, compute max/mean/median error per pair.

**5b: Mass conservation** — `mps_sum_value(rho)` vs `np.sum(rho_decomp)` vs `np.sum(rho_vanilla)`, single scalar comparison.

**5c: Coarse field accuracy** — levels 1, 2, 3 (n/2, n/4, n/8 grids). Per level: `mps_coarse_field` vs `decomp.reshape().sum()` vs `vanilla.reshape().sum()`. Record max error per cell.

**5d: Drag/lift** (cylinder only) — `compute_drag_lift_mps` (per-direction masks) vs decompressed momentum exchange vs vanilla. N=128, 256.

**5e: Convergence criterion** — run 2 MPS steps, compute `check_convergence_mps` (uses moments). Decompress before/after, compute du from numpy. Compare values + timing.

**5f: Cost scaling** — measure extraction time of each observable per (N, chi). Plot vs N. Find crossover where MPS-direct beats decompress+numpy.

### Job structure

Reuse Exp 1/2 saved MPS final states + vanilla baselines. Each job: load state, run all observables, save results.

| Test case | N values | Jobs |
|---|---|---|
| TG | 32, 64, 128, 256, 512 | 10 (× 2 chi) |
| Cavity | 32, 64, 128, 256, 512 | 10 |
| Cylinder | 128, 256 | 4 |

**Total: 24 jobs** (fast — just observable extraction, ~minutes each)

### Data per job (JSON)

```json
{
  "experiment": "exp5_observables",
  "params": {"test_case", "n", "chi", "Re", "u"},
  "observables": {
    "point_eval": {
      "n_probes": 50,
      "errors": {
        "mps_vs_decomp": {"max": 1e-16, "mean": 5e-17},
        "decomp_vs_vanilla": {"max": 0.05, "mean": 0.02},
        "mps_vs_vanilla": {"max": 0.05, "mean": 0.02}
      },
      "timing": {
        "mps_per_call_us": 50,
        "decompress_us": 12000,
        "numpy_lookup_us": 0.1
      }
    },
    "mass": {"mps": 4096.001, "decomp": 4096.001, "vanilla": 4096.000,
             "errors": {...}, "timing": {...}},
    "coarse_field": {
      "level_1": {"errors": {...}, "timing": {...}},
      "level_2": {...},
      "level_3": {...}
    },
    "drag_lift": {...},
    "convergence": {...},
    "inverse_density": {...}
  }
}
```

### Thesis output
1. **Table:** Observable accuracy at multiple scales — should be near machine precision for `mps_vs_decomp`
2. **Fig:** Cost scaling — extraction time vs N (log-log). MPS-direct sub-linear, decompress+numpy linear in N²
3. **Fig:** Crossover plot — at what N does MPS-direct beat decompress+numpy?
4. **Discussion:** Practical implications — real-time monitoring at large N without materializing dense field

### Files
- `experiments/stage3/exp5_observables.py`
- `experiments/stage3/jobs/submit_exp5.sh`
- `experiments/stage3/jobs/params/exp5_params.csv`

---

## Implementation Order

1. `common.py` — shared infrastructure
2. Experiment 1 (chi sweep) — validates pattern, most data
3. Experiment 4 (memory comparison) — extends stage 2
4. Experiment 2 (grid scaling) — key thesis figure
5. Experiment 3 (error decomposition) — understanding
6. Experiment 5 (observables) — formalize notebook results

---

## Status (2026-05-13)

### Code infrastructure — DONE

- `common.py` — Job dataclass + CSV I/O, baseline cache, checkpoint
  save/load, Snapshot dataclass + JSON writer, NVPS memory accounting
  (Gross Eq 46), L2 helper.
- `test_cases.py` — TG / cavity / cylinder setup with uniform contract:
  `apply_bc_vanilla(lattice, f_post_collision)` returns post-stream+BC.
  Cavity uses FWBB (matches MPS apply_mps_boundary). Cylinder uses
  apply_fwbb_cylinder. Boundary masks for drag/lift precomputed for
  cylinder.
- `runners.py` — `run_vanilla` (with baseline cache) and `run_mps` (with
  per-step convergence + 2-consecutive count, checkpoint save/resume,
  wall-clock limit). Safe BGK (uses `_safe_moments` for cylinder solid
  nodes — matches notebook pattern).
- `build_master_jobs.py` — enumerates all 5 experiments, dedups,
  produces `jobs/master_jobs.csv` (460 unique jobs).
- Code validated against the three canonical notebooks (TG / cavity /
  cylinder) at full rank — produces expected error floors.

### Master CSV — DONE

- Total: **460 unique jobs** (after deduplication, removing 75 overlaps)
- Breakdown: Exp 1 = 192, Exp 2 = 136, Exp 3 = 107, Exp 4 = 96, Exp 5 = 4 (cylinder demo for drag/lift)
- Cylinder removed from Exp 1 — only appears in Exp 5
- Exp 2 grid extended 23 → 34 (N, Re) pairs for denser fits

### Remaining work

1. **`run_job.py`** — CLI entry point (`python run_job.py --job-id N`) that
   reads `master_jobs.csv`, calls `run_mps(job)`. Glue layer (~40 lines).
2. **`jobs/submit.sh`** — SLURM array script (`#SBATCH --array=0-459`).
   Reads `SLURM_ARRAY_TASK_ID`, calls `run_job.py`. Sets wall-clock
   limit so checkpointing kicks in before kill.
3. **Analysis scripts** (one per experiment): `analyze_exp{1..5}.py`
   read relevant JSONs from `results/`, filter by experiment criteria,
   produce thesis figures + tables.
4. **Cluster submission + monitoring** — actually run the jobs.
5. **Figure generation + thesis writing**.

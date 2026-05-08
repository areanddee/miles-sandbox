# Test Specifications — miles-sandbox

Per-test specifications for the seven risk-reduction tests. Each test
is a self-contained `.py` file in `tests/` with a thin Colab notebook
wrapper. Tests are ordered by information-per-hour for the Gift-stage
risk register (see `README.md`); run in order, don't skip ahead.

Format for each test: **Risk addressed**, **Procedure**,
**Pre-stated success criteria** (green/yellow/red, committed *before*
measurement to prevent post-hoc reasoning), **Output**, **Notes**.

---

## Test 01 — `s2fft` round-trip on TPU at ERA5 grids

**File:** `tests/test_01_s2fft_roundtrip.py`

**Risk addressed:** R1 — SHT performance on TPU. The single
highest-severity risk in the register. If `s2fft` is too slow at
CREDIT's grid resolutions on TPU v6e, the polar Laplacian filter port
strategy needs to change before any port effort is committed.

**Procedure:**

1. Install `s2fft` (JAX-native spherical harmonic transforms). Pin
   the version in `requirements.txt`. If the install fails on Colab
   TPU, that itself is a finding — document and stop the test;
   this changes the project's risk picture significantly.
2. For each grid configuration in `[(192, 288), (640, 1280), (721, 1440)]`:
   - Generate a random complex spherical-harmonic spectrum at the
     appropriate band-limit (lmax = nlat - 1).
   - Run forward `s2fft.transform.spherical.inverse_jax` then
     `forward_jax`, jit-compiled.
   - Measure: (a) compile time of the jit, (b) per-call wall-clock
     for 5 trials after warmup, (c) round-trip max absolute error
     vs the starting coefficients.
   - Repeat in fp32 and bf16. For bf16, measure round-trip error
     against fp32 reference.
3. Generate HLO for the jitted round-trip; count gather and dot
   operations. Save HLO text to `results/test_01_s2fft/hlo_<isodate>.txt`.

**Pre-stated success criteria:**

- 🟢 0.25° (721×1440) round-trip in fp32 < 200 ms median per call.
- 🟡 200 ms – 1 s — mitigations needed (reduce frequency of filter
  application, batch multiple SHTs, host-side fallback).
- 🔴 > 1 s — must redesign filter strategy entirely.
- Additional fail: bf16 round-trip max absolute error > 1e-2
  (relative to L2 norm of field) → bf16 SHT is unusable; filter must
  run in fp32.

**Output:**
- `results/test_01_s2fft/<isodate>.json` with all timings + errors.
- `reports/test_01_s2fft.md` — 1-page memo: conclusion, numbers,
  HLO op counts, recommendation.

**Notes:**
- `s2fft` may need a TPU-specific install path. Check the `s2fft`
  docs for current install instructions and document whatever works
  in the test file's docstring.
- The 0.25° grid is the operational case; the smaller grids exist to
  show scaling shape so we can extrapolate.

---

## Test 02 — bf16 global-sum precision audit

**File:** `tests/test_02_bf16_global_sum.py`

**Risk addressed:** R4 — Whether CREDIT's conservation fixers
(mass, water, energy, tracer) can run in bf16. The fixers compute
area-weighted global means and *differences* of large global sums;
bf16 has only ~7 mantissa bits, and global sums lose precision
proportional to log(N).

**Procedure:**

1. Construct a synthetic ERA5-shaped specific-humidity field at
   (721, 1440) with realistic dynamic range (mean ~5e-3 kg/kg,
   surface values up to 2e-2, stratospheric values down to 1e-7).
   Use a seeded PRNG for reproducibility.
2. Compute area weights from a Gaussian latitude grid
   (`cos(latitude)` approximation is fine).
3. Compute area-weighted global sum in five precisions:
   - fp64 reference (use `jax.config.update('jax_enable_x64', True)`)
   - fp32 throughout
   - bf16 input, bf16 accumulator (worst case)
   - bf16 input, fp32 accumulator (`x.astype(jnp.float32).sum()`)
   - bf16 input, hierarchical fp32 sum (sum each row in fp32, then
     sum rows)
4. Compute *relative error* of each against the fp64 reference.
5. Repeat for the conservation-fixer correction term:
   `(target_mass - current_mass) / total_area`. This is a difference
   of two large numbers and is the actually precision-sensitive
   quantity.
6. Repeat with synthetic temperature field (mean 250K, range
   180–320K) — different conditioning, different result.
7. Run on TPU and on CPU; the answer should be the same to within
   determinism noise. Differences indicate a TPU bf16 quirk worth
   knowing about.

**Pre-stated success criteria:**

- 🟢 bf16/fp32-accumulator relative error on global mean < 1e-5 →
  bf16 conservation fixers are viable.
- 🟡 1e-5 to 1e-3 → marginal; need fp32 for the *difference* but
  bf16 inputs may be OK.
- 🔴 > 1e-3 → conservation fixers must run in fp32 throughout.

**Output:**
- `results/test_02_bf16/<isodate>.json`
- `reports/test_02_bf16.md` — 1-page memo with a precision table
  and a recommendation for each of the four CREDIT fixers.

**Notes:**
- This test does not strictly *need* TPU to answer the precision
  question — bf16 arithmetic is identical across hardware. Run on
  TPU anyway to (a) measure the fp32 reduction *cost* on TPU,
  (b) catch any libtpu quirks.
- The fp64 reference must be computed with `jax_enable_x64=True`,
  set BEFORE any JAX operation. Use a separate Python process if
  needed; jax x64 is one-shot per session.

---

## Test 03 — Attention head-dim MXU utilization

**File:** `tests/test_03_attention_tile.py`

**Risk addressed:** R5 — TPU MXU is sized for 128- or 256-lane
matmuls, while transformer codebases often use head_dim=64. WxFormer's
`dim: [256, 512, 1024, 2048]` and unknown `num_heads` may give
suboptimal head_dim. Also folds in R6 (ERA5 grid tile alignment).

**Procedure:**

1. Read `../miles-credit/credit/models/wxformer/crossformer.py` and
   extract the actual default `num_heads` per layer. Cite line
   numbers in the test docstring. **Do not modify** the CREDIT file.
2. For each (dim, heads) combination representing WxFormer's
   pyramid layers:
   - Build a single multi-head attention call: `(B, H, N, head_dim)
     × (B, H, head_dim, N) → (B, H, N, N)`, softmax,
     `× (B, H, N, head_dim)`. Use realistic seq lengths (image
     tokens at the appropriate pyramid level).
   - jit, warm up, time 20 trials, report median + p10/p90.
   - Compute achieved TFLOP/s assuming `4·B·H·N²·head_dim` FLOPs
     per attention call.
3. Report % of theoretical peak. Trillium peak bf16 is ~926 TFLOP/s
   per chip (single-core slice will be a fraction).
4. For comparison, run the same test with head_dim padded to 128
   (use head_dim=128 with fewer heads to keep total params constant).

**Pre-stated success criteria:**

- 🟢 WxFormer default head_dim achieves > 50% of peak → no concern.
- 🟡 25–50% → flag for the port: padding heads to 128 is a free win.
- 🔴 < 25% → significant — recommend reconfiguring WxFormer for the
  port and noting in pitch as a "TPU-aware reconfiguration" win.

**Output:**
- `results/test_03_attention/<isodate>.json`
- `reports/test_03_attention.md`

**Notes:**
- Use `jax.nn.dot_product_attention` if available in the pinned JAX
  version (added in 0.4.30). Falls back to einsum-based attention
  otherwise. Both should compile to the same XLA SDPA on TPU;
  verify via HLO.
- The seq length in WxFormer depends on `image_height/image_width`
  and patch sizes — derive from config, don't guess.

---

## Test 04 — `lax.scan` rollout compile cost

**File:** `tests/test_04_scan_compile.py`

**Risk addressed:** R3 — Autoregressive training rolls the model
forward N steps under a single jit. For large models the resulting
HLO can be huge and compile times can be 30+ minutes on TPU.
Folds in R7 (dynamic-shape recompile cost).

**Procedure:**

1. Build a stand-in WxFormer-shaped model in pure JAX: same
   depth/dim pyramid as `wxformer_1dg_6hr_v2.yml`
   (dim=[256, 512, 1024, 2048], depth=[2, 2, 18, 2]). Random
   weights, no pretraining. Use Flax NNX or Equinox — pick one
   and document the choice.
2. Wrap in `lax.scan` for an N-step rollout (start with N=2, then
   4, then 6).
3. Build the loss: MSE against a target trajectory (random tensor
   of correct shape). Take `grad`.
4. jit. Measure:
   - First-call wall-clock (compile time).
   - HLO size in bytes (`jit.lower(...).compile().as_text()`).
   - Steady-state per-step time after compile.
   - Peak HBM usage during the call (if accessible; otherwise
     note "OOM at N=X" as the data point).
5. Repeat with `jax.checkpoint` (remat) on the inner step function.
   Compare HLO size and compile time.

**Pre-stated success criteria:**

- 🟢 N=6 rollout compiles in < 5 min and steady-state per-step time
  is reasonable.
- 🟡 5–30 min compile — document but proceed.
- 🔴 > 30 min compile or OOM at N=4 — pitch must explicitly scope
  to shorter rollouts or include remat as required.

**Output:**
- `results/test_04_scan/<isodate>.json`
- `results/test_04_scan/hlo_size_table.md`
- `reports/test_04_scan.md`

**Notes:**
- Single-core v6e has finite HBM. The full WxFormer-1° model may
  not fit at N=6 with backprop. If OOM, *that's the result* — don't
  try to make it fit by shrinking dims; document the OOM point and
  conclude that multi-device sharding is required.
- Use `jax.config.update('jax_log_compiles', True)` to see compile
  events explicitly.

---

## Test 05 — `tf.data` + xarray throughput

**File:** `tests/test_05_xarray_pipeline.py`

**Risk addressed:** R8 — Host data pipeline must feed the TPU faster
than the TPU consumes. If not, TPU is starved and money is wasted.

**Procedure:**

1. Set up a small synthetic ERA5-like zarr store on local disk
   (Colab `/tmp/`): 100 timesteps of a (5 levels, 192, 288) field
   with realistic chunking.
2. Build three pipelines:
   - **Pipeline A:** xarray → numpy → `jax.numpy.array` directly,
     sequential.
   - **Pipeline B:** `tf.data.Dataset.from_generator` wrapping an
     xarray iterator, `prefetch(4)`, `batch(1)`.
   - **Pipeline C:** Grain dataset (if available; skip if Grain
     install fails on Colab).
3. Measure samples/sec for each, host-side only (no TPU consumption).
   Run 1000 samples, report median + p10/p90.
4. Cross-reference with Test 04's TPU step time: is the pipeline
   serving > 2× the rate the TPU can consume? (2× for safety
   margin given prefetch.)

**Pre-stated success criteria:**

- 🟢 Pipeline B serves at ≥ 2× TPU step rate.
- 🟡 Pipeline B serves at 1–2× — prefetch tuning required.
- 🔴 Pipeline B serves at < 1× — host pipeline is the bottleneck;
  pitch must scope dataset preprocessing as a real cost item.

**Output:**
- `results/test_05_pipeline/<isodate>.json`
- `reports/test_05_pipeline.md`

**Notes:**
- Colab's `/tmp/` is local SSD; on a real TRC instance you'd be
  reading from GCS. The numbers from this test are an *upper bound*
  on real-world throughput. Note this clearly in the report.
- If the synthetic data is too small to be representative, scale up
  until it's at least 1 GB on disk.

---

## Test 06 — Spectral-norm parametrization in JAX

**File:** `tests/test_06_spectral_norm_jax.py`

**Risk addressed:** R9 — Spectral norm is the most-replicated
custom-pattern in CREDIT's models (10 of 17 files in the autograd
grep). The JAX port needs a clean parametrization-style implementation
that matches PyTorch's legacy `nn.utils.spectral_norm` numerically.

**Procedure:**

1. In a separate Python process (no JAX), build a small PyTorch
   `nn.Conv2d` and apply `nn.utils.spectral_norm`. Run 100 forward
   passes with random inputs to converge the power-iteration vector
   `u`. Save weights, `u`, and outputs to a numpy file.
2. In JAX, build the equivalent: a custom layer that holds `u` in
   state, performs power iteration, and normalizes the weight by
   the estimated spectral norm.
3. Initialize the JAX layer with the saved PyTorch weights and `u`.
   Run on the same random inputs. Compare outputs to PyTorch
   reference.
4. Verify match to within 1e-6 in fp32 and 1e-3 in bf16.
5. Confirm the JAX implementation works as a Flax/Equinox parameter
   (gets correctly handled by `jax.grad`, `jax.jit`, and pytree
   operations).

**Pre-stated success criteria:**

- 🟢 Output match better than 1e-6 in fp32 → done.
- 🔴 Worse → debug; do not proceed to Test 07 until spectral-norm
  is settled, since it's load-bearing for almost every WxFormer
  layer.

**Output:**
- `results/test_06_spectral/<isodate>.json`
- `lib/spectral_norm.py` — the validated implementation, ready for
  reuse in the eventual port.
- `reports/test_06_spectral.md`

**Notes:**
- This is the only test that produces a *reusable artifact* (the
  validated `spectral_norm.py`). Make it library-quality: docstrings,
  type hints, test coverage.
- PyTorch's `nn.utils.spectral_norm` uses the *legacy* hook-based
  API (confirmed in CREDIT's `crossformer.py`); the modern
  `parametrizations.spectral_norm` is slightly different. Match the
  legacy behavior, since that's what CREDIT uses.

---

## Test 07 — `jax.checkpoint` (remat) memory/compile tradeoff

**File:** `tests/test_07_remat_savings.py`

**Risk addressed:** R10 — Backup plan if Test 04 shows trouble.
Remat trades compute for memory; understanding the cost shape lets
us recommend an explicit remat strategy if rollout compile is the
blocker.

**Procedure:**

1. Take the stand-in WxFormer model from Test 04. Apply
   `jax.checkpoint` at three granularities:
   - No remat (baseline, from Test 04).
   - Remat per CrossFormer block (coarse).
   - Remat per attention call (fine).
2. For each, measure: compile time, HLO size, peak HBM, steady-state
   per-step wall-clock with backprop.
3. Build a 2D table: (memory savings %) vs (compute slowdown %).

**Pre-stated success criteria:**

- 🟢 Identifies a remat granularity that allows N=6 rollout to fit
  in HBM with < 30% slowdown → recommend it as the default.
- 🔴 No granularity fits → must use multi-device sharding even for
  a single rollout.

**Output:**
- `results/test_07_remat/<isodate>.json` with the tradeoff table.
- `reports/test_07_remat.md`

**Notes:**
- Only run this if Test 04 produced a yellow or red result. If
  Test 04 is green, defer Test 07.

---

## Tests deferred (require multi-device hardware)

These were considered for the original test list but require
multi-device hardware and are deferred until TRC access lands:

- **`lax.ppermute` halo exchange** — needs ≥ 4 TPU cores for a
  meaningful 2D mesh test. Single-core can't characterize torus-
  topology behavior. Retires R2.
- **Full WxFormer training step at scale** — needs multi-host SPMD.
- **End-to-end `tf.data` → TPU multi-host streaming** —
  single-host pipeline is informative but not the operational case.

These are the obvious "Phase 2" tests once a TRC allocation or
Gift-funded compute lands. They are noted in the Gift pitch as the
follow-on validation work.

---

## Reporting structure for the Gift pitch

After Tests 01, 02, and 05 complete (the must-haves), produce
`reports/risk_register_v2.md` — an updated version of the risk
register from `README.md` with each risk's severity tagged
green/yellow/red based on measurement. *That document* is what goes
into the Gift proposal as the feasibility appendix.

Pre-stated proposal language to slot in (only after measurements
land):

> *"Standalone TPU validation has been executed for the three
> highest-severity risks in the proposed port. Spherical harmonic
> transforms at 0.25° resolution measured at [X] ms per round-trip
> on TPU v6e (well within the per-step budget); bf16
> conservation-fixer arithmetic validated to [Y] relative error
> against fp64 reference; host data pipeline measured at [Z]
> samples/sec, [W]× the projected TPU consumption rate. Detailed
> measurements and HLO analyses are available at [GitHub URL]."*

That paragraph is what makes the pitch read as executable. Don't
write it until the numbers land.

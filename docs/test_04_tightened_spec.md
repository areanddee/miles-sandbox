# Test 04 — `lax.scan` rollout compile cost (tightened spec)

This supersedes the Test 04 spec in `docs/test_specs.md` for implementation
purposes. The original spec stands as the high-level definition; this
document is the implementation-level tightening that reflects what we've
learned since the test was first scoped (single-core v6e, no PyTorch
dependency, Dinosaur as the JAX numerics reference).

## What this test is actually answering

**The risk:** CREDIT trains autoregressively. A future JAX port will wrap
the model's single-step forward + post-block in a `lax.scan` for an
N-step rollout, then `grad` the whole thing. On TPU, the resulting HLO
can be large enough that XLA compile times become unacceptable (30–90
minutes have been reported on similar workloads), and the compiled
program may not fit in single-core HBM at meaningful rollout lengths.

**The decision the test informs:** whether the Gift pitch can credibly
claim a 10–12-week lighthouse port, or whether the port must scope to
shorter rollouts and treat multi-step training as Phase 2 work.

## What this test is NOT

- Not a correctness check. Random weights are fine; the loss function
  doesn't need to mean anything; the target trajectory doesn't need
  to come from real data.
- Not a comparison against PyTorch. Pure JAX-on-TPU measurement.
- Not a multi-device test. Single-core v6e only; multi-device sharding
  is the obvious mitigation for a red result here and is Phase 2 work.
- Not a Dinosaur or SHT test. The stand-in model is a transformer-
  shaped JAX module; no spherical operations.

## Stand-in model design

The test needs a model that has the *shape* of WxFormer at 1° resolution,
not its functional behavior. Build a minimal JAX module with these
characteristics, derived from `wxformer_1dg_6hr_v2.yml` (consult
CREDIT's config files for the actual values, treating CREDIT as
read-only reference — do NOT import from `../miles-credit/`):

- **Input shape:** `(B=1, C, H, W)` where C is the number of channels
  (combining prognostic, diagnostic, dynamic forcing, static — pick
  a realistic total like 64 or read from the YAML), and (H, W) matches
  the 1° image dimensions CREDIT uses.
- **Patch embedding:** standard 2D patchify, patch size from config
  (typically 2 or 4).
- **Transformer pyramid:** 4 stages with `dim=[256, 512, 1024, 2048]`
  and `depth=[2, 2, 8, 2]` (corrected 2026-05-12 from `[2, 2, 18, 2]`
  to match CREDIT's v1 1° config — `config/gen_1/archive/v1/wxformer_1dg_6hr.yml`
  in NCAR/miles-credit — which is the source of truth for this stand-in;
  no v2 1° config exists in miles-credit yet). Use a *generic*
  transformer block (LayerNorm → MHA → LayerNorm → MLP, residual
  connections), NOT the CrossFormer-specific attention pattern. The
  point is to get the parameter count and FLOP shape right, not to
  reproduce CREDIT's architectural details. With this correction the
  stand-in lands at ~231 M params (computed in the test sketch),
  comfortably inside the 150–250 M range CREDIT's documented 1° config
  occupies — smaller than the original "hundreds of millions to ~1B"
  framing below.
- **Output head:** un-patchify back to (B, C, H, W).
- **Framework:** Flax NNX or Equinox — pick one and document. Flax
  NNX is the more current Google idiom; Equinox is simpler. Either
  works.

Document the parameter count of the constructed model. WxFormer 1°
should land in the hundreds of millions to ~1B parameters; if your
stand-in is off by 10×, the compile-time numbers won't transfer.

## Test procedure

### 1. Baseline: single forward step
Build the model, jit a single forward pass with the input shape above.
Measure:
- compile time of the jit (first call wall-clock, NOT including model
  construction)
- HLO size in bytes (`jit.lower(...).compile().as_text()` length)
- steady-state per-step time after 5 warmup calls (median of 10 timed)
- output `.block_until_ready()` on every measurement

### 2. Scan rollout, forward only
Wrap the single step in `lax.scan` for `N ∈ [2, 4, 6]`. The scan
carry is the model state (just the prediction tensor for this
stand-in; in real CREDIT it would also include accumulated diagnostics).

For each N:
- jit, measure compile time and HLO size
- per-step steady-state time
- record peak HBM if accessible (see "HBM measurement" below)

### 3. Scan rollout, with backprop
Same as #2 but wrap the rollout in a loss + `jax.grad`. Loss: MSE
against a random target trajectory of shape `(N, B, C, H, W)`.

For each N:
- jit, measure compile time, HLO size
- per-step steady-state time
- whether it OOMs at this N

### 4. Scan rollout, with backprop + `jax.checkpoint`
Add `jax.checkpoint` (remat) on the inner step function. Re-run #3.

This is the "mitigation preview" — if #3 OOMs or compiles for too
long at N=6, does adding remat fix it? A green result here with
an unacceptable result in #3 is the data point that says "remat
is required, port must use it by default."

## HBM measurement

JAX's HBM observability on TPU is limited. Use what's available:

- `jax.live_arrays()` is informative but lazy.
- `device.memory_stats()` may work on v6e (check; not all platforms
  expose it).
- `jax.devices()[0].memory_stats()` if that returns a dict, capture
  `bytes_in_use` and `peak_bytes_in_use` before and after the call.
- If neither works, fall back to "did it OOM?" as a binary signal,
  and use the HLO size as a proxy for memory pressure.

If OOM happens, that IS a data point. Don't try to make it fit by
shrinking the model — the model is calibrated to WxFormer 1°. An
OOM at N=4 means "single-core v6e cannot do a 4-step backprop of
this model; multi-device sharding required."

## Pre-stated success criteria (committed before measurement)

These are evaluated against the **N=6, backprop, no remat** configuration
(test step #3 at N=6), which is the headline case for the Gift pitch:

- 🟢 **Green:** N=6 backprop compiles in < 5 min on a cold cache, fits
  in single-core HBM, steady-state per-step time < 10× the single-
  forward-step baseline.
- 🟡 **Yellow:** 5–30 min compile, OR OOMs without remat but green
  with remat, OR per-step time 10–30× the baseline. Pitch must
  scope remat as a port requirement.
- 🔴 **Red:** > 30 min compile, OR OOMs even with remat, OR per-step
  time > 30× baseline. Pitch must scope to shorter rollouts (N≤3)
  on single-device and treat full-length rollout training as a
  multi-device Phase 2 milestone.

Additional diagnostic tags (don't gate green/yellow/red but inform
the report):
- `compile_scales_with_N`: how steeply compile time grows with N.
  Linear is fine; super-linear is a warning.
- `hlo_size_scales_with_N`: same for HLO bytes.
- `remat_overhead`: ratio of per-step time with remat to without
  (when both are measurable).

## Outputs

`results/test_04_scan/<isodate>.json` with this top-level schema:

```json
{
  "_meta": { ... },
  "model": {
    "framework": "flax_nnx | equinox",
    "param_count": <int>,
    "dim_pyramid": [256, 512, 1024, 2048],
    "depth_pyramid": [2, 2, 8, 2],
    "input_shape": [1, C, H, W],
    "patch_size": <int>
  },
  "baseline_single_step": { "compile_s": ..., "step_s": ...,
                            "hlo_bytes": ... },
  "scan_forward":   { "N=2": {...}, "N=4": {...}, "N=6": {...} },
  "scan_backprop":  { "N=2": {...}, "N=4": {...}, "N=6": {...} },
  "scan_backprop_remat": { "N=2": {...}, "N=4": {...}, "N=6": {...} },
  "decision": {
    "tag": "green | yellow | red",
    "compile_scales_with_N": "linear | superlinear | unknown",
    "remat_required": <bool>,
    "max_feasible_N_singlecore": <int>
  }
}
```

Per-N records include `compile_s`, `hlo_bytes`, `step_s_median`,
`step_s_p10`, `step_s_p90`, `hbm_peak_bytes_or_oom`,
`hlo_text_path` (sibling .txt file with the actual HLO).

Write incrementally via `_common.atomic_write_json` after each
(rollout_mode, N) completes, so a Colab timeout preserves progress.

## Process

1. Read this spec end to end.
2. Sketch the stand-in model construction in a comment block at the
   top of `tests/test_04_scan_compile.py`. Don't write executable
   code yet. Confirm parameter count is in the right ballpark
   (target: 200M–1B params; document the exact number).
3. Bring the sketch back for human review.
4. After approval: implement the test body, the JSON schema, the
   atomic write, the decision logic.
5. Stage. Bring the diff back. Human reviews, then runs on Colab.

## Risks specific to this test

- **Compile time on cold cache could be hours.** If your first run
  hits 30+ minutes compile on N=2, abort and report; we will
  reduce model size (with the human's approval) rather than wait
  out a multi-hour compile.
- **OOM during model construction itself.** If 1B-parameter model
  weights don't fit in v6e single-core HBM, the test can't run as
  designed. Document and reduce dim_pyramid scaling factor with
  human approval.
- **Flax NNX vs Equinox API drift.** Both libraries have moved
  rapidly. Pin the version in `requirements.txt` and document.
- **HBM measurement may be unavailable.** Document what you tried;
  fall back to OOM-as-binary-signal.
- **First Colab run may need TPU v6e specifically.** v6e and v4 have
  different compile-cache behavior. The test runs on v6e (per CLAUDE.md);
  results are not necessarily generalizable to v4 or v5.

## What stays the same as the original spec in docs/test_specs.md

- Three rollout lengths (2, 4, 6).
- Three configurations (forward, backprop, backprop+remat) — *expanded
  from the original "with and without remat" framing*.
- HLO captured before timing.
- Atomic incremental JSON writes.
- Per-test working-directory guard at top of file.

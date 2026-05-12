# Test 03 — Attention head-dim MXU utilization (tightened spec)

This supersedes the Test 03 spec in `docs/test_specs.md` for implementation
purposes. The original spec stands as the high-level definition; this
document is the implementation-level tightening.

## What this test is actually answering

**The risk:** TPU MXUs are 128×128 (TPU v4/v5) or 256×256 (TPU v6e
Trillium) systolic arrays. They achieve peak throughput when matrix
dimensions divide evenly into 128 (or 256). Transformer codebases
commonly hardcode `head_dim = 64`, which uses only half a TPU v6e MXU
tile per head, leaving ~50% of MXU FLOPs on the table.

WxFormer's actual config uses windowed attention with specific
(num_heads, head_dim) values per pyramid stage. We want to know:
**how well does WxFormer's head configuration utilize TPU v6e's MXU
in practice?**

**The decision the test informs:**
- If WxFormer's head_dim achieves > 50% of peak → no concern;
  inherit the config in the JAX port.
- If 25–50% → flag for the port; padding head_dim to 128 is a
  free win.
- If < 25% → significant — recommend reconfiguring WxFormer for the
  port. Note in the Gift pitch as a "TPU-aware reconfiguration"
  result.

**Secondary purpose: TPU MXU intuition.** This is the first test
that measures TPU compute performance directly. It also gives a
calibrated answer to "what FLOPs/s can a single TPU v6e core
actually deliver" on a workload that matters for CREDIT.

## What this test is NOT

- Not a full forward pass benchmark. We isolate the attention
  operation specifically.
- Not a windowed-attention test. Use full attention here; we're
  measuring matmul efficiency at specific shapes, and the windowed
  pattern is irrelevant to that question. (The windowed mask is
  a separate concern that doesn't affect head_dim × MXU alignment.)
- Not a comparison across hardware. v6e only.

## Architecture extraction

Read WxFormer's actual config to extract the per-stage attention
parameters. Source of truth:
`../miles-credit/config/gen_1/archive/v1/wxformer_1dg_6hr.yml`

The four pyramid stages have:
- stage 0: dim=256, depth=2
- stage 1: dim=512, depth=2
- stage 2: dim=1024, depth=8
- stage 3: dim=2048, depth=2

Look up `num_heads` for each stage in the config. The head_dim per
stage is then dim / num_heads. If the config doesn't make num_heads
explicit per stage, read the model source at
`../miles-credit/credit/models/wxformer/crossformer.py` to find the
default. Cite the file:line in the test docstring.

Record the actual (dim, num_heads, head_dim) per stage in the JSON
output. Don't hardcode the numbers in the test file beyond what's
needed for the sweep — read them from a config or constants section
at the top of the test.

## Sweep design

For each pyramid stage, run two variants:

### Variant A — WxFormer-native config
The (dim, num_heads, head_dim) from the config as-is. This is the
measurement we care most about.

### Variant B — head_dim padded to 128 with constant total params
Same total `dim`, but `num_heads` reduced so that head_dim = 128.
For example, if WxFormer-native has (dim=512, heads=8, head_dim=64),
the padded variant is (dim=512, heads=4, head_dim=128). Same FLOPs
per attention call to first order, same memory footprint, but
head_dim is tile-aligned.

If WxFormer-native already has head_dim ≥ 128, Variant B is skipped
for that stage (already aligned).

## Sequence length per stage

WxFormer's image dims are (192, 288) at 1°. Patches are 1×1 (verified
in the config Grunt read for Test 04), so the stage-0 token count
is 192 × 288 = 55,296. The pyramid downsamples by 2× per stage,
giving:
- stage 0: 55,296 tokens
- stage 1: 13,824 tokens
- stage 2: 3,456 tokens
- stage 3: 864 tokens

Use these as the sequence lengths in the attention measurement at
each stage.

## What we measure per (stage, variant)

The attention operation has the form:
```
Q, K, V: shape (B=1, num_heads, N, head_dim)
scores = Q @ K^T          # → (B, num_heads, N, N)
attn   = softmax(scores)  # → (B, num_heads, N, N)
out    = attn @ V         # → (B, num_heads, N, head_dim)
```

**FLOP count** (per single attention call):
- Q@K^T: 2 * B * num_heads * N * N * head_dim
- attn@V: 2 * B * num_heads * N * N * head_dim
- Total: 4 * B * num_heads * N² * head_dim
  (ignoring softmax, which is bandwidth-bound and small)

**Measurements:**
- Compile time of the jit (first call wall-clock)
- Steady-state per-call wall-clock (20 trials, median + p10/p90,
  with `.block_until_ready()` per call)
- Achieved TFLOP/s = FLOPs / median_seconds / 1e12
- Percent of theoretical peak (see below)

**Theoretical peak for TPU v6e Trillium:**
~926 TFLOP/s bf16 per chip, but single-core slice gets a fraction.
The single-core MXU peak depends on chip configuration; **document
whatever JAX reports for `jax.devices()[0]` and compute peak from
the chip spec**. Cite the source for the peak number.

If the peak number is uncertain, report achieved TFLOP/s in absolute
units AND as a ratio relative to a known-good attention benchmark
(e.g., a stage with dim=2048 and head_dim=128 should be near peak).
The *relative* utilization across stages and variants is what matters,
not the absolute peak number.

## Precision and JAX attention API

Use `jax.nn.dot_product_attention` if available in pinned JAX 0.7.2.
This dispatches to XLA's SDPA on TPU, which uses bf16 multiply +
fp32 accumulate on the MXU by default — the actually-realistic case
for a transformer.

If `jax.nn.dot_product_attention` is not available, fall back to
einsum-based attention:
```python
scores = jnp.einsum('bhnd,bhmd->bhnm', q, k) / jnp.sqrt(head_dim)
attn = jax.nn.softmax(scores, axis=-1)
out = jnp.einsum('bhnm,bhmd->bhnd', attn, v)
```

Both should compile to the same XLA SDPA on TPU; verify by inspecting
HLO. If they don't, document the difference and use whichever
JAX-native path performs better (which is the path WxFormer would
actually use).

## Pre-stated success criteria (committed before measurement)

Evaluated against the **dim=512 stage** (representative middle of
the pyramid; stage 0 is dominated by sequence length cost, stage 3
is dominated by attention-on-tiny-sequence overhead):

- 🟢 **Green:** WxFormer-native achieves > 50% of theoretical peak
  TFLOP/s. No concern; port inherits config as-is.
- 🟡 **Yellow:** 25–50% of peak. Variant B (head_dim=128) measurably
  better. Port should reconfigure heads to 128 — a free win.
- 🔴 **Red:** < 25% of peak. Significant — pitch can present this
  as "TPU-aware reconfiguration improves throughput by X%."

The pyramid-average utilization is also reported but doesn't gate
the tag.

Additional diagnostic tags:
- `padding_speedup_factor`: ratio of Variant B / Variant A throughput
  at the dim=512 stage. Numbers > 1.2× indicate meaningful win from
  the head_dim reconfiguration.
- `stage_0_oom`: whether stage 0 with full attention OOMs (Test 04
  showed full MHA at 55K tokens is 48 GB; expect this to OOM in
  bf16 too, but worth confirming).
- `worst_stage`: which pyramid stage has lowest MXU utilization.

## Outputs

`results/test_03_attention/<isodate>.json` with this top-level schema:

```json
{
  "_meta": { ... },
  "config_source": "../miles-credit/config/gen_1/archive/v1/wxformer_1dg_6hr.yml",
  "tpu_peak_tflops_assumed": <float>,
  "peak_source_citation": "TPU v6e Trillium spec sheet, etc.",
  "stages": {
    "stage_0": { "dim": 256, "wxformer_native": {...}, "padded_128": {...} },
    "stage_1": { ... },
    "stage_2": { ... },
    "stage_3": { ... }
  },
  "decision": {
    "tag": "green | yellow | red",
    "midpyramid_util_percent": <float>,
    "padding_speedup_factor": <float>,
    "worst_stage": "stage_N",
    "stage_0_oom": <bool>
  }
}
```

Per-(stage, variant) records contain `dim`, `num_heads`, `head_dim`,
`seq_len`, `flops_per_call`, `compile_s`, `step_s_median`,
`step_s_p10`, `step_s_p90`, `achieved_tflops`, `util_percent`,
`hlo_bytes`, `hlo_text_path`.

Write incrementally via `_common.atomic_write_json`.

## Process

1. Read this spec.
2. Read WxFormer's config and source to extract the actual
   (num_heads, head_dim) per stage. Cite file:line. **READ ONLY —
   do not import or copy CREDIT source into the test; the per-stage
   numbers can be hardcoded in the test file with a citation comment.**
3. Sketch the test structure in a comment block at the top of
   `tests/test_03_attention_tile.py`. Note the JAX attention API
   choice (`dot_product_attention` vs einsum fallback) based on
   what's available.
4. Bring the sketch back for human review.
5. Implement. Stage. Bring the diff back.

## Risks specific to this test

- **Stage 0 with full attention will OOM** (we already know this
  from Test 04). Handle this gracefully: catch the OOM, record
  `stage_0_oom: true`, continue with stages 1–3. The stage-0
  measurement isn't needed for the decision tag (which is on
  the dim=512 stage).
- **TPU peak TFLOP/s is hardware-specific and version-specific.**
  The number for v6e single-core depends on whether you're on
  a v6e-1 slice (single chip, multiple cores) or counting the
  whole chip. Document what JAX reports, cite the chip spec,
  and report util_percent against that. If the peak number is
  off, the *relative* numbers across variants are still valid.
- **`jax.nn.dot_product_attention` may not exist in the pinned
  JAX 0.7.2.** It was added in 0.4.30 but the API surface has
  evolved. Verify on Colab first; fall back to einsum if needed.
- **Compile time at stage 0 (if it doesn't OOM) might be long
  due to the 55K-token shape.** Cap timing at 5 trials at stage 0
  if compile is > 60 seconds.

## What stays the same as the original spec

- Per-stage measurement of (dim, num_heads, head_dim).
- Padded-to-128 comparison variant.
- TFLOP/s and % peak reporting.
- HLO captured before timing.
- Atomic incremental JSON writes.
- Working-directory guard at top of file.

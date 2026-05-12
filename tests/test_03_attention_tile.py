"""Test 03 -- attention head-dim MXU utilization on TPU v6e.

Retires risks: R5 (medium) -- attention head-dim MXU utilization on TPU,
               R6 (medium, folded in) -- ERA5 grid dims vs TPU tile alignment.

Goal: measure how well WxFormer's per-stage attention configuration
utilizes TPU v6e's MXU. Four-point head_dim sweep at the mid-pyramid
(dim=512) stage (head_dim in {32, 64, 128, 256}) reveals the
utilization curve's elbow; other stages take 2 points each (native
head_dim=32 vs padded head_dim=128). At stage_1 native, both
``jax.nn.dot_product_attention`` (XLA SDPA) and an einsum-based
attention are measured in parallel to confirm they compile to the
same XLA SDPA on TPU.

Decision boundary (per docs/test_03_tightened_spec.md): tag from the
stage_1 native variant util_percent::

    green   util_percent > 50
    yellow  25 <= util_percent <= 50
    red     util_percent < 25

Diagnostic tags: padding_speedup_factor, worst_stage, stage_0_oom.

Procedure: ``docs/test_03_tightened_spec.md`` supersedes the Test 03
entry in ``docs/test_specs.md``.

Architecture (CREDIT references, consulted read-only per CLAUDE.md rule 14)::

    config/gen_1/archive/v1/wxformer_1dg_6hr.yml:142
        dim: [256, 512, 1024, 2048]   (num_heads not in YAML)

    credit/models/wxformer/crossformer.py:234
        def __init__(self, dim, attn_type, window_size, dim_head=32, ...)
                                                       ^^^^^^^^^^^^
    credit/models/wxformer/crossformer.py:245
        heads = dim // dim_head

    credit/models/wxformer/crossformer.py:364, 424
        Higher-level classes also default dim_head=32.

Per-stage WxFormer-native config (inherited dim_head=32)::

    stage 0   dim=256   heads=8    head_dim=32   seq_len=55,296
    stage 1   dim=512   heads=16   head_dim=32   seq_len=13,824
    stage 2   dim=1024  heads=32   head_dim=32   seq_len=3,456
    stage 3   dim=2048  heads=64   head_dim=32   seq_len=864

Design notes
------------
- head_dim=32 is 1/4 of TPU v6e's 256-wide MXU tile. The native config
  is expected to under-utilize the MXU. The 4-point head_dim sweep at
  stage_1 (head_dim in {32, 64, 128, 256}) is designed to reveal the
  utilization curve's "elbow" -- three points would not.
- Stage 0 with N=55,296 is expected to OOM at bf16 attention scores
  (Test 04 evidence: ~24 GB at bf16 forward, ~48 GB at fp32 backprop;
  v6e single-core is ~31 GB HBM). Caught gracefully; ``stage_0_oom``
  is a diagnostic tag that confirms Test 04's finding on a different
  code path (attention-only vs scan+backprop).
- Plain ``jax.jit`` (not ``eqx.filter_jit``) -- this test has no
  Equinox modules, so ``compiled.as_text()`` works directly. No
  HLO-extraction helper needed.
- bf16 Q / K / V (the realistic TPU transformer case; XLA SDPA uses
  bf16-mul / fp32-acc on the MXU by default). Matmul precision left
  at JAX default -- this test measures throughput at typical
  precision, not numerical accuracy.
- v6e bf16 peak: 918 TFLOP/s per Google's Oct 2024 TPU v6e
  announcement (v6e-1 slice = 1 chip). If the actual peak on this
  hardware differs, ``achieved_tflops`` is absolute and util_percent
  can be re-derived.
- ``jax.nn.dot_product_attention`` (SDPA) is the primary path; einsum
  is also measured at stage_1/native_h32 to verify both paths compile
  equivalently. If they don't compile identically (``speedup`` notably
  > 1 or HLO op counts differ), the port should use SDPA explicitly.

Out of scope
------------
- Windowed-attention mask (CrossFormer's Swin pattern). This test
  isolates the attention matmul efficiency at fixed (B, H, N, D)
  shapes; masking is orthogonal.
- Q / K / V projection matmuls. Only the SDPA core is timed.
- bf16 vs fp32 precision comparison.
- Multi-device sharding (single-core only).
"""

import os

assert os.path.basename(os.getcwd()) == "miles-sandbox", (
    f"Tests must run from miles-sandbox/, got {os.getcwd()}"
)

import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from _common import is_tpu, tpu_info, op_counts, atomic_write_json


# --- Configuration ---------------------------------------------------------

SEED = 42
N_WARMUP = 3
N_TRIALS = 20
TPU_PEAK_TFLOPS = 918.0   # v6e bf16 per chip; v6e-1 = 1 chip

# Decision thresholds (docs/test_03_tightened_spec.md).
GREEN_UTIL_PERCENT = 50.0
YELLOW_UTIL_PERCENT = 25.0
PADDING_MEANINGFUL_RATIO = 1.2

# SDPA presence check at module import (no JAX op fired).
SDPA_AVAILABLE = hasattr(jax.nn, "dot_product_attention")

# Sweep matrix. Stage 1 has the 4-point head_dim curve plus an einsum
# variant for the API comparison; other stages take native vs padded.
SWEEP = [
    {
        "stage": "stage_0",
        "dim": 256,
        "seq_len": 192 * 288,           # 55,296
        "variants": [
            {"label": "native_h32",  "heads": 8, "head_dim": 32,  "api": "sdpa"},
            {"label": "padded_h128", "heads": 2, "head_dim": 128, "api": "sdpa"},
        ],
    },
    {
        "stage": "stage_1",
        "dim": 512,
        "seq_len": 96 * 144,            # 13,824
        "variants": [
            {"label": "native_h32",        "heads": 16, "head_dim": 32,  "api": "sdpa"},
            {"label": "h64",               "heads": 8,  "head_dim": 64,  "api": "sdpa"},
            {"label": "padded_h128",       "heads": 4,  "head_dim": 128, "api": "sdpa"},
            {"label": "v6e_full_h256",     "heads": 2,  "head_dim": 256, "api": "sdpa"},
            {"label": "native_h32_einsum", "heads": 16, "head_dim": 32,  "api": "einsum"},
        ],
    },
    {
        "stage": "stage_2",
        "dim": 1024,
        "seq_len": 48 * 72,             # 3,456
        "variants": [
            {"label": "native_h32",  "heads": 32, "head_dim": 32,  "api": "sdpa"},
            {"label": "padded_h128", "heads": 8,  "head_dim": 128, "api": "sdpa"},
        ],
    },
    {
        "stage": "stage_3",
        "dim": 2048,
        "seq_len": 24 * 36,             # 864
        "variants": [
            {"label": "native_h32",  "heads": 64, "head_dim": 32,  "api": "sdpa"},
            {"label": "padded_h128", "heads": 16, "head_dim": 128, "api": "sdpa"},
        ],
    },
]

OPS_OF_INTEREST = [
    # HLO dialect (post-XLA).
    "dot", "dot-general", "convolution",
    "gather", "scatter", "all-reduce", "reduce", "transpose",
    "custom-call",          # XLA SDPA may compile to a single fused custom call
    # StableHLO dialect (pre-compile, in case as_text returns StableHLO).
    "stablehlo.dot", "stablehlo.dot_general",
    "stablehlo.reduce", "stablehlo.transpose",
    "stablehlo.custom_call",
]


# --- Attention functions ---------------------------------------------------

def _attention_sdpa(q, k, v):
    """JAX SDPA. Q/K/V shape: (B, T, N, H). Returns (B, T, N, H)."""
    return jax.nn.dot_product_attention(q, k, v)


def _attention_einsum(q, k, v):
    """Manual einsum + softmax. Q/K/V shape: (B, T, N, H). Returns (B, T, N, H)."""
    head_dim = q.shape[-1]
    scale = jnp.asarray(head_dim ** -0.5, dtype=q.dtype)
    # (B, T, N, H) x (B, S, N, H) -> (B, N, T, S)
    scores = jnp.einsum("btnh,bsnh->bnts", q, k) * scale
    attn = jax.nn.softmax(scores, axis=-1)
    # (B, N, T, S) x (B, S, N, H) -> (B, T, N, H)
    return jnp.einsum("bnts,bsnh->btnh", attn, v)


def make_qkv(seq_len: int, num_heads: int, head_dim: int, seed: int):
    """Build random Q/K/V tensors in bfloat16, shape (B=1, T, N, H)."""
    key = jax.random.PRNGKey(seed)
    k_q, k_k, k_v = jax.random.split(key, 3)
    shape = (1, seq_len, num_heads, head_dim)
    q = jax.random.normal(k_q, shape, dtype=jnp.bfloat16)
    k = jax.random.normal(k_k, shape, dtype=jnp.bfloat16)
    v = jax.random.normal(k_v, shape, dtype=jnp.bfloat16)
    return q, k, v


def flops_attention(B: int, num_heads: int, seq_len: int, head_dim: int) -> int:
    """4 * B * H * N^2 * D per spec (Q@K^T + attn@V; softmax not counted)."""
    return 4 * B * num_heads * seq_len * seq_len * head_dim


# --- Helpers ---------------------------------------------------------------

def is_oom(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in msg
        or "OOM" in msg.upper()
        or "out of memory" in msg.lower()
    )


def measure_compiled(compiled, args, label: str):
    """Warmup + timed trials. Returns (timing_dict, exc_or_None)."""
    try:
        for i in range(N_WARMUP):
            t0 = time.perf_counter()
            out = compiled(*args)
            jax.block_until_ready(out)
            dt = time.perf_counter() - t0
            print(
                f"    [{label}] warmup {i+1}/{N_WARMUP}: {dt*1000:.2f} ms",
                flush=True,
            )

        trials = []
        for i in range(N_TRIALS):
            t0 = time.perf_counter()
            out = compiled(*args)
            jax.block_until_ready(out)
            dt = time.perf_counter() - t0
            trials.append(dt)
            print(
                f"    [{label}] trial {i+1}/{N_TRIALS}: {dt*1000:.2f} ms",
                flush=True,
            )
        return {
            "trials_s": trials,
            "step_s_median": float(np.median(trials)),
            "step_s_p10": float(np.percentile(trials, 10)),
            "step_s_p90": float(np.percentile(trials, 90)),
        }, None
    except Exception as e:
        if is_oom(e):
            return {}, e
        raise


def measure_cell(stage_label: str, variant_label: str, dim: int,
                 num_heads: int, head_dim: int, seq_len: int, api: str,
                 out_dir: Path, run_stamp: str) -> dict:
    """Measure compile + HLO + warm trials for one (stage, variant) cell.

    On OOM at compile or warm phase, returns a record with oom=True and
    the error message; does NOT short-circuit other cells (this differs
    from Test 04's per-mode short-circuit because here each cell is
    independent in HBM footprint).
    """
    label = f"{stage_label}/{variant_label}"
    base = {
        "dim": dim, "num_heads": num_heads, "head_dim": head_dim,
        "seq_len": seq_len, "api": api,
    }

    if api == "sdpa" and not SDPA_AVAILABLE:
        return {
            **base,
            "skipped": True,
            "reason": "jax.nn.dot_product_attention not available in this JAX",
        }

    fn = _attention_sdpa if api == "sdpa" else _attention_einsum
    jit_fn = jax.jit(fn)
    q, k, v = make_qkv(seq_len, num_heads, head_dim, SEED)

    # Compile (timed).
    try:
        t0 = time.perf_counter()
        compiled = jit_fn.lower(q, k, v).compile()
        compile_s = time.perf_counter() - t0
    except Exception as e:
        if is_oom(e):
            print(f"    [{label}] OOM during compile: {str(e)[:200]}", flush=True)
            return {**base, "oom": True, "oom_phase": "compile",
                    "error_msg": str(e)[:500]}
        raise

    print(f"    [{label}] compile_s = {compile_s:.2f}", flush=True)

    # HLO before warm calls (CLAUDE.md rule 6). Plain jax.jit here, so
    # compiled.as_text() works directly.
    hlo = compiled.as_text()
    hlo_bytes = len(hlo.encode("utf-8"))
    counts = op_counts(hlo, OPS_OF_INTEREST)
    hlo_path = out_dir / f"hlo_{stage_label}_{variant_label}_{api}_{run_stamp}.txt"
    hlo_path.write_text(hlo)

    dots = counts.get("dot", 0) + counts.get("dot-general", 0)
    custom_calls = counts.get("custom-call", 0) + counts.get("stablehlo.custom_call", 0)
    print(
        f"    [{label}] HLO bytes={hlo_bytes:,}, "
        f"dots={dots}, custom-calls={custom_calls}",
        flush=True,
    )

    timing, oom_exc = measure_compiled(compiled, (q, k, v), label)
    if oom_exc is not None:
        print(f"    [{label}] OOM during warm trials: {str(oom_exc)[:200]}", flush=True)
        return {
            **base, "oom": True, "oom_phase": "warm_trials",
            "error_msg": str(oom_exc)[:500],
            "compile_s": compile_s,
            "hlo_bytes": hlo_bytes,
            "hlo_op_counts": counts,
            "hlo_path": str(hlo_path),
        }

    flops = flops_attention(1, num_heads, seq_len, head_dim)
    median_s = timing["step_s_median"]
    achieved_tflops = flops / median_s / 1e12 if median_s > 0 else 0.0
    util_percent = 100.0 * achieved_tflops / TPU_PEAK_TFLOPS

    print(
        f"    [{label}] median={median_s*1000:.2f} ms, "
        f"achieved={achieved_tflops:.1f} TFLOP/s, util={util_percent:.1f}%",
        flush=True,
    )

    return {
        **base,
        "compile_s": compile_s,
        "hlo_bytes": hlo_bytes,
        "hlo_op_counts": counts,
        "hlo_path": str(hlo_path),
        **timing,
        "flops_per_call": flops,
        "achieved_tflops": achieved_tflops,
        "util_percent": util_percent,
    }


# --- Decision + API comparison ---------------------------------------------

def compute_decision(payload: dict) -> dict:
    """Tag against stage_1 native variant util_percent."""
    thresholds = {
        "util_percent_green": GREEN_UTIL_PERCENT,
        "util_percent_yellow": YELLOW_UTIL_PERCENT,
        "padding_meaningful_ratio": PADDING_MEANINGFUL_RATIO,
        "tpu_peak_tflops": TPU_PEAK_TFLOPS,
    }

    stages = payload.get("stages") or {}
    s1 = stages.get("stage_1") or {}
    native = s1.get("native_h32")

    if (native is None or native.get("oom") or native.get("skipped")
            or "util_percent" not in native):
        return {
            "tag": "unknown",
            "reason": "stage_1 native_h32 (dim=512, head_dim=32) not measured",
            "tag_thresholds": thresholds,
        }

    util = native["util_percent"]
    if util > GREEN_UTIL_PERCENT:
        tag = "green"
    elif util >= YELLOW_UTIL_PERCENT:
        tag = "yellow"
    else:
        tag = "red"

    reason = (
        f"stage_1 native_h32 (dim=512, head_dim=32) util = {util:.1f}% "
        f"({native['achieved_tflops']:.1f} TFLOP/s of "
        f"{TPU_PEAK_TFLOPS:.0f} peak)"
    )

    # Padding speedup at stage_1 (native -> padded_h128).
    padded = s1.get("padded_h128")
    padding_speedup = None
    if padded and not padded.get("oom") and not padded.get("skipped"):
        if "step_s_median" in padded and padded["step_s_median"] > 0:
            padding_speedup = native["step_s_median"] / padded["step_s_median"]

    # Worst native stage among those that completed.
    worst_stage = None
    worst_util = None
    for stage_label, stage in stages.items():
        if not isinstance(stage, dict):
            continue
        n = stage.get("native_h32")
        if n and not n.get("oom") and not n.get("skipped") and "util_percent" in n:
            if worst_util is None or n["util_percent"] < worst_util:
                worst_util = n["util_percent"]
                worst_stage = stage_label

    # Stage 0 OOM signal (any native or padded variant OOMed).
    s0 = stages.get("stage_0") or {}
    stage_0_oom = bool(
        (s0.get("native_h32") or {}).get("oom")
        or (s0.get("padded_h128") or {}).get("oom")
    )

    return {
        "tag": tag,
        "reason": reason,
        "midpyramid_util_percent": util,
        "midpyramid_achieved_tflops": native["achieved_tflops"],
        "padding_speedup_factor": padding_speedup,
        "worst_stage": worst_stage,
        "worst_stage_util_percent": worst_util,
        "stage_0_oom": stage_0_oom,
        "tag_thresholds": thresholds,
    }


def compute_api_comparison(payload: dict) -> dict:
    """SDPA vs einsum at stage_1/native_h32 (both shapes match)."""
    s1 = (payload.get("stages") or {}).get("stage_1") or {}
    sdpa_cell = s1.get("native_h32")
    einsum_cell = s1.get("native_h32_einsum")

    def _ok(c):
        return (c is not None and not c.get("oom") and not c.get("skipped")
                and "step_s_median" in c)

    if not (_ok(sdpa_cell) and _ok(einsum_cell)):
        return {
            "available": False,
            "reason": "SDPA and/or einsum cell did not complete at stage_1/native_h32",
        }

    speedup = einsum_cell["step_s_median"] / sdpa_cell["step_s_median"]
    return {
        "available": True,
        "sdpa_hlo_bytes": sdpa_cell["hlo_bytes"],
        "einsum_hlo_bytes": einsum_cell["hlo_bytes"],
        "sdpa_op_counts": sdpa_cell["hlo_op_counts"],
        "einsum_op_counts": einsum_cell["hlo_op_counts"],
        "sdpa_step_s_median": sdpa_cell["step_s_median"],
        "einsum_step_s_median": einsum_cell["step_s_median"],
        "sdpa_achieved_tflops": sdpa_cell["achieved_tflops"],
        "einsum_achieved_tflops": einsum_cell["achieved_tflops"],
        "speedup_sdpa_over_einsum": speedup,
        "interpretation": (
            "If speedup > 1, SDPA is faster than the einsum path -- meaning "
            "einsum is leaving performance on the table and the port should "
            "use jax.nn.dot_product_attention explicitly. Compare HLO files "
            "at sdpa.hlo_path vs einsum.hlo_path to see whether both paths "
            "compile to the same XLA SDPA kernel."
        ),
    }


# --- Main ------------------------------------------------------------------

def main():
    if not is_tpu():
        print(
            f"WARNING: not on TPU. JAX devices: {jax.devices()}. "
            "Continuing for CPU sanity check; spec criteria are TPU-specific.",
            flush=True,
        )

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("results") / "test_03_attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_stamp}.json"

    payload = {
        "test_name": "test_03_attention_tile",
        "config_source": "../miles-credit/config/gen_1/archive/v1/wxformer_1dg_6hr.yml",
        "source_citation": (
            "credit/models/wxformer/crossformer.py:234 (dim_head=32 default), "
            ":245 (heads = dim // dim_head)"
        ),
        "tpu_peak_tflops_assumed": TPU_PEAK_TFLOPS,
        "peak_source_citation": (
            "Google Cloud TPU v6e (Trillium) announcement, Oct 2024; "
            "918 TFLOP/s bf16 per chip; v6e-1 = 1 chip"
        ),
        "attention_api_primary": "sdpa" if SDPA_AVAILABLE else "einsum",
        "sdpa_available": SDPA_AVAILABLE,
        "seed": SEED,
        "n_warmup": N_WARMUP,
        "n_trials": N_TRIALS,
        "ops_counted": OPS_OF_INTEREST,
        "stages": {},
        "api_comparison": None,
        "decision": None,
    }

    def write_partial():
        full = {
            **payload,
            "_meta": {
                **tpu_info(),
                "timestamp_utc": run_stamp,
                "test_name": payload["test_name"],
            },
        }
        atomic_write_json(json_path, full)

    write_partial()

    print(f"sdpa_available = {SDPA_AVAILABLE}", flush=True)
    print(f"TPU peak (assumed) = {TPU_PEAK_TFLOPS} TFLOP/s bf16", flush=True)

    for stage_entry in SWEEP:
        stage_label = stage_entry["stage"]
        dim = stage_entry["dim"]
        seq_len = stage_entry["seq_len"]
        if stage_label not in payload["stages"]:
            payload["stages"][stage_label] = {"dim": dim, "seq_len": seq_len}
        print(f"\n=== {stage_label}: dim={dim}, seq_len={seq_len} ===", flush=True)

        for variant in stage_entry["variants"]:
            label = variant["label"]
            heads = variant["heads"]
            head_dim = variant["head_dim"]
            api = variant["api"]
            print(
                f"\n--- {stage_label}/{label}: heads={heads}, "
                f"head_dim={head_dim}, api={api} ---",
                flush=True,
            )
            cell = measure_cell(
                stage_label, label, dim, heads, head_dim, seq_len,
                api, out_dir, run_stamp,
            )
            payload["stages"][stage_label][label] = cell
            write_partial()

    payload["api_comparison"] = compute_api_comparison(payload)
    payload["decision"] = compute_decision(payload)
    write_partial()

    print(f"\nDecision tag: {payload['decision'].get('tag')}", flush=True)
    print(f"Reason: {payload['decision'].get('reason')}", flush=True)
    print(f"Results: {json_path}", flush=True)
    return json_path


if __name__ == "__main__":
    main()

"""Test 04 -- lax.scan rollout compile cost (stand-in WxFormer on TPU).

Retires risks: R3 (high)   -- lax.scan rollout compile cost on TPU,
               R7 (medium) -- dynamic-shape recompile cost.

Goal: build a stand-in JAX module with WxFormer-1deg's *parameter count
and FLOP shape* (not its functional behaviour); wrap in ``lax.scan``
for an N-step autoregressive rollout; jit; measure cold compile time,
HLO size, steady-state per-step time, and peak HBM. Sweep N in {2, 4, 6}
across three rollout modes (forward only, backprop, backprop + remat).

Decision boundary: the timing tag is evaluated against the **N=6
backprop, no remat** configuration. See ``docs/test_04_tightened_spec.md``
for pre-stated thresholds and the diagnostic tags that travel alongside
the green / yellow / red.

Procedure: ``docs/test_04_tightened_spec.md`` supersedes the Test 04
entry in ``docs/test_specs.md`` for implementation purposes.

Architecture (matches CREDIT's v1 1-deg WxFormer config at
``config/gen_1/archive/v1/wxformer_1dg_6hr.yml``, consulted as read-only
reference per CLAUDE.md rule 14; depth corrected 2026-05-12 from
``[2,2,18,2]`` to ``[2,2,8,2]`` to match the actual v1 config)::

    Input shape         (C=74, H=192, W=288)  [B=1, single-sample model]
    Patch embed         2D conv, p=1 (matches v1 config)
    Pyramid             dim   = [256, 512, 1024, 2048]
                        depth = [2, 2, 8, 2]
                        heads = [2, 4, 8, 16]   (head_dim = 128)
    Stage transitions   strided 2x2 conv  (2x spatial downsample)
    Decoder             symmetric mirror -- three 2x2 transposed convs
                        + 1x1 conv to C output channels. No skip
                        connections; the decoder is not the test target.
    Block               LayerNorm -> MHA -> residual ->
                        LayerNorm -> MLP (4x expansion, GELU) -> residual
    Param count         ~231 M  (computed exactly at runtime; recorded
                        in the JSON under model.param_count)

Design notes
------------
- **Swin attention substituted by generic full MHA** per the spec's
  framework-portability intent. The real WxFormer uses windowed
  attention (``global_window_size = [8, 4, 2, 1]`` per stage); our
  substitution amplifies stage-0 attention compute / memory by ~6,900x
  (55,296 tokens N^2 vs (N/W)*W^2 with W=8). Stage-0 attention scores
  alone occupy ~24 GB per layer at fp32 / ~12 GB at bf16. On single-
  core v6e (~95 GB HBM), N=6 backprop is very likely to OOM at stage 0
  -- that IS a legitimate R3 / R7 finding ("full-MHA 1-deg backprop
  does not fit on single-core v6e; use windowed attention or shard
  across cores"). The test catches OOMs and short-circuits higher-N
  cells in the affected mode rather than aborting.
- **Framework: Equinox.** Simpler than Flax NNX; we are building a
  one-off stand-in, not infrastructure. Installed version recorded in
  the JSON under ``model.equinox_version``.
- **HBM measurement** via ``jax.devices()[0].memory_stats()`` when
  available; falls back to OOM-as-binary signal. Recorded per cell.
- **Compilation cache observation only:** ``JAX_COMPILATION_CACHE_DIR``
  env var is recorded in ``_meta`` but not actively managed. If
  non-null, ``compile_s`` may reflect a cache hit and is not a true
  cold start -- flag that to the report writer rather than try to
  clear the cache.
- **HLO captured before warm-call timing** (CLAUDE.md rule 6) by
  reading ``compiled.as_text()`` immediately after the timed compile,
  sharing one compile across both timing and inspection.
- **Per-warmup AND per-trial progress prints with flush=True** so a
  live notebook surfaces timing as it accumulates rather than after
  ``measure_*`` returns (Test 01 lesson). Each of the 5 warmup calls
  gets its own labeled wall-clock line before the 10 timed trials,
  eliminating any silent gap between the HLO summary and trial 1/10
  -- which at L=192 with full MHA could otherwise hide minutes of
  warmup-side work.
- **JAX's auto compile-chatter suppressed.** The
  ``Finished XLA compilation of jit(...) in N sec`` WARNING lines
  from ``jax._src.dispatch`` are silenced via a logger-level bump.
  We print ``compile_s`` ourselves with a controlled per-cell label;
  the JAX-emitted lines duplicate that signal and clutter notebook
  output. Real warnings from other modules still surface.

Out of scope
------------
- Functional correctness against CREDIT (stand-in shape only).
- Multi-device sharding / SPMD (deferred to Phase 2 per spec).
- CrossFormer-specific windowed attention (spec says generic MHA).
- Real ERA5 data, real forcing fan-in, real loss target.
- Decoder skip connections (decoder is not the test target).
"""

import os

assert os.path.basename(os.getcwd()) == "miles-sandbox", (
    f"Tests must run from miles-sandbox/, got {os.getcwd()}"
)

import logging
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

# Suppress JAX's per-compile chatter (the "Finished XLA compilation..."
# WARNING lines emitted by jax._src.dispatch). We print compile_s
# ourselves with a controlled per-cell label; the JAX-emitted lines
# duplicate that signal and clutter notebook output. Real warnings from
# other modules still surface at WARNING and above.
logging.getLogger("jax._src.dispatch").setLevel(logging.ERROR)

from _common import is_tpu, tpu_info, op_counts, atomic_write_json


# --- Configuration ---------------------------------------------------------

# Architecture per v1 1-deg WxFormer config (consulted, not imported per
# CLAUDE.md rule 14); depth corrected 2026-05-12 per the tightened spec.
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 288
INPUT_CHANNELS = 74            # 4 * 16 (upper-air * levels) + 7 surface + 3 dyn/static
PATCH_SIZE = 1                 # matches v1 1-deg config
DIM_PYRAMID = (256, 512, 1024, 2048)
DEPTH_PYRAMID = (2, 2, 8, 2)   # v1 1-deg; spec corrected from [2,2,18,2]
HEAD_DIM = 128                 # TPU MXU-friendly tile dim (Test 03's R5 lever)
NUM_HEADS = tuple(d // HEAD_DIM for d in DIM_PYRAMID)  # (2, 4, 8, 16)
MLP_EXPANSION = 4

# Sweep parameters
SEED = 42
N_VALUES = (2, 4, 6)
N_WARMUP = 5
N_TRIALS = 10
ROLLOUT_MODES = ("forward", "backprop", "backprop_remat")

# Pre-stated success criteria from docs/test_04_tightened_spec.md, evaluated
# against the N=6 backprop no-remat cell.
GREEN_COMPILE_S = 5 * 60       # < 5 min compile = green
YELLOW_COMPILE_S = 30 * 60     # < 30 min compile = yellow else red
GREEN_STEP_RATIO = 10          # step_s_median / baseline_step_s < 10 = green
YELLOW_STEP_RATIO = 30         # < 30 = yellow else red
ABORT_COMPILE_S = 30 * 60      # abort the run entirely if even baseline hits this

OPS_OF_INTEREST = [
    # HLO dialect names (what compiled.as_text() would return, post-XLA).
    "dot", "dot-general", "convolution", "gather", "scatter",
    "all-reduce", "reduce", "transpose", "while", "scan",
    # StableHLO dialect names (what lowered.as_text() typically returns
    # in modern JAX). Whichever dialect _get_hlo_text recovers from the
    # Equinox-wrapped Lowered will have meaningful counts.
    "stablehlo.dot", "stablehlo.dot_general", "stablehlo.convolution",
    "stablehlo.gather", "stablehlo.scatter",
    "stablehlo.reduce", "stablehlo.transpose", "stablehlo.while",
]


# --- Model -----------------------------------------------------------------

class TransformerBlock(eqx.Module):
    """LayerNorm -> MHA -> residual -> LayerNorm -> MLP -> residual.

    Single-sample (sequence-only) -- operates on x of shape (N, D).
    """

    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    mlp_up: eqx.nn.Linear
    mlp_down: eqx.nn.Linear

    def __init__(self, dim: int, num_heads: int, *, key):
        k_attn, k_up, k_down = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(shape=(dim,))
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=dim, key=k_attn
        )
        self.norm2 = eqx.nn.LayerNorm(shape=(dim,))
        self.mlp_up = eqx.nn.Linear(dim, dim * MLP_EXPANSION, key=k_up)
        self.mlp_down = eqx.nn.Linear(dim * MLP_EXPANSION, dim, key=k_down)

    def __call__(self, x):
        # x: (N, D). LayerNorm and Linear are single-sample in Equinox; vmap
        # over the sequence axis to apply per-token.
        h = jax.vmap(self.norm1)(x)
        h = self.attn(h, h, h)
        x = x + h
        h = jax.vmap(self.norm2)(x)
        h = jax.vmap(self.mlp_up)(h)
        h = jax.nn.gelu(h)
        h = jax.vmap(self.mlp_down)(h)
        return x + h


class WxFormerStandIn(eqx.Module):
    """Stand-in transformer pyramid + symmetric-mirror decoder.

    Single-sample model: takes (C, H, W) -> (C, H, W). Wrap in
    ``jax.vmap`` for batched use if ever needed (B=1 in this test).
    """

    patch_embed: eqx.nn.Conv2d
    enc_stages: list           # list of list[TransformerBlock]
    enc_transitions: list      # list of eqx.nn.Conv2d (3, between stages)
    dec_transitions: list      # list of eqx.nn.ConvTranspose2d (3, mirror)
    final_conv: eqx.nn.Conv2d
    dim_pyramid: tuple = eqx.field(static=True)

    def __init__(self, *, key):
        keys = jax.random.split(key, 64)
        ki = iter(keys)
        nk = lambda: next(ki)

        self.dim_pyramid = DIM_PYRAMID

        self.patch_embed = eqx.nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=DIM_PYRAMID[0],
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
            key=nk(),
        )

        self.enc_stages = [
            [
                TransformerBlock(dim=DIM_PYRAMID[s], num_heads=NUM_HEADS[s], key=nk())
                for _ in range(DEPTH_PYRAMID[s])
            ]
            for s in range(len(DIM_PYRAMID))
        ]

        # 3 transitions: between stages 0->1, 1->2, 2->3 (none after stage 3).
        self.enc_transitions = [
            eqx.nn.Conv2d(
                in_channels=DIM_PYRAMID[i],
                out_channels=DIM_PYRAMID[i + 1],
                kernel_size=2,
                stride=2,
                key=nk(),
            )
            for i in range(3)
        ]

        # Symmetric mirror: 3 transposed convs reversing the encoder
        # transitions, in reverse order (3->2, 2->1, 1->0).
        self.dec_transitions = [
            eqx.nn.ConvTranspose2d(
                in_channels=DIM_PYRAMID[3 - i],
                out_channels=DIM_PYRAMID[2 - i],
                kernel_size=2,
                stride=2,
                key=nk(),
            )
            for i in range(3)
        ]

        # Final 1x1 conv to C output channels (no pixel-shuffle since p=1).
        self.final_conv = eqx.nn.Conv2d(
            in_channels=DIM_PYRAMID[0],
            out_channels=INPUT_CHANNELS,
            kernel_size=1,
            stride=1,
            key=nk(),
        )

    def __call__(self, x):
        # x: (C, H, W)
        x = self.patch_embed(x)  # (D[0], H, W)  -- p=1, no downsample

        for stage_i, blocks in enumerate(self.enc_stages):
            D, H, W = x.shape
            # Flatten spatial to sequence for transformer blocks.
            x_seq = x.transpose(1, 2, 0).reshape(H * W, D)
            for block in blocks:
                x_seq = block(x_seq)
            x = x_seq.reshape(H, W, D).transpose(2, 0, 1)

            # Stage transition (3 of them; not after the last stage).
            if stage_i < 3:
                x = self.enc_transitions[stage_i](x)

        # Decoder: 3 transposed convs (symmetric mirror).
        for dec_trans in self.dec_transitions:
            x = dec_trans(x)

        # Final 1x1 conv to (C, H, W).
        return self.final_conv(x)


# --- Functional builders ---------------------------------------------------

def make_forward(N: int):
    """Pre-jitted forward-only rollout. Returns all step outputs (N, C, H, W)."""

    @eqx.filter_jit
    def rollout(model, x0):
        def step(carry, _):
            next_x = model(carry)
            return next_x, next_x

        _, outputs = jax.lax.scan(step, init=x0, xs=None, length=N)
        return outputs

    return rollout


def make_loss_grad(N: int, remat: bool):
    """Pre-jitted MSE loss + grad over an N-step rollout.

    With remat=True, wraps the per-step body in jax.checkpoint so the
    backward pass rematerialises activations rather than storing them.
    """

    def loss(model, x0, targets):
        def step(carry, _):
            next_x = model(carry)
            return next_x, next_x

        if remat:
            step = jax.checkpoint(step)

        _, outputs = jax.lax.scan(step, init=x0, xs=None, length=N)
        return jnp.mean((outputs - targets) ** 2)

    @eqx.filter_jit
    def loss_grad(model, x0, targets):
        return eqx.filter_value_and_grad(loss)(model, x0, targets)

    return loss_grad


# --- Helpers ---------------------------------------------------------------

def safe_version(pkg: str):
    """Best-effort package version lookup. Returns None on failure."""
    try:
        return pkg_version(pkg)
    except PackageNotFoundError:
        return None


def count_params(model) -> int:
    """Total parameter count across all array leaves of the model pytree."""
    return sum(leaf.size for leaf in jax.tree.leaves(eqx.filter(model, eqx.is_array)))


def read_hbm() -> dict:
    """Best-effort TPU HBM stats via jax.devices()[0].memory_stats().

    Returns a dict with bytes_in_use, peak_bytes_in_use (when available),
    or an error string if the platform does not expose memory_stats().
    """
    try:
        stats = jax.devices()[0].memory_stats()
    except (AttributeError, RuntimeError) as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    if stats is None:
        return {"error": "memory_stats() returned None"}
    return {
        "bytes_in_use": stats.get("bytes_in_use"),
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "bytes_limit": stats.get("bytes_limit"),
    }


def is_oom(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in msg
        or "OOM" in msg.upper()
        or "out of memory" in msg.lower()
    )


def _get_hlo_text(lowered) -> tuple[str, str]:
    """Best-effort HLO/StableHLO text extraction from a Lowered.

    Equinox's ``filter_jit`` wraps the underlying ``jax.stages.Lowered`` in
    an ``eqx.Module`` (and similarly for ``Compiled``) that does not
    proxy ``as_text`` through. Walk a few plausible attribute paths to
    reach the underlying JAX object, and try the HLO dialect first
    (matches ``OPS_OF_INTEREST`` naming) then fall back to the default
    dialect (typically StableHLO in modern JAX).

    Returns ``(text, dialect_label)`` where ``dialect_label`` is one of
    ``"hlo"``, ``"stablehlo"``. Raises ``AttributeError`` if nothing
    plausible exposes ``.as_text``.
    """
    candidates = [lowered]
    for attr in ("_lowered", "lowered", "_fun", "_inner"):
        inner = getattr(lowered, attr, None)
        if inner is not None and inner not in candidates:
            candidates.append(inner)

    for cand in candidates:
        for call, label in (
            (lambda c=cand: c.as_text(dialect="hlo"), "hlo"),
            (lambda c=cand: c.as_text("hlo"), "hlo"),
            (lambda c=cand: c.as_text(), "stablehlo"),
        ):
            try:
                txt = call()
            except (AttributeError, TypeError, ValueError, KeyError):
                continue
            if isinstance(txt, str) and txt:
                return txt, label

    raise AttributeError(
        f"Could not extract HLO from lowered of type "
        f"{type(lowered).__name__} (tried attrs: _lowered, lowered, "
        f"_fun, _inner)"
    )


def trial_stats(trials: list[float]) -> dict:
    return {
        "trials_s": trials,
        "step_s_median": float(np.median(trials)),
        "step_s_p10": float(np.percentile(trials, 10)),
        "step_s_p90": float(np.percentile(trials, 90)),
    }


def measure_compiled(compiled, args, label: str) -> tuple[dict, BaseException | None]:
    """Run N_WARMUP + N_TRIALS calls; return timing dict + None, or {} + exc on OOM."""
    try:
        for i in range(N_WARMUP):
            t0 = time.perf_counter()
            out = compiled(*args)
            jax.block_until_ready(out)
            dt = time.perf_counter() - t0
            print(
                f"    [{label}] warmup {i+1}/{N_WARMUP}: {dt:.3f} s",
                flush=True,
            )

        trials = []
        for i in range(N_TRIALS):
            t0 = time.perf_counter()
            out = compiled(*args)
            jax.block_until_ready(out)
            dt = time.perf_counter() - t0
            trials.append(dt)
            print(f"    [{label}] trial {i+1}/{N_TRIALS}: {dt:.3f} s", flush=True)
        return trial_stats(trials), None
    except Exception as e:
        if is_oom(e):
            return {}, e
        raise


def measure_cell(
    model,
    x0,
    targets,
    mode: str,
    N: int | None,
    out_dir: Path,
    run_stamp: str,
    baseline_step_s: float | None,
) -> dict:
    """Compile + HLO capture + warm trials for one sweep cell.

    mode in {"baseline", "forward", "backprop", "backprop_remat"};
    N is None for the baseline single-forward-step case.
    """
    if mode == "baseline":
        fn = eqx.filter_jit(lambda m, x: m(x))
        args = (model, x0)
        label = "baseline"
    elif mode == "forward":
        fn = make_forward(N)
        args = (model, x0)
        label = f"forward N={N}"
    elif mode == "backprop":
        fn = make_loss_grad(N, remat=False)
        args = (model, x0, targets)
        label = f"backprop N={N}"
    elif mode == "backprop_remat":
        fn = make_loss_grad(N, remat=True)
        args = (model, x0, targets)
        label = f"remat N={N}"
    else:
        raise ValueError(f"unknown mode: {mode}")

    hbm_before = read_hbm()

    # Compile (timed). Split lower + compile so HLO can be extracted from
    # the lowered side: Equinox's filter_jit wraps Compiled in an
    # eqx.Module that does not proxy .as_text() through. The lowered
    # wrapper may or may not -- _get_hlo_text walks both possibilities.
    try:
        t0 = time.perf_counter()
        lowered = fn.lower(*args)
        compiled = lowered.compile()
        compile_s = time.perf_counter() - t0
    except Exception as e:
        if is_oom(e):
            print(f"    [{label}] OOM during compile: {str(e)[:200]}", flush=True)
            return {
                "oom": True,
                "oom_phase": "compile",
                "error_msg": str(e)[:500],
                "hbm_before": hbm_before,
            }
        raise

    print(f"    [{label}] compile_s = {compile_s:.2f}", flush=True)

    # HLO inspection before warm calls (CLAUDE.md rule 6).
    try:
        hlo, hlo_dialect = _get_hlo_text(lowered)
    except AttributeError as e:
        print(f"    [{label}] HLO extraction failed: {e}", flush=True)
        hlo, hlo_dialect = "", "unavailable"
    hlo_bytes = len(hlo.encode("utf-8"))
    counts = op_counts(hlo, OPS_OF_INTEREST) if hlo else {}
    hlo_name = "baseline" if mode == "baseline" else f"{mode}_N{N}"
    hlo_path = out_dir / f"hlo_{hlo_name}_{run_stamp}.txt"
    if hlo:
        hlo_path.write_text(hlo)
    # Surface either HLO-dialect or StableHLO-dialect counts in the log line,
    # whichever has nonzero values (dialect label tells us which we got).
    dots = counts.get("dot", 0) + counts.get("dot-general", 0)
    stablehlo_dots = counts.get("stablehlo.dot", 0) + counts.get("stablehlo.dot_general", 0)
    print(
        f"    [{label}] HLO bytes={hlo_bytes:,} dialect={hlo_dialect}, "
        f"dots={dots if hlo_dialect == 'hlo' else stablehlo_dots}, "
        f"scans={counts.get('scan', 0) + counts.get('stablehlo.while', 0)}, "
        f"whiles={counts.get('while', 0) + counts.get('stablehlo.while', 0)}",
        flush=True,
    )

    # Warm + timed trials.
    timing, oom_exc = measure_compiled(compiled, args, label)
    hbm_after = read_hbm()

    if oom_exc is not None:
        return {
            "oom": True,
            "oom_phase": "warm_trials",
            "error_msg": str(oom_exc)[:500],
            "compile_s": compile_s,
            "hlo_bytes": hlo_bytes,
            "hlo_dialect": hlo_dialect,
            "hlo_op_counts": counts,
            "hlo_path": str(hlo_path),
            "hbm_before": hbm_before,
        }

    record = {
        "compile_s": compile_s,
        "hlo_bytes": hlo_bytes,
        "hlo_dialect": hlo_dialect,
        "hlo_op_counts": counts,
        "hlo_path": str(hlo_path),
        "hbm_before": hbm_before,
        "hbm_after": hbm_after,
        **timing,
    }
    if baseline_step_s is not None and baseline_step_s > 0:
        record["step_s_ratio_to_baseline"] = record["step_s_median"] / baseline_step_s
    return record


# --- Decision logic --------------------------------------------------------

def classify_scaling(ratio: float | None, expected_linear: float = 3.0) -> str:
    if ratio is None:
        return "unknown"
    return "linear" if ratio < expected_linear * 1.3 else "superlinear"


def compute_decision(payload: dict) -> dict:
    """Apply pre-stated thresholds; derive diagnostic tags."""
    base = payload.get("baseline_single_step") or {}
    thresholds = {
        "compile_green_s": GREEN_COMPILE_S,
        "compile_yellow_s": YELLOW_COMPILE_S,
        "step_ratio_green": GREEN_STEP_RATIO,
        "step_ratio_yellow": YELLOW_STEP_RATIO,
    }

    if base.get("oom") or "step_s_median" not in base:
        return {
            "tag": "unknown",
            "reason": "baseline_single_step did not complete",
            "tag_thresholds": thresholds,
        }

    baseline_step_s = base["step_s_median"]

    cell_f = (payload.get("scan_backprop") or {}).get("N=6")
    cell_i = (payload.get("scan_backprop_remat") or {}).get("N=6")

    # Tag against cell (f); cell (i) is the remat-mitigation fallback.
    remat_required = False
    if cell_f is None or cell_f.get("skipped"):
        tag = "unknown"
        reason = "scan_backprop N=6 not measured"
    elif cell_f.get("oom"):
        # Backprop fails; remat may rescue.
        if cell_i and not cell_i.get("oom") and not cell_i.get("skipped") and "step_s_median" in cell_i:
            tag = "yellow"
            reason = "N=6 backprop OOMs without remat but succeeds with remat"
            remat_required = True
        else:
            tag = "red"
            reason = "N=6 backprop OOMs even with remat (or remat not measured)"
            remat_required = True
    else:
        compile_s = cell_f["compile_s"]
        step_s = cell_f["step_s_median"]
        ratio = step_s / baseline_step_s if baseline_step_s > 0 else float("inf")
        if compile_s >= YELLOW_COMPILE_S or ratio >= YELLOW_STEP_RATIO:
            tag = "red"
            reason = (
                f"N=6 backprop compile_s={compile_s:.1f}s "
                f"or step_ratio={ratio:.1f}x exceeds yellow"
            )
        elif compile_s >= GREEN_COMPILE_S or ratio >= GREEN_STEP_RATIO:
            tag = "yellow"
            reason = (
                f"N=6 backprop compile_s={compile_s:.1f}s; "
                f"step_ratio={ratio:.1f}x"
            )
        else:
            tag = "green"
            reason = (
                f"N=6 backprop compile_s={compile_s:.1f}s; "
                f"step_ratio={ratio:.1f}x"
            )

    # Scaling diagnostics on the backprop row (N=6 vs N=2).
    def cell_field(mode_key: str, N: int, field: str):
        c = (payload.get(mode_key) or {}).get(f"N={N}")
        if c is None or c.get("oom") or c.get("skipped"):
            return None
        v = c.get(field)
        return v if isinstance(v, (int, float)) else None

    bp_compile_n2 = cell_field("scan_backprop", 2, "compile_s")
    bp_compile_n6 = cell_field("scan_backprop", 6, "compile_s")
    bp_hlo_n2 = cell_field("scan_backprop", 2, "hlo_bytes")
    bp_hlo_n6 = cell_field("scan_backprop", 6, "hlo_bytes")

    compile_ratio = (
        bp_compile_n6 / bp_compile_n2
        if bp_compile_n2 and bp_compile_n6 and bp_compile_n2 > 0
        else None
    )
    hlo_ratio = (
        bp_hlo_n6 / bp_hlo_n2
        if bp_hlo_n2 and bp_hlo_n6 and bp_hlo_n2 > 0
        else None
    )

    remat_overhead = None
    if (
        cell_f and not cell_f.get("oom") and cell_f.get("step_s_median")
        and cell_i and not cell_i.get("oom") and cell_i.get("step_s_median")
    ):
        remat_overhead = cell_i["step_s_median"] / cell_f["step_s_median"]

    # Max feasible N across any mode (largest N that produced step times).
    max_N = 0
    for mode in ROLLOUT_MODES:
        for N in N_VALUES:
            c = (payload.get(f"scan_{mode}") or {}).get(f"N={N}")
            if c and not c.get("oom") and not c.get("skipped") and c.get("step_s_median"):
                max_N = max(max_N, N)

    return {
        "tag": tag,
        "reason": reason,
        "compile_scales_with_N": classify_scaling(compile_ratio),
        "compile_ratio_backprop_N6_over_N2": compile_ratio,
        "hlo_size_scales_with_N": classify_scaling(hlo_ratio),
        "hlo_ratio_backprop_N6_over_N2": hlo_ratio,
        "remat_required": remat_required,
        "remat_overhead": remat_overhead,
        "max_feasible_N_singlecore": max_N,
        "tag_thresholds": thresholds,
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
    out_dir = Path("results") / "test_04_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_stamp}.json"

    # Inputs.
    key = jax.random.PRNGKey(SEED)
    model_key, input_key, *target_keys = jax.random.split(key, 3 + len(N_VALUES))

    # Build model.
    print("Building stand-in WxFormer model...", flush=True)
    t0 = time.perf_counter()
    model = WxFormerStandIn(key=model_key)
    model_build_s = time.perf_counter() - t0
    param_count = count_params(model)
    print(
        f"  param_count = {param_count:,}  "
        f"(target ~231 M; build {model_build_s:.2f} s)",
        flush=True,
    )

    x0 = jax.random.normal(input_key, (INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))
    targets_per_N = {
        N: jax.random.normal(
            target_keys[i], (N, INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        )
        for i, N in enumerate(N_VALUES)
    }

    payload = {
        "test_name": "test_04_scan_compile",
        "model": {
            "framework": "equinox",
            "equinox_version": safe_version("equinox"),
            "jax_version": jax.__version__,
            "param_count": param_count,
            "dim_pyramid": list(DIM_PYRAMID),
            "depth_pyramid": list(DEPTH_PYRAMID),
            "num_heads": list(NUM_HEADS),
            "head_dim": HEAD_DIM,
            "mlp_expansion": MLP_EXPANSION,
            "input_shape": [1, INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
            "patch_size": PATCH_SIZE,
            "calibration_source": (
                "miles-credit/config/gen_1/archive/v1/wxformer_1dg_6hr.yml "
                "(v1 1-deg WxFormer; depth corrected per "
                "docs/test_04_tightened_spec.md 2026-05-12)"
            ),
            "swin_substituted_by_full_mha": True,
            "model_build_s": model_build_s,
        },
        "compilation_cache": {
            "persistent_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
            "note": (
                "Recorded for observation only -- not actively managed by "
                "this test. If persistent_dir is non-null, compile_s may "
                "reflect a cache hit and is not a true cold start."
            ),
        },
        "seed": SEED,
        "n_warmup": N_WARMUP,
        "n_trials": N_TRIALS,
        "ops_counted": OPS_OF_INTEREST,
        "n_values": list(N_VALUES),
        "rollout_modes": list(ROLLOUT_MODES),
        "thresholds": {
            "compile_green_s": GREEN_COMPILE_S,
            "compile_yellow_s": YELLOW_COMPILE_S,
            "step_ratio_green": GREEN_STEP_RATIO,
            "step_ratio_yellow": YELLOW_STEP_RATIO,
        },
        "baseline_single_step": None,
        "scan_forward": {},
        "scan_backprop": {},
        "scan_backprop_remat": {},
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

    write_partial()  # skeleton early so the file is monitorable

    # Baseline: single forward step.
    print("\n=== Baseline: single forward step ===", flush=True)
    base = measure_cell(model, x0, None, "baseline", None, out_dir, run_stamp, None)
    payload["baseline_single_step"] = base
    write_partial()

    if base.get("oom"):
        print(
            "ABORT: baseline OOM. Cannot establish step-time denominator; "
            "writing decision and exiting.",
            flush=True,
        )
        payload["decision"] = compute_decision(payload)
        write_partial()
        return json_path

    if base["compile_s"] >= ABORT_COMPILE_S:
        print(
            f"ABORT: baseline compile_s={base['compile_s']:.0f}s exceeds "
            f"the {ABORT_COMPILE_S}s abort threshold (cold-cache compile cliff). "
            "Writing decision and exiting.",
            flush=True,
        )
        payload["decision"] = compute_decision(payload)
        write_partial()
        return json_path

    baseline_step_s = base["step_s_median"]

    # 9-cell sweep: 3 modes x 3 N. Run row by row; on OOM in a mode, skip
    # higher N in that mode (record as "skipped" with a reason).
    for mode in ROLLOUT_MODES:
        mode_key = f"scan_{mode}"
        print(f"\n=== Mode: {mode} ===", flush=True)
        for N in N_VALUES:
            print(f"\n--- {mode} N={N} ---", flush=True)
            targets = targets_per_N[N] if mode != "forward" else None
            cell = measure_cell(
                model, x0, targets, mode, N, out_dir, run_stamp, baseline_step_s
            )
            payload[mode_key][f"N={N}"] = cell
            write_partial()

            if cell.get("oom"):
                higher = [n for n in N_VALUES if n > N]
                if higher:
                    print(
                        f"  OOM at {mode} N={N}; skipping {higher} in this mode.",
                        flush=True,
                    )
                    for h in higher:
                        payload[mode_key][f"N={h}"] = {
                            "skipped": True,
                            "reason": f"OOM at N={N} in mode={mode}",
                        }
                    write_partial()
                break

    # Decision.
    payload["decision"] = compute_decision(payload)
    write_partial()

    print(f"\nDecision tag: {payload['decision'].get('tag')}", flush=True)
    print(f"Reason: {payload['decision'].get('reason')}", flush=True)
    print(f"Results: {json_path}", flush=True)
    return json_path


if __name__ == "__main__":
    main()

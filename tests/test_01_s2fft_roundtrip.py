"""Test 01 -- s2fft round-trip on TPU at ERA5 grids.

Retires risk: R1 (high) -- spherical harmonic transform performance on TPU.

Goal: round-trip an ERA5 0.25-degree (1440 x 721) field through s2fft
inverse and forward spherical-harmonic transforms; measure cold compile
time, warm per-call time, and round-trip accuracy in coefficient space.
Compare bf16 against an fp32 reference.

Decision boundary: if s2fft cannot complete the round-trip in well under
one second on TPU v6e, the polar Laplacian filter strategy in
``$CREDIT/credit/pol_lapdiff_filt.py`` needs to be rethought before any
JAX port is undertaken.

Procedure: see ``docs/test_specs.md``. This file is the implementation;
the spec governs the procedure and the pre-stated success criteria.

CREDIT reference: ``$CREDIT/credit/pol_lapdiff_filt.py`` (uses
``torch_harmonics.RealSHT``, ``InverseRealSHT``, ``RealVectorSHT``,
``InverseRealVectorSHT``).

Design notes
------------
- The bf16 path tests **matmul-precision degradation** (the question
  CREDIT actually faces on TPU MXU), not bf16 storage. Implemented via
  ``jax.default_matmul_precision("bfloat16")`` context around the jit
  lowering. Storage stays complex64 throughout.
- The decision block separates the timing tag (green / yellow / red,
  drawn from the fp32 721x1440 median per ``docs/test_specs.md``) from
  ``bf16_acceptable`` (boolean, drawn from the worst-case vs-fp32
  ``max_abs / ||flm||_2`` across grids; threshold 1e-2 per spec).
- Results JSON is written incrementally after each (grid, precision)
  using ``_common.atomic_write_json`` so a Colab session timeout
  preserves all completed measurements. HLO files are written
  alongside (not atomically -- the JSON only references their path
  after the .txt write returns, so a crash during the .txt write
  leaves an orphan .txt and a JSON that doesn't reference it).
- HLO is captured BEFORE warm-call timing (CLAUDE.md rule 6) by reading
  ``compiled.as_text()`` immediately after the timed compile, sharing
  one compile across both timing and inspection.
- Each grid creates a fresh closure over ``L``, so JAX sees a different
  function per grid and compiles fresh. Persistent compilation cache
  is assumed disabled (no ``JAX_COMPILATION_CACHE_DIR`` set); the
  ``_meta`` block records whether it was, so re-runs against a populated
  cache are flagged.
- Each grid_record carries both the **requested** grid (``nlat``,
  ``nlon`` from ``docs/test_specs.md``) and the s2fft **produced**
  spatial shape (``produced_spatial_shape``), plus a ``shape_note``
  string the report writer can quote. The (192, 288) target in the
  spec represents the WxFormer 1-degree config resolution; ``gl``
  sampling at L=192 produces (192, 383) instead. The fp32 timing tag
  is meaningful at L regardless of nphi mismatch -- L sets the
  compute volume, nphi details affect padding overhead but not the
  order-of-magnitude that determines the green/yellow/red thresholds.

Out of scope
------------
- Vector SHTs (CREDIT also uses ``RealVectorSHT`` in
  ``pol_lapdiff_filt.py``). Natural follow-up if scalar SHT tests
  green; flag in the report.
- ``torch_harmonics`` GPU baseline. Defer unless TPU number lands in
  the yellow band and a comparison is needed for the report.
- Real-field structure of the spectrum. Random complex flm is fine for
  the precision and timing question -- round-trip behavior is
  independent of conjugate-symmetry constraints.
- Spin > 0 SHTs. ``SPIN = 0``, ``REALITY = False``.
- s2fft precomputed Wigner-d matrices (``precomps`` left at default
  ``None``). If TPU SHT is bottlenecked by Wigner-d recomputation,
  that's a finding to report, not something to silently optimize away.
"""

import os

assert os.path.basename(os.getcwd()) == "miles-sandbox", (
    f"Tests must run from miles-sandbox/, got {os.getcwd()}"
)
# CREDIT is referenced only by path in docstrings for this test; not
# required at runtime. Many Colab sessions won't have it cloned (it's
# a 300 MB repo, mostly git history). Soft-warn if absent rather than
# blocking. Tests that actually read CREDIT source at runtime should
# hard-assert; this one doesn't.
CREDIT = os.path.abspath("../miles-credit")
if not os.path.isdir(CREDIT):
    print(f"NOTE: ../miles-credit not present at {CREDIT} -- not required for Test 01.")

# Pin matmul precision BEFORE any JAX op so the fp32 path actually runs
# at fp32 (TPU's default would silently downgrade matmuls). CLAUDE.md
# rule for any bf16-vs-fp32 numerics test.
import jax
jax.config.update("jax_default_matmul_precision", "highest")

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import s2fft

from _common import is_tpu, tpu_info, op_counts, atomic_write_json


# --- Configuration ---------------------------------------------------------

# (nlat, nlon) per docs/test_specs.md. lmax = nlat - 1 -> s2fft L = nlat.
GRIDS = [(192, 288), (640, 1280), (721, 1440)]
SEED = 42
N_TRIALS = 5
N_THROWAWAY = 1
SAMPLING = "gl"   # gauss-legendre -- matches CREDIT's pol_lapdiff_filt.py grid
SPIN = 0
REALITY = False   # generic complex spectrum; round-trip is reality-independent

# Pre-stated success criteria from docs/test_specs.md.
TIMING_GREEN_MS = 200          # fp32 721x1440 median per call
TIMING_YELLOW_MS = 1000
BF16_REL_L2_THRESHOLD = 1e-2   # vs-fp32 max_abs / ||flm||_2

OPS_OF_INTEREST = [
    "dot", "dot-general", "gather", "scatter",
    "convolution", "all-reduce", "fft",
]


# --- Helpers ---------------------------------------------------------------

def spatial_shape(L: int, sampling: str) -> tuple[int, int]:
    """s2fft-produced spatial grid shape (n_theta, n_phi) for given L and sampling.

    Hardcoded from s2fft sampling conventions; documented in one place so
    each grid_record can record what s2fft actually computes on, separate
    from the nominal (nlat, nlon) docs/test_specs.md requested.
    """
    if sampling in ("mw", "gl"):
        return (L, 2 * L - 1)
    if sampling == "mwss":
        return (L + 1, 2 * L)
    if sampling == "dh":
        return (2 * L, 2 * L)
    raise ValueError(f"Unknown sampling: {sampling!r}")


def generate_random_flm(L: int, seed: int):
    """Seeded random complex SH spectrum, shape (L, 2L-1), |m|>l zeroed."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    real = jax.random.normal(k1, (L, 2 * L - 1))
    imag = jax.random.normal(k2, (L, 2 * L - 1))
    flm = (real + 1j * imag).astype(jnp.complex64)
    l_idx = jnp.arange(L).reshape(-1, 1)
    m_idx = (jnp.arange(2 * L - 1) - (L - 1)).reshape(1, -1)
    mask = jnp.abs(m_idx) <= l_idx
    return jnp.where(mask, flm, 0.0)


def make_round_trip(L: int):
    """Closure capturing L; returns the SHT round-trip function flm -> flm."""
    def round_trip(flm):
        f = s2fft.transform.spherical.inverse_jax(
            flm, L=L, sampling=SAMPLING, spin=SPIN, reality=REALITY
        )
        return s2fft.transform.spherical.forward_jax(
            f, L=L, sampling=SAMPLING, spin=SPIN, reality=REALITY
        )
    return round_trip


def measure_path(L: int, flm_input, precision: str):
    """Compile (timed), capture HLO, then 5 warm trials.

    Returns (record, last_output, hlo_text).
    """
    matmul_prec = "bfloat16" if precision == "bf16" else "highest"

    with jax.default_matmul_precision(matmul_prec):
        jit_rt = jax.jit(make_round_trip(L))
        t0 = time.perf_counter()
        compiled = jit_rt.lower(flm_input).compile()
        compile_s = time.perf_counter() - t0

    # HLO inspection BEFORE warm-call timing (CLAUDE.md rule 6). Sharing
    # the compile we just timed avoids a second compile for HLO extraction.
    hlo = compiled.as_text()
    counts = op_counts(hlo, OPS_OF_INTEREST)

    # Throw-away warm call (HBM staging, dispatch warmup).
    for _ in range(N_THROWAWAY):
        out = compiled(flm_input)
        out.block_until_ready()

    # Timed warm trials.
    trials_s = []
    rt_output = None
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        out = compiled(flm_input)
        out.block_until_ready()  # JAX is async; without this we time launch only
        trials_s.append(time.perf_counter() - t0)
        rt_output = out

    record = {
        "precision": precision,
        "matmul_precision": matmul_prec,
        "compile_s": compile_s,
        "trials_s": trials_s,
        "median_s": float(np.median(trials_s)),
        "p10_s": float(np.percentile(trials_s, 10)),
        "p90_s": float(np.percentile(trials_s, 90)),
        "hlo_op_counts": counts,
        "hlo_lines": hlo.count("\n") + 1,
    }
    return record, rt_output, hlo


def errors_vs(rt_output, reference, denom_l2: float) -> dict:
    """Round-trip error metrics for a single comparison."""
    diff = rt_output - reference
    max_abs = float(jnp.max(jnp.abs(diff)))
    l2 = float(jnp.linalg.norm(diff.ravel()))
    return {
        "max_abs": max_abs,
        "max_abs_rel_l2": max_abs / denom_l2,
        "l2_rel": l2 / denom_l2,
    }


def compute_decision(grids: list) -> dict:
    """Apply pre-stated success criteria from docs/test_specs.md."""
    decision = {
        "tag_thresholds_ms": [TIMING_GREEN_MS, TIMING_YELLOW_MS],
        "bf16_threshold_max_abs_rel_l2": BF16_REL_L2_THRESHOLD,
    }

    target = next(
        (g for g in grids if g["nlat"] == 721 and g["nlon"] == 1440 and "fp32" in g),
        None,
    )
    if target is None:
        decision["tag"] = "unknown"
        decision["reason"] = "721x1440 fp32 measurement missing"
    else:
        fp32_median_ms = target["fp32"]["median_s"] * 1000.0
        if fp32_median_ms < TIMING_GREEN_MS:
            tag = "green"
        elif fp32_median_ms < TIMING_YELLOW_MS:
            tag = "yellow"
        else:
            tag = "red"
        decision["fp32_721x1440_median_ms"] = fp32_median_ms
        decision["tag"] = tag

    bf16_grids = [g for g in grids if "bf16" in g and "vs_fp32" in g["bf16"]]
    if not bf16_grids:
        decision["bf16_acceptable"] = None
        decision["bf16_reason"] = "no bf16 vs-fp32 measurements completed"
    else:
        worst = max(g["bf16"]["vs_fp32"]["max_abs_rel_l2"] for g in bf16_grids)
        decision["bf16_acceptable"] = worst < BF16_REL_L2_THRESHOLD
        decision["bf16_worst_max_abs_rel_l2"] = worst
        decision["bf16_reason"] = (
            f"worst-case vs-fp32 max_abs / ||flm||_2 = {worst:.3e}; "
            f"threshold = {BF16_REL_L2_THRESHOLD:.0e}"
        )

    return decision


# --- Main ------------------------------------------------------------------

def main():
    if not is_tpu():
        print(
            f"WARNING: not on TPU. JAX devices: {jax.devices()}. "
            "Continuing for CPU sanity check; spec criteria are TPU-specific."
        )

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("results") / "test_01_s2fft"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_stamp}.json"

    payload = {
        "test_name": "test_01_s2fft_roundtrip",
        "s2fft_version": s2fft.__version__,
        "sampling": SAMPLING,
        "spin": SPIN,
        "reality": REALITY,
        "matmul_precision_pinned": "highest",
        "seed": SEED,
        "n_trials": N_TRIALS,
        "n_throwaway": N_THROWAWAY,
        "ops_counted": OPS_OF_INTEREST,
        "compilation_cache": {
            "persistent_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
            "note": (
                "Test assumes the persistent compilation cache is disabled; "
                "compile_s reflects fresh in-process compilation per grid. "
                "If persistent_dir is non-null, compile_s may reflect a "
                "cache hit and is not a true cold-start measurement."
            ),
        },
        "grids": [],
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

    write_partial()  # write skeleton early so the file exists for monitoring

    for nlat, nlon in GRIDS:
        L = nlat   # spec: lmax = nlat - 1, so L = lmax + 1 = nlat
        print(f"\n--- Grid (nlat={nlat}, nlon={nlon}, L={L}) ---")

        flm = generate_random_flm(L, SEED)
        flm_l2 = float(jnp.linalg.norm(flm.ravel()))

        produced = spatial_shape(L, SAMPLING)
        grid_record = {
            "nlat": nlat,                               # requested per docs/test_specs.md
            "nlon": nlon,                               # requested per docs/test_specs.md
            "L": L,
            "lmax": L - 1,
            "produced_spatial_shape": list(produced),   # what s2fft actually computes on
            "flm_shape": list(flm.shape),
            "flm_l2_norm": flm_l2,
            "shape_note": (
                f"Spec requested (nlat={nlat}, nlon={nlon}); s2fft "
                f"sampling='{SAMPLING}' at L={L} produces spatial shape "
                f"{tuple(produced)}. The fp32 timing tag is meaningful at "
                f"L={L} regardless of any (requested vs produced) nphi "
                f"mismatch -- L sets the compute volume; nphi details affect "
                f"padding overhead but not the order-of-magnitude that "
                f"determines the green/yellow/red thresholds. See "
                f"docs/test_specs.md."
            ),
        }

        # FP32 path
        print("  fp32: compile + 5 warm trials...")
        fp32_meas, rt_fp32, hlo_fp32 = measure_path(L, flm, "fp32")
        hlo_fp32_path = out_dir / f"hlo_{nlat}x{nlon}_fp32_{run_stamp}.txt"
        hlo_fp32_path.write_text(hlo_fp32)
        fp32_meas["hlo_path"] = str(hlo_fp32_path)
        fp32_meas["vs_input"] = errors_vs(rt_fp32, flm, flm_l2)
        grid_record["fp32"] = fp32_meas
        print(
            f"    median {fp32_meas['median_s']*1000:.2f} ms; "
            f"compile {fp32_meas['compile_s']:.2f} s; "
            f"vs-input l2_rel {fp32_meas['vs_input']['l2_rel']:.3e}"
        )

        payload["grids"].append(grid_record)
        write_partial()

        # BF16 path
        print("  bf16: compile + 5 warm trials...")
        bf16_meas, rt_bf16, hlo_bf16 = measure_path(L, flm, "bf16")
        hlo_bf16_path = out_dir / f"hlo_{nlat}x{nlon}_bf16_{run_stamp}.txt"
        hlo_bf16_path.write_text(hlo_bf16)
        bf16_meas["hlo_path"] = str(hlo_bf16_path)
        bf16_meas["vs_input"] = errors_vs(rt_bf16, flm, flm_l2)
        bf16_meas["vs_fp32"] = errors_vs(rt_bf16, rt_fp32, flm_l2)
        grid_record["bf16"] = bf16_meas
        print(
            f"    median {bf16_meas['median_s']*1000:.2f} ms; "
            f"vs-fp32 max_abs_rel_l2 {bf16_meas['vs_fp32']['max_abs_rel_l2']:.3e}"
        )

        write_partial()

    payload["decision"] = compute_decision(payload["grids"])
    write_partial()

    print(
        f"\nDecision: tag={payload['decision'].get('tag')}, "
        f"bf16_acceptable={payload['decision'].get('bf16_acceptable')}"
    )
    print(f"Results: {json_path}")
    return json_path


if __name__ == "__main__":
    main()

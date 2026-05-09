"""Shared utilities for the miles-sandbox test suite.

Per CLAUDE.md rule 9, this is the ONLY allowed cross-test import. Tests may
import from stdlib, pinned third-party packages, and ``tests._common`` --
never from another ``test_NN_*.py`` file.

Functions:
    is_tpu()       Lazy, cached check for whether JAX sees a TPU device.
    tpu_info()     Hardware/software identifiers for the results registry.
    dump_result()  Write a measurement payload to results/<test>/<stamp>.json.
    hlo_text()     Compiled HLO as text, for op-counting before timing.
    op_counts()    Count named ops in HLO text (rule 6: HLO before timing).

This module avoids importing JAX at top level so test files can be inspected
or fail fast (e.g., on the working-directory guard) without paying the JAX
import cost. JAX is imported lazily inside the functions that need it.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_TPU_DETECTED: bool | None = None


def is_tpu() -> bool:
    """Return True iff JAX sees a TPU device. Lazy and cached.

    Resolves at first call rather than at module import. Colab's TPU is not
    visible to JAX until the first JAX call, so import-time detection
    returns False permanently (CLAUDE.md rule 10).

    The result is cached for the lifetime of the process. If the JAX import
    or device query fails (e.g., running locally without JAX installed),
    returns False.
    """
    global _TPU_DETECTED
    if _TPU_DETECTED is None:
        try:
            import jax

            _TPU_DETECTED = any(d.platform == "tpu" for d in jax.devices())
        except Exception:
            _TPU_DETECTED = False
    return _TPU_DETECTED


def tpu_info() -> dict[str, Any]:
    """Return hardware/software identifiers for the results registry.

    Used by ``dump_result`` to satisfy CLAUDE.md rule 7 (performance claims
    require hardware identifier and library versions). Best-effort:
    missing libraries produce ``None`` values rather than exceptions, so
    this function never blocks a result from being written.
    """
    info: dict[str, Any] = {"tpu_detected": is_tpu()}

    try:
        import jax

        info["jax_version"] = jax.__version__
        devices = jax.devices()
        info["device_count"] = len(devices)
        info["platform"] = devices[0].platform if devices else None
        info["device_kind"] = (
            getattr(devices[0], "device_kind", None) if devices else None
        )
    except Exception as e:
        info["jax_error"] = repr(e)

    try:
        import jaxlib

        info["jaxlib_version"] = getattr(jaxlib, "__version__", None)
    except Exception:
        info["jaxlib_version"] = None

    try:
        import libtpu  # type: ignore[import-not-found]

        info["libtpu_version"] = getattr(libtpu, "__version__", None)
    except Exception:
        info["libtpu_version"] = None

    info["colab_release_tag"] = os.environ.get("COLAB_RELEASE_TAG")
    info["tpu_name"] = os.environ.get("TPU_NAME")
    return info


def dump_result(test_name: str, payload: dict[str, Any]) -> Path:
    """Write a result file under ``results/<test_name>/<utc-stamp>.json``.

    Called *during* test execution, not at the end. Per CLAUDE.md, Colab
    sessions can die at any time; results must land on disk as they are
    produced, not be buffered to a single end-of-run dump.

    Augments the caller's payload with a ``_meta`` block carrying
    ``tpu_info()`` plus the timestamp and test name, so every committed
    result row carries its own provenance.

    Returns the absolute path written.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("results") / test_name
    out_dir.mkdir(parents=True, exist_ok=True)
    enriched = {
        **payload,
        "_meta": {
            **tpu_info(),
            "timestamp_utc": stamp,
            "test_name": test_name,
        },
    }
    path = out_dir / f"{stamp}.json"
    path.write_text(json.dumps(enriched, indent=2, default=str))
    return path.resolve()


def hlo_text(jit_fn, *args, **kwargs) -> str:
    """Return compiled HLO text for a jitted callable applied to sample inputs.

    Per CLAUDE.md rule 6, HLO analysis precedes timing for any test where
    TPU compiler behavior is in question (Tests 3, 4, 6, 7).

    Caller is responsible for passing inputs whose shape and dtype match
    the intended timing run; HLO depends on both.

    Equivalent to ``jit_fn.lower(*args, **kwargs).compile().as_text()``.
    """
    return jit_fn.lower(*args, **kwargs).compile().as_text()


def op_counts(hlo: str, ops: Iterable[str]) -> dict[str, int]:
    """Count named-op occurrences in compiled HLO text.

    HLO text dumps operations roughly as ``<dtype>[<shape>] <op>(<args>)``.
    This counts whitespace-then-op-name-then-open-paren occurrences, which
    is robust to op-name substrings (e.g., counting ``dot`` will not match
    ``dot-general`` -- count both explicitly when both are of interest).

    Per CLAUDE.md rule 6, useful ops to count include:
        dot, dot-general, convolution, all-reduce, all-gather,
        gather, scatter, dynamic-slice, dynamic-update-slice,
        reduce, while, conditional, custom-call.

    Returns ``{op: count}`` for every op in ``ops``.
    """
    counts: dict[str, int] = {}
    for op in ops:
        pattern = rf"\s{re.escape(op)}\("
        counts[op] = len(re.findall(pattern, hlo))
    return counts

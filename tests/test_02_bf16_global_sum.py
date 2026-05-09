"""Test 02 -- bf16 global-sum precision audit.

Retires risk: R4 (high) -- bf16 numerics for conservation fixers.

Goal: compute area-weighted global sums of ERA5-shaped tensors in bf16,
fp32, and fp64; measure relative error of the *difference* between two
sums (which is the operation CREDIT's mass / water / energy / tracer
fixers actually perform). Sweep representative perturbation magnitudes
to identify where bf16's seven mantissa bits stop being enough.

Decision boundary: a "fix" that introduces more error than it removes is
worse than no fix at all. If bf16 cannot represent the difference of
large global sums to acceptable precision, the conservation post-blocks
must run in fp32 (with bf16 elsewhere) or use Kahan / Neumaier summation.

CREDIT reference: ``$CREDIT/credit/postblock/`` -- the global mass /
water / energy / tracer fixers.

Status: STUB -- implementation pending. See ``docs/test_specs.md`` for
the full procedure and pre-stated success criteria. The fp64 reference
requires ``jax.config.update('jax_enable_x64', True)`` set BEFORE any
JAX operation -- x64 is one-shot per session, so use a separate Python
process if a non-x64 path has already touched JAX. This test is one of
the three Gift-pitch must-haves.
"""

import os

assert os.path.basename(os.getcwd()) == "miles-sandbox", (
    f"Tests must run from miles-sandbox/, got {os.getcwd()}"
)
CREDIT = os.path.abspath("../miles-credit")
assert os.path.isdir(CREDIT), f"CREDIT clone not found at {CREDIT}"


raise NotImplementedError(
    "test_02_bf16_global_sum: stub. Implementation pending; see docstring."
)

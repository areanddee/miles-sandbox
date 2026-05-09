"""Test 06 -- spectral-norm parametrization in JAX.

Retires risk: R9 (low) -- spectral-norm parametrization correctness.

Goal: implement a spectral-norm constraint in JAX with parametrization-
style semantics equivalent to PyTorch's legacy ``nn.utils.spectral_norm``
(power-iteration estimate of the largest singular value, weight rescaled
each forward). Verify numerically against the PyTorch implementation on
identical weight matrices.

Decision boundary: output match within 1e-6 in fp32 (and 1e-3 in bf16)
on representative weight shapes is the acceptance criterion. Spectral
norm is load-bearing for almost every WxFormer layer; if this test
fails, do not proceed to Test 07 until it is settled (per
``docs/test_specs.md``). The validated implementation is promoted to
``lib/spectral_norm.py`` as a reusable artifact for the eventual port.

CREDIT reference: ``$CREDIT/credit/models/wxformer/crossformer.py``
lines 23-27 -- ``apply_spectral_norm`` walks the module tree and calls
``nn.utils.spectral_norm`` on every Conv2d / Linear / ConvTranspose2d
with non-empty weights. Many CREDIT models call this; faithful semantics
matter.

Status: STUB -- implementation pending. Discuss approach before coding
(CLAUDE.md rule 3) -- the JAX parametrization API has multiple acceptable
shapes (e.g., Equinox-style transform vs. Flax-style decorated module),
and the choice depends on the eventual port's framework selection.
"""

import os

assert os.path.basename(os.getcwd()) == "miles-sandbox", (
    f"Tests must run from miles-sandbox/, got {os.getcwd()}"
)
CREDIT = os.path.abspath("../miles-credit")
assert os.path.isdir(CREDIT), f"CREDIT clone not found at {CREDIT}"


raise NotImplementedError(
    "test_06_spectral_norm_jax: stub. Implementation pending; see docstring."
)

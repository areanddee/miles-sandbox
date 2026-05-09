"""Test 07 -- jax.checkpoint memory / compile tradeoff.

Retires risk: R10 (low) -- jax.checkpoint memory-vs-compute tradeoff.

Goal: rerun the Test-04-style scan rollout with several
``jax.checkpoint`` (remat) policies (e.g., ``checkpoint_dots``,
``offload``, custom policy by op name); measure peak HBM, compile time,
and step time. Map the tradeoff curve and recommend a default
granularity for the port.

Decision boundary: this is a backup-plan test. If Test 04 retires R3
green (rollouts compile and fit in HBM with acceptable step time), this
test is informational only. If Test 04 shows trouble, this test's
recommendation goes into the port's training loop from day one.

CREDIT reference: same as Test 04 -- ``$CREDIT/credit/trainers/`` and
``$CREDIT/credit/models/wxformer/crossformer.py``.

Status: STUB -- implementation pending. Run order: only after Test 04
results inform whether remat is needed. Defensible to defer relative to
Tests 1, 2, 5.
"""

import os

assert os.path.basename(os.getcwd()) == "miles-sandbox", (
    f"Tests must run from miles-sandbox/, got {os.getcwd()}"
)
CREDIT = os.path.abspath("../miles-credit")
assert os.path.isdir(CREDIT), f"CREDIT clone not found at {CREDIT}"


raise NotImplementedError(
    "test_07_remat_savings: stub. Implementation pending; see docstring."
)

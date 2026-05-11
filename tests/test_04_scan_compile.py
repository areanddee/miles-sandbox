"""Test 04 -- lax.scan rollout compile cost.

Retires risks: R3 (high) -- lax.scan rollout compile cost,
               R7 (medium) -- dynamic-shape data-pipeline recompile cost.

Goal: build a stand-in WxFormer-shaped layer; jit a ``lax.scan`` rollout
of N autoregressive steps with full backprop; measure cold compile time,
warm step time, and HLO size as N varies. Trigger and time recompiles
for varying-shape inputs to characterize the bucketing / padding strategy
the port will need.

Decision boundary: TPU compile times of 30-90 minutes for large N are
not unheard of on this kind of workload, and first-step latency dominates
wall-clock for short jobs. If compile dominates short training runs,
``jax.checkpoint`` (Test 07) is the standard mitigation. Bucketing the
multistep batcher's trajectory lengths is standard but must be designed
in from day one rather than retrofitted.

CREDIT reference: ``$CREDIT/credit/trainers/trainerERA5gen2.py`` for the
multistep rollout shape; ``$CREDIT/credit/models/wxformer/crossformer.py``
for the model shape.

Status: STUB -- implementation pending. HLO analysis must precede timing
(CLAUDE.md rule 6). Discuss approach before coding (rule 3) -- ``lax.scan``
with carry interacts subtly with backprop and remat.
"""

import os

assert os.path.basename(os.getcwd()) == "miles-sandbox", (
    f"Tests must run from miles-sandbox/, got {os.getcwd()}"
)


raise NotImplementedError(
    "test_04_scan_compile: stub. Implementation pending; see docstring."
)

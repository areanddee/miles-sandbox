"""Test 03 -- attention head-dim MXU utilization.

Retires risks: R5 (medium) -- attention head-dimension MXU utilization on TPU,
               R6 (medium) -- ERA5 grid dimensions vs. TPU tile alignment.

Goal: sweep ``head_dim`` in {64, 128, 256} on a stand-in WxFormer attention
block at ERA5 0.25-degree grid widths (1440 columns, 721 rows pre-padding;
XLA will pad 721 to 768). Measure MXU utilization via HLO op counts and
timing.

Decision boundary: TPU MXUs achieve peak efficiency at matmul tile
dimensions of 128 or 256. Many transformer codebases hardcode
``head_dim=64``. If WxFormer's defaults leave significant MXU headroom
on TPU v6e, the JAX port should reconfigure (a free win) rather than
inherit the GPU-tuned config. The 721-axis padding (~6.5 percent
overhead on every matmul touching the lat axis) is almost certainly
acceptable but should be characterized.

CREDIT reference: ``$CREDIT/credit/models/wxformer/crossformer.py``
(attention block), ``$CREDIT/credit/attend.py`` (SDPA dispatcher).

Status: STUB -- implementation pending. HLO analysis must precede
timing (CLAUDE.md rule 6). Defensible to defer relative to Tests 1, 2, 5.
"""

import os

assert os.path.basename(os.getcwd()) == "miles-sandbox", (
    f"Tests must run from miles-sandbox/, got {os.getcwd()}"
)
CREDIT = os.path.abspath("../miles-credit")
assert os.path.isdir(CREDIT), f"CREDIT clone not found at {CREDIT}"


raise NotImplementedError(
    "test_03_attention_tile: stub. Implementation pending; see docstring."
)

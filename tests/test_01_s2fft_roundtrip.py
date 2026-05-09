"""Test 01 -- s2fft round-trip on TPU at ERA5 grids.

Retires risk: R1 (high) -- spherical harmonic transform performance on TPU.

Goal: round-trip an ERA5 0.25-degree (1440 x 721) field through s2fft
forward and inverse spherical-harmonic transforms; measure cold and warm
wall-clock time and round-trip accuracy. Compare against torch_harmonics
on a GPU baseline where available.

Decision boundary: if s2fft cannot complete the round-trip in well under
one second on TPU v6e, the polar Laplacian filter strategy in
``$CREDIT/credit/pol_lapdiff_filt.py`` needs to be rethought before any
JAX port is undertaken. SHTs are bandwidth-bound and dominated by small
matmuls -- exactly the workload TPUs are weakest at relative to GPUs.

CREDIT reference: ``$CREDIT/credit/pol_lapdiff_filt.py`` (uses
``torch_harmonics.RealSHT``, ``InverseRealSHT``, ``RealVectorSHT``,
``InverseRealVectorSHT``).

Status: STUB -- implementation pending. See ``docs/test_specs.md`` for
the full procedure and pre-stated success criteria. Per CLAUDE.md rule 6,
HLO analysis must precede timing. This test is the highest-information-
per-hour entry in the risk register and is one of the three Gift-pitch
must-haves.
"""

import os

assert os.path.basename(os.getcwd()) == "miles-sandbox", (
    f"Tests must run from miles-sandbox/, got {os.getcwd()}"
)
CREDIT = os.path.abspath("../miles-credit")
assert os.path.isdir(CREDIT), f"CREDIT clone not found at {CREDIT}"


raise NotImplementedError(
    "test_01_s2fft_roundtrip: stub. Implementation pending; see docstring."
)

"""Test 05 -- tf.data + xarray throughput.

Retires risk: R8 (medium) -- host data pipeline throughput.

Goal: build a minimal ``tf.data`` pipeline reading ERA5-shaped synthetic
xarray / zarr data; measure samples per second on the Colab host CPU and
compare to the TPU consumption rate (use timings from Test 04 once
available; otherwise a plausible WxFormer step-time estimate from
published numbers).

Decision boundary: TPUs are very expensive idle. The host pipeline must
serve samples at >= 2x the rate the TPU consumes them, or the project
wastes compute. CREDIT's PyTorch ``DataLoader`` cannot be reused on TPU
and must be replaced with ``tf.data`` or Grain.

CREDIT reference: ``$CREDIT/credit/data/`` (``era5_singlestep``,
``era5_multistep_batcher`` on the v2 schema, xarray / zarr-backed).

Status: STUB -- implementation pending. One of the three Gift-pitch
must-haves. CPU-only ``tensorflow`` is fine on the Colab host; the TPU
itself does not need TF.
"""

import os

assert os.path.basename(os.getcwd()) == "miles-sandbox", (
    f"Tests must run from miles-sandbox/, got {os.getcwd()}"
)
CREDIT = os.path.abspath("../miles-credit")
assert os.path.isdir(CREDIT), f"CREDIT clone not found at {CREDIT}"


raise NotImplementedError(
    "test_05_xarray_pipeline: stub. Implementation pending; see docstring."
)

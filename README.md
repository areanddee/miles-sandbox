# miles-sandbox

Feasibility reconnaissance for a possible JAX/TPU port of
[NCAR/miles-credit](https://github.com/NCAR/miles-credit), the MILES
Community Research Earth Digital Intelligence Twin (CREDIT) framework
for AI-based atmospheric prediction.

This repo is a **standalone risk-reduction sandbox**, not a port. It
produces microbenchmarks, HLO analyses, and short technical memos that
answer specific TPU-feasibility questions *before* any engineering hours
are committed to a port. The CREDIT repo itself is treated as read-only
reference material; nothing in `miles-sandbox` modifies, vendors, or
forks CREDIT source.

Work here is conducted in coordination with the MILES group at NSF
NCAR. It is not a UCAR/NCAR product.

## Why this exists

CREDIT is a PyTorch/CUDA framework that targets NVIDIA GPUs and the
NSF NCAR Derecho system. As of mid-2026, running PyTorch on Google
TPUs at the scale and dynamic-shape complexity of a CREDIT trainer is
not a solved engineering problem — `PyTorch/XLA` is mature but pays
real performance penalties on CREDIT-shaped code, and the successor
project `TorchTPU` is not yet publicly available. JAX, by contrast,
is the native TPU programming model and is the language of every
production-scale AI weather model that runs on TPU today
(GraphCast/GenCast, NeuralGCM, Dinosaur).

A clean JAX implementation of a useful subset of CREDIT — the
"lighthouse" port — would unlock TPU as a viable training and
deployment target for the CREDIT scientific community. Whether that
port is worth doing depends on a small set of TPU-specific risk
questions that **cannot be answered from documentation alone**.
This sandbox answers them.

## What "lighthouse subset" means

For scoping, the lighthouse port covers exactly:

- **One trainer** — `trainerERA5gen2` (the v2 schema is the
  recommended path).
- **One model** — WxFormer (CrossFormer) at the published config
  (e.g., `wxformer_1dg_6hr_v2.yml`).
- **One dataset path** — `era5_singlestep` + `era5_multistep_batcher`
  on the v2 schema, xarray/zarr-backed.
- **The conservation post-blocks** — global mass, water, energy, and
  tracer fixers. These are CREDIT's primary scientific differentiator
  and must be ported faithfully or the physics is wrong.

Excluded from the lighthouse: ensemble methods, diffusion models,
WRF/LES regional trainers, the Samudra ocean coupling, downscaling
pipelines, the graph transformer model, the AI configuration agent,
and the inference server. Those are real CREDIT capabilities; they
are simply out of scope for the first port.

## Risks this sandbox is designed to retire

The following risk register is the working hypothesis. Each risk is
addressed by one or more numbered tests under `tests/`. Severity tags
are *pre-measurement*; the project's job is to convert them to
post-measurement green/yellow/red findings backed by reproducible
numbers in `results/`.

### 🔴 High-severity TPU risks

**R1 — Spherical harmonic transform performance on TPU.**
CREDIT's polar Laplacian filter (`pol_lapdiff_filt.py`) uses NVIDIA's
`torch_harmonics` library. The JAX-native equivalent is `s2fft`. SHTs
are bandwidth-bound and dominated by small matmuls — exactly the
workload TPUs are *least* good at relative to GPUs. If `s2fft` cannot
round-trip an ERA5 0.25° field in well under a second on TPU, the
entire filter strategy needs to be rethought.
*Test 01 retires this risk.*

**R2 — Halo exchange on the TPU torus topology.**
CREDIT's `domain_parallel` package implements 2D lat/lon
decomposition with halo exchange. JAX maps this to `lax.ppermute` over
a `shard_map` mesh. On TPU pods the inter-chip interconnect is wired
as a 2D/3D torus, and "nearest-neighbor" halo exchange behaves
pathologically if the user-requested mesh shape doesn't align with
the physical topology.
*Deferred to Phase 2 (multi-device); single-core TPU cannot
characterize this risk.*

**R3 — `lax.scan` rollout compile cost.**
CREDIT trains autoregressively, rolling the model forward N steps
under a single `jit`. For a 1B-parameter WxFormer with full backprop,
the compiled HLO can be enormous, and TPU compile times of 30–90
minutes are not unheard of on this kind of workload. First-step
latency dominates wall-clock for short jobs.
*Test 04 retires this risk; Test 07 maps the `jax.checkpoint`
mitigation if Test 04 shows trouble.*

**R4 — bf16 numerics for conservation fixers.**
TPU MXUs are bf16-native. CREDIT's mass/water/energy/tracer fixers
compute area-weighted *differences* of large global sums — a
conditioning-sensitive operation that bf16's seven mantissa bits may
not survive. A "fix" that introduces more error than it removes is
worse than no fix at all.
*Test 02 retires this risk.*

### 🟡 Medium-severity TPU risks

**R5 — Attention head-dimension MXU utilization.**
TPU MXUs achieve peak efficiency at matmul tile dimensions of 128 or
256. Many transformer codebases hardcode `head_dim=64`. WxFormer's
defaults need to be measured against actual MXU utilization; if
they're suboptimal, the port should reconfigure (a free win) rather
than inherit the GPU-tuned config.
*Test 03 retires this risk.*

**R6 — ERA5 grid dimensions vs. TPU tile alignment.**
ERA5 0.25° is 1440×721. The 721 latitude dimension is awkward (XLA
will pad to 768, costing ~6.5% on every matmul touching that axis).
Worth measuring; almost certainly acceptable but should be
characterized.
*Folded into Tests 03 and 04.*

**R7 — Dynamic-shape data pipeline recompile cost.**
JAX recompiles whenever input shapes change. CREDIT's multi-step
batcher likely yields trajectories of varying length during training.
The mitigation (bucketing + padding) is standard but must be designed
in from day one rather than retrofitted.
*Folded into Test 04 (compile cost) and Test 05 (pipeline shape).*

**R8 — Host data pipeline throughput.**
TPUs are very expensive idle. The host-side pipeline must serve
samples at least 2× the rate the TPU consumes them, or the project
wastes compute. CREDIT's PyTorch `DataLoader` has to be replaced
with `tf.data` or Grain.
*Test 05 retires this risk.*

### 🟢 Lower-severity risks (tracked but not blocking)

**R9 — Spectral norm parametrization.**
CREDIT uses PyTorch's legacy `nn.utils.spectral_norm` extensively
across model files. The JAX port needs a clean parametrization-style
implementation that matches the legacy hook-based behavior
numerically.
*Test 06 retires this risk and produces a reusable
`lib/spectral_norm.py` artifact.*

**R10 — `jax.checkpoint` (remat) memory-vs-compute tradeoff.**
If Test 04 shows that full-rollout backprop doesn't fit in HBM,
explicit remat is the standard mitigation. The cost shape needs
characterization to recommend a default granularity.
*Test 07 maps this tradeoff if Test 04 motivates it.*

### Risks NOT addressed by this sandbox

These are real risks that will need to be retired later, but they
require resources beyond a single Colab TPU v6e core and are
therefore deferred:

- **Multi-host SPMD scaling** — needs a TRC pod allocation.
- **Halo exchange on actual torus topology (R2)** — needs ≥4 cores.
- **End-to-end training equivalence with the published CREDIT
  baseline** — needs production-scale compute, weeks of runtime, and
  a porting effort that this sandbox is explicitly not undertaking.
- **Operational deployment, inference serving, ensemble
  forecasting** — out of scope.

If risks R1, R2, R4, R8 retire green, the case for the lighthouse
port is strong and a Phase 2 proposal can credibly request the
resources to retire R-deferred. If any of R1, R2, R4, R8 retire red,
the port scope or strategy must change before resources are
committed.

## Test ordering

Tests are ordered by **highest-information-per-hour for a Gift-stage
risk register**, not by ease of implementation. The first three tests
are the "must-haves": they answer the questions a Google reviewer
will actually ask about feasibility.

| # | Test | Retires |
|---|------|---------|
| 01 | `s2fft` round-trip on TPU at ERA5 grids | R1 |
| 02 | bf16 global-sum precision audit | R4 |
| 03 | Attention head-dim MXU utilization | R5, R6 |
| 04 | `lax.scan` rollout compile cost | R3, R7 |
| 05 | `tf.data` + xarray throughput | R8 |
| 06 | Spectral-norm parametrization in JAX | R9 |
| 07 | `jax.checkpoint` memory/compile tradeoff | R10 |

See `CLAUDE.md` for execution rules and `docs/` (when populated) for
per-test specifications.

## Repository layout

```
miles-sandbox/                THIS REPO
  tests/                      Test implementations (.py files)
    _common.py                Shared utilities (only allowed cross-test
                              import)
    test_01_*.py … test_07_*.py
  lib/                        Reusable artifacts (e.g., validated
                              spectral_norm.py from Test 06)
  notebooks/                  Thin Colab wrappers (clone, install, %run)
  results/                    Generated measurements (gitignored;
                              regenerable from tests)
  reports/                    Technical memos (committed; what the
                              Gift pitch quotes from)
  CLAUDE.md                   Execution rules for AI coding agents
  README.md                   This file
  LICENSE                     Apache-2.0
  NOTICE                      Provenance attribution
  requirements.txt            Pinned dependencies

../miles-credit/              READ-ONLY sibling — the CREDIT repo.
                              Not modified by anything in this sandbox.
```

## Execution model

Development happens locally and runs on Google Colab TPU v6e (single
core for now). The notebook pattern is git-based: notebooks clone this
repo via a fine-scoped GitHub PAT stored in Colab Secrets, install
pinned dependencies, and `%run` the test of interest. See `CLAUDE.md`
for the standard four-cell notebook template.

Multi-core / pod-scale tests are deferred until either a TPU Research
Cloud (TRC) allocation lands or a Gift funds dedicated TPU access.

## Relationship to upstream CREDIT

This sandbox is licensed Apache-2.0, matching CREDIT. Code that
proves useful here can be contributed back to CREDIT (or to a future
`credit-jax` companion repo) without license friction. Conversely,
nothing in this sandbox depends on or imports from CREDIT — it
references CREDIT source only for path:line citations in test
documentation.

If the lighthouse-port case is made by this work, the natural next
step is a discussion with the MILES team about whether a `credit-jax`
companion (consuming the same YAML configs, swapping the model +
trainer + post-blocks for JAX equivalents) is something they would
want to host upstream.

## Status

Pre-execution. Tests are specified; implementation begins after the
sandbox repository is scaffolded.

## License

Apache License 2.0. See `LICENSE`.

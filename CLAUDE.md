# CLAUDE.md — CREDIT JAX/TPU Risk Reduction (pre-port reconnaissance)

## Project overview

This is a **risk-reduction reconnaissance project**, not a port. Goal: answer
TPU-specific feasibility questions about a future JAX port of the MILES-CREDIT
weather modeling framework, *before* committing engineering hours to the port
itself. Outputs are microbenchmarks, HLO analyses, and short technical memos —
not production code, not a CREDIT fork.

**This repo is `miles-sandbox/`.** It sits as a sibling directory next to
the read-only `miles-credit/` clone:

```
<workspace>/
  miles-credit/      READ-ONLY. The CREDIT repo. Never modify, never
                     commit to, never import from. Reference only.
  miles-sandbox/     THIS REPO. All work lives here.
```

If a question requires reading CREDIT source, read it in place at
`../miles-credit/` and quote `path:line` references; do not vendor or
copy CREDIT files into this repo.

Target hardware for now: **single-core TPU v6e on Google Colab** (lazy
initialization, must be resolved at first JAX call). Multi-core / pod-scale
tests are deferred until either (a) a TRC allocation lands, or (b) a Gift
funds dedicated TPU access.

Deployment to Colab is **git-based**: this repo is pushed to a private
GitHub remote, and Colab clones it at session start. See "Colab
deployment" below.

## Repo layout

```
miles-sandbox/                 THIS REPO (writable, version controlled).
  tests/
    _common.py                     Shared utilities. ONLY allowed cross-test
                                   import. See test rules below.
    test_01_s2fft_roundtrip.py     SHT performance + accuracy on TPU
    test_02_bf16_global_sum.py     Conservation-fixer numerics audit
    test_03_attention_tile.py      WxFormer head-dim MXU utilization
    test_04_scan_compile.py        lax.scan compile cost for rollouts
    test_05_xarray_pipeline.py     tf.data + xarray throughput
    test_06_spectral_norm_jax.py   Spectral-norm parametrization correctness
    test_07_remat_savings.py       jax.checkpoint memory/compile tradeoff
  lib/                         Reusable artifacts that survive past this
                               project (e.g., spectral_norm.py from Test 06).
  results/                     Registry of every measurement (JSON + HLO
                               text). One subdir per test, dated. **Not
                               committed** — see .gitignore.
  reports/                     Short technical memos. One per test, plus
                               an overall risk-register update. **Committed.**
  notebooks/                   Thin Colab .ipynb wrappers. Two-or-three-cell
                               structure: clone, install, %run. No logic in
                               notebook cells beyond the boilerplate.
  requirements.txt             Pinned. See dependency rule below.
  .gitignore                   Excludes results/ payloads, __pycache__,
                               .ipynb_checkpoints, sessions, ad-hoc dumps.
  CLAUDE.md                    This file.

../miles-credit/               READ-ONLY sibling. Local clone of CREDIT.
                               Path is hardcoded as ../miles-credit
                               (see path convention below).
                               Grunt MUST NOT write here.
```

## Source-of-truth order

1. Explicit human decisions in the current conversation
2. This file (rules, conventions, parameters)
3. Code in `miles-sandbox/`
4. The Claude assessment doc `credit_jax_tpu_assessment.md`
   (project-level reference, do not modify)
5. CREDIT source at `../miles-credit/` (read-only reference)

## Path convention

All paths in this project are **relative to the repo root** (`miles-sandbox/`).

- `$SANDBOX = .` — current repo, writable.
- `$CREDIT  = ../miles-credit` — read-only sibling.

Every test file must begin with:

```python
import os
assert os.path.basename(os.getcwd()) == "miles-sandbox", (
    f"Tests must run from miles-sandbox/, got {os.getcwd()}"
)
CREDIT = os.path.abspath("../miles-credit")
assert os.path.isdir(CREDIT), f"CREDIT clone not found at {CREDIT}"
```

This catches the two most likely failure modes (wrong working directory,
missing CREDIT clone) before any other code runs. Notebooks `cd` into
`miles-sandbox/` after `git clone` and inherit the convention.

## Colab deployment (git-based)

Tests are deployed to Colab via private GitHub clone. **The human pushes;
Grunt commits locally only.** See Rule 2 below.

### One-time setup (human, already done by the time Grunt reads this)

1. `git init` `miles-sandbox/`, push to private GitHub repo `miles-sandbox`.
2. Create a fine-scoped GitHub PAT with **read-only** access to that repo.
3. Add the PAT to Colab Secrets as `GH_PAT_MILES_SANDBOX`.
4. Clone `miles-credit` somewhere accessible if Colab tests need to read
   CREDIT source directly (most tests don't — they reference specific
   `path:line` quotes captured ahead of time).

### Standard Colab notebook structure

Every notebook in `notebooks/` has exactly this shape, and no more:

```python
# Cell 1 — clone (run once per session)
from google.colab import userdata
import os, subprocess
PAT = userdata.get("GH_PAT_MILES_SANDBOX")
USER = "REPLACE_ME"  # GitHub user/org
if not os.path.isdir("/content/miles-sandbox"):
    subprocess.run(
        ["git", "clone",
         f"https://{PAT}@github.com/{USER}/miles-sandbox.git",
         "/content/miles-sandbox"],
        check=True,
    )
%cd /content/miles-sandbox
```

```python
# Cell 2 — install pinned deps (run once per session)
!pip install -q -r requirements.txt
```

```python
# Cell 3 — run the test
%run tests/test_NN_<name>.py
```

```python
# Cell 4 (optional) — surface the result file
from pathlib import Path
latest = sorted(Path("results").rglob("*.json"))[-1]
print(latest)
print(latest.read_text())
```

**Rules for notebooks:**

- No logic in notebooks beyond the four cells above. Every test is a
  `.py` file under `tests/`. If you find yourself adding a fifth cell
  with a new function, that function belongs in `tests/_common.py` or
  in the test file itself.
- Use `%run`, never `!python3`. Subprocesses can't see the TPU on Colab.
- Do not commit notebooks with stale outputs. Either commit them
  cleared (preferred) or commit them with the outputs from the
  reported run; don't mix.
- Do not paste PATs into cells. Always read from `userdata.get(...)`.

### Pulling new tests during a live session

```python
%cd /content/miles-sandbox
!git pull
%run tests/test_NN_<name>.py
```

### Saving results back

Default: **don't.** Results JSON is gitignored. Download manually with
`files.download(...)` if needed, or push from a fresh local clone after
inspecting the JSON. We do not push from Colab — the PAT is read-only
and Colab is treated as ephemeral compute.

## Hardware-specific context (Colab TPU v6e, single core)

- **Lazy TPU init.** Detect TPU at first JAX call, not at import. Colab's
  TPU is not visible until JAX touches it. Cache the result.
- **`%run`, not `!python3`.** Subprocesses can't see the TPU on Colab.
- **`jax_default_matmul_precision='highest'` BEFORE any JAX operations,**
  for any test where bf16 vs fp32 numerics is the point of the test.
- **Single core means no `shard_map`, no `ppermute`, no multi-host SPMD.**
  Tests that require multi-device (halo exchange, full pod scaling) are
  out of scope here. Document what the single-core test *does* tell us
  and what it leaves unanswered for the multi-device case.
- **TPU v6e (Trillium) specs to remember:** bf16-native MXU, 128-lane vector
  unit, prefers tile dimensions divisible by 128 (and ideally 256). HBM
  bandwidth is the typical bottleneck for memory-bound ops.
- **Colab session ephemerality.** Sessions die. Every test must dump its
  results to `results/<test>/<datestamp>.json` *during* execution, not at
  the end. Don't lose data to a 12-hour timeout.

## Non-negotiable rules

These adapt the Wraptor rules. Read them carefully.

1. **READ-ONLY CREDIT REPO.** Never write to, modify, stage, or commit
   anything in `../miles-credit/`. Never `git add` a file in that
   directory. If you need to extract data from CREDIT source, copy
   the relevant *function* into a `miles-sandbox/` file with a comment
   citing the original `path:line`. Do not vendor whole modules. The
   `assert os.path.basename(os.getcwd()) == "miles-sandbox"` guard
   at the top of every test catches the most common way this rule
   gets violated.

2. **NO COMMITS WITHOUT EXPLICIT HUMAN GO. NO PUSHES, EVER.** Stage
   changes, run tests, produce a diff summary. Wait for "commit" from
   human before `git commit`. Never run `git push`. The human is the
   sole pusher to the GitHub remote. The PAT used by Colab is
   read-only and cannot push; this is by design. If you find yourself
   wanting to push a fix during a Colab session, the answer is
   "commit locally, hand off the diff, human pushes."

3. **Discuss implementation approach before writing code.** Especially
   true for tests that involve nontrivial JAX patterns (custom_vjp,
   shard_map, scan with carry). Sketch the test in a comment block,
   confirm the approach, then write it.

4. **No trial-and-error debugging.** When a test produces an unexpected
   number: add diagnostic prints to localize the discrepancy, report
   findings with concrete numbers, wait for direction. Every code
   change must be justified by a specific reference or measured data,
   not by hypothesis.

5. **No inline `python3 -c` one-liners.** Always write a `.py` file
   and run it. One-liners break on copy-paste and waste time on
   shell escaping.

6. **HLO analysis before timing.** For any test where TPU compiler
   behavior is in question (Tests 3, 4, 6, 7), produce HLO via
   `jit.lower(...).compile().as_text()` *first*, count the relevant
   ops (gather, scatter, dynamic-slice, all-reduce, matmul, convolution),
   state a prediction for what timing should show, then time. If timing
   contradicts the HLO prediction, debug the test code or the JAX
   version, not the hardware.

7. **Performance claims require registry evidence.** No measurement is
   considered "the answer" without:
   (a) hardware identifier (Colab session UUID + TPU version string),
   (b) JAX/jaxlib/libtpu versions,
   (c) compilation cache state (cold vs warm),
   (d) at least 5 trials with median + p10/p90,
   (e) results JSON committed to `results/<test>/`.

8. **No platform-conditional branches inside test code.** A test runs on
   TPU; if it can't, it skips with a clear message. Don't write
   `if TPU: ... else: ...` paths inside the measurement code — that
   produces tests that pass on CPU and lie about TPU.

9. **Cross-test imports go through `_common.py` only.** A test file may
   import from the standard library, pinned third-party packages
   (`jax`, `numpy`, `xarray`, `tensorflow`, `s2fft`, etc.), and
   `tests/_common`. **It may not import from another `test_NN_*.py`
   file.** If two tests need the same helper, that helper goes in
   `_common.py`. This keeps every test independently runnable from
   a fresh Colab session via `%run tests/test_NN_*.py` and avoids
   the failure mode where Test 04 silently depends on Test 03 having
   been imported first.

10. **Lazy TPU detection.** Resolve `_is_tpu()` at first call, not at
    import time. Colab's lazy TPU initialization means import-time
    detection returns False permanently.

11. **YAML configs: floats with decimal point + explicit exponent sign.**
    Write `1.0e-3`, not `1e-3`; write `3.0e+8`, not `3e8`. PyYAML's safe
    loader requires a decimal point in the mantissa to tag a scalar as
    float; without it the value arrives as a string and relies on
    pydantic coercion. Same convention as Wraptor — keep it consistent.

12. **Numbers in reports must be reproducible.** Every quoted figure in
    a `reports/` memo includes a path to the JSON in `results/` that
    contains it. Reviewer must be able to grep for the number and find
    its provenance.

13. **Do not silently change JAX/jax-related dependencies.**
    `requirements.txt` is pinned. Upgrading jax or jaxlib is a human
    decision because TPU/JAX/libtpu version interactions are real.

## Test scope and ordering

Tests are ordered by **highest-information-per-hour** for a Gift-stage
risk register. Run in order; don't skip ahead.

| # | Test | Hours | Answers |
|---|------|-------|---------|
| 1 | s2fft round-trip on TPU at ERA5 grids | 4–8 | SHT performance — biggest single risk |
| 2 | bf16 global-sum precision audit | 2–4 | Whether conservation fixers can run in bf16 |
| 3 | Attention head-dim MXU utilization | 4–8 | WxFormer head sizing on TPU MXU |
| 4 | lax.scan rollout compile cost | 8–16 | Whether autoregressive training is tractable |
| 5 | tf.data + xarray throughput | 4–8 | Whether host pipeline can feed TPU |
| 6 | Spectral-norm parametrization in JAX | 4–8 | One-time pattern, validate against torch |
| 7 | jax.checkpoint memory/compile tradeoff | 8–16 | Backup plan if Test 4 shows trouble |

**Tests 1, 2, 5 are the must-haves for the Gift pitch.** They answer the
three questions a Google reviewer will actually ask. Tests 3, 4, 6, 7
are valuable but defensible to defer.

## What this project is NOT

- **Not a CREDIT port.** We do not produce JAX implementations of CREDIT
  models, trainers, datasets, or post-blocks. Stand-in models with the
  right *shape* are fine for compile/timing tests; functional correctness
  vs CREDIT is out of scope.

- **Not a product.** No CLI, no config files beyond what individual
  tests need, no setup.py beyond requirements pinning. Scripts and
  notebooks only.

- **Not a benchmarking framework.** Each test is a self-contained .py
  file. We do not build a generalized harness.

- **Not multi-device.** Single core only. Multi-device tests will live
  in a successor repo when TRC access lands.

## Token efficiency rules

1. Do NOT re-read the assessment doc on every interaction. It is
   reference; cite section numbers.
2. Do NOT re-derive what TPU v6e bf16 throughput is. If the value is
   needed, look it up once and pin it in this file.
3. Do NOT browse the CREDIT repo to "understand context." Specific
   line-number questions only.
4. Start responses with the measurement or the code, not with
   re-establishing context.
5. When timing results come in, report the median and the spread.
   Don't paste raw timing arrays — link to the JSON.

## Multi-agent workflow

- **Claude (architecture chat):** Test design, risk interpretation,
  what-does-this-result-mean analysis, report writing.
- **Grunt (Claude Code, this repo):** Test implementation, Colab
  notebook construction, HLO extraction, JSON results emission.
  Operates in `risk_tests/` only. Read-only access to
  `$CREDIT_REPO_PATH`.
- **Human:** Runs Colab sessions (TPU v6e currently), commits when
  approved, drives prioritization decisions.

# Test 01 Refactor: Swap s2fft for Dinosaur SHT

## Why

The s2fft library, while JAX-native and supporting GL sampling, is built
by an astrophysics group whose accuracy requirements assume fp64. Empirical
evidence from a human-run A100 benchmark: s2fft fp64 GL round-trip works
fine; s2fft **fp32** GL round-trip **takes forever to compile** on the same
hardware. The library emits an explicit warning when x64 is not enabled.

For our Test 01 question — "is SHT viable for CREDIT's polar Laplacian
filter on TPU" — we want the JAX SHT implementation that Google's own
NeuralGCM team built and runs on TPU in production: **Dinosaur**.

Dinosaur is the SHT/dycore library inside NeuralGCM. It's Apache-2.0,
JAX-native, and explicitly designed for "modern accelerator hardware
(GPU/TPU)" per its README. Repo:
https://github.com/neuralgcm/dinosaur. Install: `pip install dinosaur`.

Testing against Dinosaur is also a stronger claim for the eventual Gift
pitch: "we benchmarked the JAX SHT used by NeuralGCM on TPU" reads better
than "we benchmarked an astrophysics library against its authors'
warnings."

## What changes

1. `requirements.txt`: add `dinosaur` (pinned to whatever version installs
   cleanly on Colab — record the version observed). Remove `s2fft`.
   Keep `s2fft` commented out with a note linking to this refactor doc,
   for future-archaeology purposes.

2. `tests/test_01_s2fft_roundtrip.py` → rename to
   `tests/test_01_sht_roundtrip.py`. The test is about SHT round-trip,
   not about a specific library. Update the docstring accordingly.

3. Test body: replace s2fft imports and calls with Dinosaur equivalents.
   API discovery is part of the work — see "API discovery" below.

4. Decision block, success criteria, JSON schema: unchanged structurally,
   but record the SHT library + version explicitly in `_meta` so future
   readers know which library produced which numbers.

5. Update the test docstring header to cite the Dinosaur paper / repo
   and explain the swap-from-s2fft.

## API discovery

The Dinosaur public API isn't well-indexed by search, so the first
implementation step is to discover the actual entry points. Do this
on Colab (NOT on EC2 — Dinosaur may need JAX configured, and we want
to verify the install path that the test will use):

1. `pip install dinosaur` in a Colab cell. Record the installed version.
2. `import dinosaur; print(dir(dinosaur))` — top-level surface.
3. `from dinosaur import spherical_harmonic; print(dir(spherical_harmonic))`
   — this is the module the source-tree layout strongly implies exists.
4. Find:
   - A grid/sampling configuration object (likely a `Grid` class with
     `gaussian(...)` or `gauss_legendre(...)` constructor)
   - A forward transform (grid → spectral)
   - An inverse transform (spectral → grid)
   - Whatever precision/dtype controls exist
5. Print the docstrings of those entry points.
6. Construct a minimal round-trip on a small grid (L=32 or so) to
   confirm the API works end-to-end. Verify max-abs error is small.

Paste the discovery output back BEFORE writing the full test. The
discovery step gives us the canonical API, then the test is a
mechanical rewrite around it.

If `dinosaur.spherical_harmonic` doesn't exist, try `dinosaur.transforms`,
`dinosaur.spectral`, `dinosaur.shc`, or whatever the top-level `dir()`
suggests. If the public API isn't obvious, fall back to reading the
Dinosaur source on GitHub at
https://github.com/neuralgcm/dinosaur/tree/main/dinosaur — it's
~10 modules, browsable in a few minutes.

## What stays the same

- Three grids: keep the (192, ~), (640, ~), (721, ~) sweep, adjusted
  to whatever spatial shape Dinosaur's GL grid naturally produces at
  each L. Record both requested L and produced (nlat, nlon) in the JSON.
- Three precisions: fp32_default (TPU MXU's bf16-multiply-fp32-acc),
  fp32_highest (true fp32 via Ozaki-scheme emulation), bf16.
- HLO captured before timing (CLAUDE.md rule 6).
- Atomic incremental JSON writes via `_common.atomic_write_json`.
- Decision block: green/yellow/red on fp32 timing at 721×~ vs
  pre-stated thresholds (200 ms / 1000 ms); bf16_acceptable separately
  from precision error.
- Working-directory guard at top of file.

## What's new in _meta

Record under `_meta`:
- `sht_library`: `"dinosaur"` (so the JSON is self-describing if we
  ever benchmark multiple libraries)
- `sht_library_version`: the installed version string from
  `dinosaur.__version__` if it exists, else `pip show dinosaur` parsed
- `replaces`: `"s2fft"` (to maintain provenance for the file rename)

## Out of scope (still)

Same as before: vector SHTs (RealVectorSHT for vorticity/divergence —
the natural Test 01b follow-up), torch_harmonics GPU baseline (only
if Dinosaur lands yellow on TPU), real-field structure of the test
spectrum.

## Process

1. Stage `requirements.txt` change (dinosaur in, s2fft out).
2. Run API discovery on Colab. Paste output back.
3. Discuss the API discovery with the human; agree on exact entry
   points before writing test logic.
4. Rename the test file (`git mv`).
5. Implement the test body around the discovered Dinosaur API.
6. Stage all changes; bring back the diff for review.
7. Human reviews, then runs on Colab, then commits.

## Note on the file rename

`git mv tests/test_01_s2fft_roundtrip.py tests/test_01_sht_roundtrip.py`
preserves history. If for any reason the rename can't preserve history
cleanly, do it in a separate atomic commit before the body changes,
not bundled with the rewrite.

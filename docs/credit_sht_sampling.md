# CREDIT SHT sampling note

**TL;DR:** miles-credit uses `torch_harmonics` with `grid="legendre-gauss"`
(Gauss-Legendre) for the polar Laplacian filter — the SHT workload that
motivates risk R1. **No `s2fft`, no `mw`, no `healpix` anywhere in
miles-credit.**

This note locks in the sampling choice for Test 01's SHT round-trip
benchmark (whether implemented via Dinosaur, s2fft, or any future
candidate): **Gauss-Legendre** matches the workload R1 is about.

## Evidence

Searched on 2026-05-12 against the read-only `../miles-credit/` clone.

| File | Lines | Finding |
|---|---|---|
| `credit/pol_lapdiff_filt.py` | 102, 118, 151–162, 172 | Default `grid="legendre-gauss"`; quadrature via `harmonics.quadrature.legendre_gauss_weights`. All four SHT objects (`RealSHT`, `InverseRealSHT`, `RealVectorSHT`, `InverseRealVectorSHT`) use this grid. |
| `credit/parser.py` | 488 | Config default: `conf["model"]["post_conf"].setdefault("grid", "legendre-gauss")` |
| `credit/trainers/utils.py` | 94 | Same `setdefault("grid", "legendre-gauss")` as above. |
| `credit/skebs.py` | 7, 452–455 | Threads `self.grid` from caller; commented-out `equiangular` alternative at line 466 (considered, not used). |
| `credit/verification/standard.py` | 35, 94 | Passes `grid=grid` through from caller; no default of its own. |
| `credit/ensemble/spherical.py` | 175, 189, 209 | **Outlier** — default `grid_type="equiangular"` for the ensemble Gaussian-noise generator. Different code path; not on R1's polar-filter critical path. |

Searches that returned zero hits in `credit/`:
- `s2fft` — CREDIT does not use it
- `healpix` — not used
- `"mw"`, `"mwss"` — torch_harmonics doesn't expose those names anyway

## Cross-library mapping

| CREDIT (`torch_harmonics`) | s2fft | Dinosaur |
|---|---|---|
| `"legendre-gauss"` (R1 path) | `"gl"` | Gauss-Legendre grid constructor |
| `"equiangular"` (ensemble noise only) | `"dh"` (Driscoll-Healy; not bit-exact) | (TBD — confirm during API discovery) |
| n/a | `"mw"`, `"mwss"`, `"healpix"` | n/a |

## Implication for Test 01

When the Dinosaur API is settled (see `docs/test_01_refactor_dinosaur.md`),
the grid constructor we want is **Gauss-Legendre** — that is the workload
`pol_lapdiff_filt.py` actually runs, so it is the workload whose TPU
performance retires R1. The earlier draft `SAMPLING = "gl"` in the
s2fft-flavored test was the right call; the Dinosaur rewrite should
mirror it on whatever Dinosaur's Gauss-Legendre grid constructor turns
out to be.

The ensemble Gaussian-noise generator's `equiangular` grid is a
secondary concern: testing the GL path is necessary and sufficient for
R1. If a separate "ensemble noise generator on TPU" risk emerges later,
that's a follow-on test, not a Test 01 scope expansion.

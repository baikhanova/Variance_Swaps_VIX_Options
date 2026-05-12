# W4 — COS pricing part

This branch is for the W4 part of the project.

The goal is to implement a Fourier/COS-based pricing block for VIX-related products and compare its output with the Monte Carlo results from W3.

## Main files

- `src/models/cir.py`
- `src/pricing/vix_cos.py`
- `tests/test_vix_cos.py`

## Scope

W4 should focus on:

- CIR variance model functions;
- characteristic function implementation;
- VIX futures or VIX option pricing structure;
- comparison with W3 Monte Carlo results;
- basic tests for numerical sanity.

## Important note

W2 derives the affine structure of VIX squared under the Heston model and also gives the characteristic function setup for the CIR/integrated variance process.

Before implementing the final COS pricing formula, the exact characteristic function used for the VIX payoff should be checked carefully. The VIX option payoff depends on the terminal VIX level, while W2 discusses integrated variance as well.

## Planned implementation

In `src/models/cir.py`:

- define CIR parameters;
- implement CIR-related characteristic function;
- implement VIX squared conversion from variance.

In `src/pricing/vix_cos.py`:

- prepare COS pricing functions;
- price VIX calls and puts;
- compare COS output with W3 Monte Carlo output.

In `tests/test_vix_cos.py`:

- check that the characteristic function equals 1 at zero;
- check that VIX squared is non-negative;
- check that option prices are non-negative.

## First target

The first W4 target is not to produce a final market-calibrated pricer immediately.

The first target is to build a small, readable COS pricing prototype that can be tested and compared against W3 Monte Carlo results.
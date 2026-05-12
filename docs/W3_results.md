# W3 — Monte Carlo results

This file summarises the current W3 implementation.

The main code is placed in:

- `src/models/heston.py`
- `src/pricing/variance_mc.py`

The notebook used for checking the results is:

- `notebooks/01_w3_monte_carlo_engine.ipynb`

## What was implemented

The W3 part currently includes:

- Heston stock and variance path simulation;
- realised variance calculation from simulated stock paths;
- analytical Heston variance strike;
- Monte Carlo variance swap pricing;
- VIX call pricing by Monte Carlo;
- confidence interval calculation;
- convergence study for different numbers of paths.

## Main validation idea

The first check is based on the comparison between:

```text
Monte Carlo estimate of realised variance and analytical Heston variance strike
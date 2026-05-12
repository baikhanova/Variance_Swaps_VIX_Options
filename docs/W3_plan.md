# W3 — Monte Carlo part

This part is responsible only for the Monte Carlo implementation.

The goal is to simulate the Heston variance process, compute realised variance, price the variance swap by Monte Carlo, and compare the result with the analytical variance strike from W2.

## Files

Main files:

- `src/models/heston.py`
- `src/pricing/variance_mc.py`
- `notebooks/01_w3_monte_carlo_engine.ipynb`

## Formulas used

The variance process is:

```text
dv_t = kappa * (theta - v_t) dt + sigma_v * sqrt(v_t) dW_t
```

The analytical fair variance strike is:

```text
K_var = theta + (v0 - theta) * (1 - exp(-kappa * T)) / (kappa * T)
```

The VIX squared approximation is:

```text
VIX_t^2 = theta + (v_t - theta) * (1 - exp(-kappa * Delta)) / (kappa * Delta)
```
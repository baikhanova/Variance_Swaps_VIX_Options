# W3 — Monte Carlo part

This part is responsible only for the Monte Carlo implementation.

The goal is to simulate the Heston variance process, compute realised variance, price the variance swap by Monte Carlo, and compare the result with the analytical variance strike from W2.

## Files

Main files:

- `src/models/heston.py`
- `src/pricing/variance_mc.py`
- `notebooks/01_w3_monte_carlo_engine.ipynb`

## Formulas used

The variance process follows the Heston/CIR form:

$$dv_t = \kappa(\theta - v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t$$

Here, $v_t$ is the instantaneous variance, $\theta$ is the long-run variance level, $\kappa$ is the mean-reversion speed, and $\sigma_v$ is the volatility of variance.

The analytical fair variance strike is:

$$K_{\text{var}} = \theta + (v_0 - \theta)\frac{1 - e^{-\kappa T}}{\kappa T}$$

This formula will be used as the first benchmark for the Monte Carlo result.

For the VIX-related Monte Carlo part, the project uses the affine relation:

$$\text{VIX}_t^2 = \theta + (v_t - \theta)\frac{1 - e^{-\kappa \Delta}}{\kappa \Delta}$$

After simulating the variance process, the terminal variance can be converted into a VIX level and used for VIX call pricing.
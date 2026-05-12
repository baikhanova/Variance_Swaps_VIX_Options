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

The first check is based on comparing the Monte Carlo estimate of realised variance with the analytical Heston variance strike.

If the simulation is implemented correctly, the Monte Carlo estimate should be close to the analytical value. The difference should also become more stable when the number of simulated paths increases.

## Current numerical output

The current run uses the base parameter set:

- $$S_0 = 100$$
- $$v_0 = 0.04$$
- $$r = 0.03$$
- $$\kappa = 2.0$$
- $$\theta = 0.04$$
- $$\sigma_v = 0.5$$
- $$\rho = -0.7$$
- maturity: 1 year
- steps: 252
- paths: 10,000
- seed: 42

The simulation produced stock and variance path arrays with shape:

| Output | Shape |
|---|---:|
| stock paths | (10000, 253) |
| variance paths | (10000, 253) |

The realised variance distribution from 10,000 paths was:

| Statistic | Value |
|---|---:|
| Count | 10000 |
| Mean | 0.040269 |
| Standard deviation | 0.031658 |
| Minimum | 0.003090 |
| 25% | 0.018492 |
| Median | 0.030821 |
| 75% | 0.051490 |
| Maximum | 0.412461 |

The main variance swap check gave:

| Quantity | Value |
|---|---:|
| Monte Carlo realised variance | 0.040269 |
| 95% CI lower | 0.039648 |
| 95% CI upper | 0.040889 |
| Analytical $$K_{\text{var}}$$ | 0.040000 |
| Used strike | 0.040000 |
| Absolute error | 0.000269 |
| Relative error | 0.006724 |
| Swap value | 0.000261 |

The VIX call Monte Carlo check with strike $$K = 20$$ gave:

| Quantity | Value |
|---|---:|
| VIX call price | 1.984952 |
| Undiscounted payoff mean | 1.989852 |
| 95% CI lower, discounted | 1.919523 |
| 95% CI upper, discounted | 2.050381 |
| Mean terminal VIX | 18.934275 |
| Strike | 20.000000 |

The convergence table was:

| Paths | MC realised variance | Analytical $$K_{\text{var}}$$ | Absolute error | Relative error | CI width |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.040907 | 0.040000 | 0.000907 | 0.022672 | 0.004002 |
| 5,000 | 0.040471 | 0.040000 | 0.000471 | 0.011784 | 0.001756 |
| 10,000 | 0.040269 | 0.040000 | 0.000269 | 0.006724 | 0.001241 |
| 50,000 | 0.040286 | 0.040000 | 0.000286 | 0.007144 | 0.000549 |

The result is close to the analytical variance strike. The confidence interval becomes narrower when the number of paths increases, which is the expected behaviour for the Monte Carlo estimator.
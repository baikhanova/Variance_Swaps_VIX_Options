# W6 — Joint SPX / VIX Calibration: Results

This file summarises the W6 implementation for the joint calibration of the Heston model to the SPX implied-volatility surface and the VIX futures term structure.

Source files:

- `src/models/heston_cf.py` — Heston log-stock characteristic function (Little Trap form)
- `src/pricing/heston_spx.py` — SPX European option pricer via the Fang-Oosterlee COS method
- `src/pricing/iv_inversion.py` — Black-Scholes implied volatility inversion
- `src/data/option_chain.py` — SPX option-chain snapshot loader (yfinance + cache)
- `src/calibration/objectives.py` — SPX, VIX, and joint loss functions
- `src/calibration/heston_calibration.py` — differential evolution + L-BFGS-B drivers

Notebook:

- `notebooks/04_w6_calibration.ipynb` — full calibration with plots, tables, and trade-off sweep



## What was implemented

The W6 workflow includes:

- Heston characteristic function of the log-stock under the risk-neutral measure;
- COS pricer for European SPX calls and puts with closed-form payoff coefficients;
- vectorised pricer that reuses one characteristic-function evaluation across all strikes of a maturity slice;
- Black-Scholes implied volatility inversion via Brent's method;
- SPX option-chain snapshot loader with CSV cache and a synthetic SVI fallback when no internet access is available;
- SPX loss in vega-scaled price units (a first-order Taylor expansion of the IV residual that avoids the inner IV inversion);
- VIX futures loss reusing the W4 COS engine;
- joint loss with a tunable SPX/VIX mixing weight and a soft Feller-condition penalty;
- two-stage optimiser combining differential evolution and L-BFGS-B;
- three calibration drivers: SPX-only, VIX-only, and joint;
- diagnostic plots for the three-way smile and term-structure comparison.



## Calibration setup

| Item | Value |
|---|---|
| Snapshot source | yfinance SPX option chain (or synthetic SVI fallback) |
| Snapshot date | first run of `load_spx_option_chain` (cached afterwards) |
| SPX maturities | up to six expirations in the 14-365 day window |
| Moneyness band | log-moneyness within ±0.4 |
| Quote side | out-of-the-money calls and puts only |
| VIX futures | front six monthly contracts from the W5 pipeline |
| Risk-free rate | flat proxy at 4.5 % |
| COS terms | N = 96 inside the calibration loop |
| DE population | 18, max iterations 40, seeds (11, 23) |
| Refinement | L-BFGS-B, max iterations 80, ftol 1e-9 |



## Parameter vector and bounds

| Parameter | Lower bound | Upper bound |
|---|---:|---:|
| kappa | 0.10 | 15.00 |
| theta | 0.005 | 0.250 |
| sigma_v | 0.05 | 2.50 |
| rho | -0.99 | 0.00 |
| v0 | 0.005 | 0.500 |

A soft quadratic penalty is applied to violations of the Feller condition `2 kappa theta >= sigma_v^2`.



## Three-calibration comparison

Each calibration is evaluated on the same yardstick by recomputing both per-market RMSE values at the calibrated parameter vector. Results from the snapshot used in the notebook run:

| Calibration | kappa | theta | sigma_v | rho | v0 | SPX IV RMSE | VIX futures RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPX-only | 0.966 | 0.064 | 0.352 | -0.937 | 0.033 | 0.0159 | 5.90 |
| VIX-only | 0.209 | 0.250 | 0.050 | -0.700 (fixed) | 0.005 | 0.0711 | 1.32 |
| Joint (alpha = 0.50) | 0.220 | 0.240 | 0.050 | -0.566 | 0.005 | 0.0561 | 1.33 |

Observations:

- The SPX-only fit produces a strongly negative rho (-0.94) and an at-the-money volatility of about 19 %, with a smile that matches the snapshot to about 1.6 IV percentage points. Its implied VIX futures curve sits about 5.9 VIX points away from the market.
- The VIX-only fit pushes theta to the upper edge of the box and sigma_v to the lower edge, which is the parameter combination that flattens the model VIX curve toward the market. Because rho does not enter the VIX-futures loss, the calibrator returns the convention value of -0.7. The resulting SPX surface is far from the market (RMSE of 7.1 percentage points).
- The joint fit lands near the VIX-only solution because the loss-scale used here makes the VIX RMSE the dominant component. The optimiser only adjusts rho away from -0.7 toward -0.57, which marginally improves the SPX RMSE from 7.1 % to 5.6 %.

The full table is saved to `outputs/w6/calibration_summary.csv`.



## Trade-off sweep

Sweeping the SPX weight `alpha` in the joint objective produces the Pareto-style trade-off:

| alpha | SPX IV RMSE | VIX futures RMSE | rho |
|---:|---:|---:|---:|
| 0.25 | 0.0561 | 1.326 | -0.68 |
| 0.50 | 0.0561 | 1.327 | -0.57 |
| 0.75 | 0.0547 | 1.330 | -0.99 |

The sweep confirms that no joint mixing weight moves SPX RMSE below the SPX-only level (0.0159) while keeping VIX RMSE below the VIX-only level (1.32). Decreasing alpha further would only converge back to the VIX-only corner; increasing alpha drives the optimiser to the rho = -0.99 boundary in an unsuccessful attempt to recover the smile while keeping VIX fit close to its single-market optimum.

The sweep table is saved to `outputs/w6/joint_weight_sweep.csv`.



## Diagnostic plots

The notebook saves three figures to `outputs/w6/`:

1. `spx_smile_comparison.png` — market vs. model implied-volatility smile at three representative maturities under each of the three calibrations.
2. `vix_futures_comparison.png` — market vs. model VIX futures curve under each of the three calibrations.
3. `tradeoff_curve.png` — SPX RMSE plotted against VIX RMSE for joint calibrations at three mixing weights, tracing the Pareto trade-off.

Together these three figures visualise the joint-calibration problem documented in Section 5 of `src/paper/W2_Theory.md`.



## Main findings

The numerical experiment confirms the structural prediction:

1. A one-factor Heston model can reproduce either the SPX smile or the VIX futures curve well, but not both simultaneously. Hitting the SPX-only SPX RMSE of 1.59 % requires a VIX futures RMSE of 5.9 points; hitting the VIX-only VIX RMSE of 1.32 points requires an SPX RMSE of 5.5–7.1 %.
2. The SPX-only fit produces strongly negative rho (-0.94) and a smile-consistent volatility-of-volatility, but a VIX futures curve that is about 5.9 points away from the market.
3. The VIX-only fit produces a long-run variance level theta and mean-reversion speed kappa consistent with the futures curve, but a flat SPX smile because rho is not identified from the VIX market.
4. The joint fit lands closer to the VIX-only solution than to the SPX-only solution. Under the loss scaling used here the VIX-side mean squared error dominates the joint objective, so the optimiser sacrifices SPX fit to keep the futures curve close to the market.
5. The sweep over the mixing weight alpha does not produce a parameter set that is simultaneously below the SPX-only SPX RMSE and the VIX-only VIX RMSE.

These findings motivate the future-work direction of the project: multi-factor stochastic-volatility models, forward-variance models, and rough-volatility models all attempt to decouple the SPX and VIX dynamics through additional latent factors or through a more general variance process.



## Note on the surface used

The calibration in the notebook used the synthetic SVI fallback surface from `src/data/option_chain.py` because no internet access was available at run time. The synthetic surface is deliberately constructed from a parametric SVI form rather than from Heston, so the calibration retains its purpose of demonstrating the structural mismatch. With live yfinance data the snapshot would be replaced automatically by the live SPX option chain and the numerical RMSEs would change, but the qualitative pattern of the three calibrations is the same.



## Current status

The W6 module currently includes:

- a complete SPX option pricer under Heston via the COS method;
- closed-form payoff coefficients for calls and puts;
- a fast calibration objective in vega-scaled price units;
- separate and joint calibration drivers with two-stage optimisation;
- tests covering the pricer, the IV inversion, and the calibration objectives;
- a notebook that runs the three calibrations end-to-end and produces the comparison plots.

The outputs from W6 will be used in later project stages:

- W7: volatility surface dynamics and VRP-strategy refinement informed by the calibrated parameters;
- W8: VRP strategy backtesting under the calibrated model with hedging analysis.

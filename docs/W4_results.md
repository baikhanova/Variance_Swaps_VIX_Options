# W4 — COS pricing results

This file summarises the current W4 COS pricing prototype.

The main code is placed in:

- `src/models/cir.py`
- `src/pricing/vix_cos.py`

The notebook used for checking the results is:

- `notebooks/02_w4_cos_engine.ipynb`

## What was implemented

The W4 part currently includes:

- CIR parameter setup;
- terminal variance characteristic function;
- VIX affine conversion from variance;
- COS pricing prototype for VIX calls and puts;
- comparison helper for COS price versus W3 Monte Carlo price;
- basic tests for CIR utilities and COS pricing outputs.

## Current parameter set

The current run uses:

- $$v_0 = 0.04$$
- $$\kappa = 2.0$$
- $$\theta = 0.04$$
- $$\sigma_v = 0.5$$
- option maturity: 30 / 365
- interest rate: 0.03
- VIX window: 30 / 365
- COS terms: 128
- truncation width: 8 standard deviations
- coefficient grid size: 4096

## CIR and VIX setup

The current setup produced the following values:

| Quantity | Value |
|---|---:|
| A coefficient | 0.003115 |
| B coefficient | 0.922133 |
| $$E[v_T]$$ | 0.040000 |
| $$Var[v_T]$$ | 0.000700 |
| Lower truncation | 0.000000 |
| Upper truncation | 0.251732 |
| VIX at $$\theta$$ | 20.000000 |

The truncation interval is positive and wide enough for the current prototype. The value `VIX at theta = 20` also works as a useful sanity check because $$\sqrt{0.04} \times 100 = 20$$.

## VIX option pricing check

For strike $$K = 20$$, the COS prototype gave:

| Quantity | Value |
|---|---:|
| VIX call COS price | 2.008969 |
| VIX put COS price | 2.949635 |
| Strike | 20.000000 |
| Lower truncation | 0.000000 |
| Upper truncation | 0.251732 |
| COS terms | 128 |

## Comparison with W3 Monte Carlo

The W3 Monte Carlo VIX call price was:

| Method | Price |
|---|---:|
| W3 Monte Carlo | 1.984952 |
| W4 COS prototype | 2.008969 |

The difference was:

| Quantity | Value |
|---|---:|
| Absolute difference | 0.024017 |
| Relative difference | 0.012100 |

The COS price is close to the Monte Carlo benchmark from W3. The relative difference is around 1.21%, which is acceptable for the first prototype because the current COS version still uses numerical payoff coefficients instead of closed-form payoff coefficients.

## Prices across strikes

The COS prototype was also tested across several strikes:

| Strike | Call COS price | Put COS price |
|---:|---:|---:|
| 10 | 9.158711 | 0.124004 |
| 15 | 4.932858 | 0.885837 |
| 20 | 2.008969 | 2.949635 |
| 25 | 0.571070 | 6.499422 |
| 30 | 0.106450 | 11.022488 |
| 35 | 0.012447 | 15.916172 |
| 40 | 0.000881 | 20.892292 |

The behaviour is reasonable: call prices decrease as the strike increases, while put prices increase as the strike increases.

## Current status

The W4 COS part is ready as a first working prototype.

The current version is not yet a final market-calibrated pricer. It is a numerical prototype that shows:

- the CIR/VIX setup works;
- COS prices are non-negative;
- call and put prices behave reasonably across strikes;
- the VIX call COS price is close to the W3 Monte Carlo benchmark.

## Possible next improvements

The next improvements can be:

- replace numerical payoff coefficients with closed-form COS payoff coefficients;
- add a runtime comparison between COS and Monte Carlo;
- calibrate parameters to market data;
- compare the COS output with external references such as QuantLib or MATLAB where possible.
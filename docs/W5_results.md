# W5 — Data Collection and Exploratory Data Analysis (EDA): Results

This file summarises the W5 implementation for market data collection, realised variance estimation, volatility risk premium analysis, and exploratory data analysis.

Source files:

- `src/data/data_collection.py` — market data pipeline, VRP computation, plotting
- `notebooks/03_w5_data_eda.ipynb` — full exploratory analysis and figures



## What was implemented

The W5 workflow includes:

- downloading VIX daily index data;
- downloading SPX market data;
- obtaining VIX futures settlement data;
- computing realised variance from SPX returns;
- computing ex-post volatility risk premium;
- computing ex-ante VRP using GARCH(1,1) forecasts;
- identifying crisis periods;
- generating exploratory plots.

The ex-post volatility risk premium is computed as:

$$
VRP_t = \frac{VIX_t^2}{12} - RV_{t,t+1M}
$$



## Data sources

Market data used:

| Dataset | Source |
|---|---|
| VIX index | Yahoo Finance / FRED |
| SPX index | Yahoo Finance |
| VIX futures | CBOE |
| Risk-free proxy | FRED |

Sample period:

| Start | End |
|---|---|
| 2019 | 2024 |



## Realised variance calculation

Realised variance is computed from log returns:

$$
RV_t = \sum_i r_i^2
$$

where:

$$
r_i = \log\left(\frac{S_i}{S_{i-1}}\right)
$$

Monthly realised variance is annualised where appropriate.



## Volatility Risk Premium (VRP)

The monthly volatility risk premium is computed as:

$$
VRP_t = IV_t - RV_{t,t+1M}
$$

In this project, one-month implied variance is approximated from the VIX index:

$$
IV_t = \frac{1}{12}\left(\frac{VIX_t}{100}\right)^2
$$

Therefore:

$$
VRP_t = \frac{1}{12}\left(\frac{VIX_t}{100}\right)^2 - RV_{t,t+1M}
$$

A positive VRP implies that implied variance is higher than subsequently realised variance.



## Generated outputs

The following figures are produced by the W5 pipeline:

1. VIX time series;
2. VRP time series;
3. scatter plot of VIX versus next-month realised variance;
4. VIX futures term structure;
5. VIX correlation with SPX returns;
6. realised variance versus implied variance decomposition.



## Crisis analysis

The following stress periods are analysed:

| Crisis period | Main feature |
|---|---|
| 2020 COVID shock | Extreme VIX spike and large realised variance |
| 2022 inflation shock | Elevated volatility regime and unstable VRP |

Expected observations:

- implied volatility increases sharply;
- realised variance rises after market shocks;
- VRP compresses or becomes negative;
- correlation between VIX changes and SPX returns becomes strongly negative.



## Main findings

The analysis confirms several stylised facts:

1. VIX rises sharply during market stress.
2. Implied variance is usually higher than future realised variance.
3. The volatility risk premium is generally positive in calm markets.
4. Crisis periods create temporary breakdowns in normal VRP behaviour.
5. VIX and SPX returns are negatively related, especially during stress periods.

These findings are consistent with the interpretation of volatility as an insurance asset class. Investors pay a premium for protection against market volatility, and sellers of variance earn this premium in normal periods while taking crisis risk.



## Current status

The W5 module currently includes:

- automated market data collection;
- realised variance computation;
- ex-post and ex-ante VRP estimation;
- exploratory plots;
- crisis-period identification;
- notebook implementation for reproducible analysis.

The outputs from W5 will be used in later project stages:

- W6: joint SPX–VIX calibration;
- W7: volatility surface and VRP analysis;
- W8: VRP strategy backtesting and risk analysis.

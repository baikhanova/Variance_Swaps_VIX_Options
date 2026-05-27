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
- computing ex-post volatility risk premium:

\[
VRP_t =
\frac{VIX_t^2}{12}
-
RV_{t,t+1M}
\]

- computing ex-ante VRP using GARCH(1,1) forecasts;
- identifying crisis periods;
- generating exploratory plots.



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

Realised variance is computed from returns:

\[
RV_t =
\sum_i r_i^2
\]

where:

\[
r_i
=
\log
\left(
\frac{S_i}{S_{i-1}}
\right)
\]

Monthly realised variance is annualised where appropriate.



## Volatility Risk Premium (VRP)

The monthly volatility risk premium is computed as:

\[
VRP_t
=
IV_t
-
RV_{t,t+1M}
\]

Positive VRP implies investors systematically overpay for volatility protection.



## Generated outputs

The following figures were produced:

1. VIX time series;
2. VRP time series;
3. Scatter: VIX vs next-month realised variance;
4. VIX futures term structure;
5. VIX correlation with SPX returns;
6. Crisis-period analysis.



## Crisis analysis

The following stress periods were analysed:

| Crisis | Main feature |
|---|---|
| 2020 COVID shock | Extreme VIX spike |
| 2022 inflation shock | Elevated volatility regime |

Expected observations:

- sharp increase in implied volatility;
- compression or inversion of VRP;
- stronger negative correlation between VIX and SPX.



## Main findings

The analysis confirms several stylised facts:

1. VIX spikes during periods of market stress.
2. Volatility risk premium is usually positive.
3. Realised variance tends to remain below implied variance.
4. Crisis periods produce temporary breakdowns in normal VRP behaviour.

These findings are consistent with the volatility risk premium literature and motivate variance swap strategies studied in later sections.



## Current status

The W5 module currently includes:

- automated market data collection;
- realised variance computation;
- ex-post and ex-ante VRP estimation;
- exploratory plots;
- crisis-period identification;
- notebook implementation for reproducible analysis.

The outputs will be used in:

- W6 calibration;
- W7 volatility surface analysis;
- W8 VRP backtesting and strategy evaluation.


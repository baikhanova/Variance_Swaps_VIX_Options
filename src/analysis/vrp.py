"""
W7 (Vol Analyst) — variance-risk-premium (VRP) time series.

The project is fully model-based (no market data), so the VRP series is
*simulated* rather than measured. The construction follows Carr & Wu (2009):

    VRP_t = IV_t^2 - RV_{t, t+Delta}

where

  * IV_t^2 is the risk-neutral expected variance over the next month, i.e. the
    model VIX squared, evaluated with the **risk-neutral (Q)** CIR parameters at
    the current variance state v_t;
  * RV_{t, t+Delta} is the variance actually realised over the following month
    along a single long path simulated under the **physical (P)** measure.

A positive variance risk premium is produced by letting the risk-neutral
long-run variance sit above the physical one (theta_Q > theta_P) — investors
pay up for variance protection. The premium turns *negative* in the rare months
where realised variance overshoots the implied level, i.e. around volatility
spikes that were not priced in a month earlier. That is the empirically
documented behaviour: VRP is usually a positive harvestable premium but
collapses (or goes negative) in crises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.heston import HestonParams, simulate_heston_paths
from src.pricing.variance_mc import compute_realized_variance, vix_squared_from_variance


@dataclass(frozen=True)
class VRPConfig:
    """Inputs for the simulated VRP experiment."""

    physical: HestonParams
    risk_neutral: HestonParams
    n_months: int = 300            # ~25 years of monthly observations
    days_per_month: int = 21       # trading days per month
    months_per_year: int = 12
    seed: int | None = 11
    realized_measure: str = "integrated"  # "integrated" (QV) or "returns"

    @property
    def delta(self) -> float:
        """VRP / VIX horizon in years (one month)."""
        return 1.0 / self.months_per_year

    @property
    def dt(self) -> float:
        return self.delta / self.days_per_month


def simulate_vrp_series(config: VRPConfig) -> pd.DataFrame:
    """Simulate a monthly VRP time series on an abstract model-time index.

    One long daily path of (S, v) is drawn under the physical parameters. At the
    start of every month the model VIX^2 (risk-neutral) is read off the current
    variance state, and the realised variance over the following month is
    measured from the simulated stock path.

    Returns a DataFrame indexed by ``month`` (0, 1, 2, ...) with columns:
        v_t          instantaneous variance at month start
        iv2          risk-neutral expected variance (model VIX^2), variance units
        rv           realised variance over the next month, annualised
        vrp          iv2 - rv (variance units)  -> short-variance P&L
        iv_vol       100 * sqrt(iv2)   (VIX index points)
        rv_vol       100 * sqrt(rv)    (realised vol, index points)
        vrp_vol      iv_vol - rv_vol   (volatility points)
    """
    cfg = config
    total_steps = cfg.n_months * cfg.days_per_month
    total_years = cfg.n_months * cfg.delta

    _, stock_paths, variance_paths = simulate_heston_paths(
        params=cfg.physical,
        maturity=total_years,
        steps=total_steps,
        paths=1,
        seed=cfg.seed,
    )
    stock = stock_paths[0]
    variance = variance_paths[0]

    rows = []
    for m in range(cfg.n_months):
        start = m * cfg.days_per_month
        end = start + cfg.days_per_month

        v_t = float(variance[start])
        # Risk-neutral expected variance over the next month (model VIX^2).
        iv2 = float(
            vix_squared_from_variance(v_t, cfg.risk_neutral, delta=cfg.delta)
        )
        # Physical expected variance over the next month (the RV forecast).
        rv_expected = float(
            vix_squared_from_variance(v_t, cfg.physical, delta=cfg.delta)
        )

        if cfg.realized_measure == "integrated":
            # Continuous-limit realised variance = quadratic variation of the
            # log-price = (1/Delta) * integral of v_s ds over the month. Using
            # the latent variance path removes the sampling noise of the
            # squared-return estimator, isolating the genuine premium.
            rv = float(np.mean(variance[start : end + 1]))
        elif cfg.realized_measure == "returns":
            window = stock[start : end + 1][np.newaxis, :]
            rv = float(compute_realized_variance(window, maturity=cfg.delta)[0])
        else:
            raise ValueError("realized_measure must be 'integrated' or 'returns'.")

        rows.append(
            {
                "month": m,
                "v_t": v_t,
                "iv2": iv2,
                "rv_expected": rv_expected,
                "rv": rv,
                # ex-ante VRP: the premium signal (IV^2 minus the RV forecast).
                "vrp_exante": iv2 - rv_expected,
                # ex-post VRP: realised short-variance P&L (IV^2 minus actual RV).
                "vrp": iv2 - rv,
                "iv_vol": 100.0 * np.sqrt(iv2),
                "rv_vol": 100.0 * np.sqrt(rv),
                "vrp_vol": 100.0 * (np.sqrt(iv2) - np.sqrt(rv)),
            }
        )

    return pd.DataFrame(rows).set_index("month")


def rolling_sharpe(
    returns: pd.Series,
    window: int = 12,
    periods_per_year: int = 12,
) -> pd.Series:
    """Annualised rolling Sharpe ratio of a periodic return series.

        Sharpe_t = mean(returns over window) / std(returns over window)
                   * sqrt(periods_per_year)

    With ``window = periods_per_year = 12`` this is the 12-month rolling Sharpe
    of the short-variance (long-VRP) strategy whose monthly return is the VRP.
    """
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    return (mean / std) * np.sqrt(periods_per_year)


def identify_negative_vrp(
    df: pd.DataFrame,
    column: str = "vrp",
) -> pd.DataFrame:
    """Group months with negative VRP into contiguous episodes.

    Returns one row per episode with the start/end model-month, its length, and
    the most negative VRP reached inside the episode.
    """
    negative = df[column] < 0
    episodes = []
    start = None
    for month, is_neg in zip(df.index, negative):
        if is_neg and start is None:
            start = month
        elif not is_neg and start is not None:
            episodes.append((start, prev))
            start = None
        prev = month
    if start is not None:
        episodes.append((start, prev))

    records = []
    for first, last in episodes:
        block = df.loc[first:last, column]
        records.append(
            {
                "start_month": first,
                "end_month": last,
                "length_months": int(last - first + 1),
                "min_vrp": float(block.min()),
            }
        )
    return pd.DataFrame(records)

import numpy as np
import pandas as pd

from src.models.cir import CIRParams
from src.models.heston import HestonParams
from src.pricing.vix_cos import COSSettings
from src.pricing.vix_options_pricer import build_vix_option_surface
from src.analysis.implied_vol import (
    black76_implied_vol,
    black76_price,
    spx_implied_vol_smile,
    vix_implied_vol_surface,
)
from src.analysis.vrp import (
    VRPConfig,
    identify_negative_vrp,
    rolling_sharpe,
    simulate_vrp_series,
)


# ----- Black-76 implied-vol inversion -----

def test_black76_implied_vol_round_trip():
    for sigma in (0.15, 0.5, 1.2):
        price = black76_price(20.0, 22.0, 0.25, sigma, 0.03, is_call=True)
        recovered = black76_implied_vol(price, 20.0, 22.0, 0.25, 0.03, is_call=True)
        assert np.isclose(recovered, sigma, atol=1e-4)


def test_black76_call_put_consistency():
    # Same vol must reprice both sides; parity then holds by construction.
    f, k, t, r, sig = 18.0, 18.0, 0.5, 0.03, 0.8
    call = black76_price(f, k, t, sig, r, is_call=True)
    put = black76_price(f, k, t, sig, r, is_call=False)
    disc = np.exp(-r * t)
    assert np.isclose(call - put, disc * (f - k), atol=1e-10)


def test_black76_implied_vol_nan_below_intrinsic():
    disc_intrinsic = np.exp(-0.03 * 0.25) * (25.0 - 18.0)
    iv = black76_implied_vol(disc_intrinsic - 0.5, 25.0, 18.0, 0.25, 0.03, is_call=True)
    assert np.isnan(iv)


def test_vix_implied_vol_surface_is_positive_and_finite_atm():
    params = CIRParams(v0=0.04, kappa=1.0, theta=0.04, sigma_v=1.5)
    mats = np.array([1 / 12, 3 / 12])
    strikes = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    surf = build_vix_option_surface(
        params, mats, strikes, settings=COSSettings(n_terms=160, truncation_std_width=12)
    )
    iv = vix_implied_vol_surface(surf, mats, strikes)
    # At least the near-the-money columns invert to a finite positive vol-of-vol.
    assert np.nanmin(iv) > 0
    assert np.isfinite(iv).sum() >= iv.size // 2


def test_spx_smile_is_negatively_skewed():
    params = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma_v=1.5, rho=-0.7)
    strikes = np.linspace(80, 120, 9)
    smile = spx_implied_vol_smile(params, 0.25, strikes, paths=60_000, seed=1)
    slope = np.polyfit(smile["moneyness"], smile["implied_vol"], 1)[0]
    assert slope < 0  # leverage effect: low strikes richer


# ----- VRP simulation -----

def _vrp_config():
    physical = HestonParams(v0=0.035, kappa=4.0, theta=0.035, sigma_v=0.45, rho=-0.7)
    risk_neutral = HestonParams(v0=0.035, kappa=6.0, theta=0.050, sigma_v=0.45, rho=-0.7)
    return VRPConfig(physical=physical, risk_neutral=risk_neutral, n_months=300, seed=11)


def test_simulate_vrp_series_shape_and_columns():
    df = simulate_vrp_series(_vrp_config())
    assert len(df) == 300
    for col in ("iv2", "rv_expected", "rv", "vrp_exante", "vrp", "vrp_vol"):
        assert col in df.columns


def test_vrp_positive_on_average_and_rarely_negative():
    df = simulate_vrp_series(_vrp_config())
    assert df["vrp_exante"].mean() > 0          # there is a premium
    assert (df["vrp_exante"] < 0).mean() < 0.25  # negative is rare


def test_rolling_sharpe_matches_manual_window():
    returns = pd.Series(np.arange(1.0, 25.0))
    rs = rolling_sharpe(returns, window=12, periods_per_year=12)
    window = returns.iloc[:12]
    expected = window.mean() / window.std(ddof=1) * np.sqrt(12)
    assert np.isclose(rs.iloc[11], expected)


def test_identify_negative_vrp_groups_contiguous_runs():
    df = pd.DataFrame({"vrp": [1, -1, -1, 1, 1, -2, 1]}, index=range(7))
    episodes = identify_negative_vrp(df, column="vrp")
    assert list(episodes["start_month"]) == [1, 5]
    assert list(episodes["end_month"]) == [2, 5]
    assert list(episodes["length_months"]) == [2, 1]
    assert np.isclose(episodes.loc[0, "min_vrp"], -1.0)

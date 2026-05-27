"""
SPX option-chain snapshot for W6 calibration.

The W6 calibration target is the SPX implied-volatility surface at a
single trading date. yfinance exposes the live option chain through
`yf.Ticker("^SPX").option_chain(expiration)`. This module wraps that
call and produces a tidy DataFrame used by `src/calibration`.

Because yfinance returns the *current* chain (not a historical one),
the snapshot date is the day on which the script first runs. The
cleaned CSV is cached under data/spx_option_chain_<YYYYMMDD>.csv so
subsequent runs reproduce identical inputs.

If the live download fails, the module falls back to a synthetic SVI
surface so the rest of the pipeline can still run offline. The same
synthetic-fallback pattern is already used in src/data/data_collection.py.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.pricing.iv_inversion import bs_call_price, bs_put_price

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def load_spx_option_chain(
    ticker: str = "^SPX",
    max_maturities: int = 6,
    min_days: int = 14,
    max_days: int = 365,
    snapshot_tag: str | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Load the SPX option-chain snapshot, downloading once and caching.

    Returns
    -------
    dict
        {
            "snapshot_date": str,
            "spot": float,
            "rate": float,
            "options": pd.DataFrame with columns
                ['expiration', 'maturity_years', 'strike',
                 'option_type', 'mid', 'iv_market'],
        }

    The "rate" returned is a constant proxy used by the BS IV inversion.
    """
    tag = snapshot_tag or datetime.utcnow().strftime("%Y%m%d")
    cache_path = DATA_DIR / f"spx_option_chain_{tag}.csv"
    meta_path = DATA_DIR / f"spx_option_chain_{tag}_meta.csv"

    if use_cache and cache_path.exists() and meta_path.exists():
        options = pd.read_csv(cache_path, parse_dates=["expiration"])
        meta = pd.read_csv(meta_path).iloc[0].to_dict()
        return {
            "snapshot_date": str(meta["snapshot_date"]),
            "spot": float(meta["spot"]),
            "rate": float(meta["rate"]),
            "options": options,
        }

    try:
        import yfinance as yf

        print(f"[W6] Downloading {ticker} option chain via yfinance …")
        ticker_obj = yf.Ticker(ticker)

        spot_history = ticker_obj.history(period="5d")
        spot = float(spot_history["Close"].iloc[-1])
        expirations = list(ticker_obj.options)[:max_maturities * 2]

        snapshot_date = datetime.utcnow().strftime("%Y-%m-%d")
        rate = _proxy_risk_free_rate()

        frames = []
        kept_maturities = 0
        for expiration_str in expirations:
            expiration = pd.Timestamp(expiration_str)
            days = (expiration - pd.Timestamp(snapshot_date)).days
            if not (min_days <= days <= max_days):
                continue

            chain = ticker_obj.option_chain(expiration_str)
            calls = chain.calls.assign(option_type="call")
            puts = chain.puts.assign(option_type="put")
            both = pd.concat([calls, puts], ignore_index=True)

            both["expiration"] = expiration
            both["maturity_years"] = days / 365.0

            frames.append(both)
            kept_maturities += 1
            if kept_maturities >= max_maturities:
                break

        if not frames:
            raise RuntimeError("No usable expirations returned.")

        raw = pd.concat(frames, ignore_index=True)
        options = _clean_option_chain(raw=raw, spot=spot, rate=rate)
    except Exception as exc:
        print(f"[W6] Live option-chain download failed ({exc}). "
              "Falling back to synthetic SVI surface.")
        spot, rate, options = _generate_synthetic_chain()
        snapshot_date = datetime.utcnow().strftime("%Y-%m-%d")

    options.to_csv(cache_path, index=False)
    pd.DataFrame([{
        "snapshot_date": snapshot_date,
        "spot": spot,
        "rate": rate,
    }]).to_csv(meta_path, index=False)

    print(f"[W6] Snapshot cached at {cache_path} "
          f"({len(options)} quotes, spot={spot:.2f}, rate={rate:.4f}).")

    return {
        "snapshot_date": snapshot_date,
        "spot": spot,
        "rate": rate,
        "options": options,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning and filtering
# ─────────────────────────────────────────────────────────────────────────────


def _clean_option_chain(
    raw: pd.DataFrame,
    spot: float,
    rate: float,
) -> pd.DataFrame:
    """
    Filter the raw yfinance option chain to a clean calibration surface.

    Steps applied:
      * keep strictly positive bid and ask with bid < ask;
      * compute mid as the midpoint of bid/ask;
      * drop ultra-deep OTM quotes (|log(K/forward)| > 0.4) — their IV
        inversion is noisy and they carry little information about
        the smile shape;
      * keep OTM side only (calls for K >= forward, puts for K < forward);
      * invert mid to a market IV via Brent.
    """
    from src.pricing.iv_inversion import implied_volatility_vector

    df = raw.copy()

    needed = {"bid", "ask", "strike", "expiration", "maturity_years", "option_type"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Option-chain columns missing: {missing}")

    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"])].copy()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])

    df["forward"] = spot * np.exp(rate * df["maturity_years"])
    df["log_moneyness"] = np.log(df["strike"] / df["forward"])
    df = df[df["log_moneyness"].abs() <= 0.4].copy()

    keep_call = (df["option_type"] == "call") & (df["strike"] >= df["forward"])
    keep_put = (df["option_type"] == "put") & (df["strike"] < df["forward"])
    df = df[keep_call | keep_put].copy()

    ivs = implied_volatility_vector(
        prices=df["mid"].to_numpy(),
        s0=spot,
        strikes=df["strike"].to_numpy(),
        maturities=df["maturity_years"].to_numpy(),
        rate=rate,
        option_types=df["option_type"].to_numpy(),
    )
    df["iv_market"] = ivs
    df = df[df["iv_market"].between(0.03, 1.5)].copy()

    columns = ["expiration", "maturity_years", "strike",
               "option_type", "mid", "iv_market"]
    return df[columns].sort_values(["expiration", "strike"]).reset_index(drop=True)


def _proxy_risk_free_rate() -> float:
    """
    Use a flat short-rate proxy. FRED would give a better term structure,
    but the dependence of SPX option prices on the rate over a one-year
    horizon is mild relative to the volatility uncertainty, so a constant
    is acceptable for coursework.
    """
    return 0.045


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallback surface
# ─────────────────────────────────────────────────────────────────────────────


def _generate_synthetic_chain() -> tuple[float, float, pd.DataFrame]:
    """
    Build an SVI-style synthetic surface so the calibration pipeline
    can run even when no internet access is available.

    The surface is intentionally not a Heston surface — otherwise the
    "calibration" would have a perfect optimum and the joint-calibration
    mismatch we want to demonstrate in W6 would disappear.
    """
    spot = 5000.0
    rate = 0.045

    maturities_days = np.array([30, 60, 90, 180, 270, 365])
    strikes_log_money = np.linspace(-0.3, 0.25, 21)

    rows = []
    base_date = pd.Timestamp(datetime.utcnow().date())
    for d in maturities_days:
        t = d / 365.0
        forward = spot * np.exp(rate * t)
        # Simple SVI-ish smile: ATM vol + skew + curvature
        atm = 0.17 + 0.02 * np.exp(-t)
        skew = -0.35 * np.sqrt(t + 0.05)
        curv = 0.6 * (1.0 - np.exp(-t))

        ivs = atm + skew * strikes_log_money + curv * strikes_log_money**2
        strikes = forward * np.exp(strikes_log_money)

        for k, iv in zip(strikes, ivs):
            otype = "call" if k >= forward else "put"
            if otype == "call":
                price = bs_call_price(spot, k, t, rate, iv)
            else:
                price = bs_put_price(spot, k, t, rate, iv)

            rows.append(
                {
                    "expiration": base_date + pd.Timedelta(days=int(d)),
                    "maturity_years": t,
                    "strike": float(k),
                    "option_type": otype,
                    "mid": float(price),
                    "iv_market": float(iv),
                }
            )

    return spot, rate, pd.DataFrame(rows)

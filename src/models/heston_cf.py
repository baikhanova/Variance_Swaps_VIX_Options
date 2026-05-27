"""
Heston characteristic function for the log-stock process.

Used by the COS method in `src/pricing/heston_spx.py` to price European
options on SPX without simulation. The implementation follows the
"Little Trap" formulation of Albrecher, Mayer, Schoutens and Tistaert
(2007), which avoids branch-cut crossings of the complex logarithm for
long maturities.

The Heston SDE is:

    dS_t / S_t = (r - q) dt + sqrt(v_t) dW_t^S
    dv_t       = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW_t^v
    d<W^S, W^v>_t = rho dt

With x_t = log(S_t), the characteristic function of x_T is:

    phi(u; T) = E[exp(i u x_T) | x_0, v_0]
              = exp(i u (x_0 + (r - q) T) + C(u,T) + D(u,T) v_0)

where C, D solve the associated Riccati ODE system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonCFParams:
    """
    Risk-neutral Heston parameters used in the characteristic function.

    These are the same five parameters that enter the SDE plus the
    risk-free rate and dividend yield. The instantaneous variance v0
    is also part of the state because the CF depends on it linearly.
    """

    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    sigma_v: float = 0.5
    rho: float = -0.7
    rate: float = 0.03
    dividend_yield: float = 0.0

    def validate(self) -> None:
        if self.v0 < 0:
            raise ValueError("v0 must be non-negative.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.theta <= 0:
            raise ValueError("theta must be positive.")
        if self.sigma_v <= 0:
            raise ValueError("sigma_v must be positive.")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be between -1 and 1.")


def heston_log_stock_cf(
    u: np.ndarray | float | complex,
    maturity: float,
    spot: float,
    params: HestonCFParams,
) -> np.ndarray | complex:
    """
    Characteristic function of x_T = log(S_T) under Heston.

    The implementation uses the Little Trap form:

        xi    = kappa - sigma_v * rho * i u
        d     = sqrt(xi^2 + sigma_v^2 * (u^2 + i u))
        g     = (xi - d) / (xi + d)
        D(T)  = (xi - d) / sigma_v^2 * (1 - exp(-d T)) / (1 - g exp(-d T))
        C(T)  = kappa * theta / sigma_v^2
                * [(xi - d) T - 2 log((1 - g exp(-d T)) / (1 - g))]

    The drift contribution i u (log(S0) + (r - q) T) is added on top.

    Returns
    -------
    np.ndarray or complex
        phi(u) with the same broadcast shape as the input u.
    """
    params.validate()

    if maturity < 0:
        raise ValueError("maturity must be non-negative.")
    if spot <= 0:
        raise ValueError("spot must be positive.")

    u_array = np.asarray(u, dtype=np.complex128)
    scalar = u_array.ndim == 0
    u_array = np.atleast_1d(u_array)

    if maturity == 0:
        result = np.exp(1j * u_array * np.log(spot))
        return complex(result[0]) if scalar else result

    kappa = params.kappa
    theta = params.theta
    sigma = params.sigma_v
    rho = params.rho
    r = params.rate
    q = params.dividend_yield
    v0 = params.v0

    xi = kappa - sigma * rho * 1j * u_array
    d = np.sqrt(xi**2 + sigma**2 * (u_array**2 + 1j * u_array))

    g = (xi - d) / (xi + d)
    exp_neg_dt = np.exp(-d * maturity)

    one_minus_g_exp = 1.0 - g * exp_neg_dt

    capital_d = (xi - d) / sigma**2 * (1.0 - exp_neg_dt) / one_minus_g_exp
    capital_c = (kappa * theta / sigma**2) * (
        (xi - d) * maturity - 2.0 * np.log(one_minus_g_exp / (1.0 - g))
    )

    drift = 1j * u_array * (np.log(spot) + (r - q) * maturity)
    result = np.exp(drift + capital_c + capital_d * v0)

    return complex(result[0]) if scalar else result


def heston_cumulants(
    maturity: float,
    params: HestonCFParams,
) -> tuple[float, float, float]:
    """
    First three cumulants of the Heston log-stock distribution.

    These are used by the Fang-Oosterlee COS method to set the
    truncation interval [a, b]. Formulae follow Fang & Oosterlee (2008)
    Section 3.4. The third cumulant is approximated by zero (a common
    choice when only c1, c2 are needed for the truncation rule).

    Returns
    -------
    (c1, c2, c4) : tuple of floats
        Mean, variance, and a rough fourth cumulant proxy.
    """
    params.validate()

    if maturity < 0:
        raise ValueError("maturity must be non-negative.")

    kappa = params.kappa
    theta = params.theta
    sigma = params.sigma_v
    rho = params.rho
    r = params.rate
    q = params.dividend_yield
    v0 = params.v0
    t = maturity

    exp_neg = np.exp(-kappa * t)

    c1 = (r - q) * t + (1.0 - exp_neg) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * t

    c2 = (
        1.0 / (8.0 * kappa**3) * (
            sigma * t * kappa * exp_neg
            * (v0 - theta)
            * (8.0 * kappa * rho - 4.0 * sigma)
            + kappa * rho * sigma * (1.0 - exp_neg) * (16.0 * theta - 8.0 * v0)
            + 2.0 * theta * kappa * t
            * (-4.0 * kappa * rho * sigma + sigma**2 + 4.0 * kappa**2)
            + sigma**2 * ((theta - 2.0 * v0) * exp_neg**2
                          + theta * (6.0 * exp_neg - 7.0) + 2.0 * v0)
            + 8.0 * kappa**2 * (v0 - theta) * (1.0 - exp_neg)
        )
    )

    c4 = 0.0

    return float(c1), float(max(c2, 1e-12)), float(c4)


def cos_truncation_interval(
    maturity: float,
    spot: float,
    params: HestonCFParams,
    level: float = 10.0,
) -> tuple[float, float]:
    """
    Truncation interval [a, b] for the COS expansion of x_T = log(S_T).

    The Fang-Oosterlee rule of thumb is:

        a = c1 - L * sqrt(c2 + sqrt(c4))
        b = c1 + L * sqrt(c2 + sqrt(c4))

    Both are expressed relative to log(S_0). Default L = 10 is the
    standard choice for stochastic volatility models.
    """
    c1, c2, c4 = heston_cumulants(maturity=maturity, params=params)
    width = level * np.sqrt(c2 + np.sqrt(max(c4, 0.0)))

    log_spot = np.log(spot)
    return float(log_spot + c1 - width), float(log_spot + c1 + width)

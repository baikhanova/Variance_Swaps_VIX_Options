# W7 Results: Volatility Surface Analysis

## 1.1 VIX Implied Volatility Surface

### 1.1.1 Setup and Model

VIX options are options written on the VIX index itself. Their implied volatility, commonly referred to as *vol-of-vol*, reflects market expectations about the magnitude of future VIX movements. This creates a second-order surface: analogous to how SPX options produce an implied volatility smile over strikes and maturities, VIX options generate their own distinct smile structure.

Under the CIR variance process adopted in this project, the spot variance $v_t$ evolves as:

$$dv_t = \kappa(\theta - v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t^v$$

The VIX index is then approximated by:

$$\text{VIX}_T = 100\sqrt{A + B\,v_T}, \qquad B = \frac{1 - e^{-\kappa\Delta}}{\kappa\Delta}, \quad A = \theta(1-B)$$

Since $v_T$ follows a scaled non-central $\chi^2$ distribution, which is strictly positive and right-skewed, the distribution of $\text{VIX}_T$ inherits that skewness directly. This is the primary mathematical source of the positive skew observed in VIX option prices.

The implied vol smile is parametrised using a cubic skew model:

$$\text{IV}(k) = \sigma_{\text{ATM}} \cdot \left(1 + \text{skew} \cdot k + \text{curl} \cdot k^2 + \text{asymm} \cdot k^3\right)$$

where $k = \ln(K/F)$ is the log-moneyness. The parameters are set to $\text{skew} = +0.30$, $\text{curl} = 0.20$, and $\text{asymm} = +0.06$, consistent with the qualitative shapes documented in Lian and Zhu (2013) and Mencía and Sentana (2013). Mean reversion under CIR attenuates the effective vol-of-vol over longer horizons through the factor:

$$\text{att}(T) = \sqrt{\frac{1 - e^{-\kappa T}}{\kappa T}}$$

so that $\sigma_{\text{ATM}}(T) = \sigma_v \cdot \text{att}(T)$, which decreases monotonically in $T$.

### 1.1.2 Results

The computed VIX forward price is **20.0** across all maturities, as the model is initialised at the long-run mean ($v_0 = \theta$, VIX$_0 = 20$). ATM vol-of-vol figures are reported in Table 1.

**Table 1: VIX and SPX forward prices and ATM implied volatilities by maturity**

| Maturity | VIX Forward | VIX ATM Vol (%) | SPX Forward | SPX ATM Vol (%) |
|----------|-------------|-----------------|-------------|-----------------|
| 1M       | 20.0        | 72.35           | 5,019       | 20.0            |
| 2M       | 20.0        | 65.90           | 5,038       | 20.0            |
| 3M       | 20.0        | 60.44           | 5,057       | 20.0            |
| 6M       | 20.0        | 48.48           | 5,114       | 20.0            |

The VIX ATM vol-of-vol at the one-month horizon is **72.35%**, declining to **48.48%** at six months. This downward-sloping term structure is a direct consequence of CIR mean reversion: over longer horizons, the variance process is pulled back toward $\theta$, reducing the dispersion of $\text{VIX}_T$ around the forward and compressing the vol-of-vol accordingly. SPX ATM implied volatility is stable at 20.0% across all maturities by model initialisation.

The sanity check confirms the directional correctness of the smile construction: the one-month VIX smile produces a 25-delta risk reversal of **+11.49 vol points** (calls minus puts), while the corresponding SPX figure is **-9.55 vol points**. These are directionally opposite, which is the core empirical fact motivating the comparison in Section 5.2.

---

## 1.2 VIX Smile vs. SPX Smile

### 1.2.1 Structural Differences

The two smiles differ not only in skew direction but in the underlying distributional properties and market forces that sustain them. Table 2 summarises the main structural contrasts.

**Table 2: Structural comparison of VIX and SPX implied volatility smiles**

| Feature | VIX Options | SPX Options |
|---|---|---|
| Skew direction | Positive (right) | Negative (left) |
| Expensive wing | OTM calls | OTM puts |
| Underlying distribution | Fat right tail | Fat left tail |
| Primary driver | VIX spikes during equity crises | Equity market crashes are fast and severe |
| Model source | Non-central $\chi^2$ distribution of $v_T$ | Negative $\rho$ in Heston ($\rho \approx -0.75$) |
| Investor behaviour | Buying VIX calls as portfolio-level tail hedge | Buying SPX puts as downside protection |
| Skew term structure | Decreasing (mean reversion dampens right tail over time) | Roughly flat or slightly increasing |

### 1.2.2 Economic Interpretation

**VIX smile: positive (right) skew.** The VIX index cannot fall to zero and tends to spike sharply during equity market dislocations. This creates a right-skewed distribution for VIX: large upward moves in VIX are more probable than symmetrically large downward moves. Mathematically, under the CIR model, $v_T$ follows a scaled non-central $\chi^2$ distribution, which is positively skewed by construction. On top of this distributional property, institutional investors routinely purchase OTM VIX calls as inexpensive portfolio-level tail-risk hedges, a strategy that adds systematic demand pressure to the right wing of the smile and pushes those implied volatilities above what the distributional argument alone would warrant.

The quantitative signature is the risk reversal: at the one-month horizon, VIX 25-delta calls trade at a **+11.49 vol point** premium over 25-delta puts. This number declines at longer maturities as mean reversion under CIR attenuates the right tail of $v_T$.

**SPX smile: negative (left) skew.** The SPX smile exhibits the opposite pattern, known in practice as the *volatility smirk*. The Heston model provides a clean first-order explanation: the negative equity-volatility correlation $\rho \approx -0.75$ means that when the SPX falls, variance rises simultaneously. This co-movement fattens the left tail of the SPX log-return distribution relative to a log-normal benchmark. In addition, the institutional demand for OTM put options as portfolio insurance is persistent and large, creating a structural excess of demand on the left wing that sustains elevated implied volatilities there. Both the leverage effect (Black, 1976) and the Heston correlation term produce the same directional outcome: IV decreases monotonically from left to right across the smile.

The one-month SPX 25-delta risk reversal of **-9.55 vol points** quantifies this effect: OTM puts are nearly 10 vol points more expensive than OTM calls at the same delta.

**Joint interpretation.** The two effects are not independent. A severe SPX decline is precisely the event that causes VIX to spike. From that perspective, the elevated left wing of the SPX smile and the elevated right wing of the VIX smile are two views of the same underlying crisis scenario, priced from different vantage points. The SPX left wing reflects the cost of insuring against the crash; the VIX right wing reflects the cost of directly expressing or hedging the associated volatility spike.

---

## References

- Black, F. (1976). Studies of stock price volatility changes. *Proceedings of the 1976 Meetings of the American Statistical Association*, 171-181.
- Carr, P. & Wu, L. (2006). A tale of two indices. *Journal of Derivatives*, 13(3), 13-29.
- Duan, J.-C. & Yeh, C.-Y. (2010). Jump and volatility risk premiums implied by VIX. *Journal of Economic Dynamics and Control*, 34(11), 2232-2244.
- Lian, G.-H. & Zhu, S.-P. (2013). Pricing VIX options with stochastic volatility and random jumps. *Decisions in Economics and Finance*, 36(1), 71-88.
- Mencía, J. & Sentana, E. (2013). Valuation of VIX derivatives. *Journal of Financial Economics*, 108(2), 367-391.

# W2: Mathematical Derivations
## Volatility as an Asset Class — Group 4

---

## 1. Model-Free Replication of VIX via Log-Contract

### Intuition
The key insight of Demeterfi et al. (1999) is that the variance swap payoff can be replicated using a static portfolio of options — without assuming any model.

### Setup
Consider a log-contract with payoff at maturity T:

$$-2\ln\frac{S_T}{F_0}$$

where $F_0 = S_0 e^{rT}$ is the forward price.

By Ito's lemma applied to $\ln S_t$:

$$d\ln S_t = \frac{dS_t}{S_t} - \frac{1}{2}\sigma^2_t dt$$

Integrating from 0 to T:

$$\ln\frac{S_T}{S_0} = \int_0^T \frac{dS_t}{S_t} - \frac{1}{2}\int_0^T \sigma^2_t dt$$

Rearranging:

$$\int_0^T \sigma^2_t dt = 2\int_0^T \frac{dS_t}{S_t} - 2\ln\frac{S_T}{S_0}$$

### Replication via Options
Any twice-differentiable payoff $f(S_T)$ can be decomposed as:

$$f(S_T) = f(F_0) + f'(F_0)(S_T - F_0) + \int_0^{F_0} f''(K)P(K)dK + \int_{F_0}^{\infty} f''(K)C(K)dK$$

Applying this to $f(S_T) = -2\ln\frac{S_T}{F_0}$, where $f''(S) = \frac{2}{S^2}$:

$$-2\ln\frac{S_T}{F_0} = \frac{2}{F_0}(F_0 - S_T) + \int_0^{F_0}\frac{2}{K^2}P(K)dK + \int_{F_0}^{\infty}\frac{2}{K^2}C(K)dK$$

### Model-Free Implied Variance
Taking expectation under Q:

$$\boxed{\text{VIX}^2_t = \frac{2}{T}\left[\int_0^{F_t}\frac{P(K)}{K^2}dK + \int_{F_t}^{\infty}\frac{C(K)}{K^2}dK\right]}$$

**Key insight:** This replication is model-free — it requires only a strip of options at all strikes.
## 2. Variance Swap Strike under Heston

Under the Heston model:

$$dv_t = \kappa(\bar{v} - v_t)dt + \sigma\sqrt{v_t}dW_t^v$$

Taking expectation under Q:

$$\mathbb{E}^{\mathbb{Q}}[v_t] = \bar{v} + (v_0 - \bar{v})e^{-\kappa t}$$

The fair variance strike is:

$$K_{var} = \frac{1}{T}\int_0^T \left[\bar{v} + (v_0 - \bar{v})e^{-\kappa t}\right] dt$$

$$\boxed{K_{var} = \bar{v} + (v_0 - \bar{v})\frac{1 - e^{-\kappa T}}{\kappa T}}$$

---

## 3. VIX² is Affine in $v_t$ under Heston

$$\text{VIX}^2_t = \mathbb{E}^{\mathbb{Q}}_t\left[\frac{1}{\Delta}\int_t^{t+\Delta} v_s ds\right]$$

$$\boxed{\text{VIX}^2_t = \bar{v} + (v_t - \bar{v})\frac{1-e^{-\kappa\Delta}}{\kappa\Delta} = A + B \cdot v_t}$$

where $A = \bar{v}\left(1 - \frac{1-e^{-\kappa\Delta}}{\kappa\Delta}\right)$ and $B = \frac{1-e^{-\kappa\Delta}}{\kappa\Delta}$.

**This is affine in $v_t$** — VIX² is a linear function of instantaneous variance.

---

## References

- Demeterfi, K., Derman, E., Kamal, M., & Zou, J. (1999). *More Than You Ever Wanted to Know About Volatility Swaps.* Goldman Sachs.
- Carr, P., & Wu, L. (2009). Variance Risk Premiums. *Review of Financial Studies*, 22(3).
- Gatheral, J. (2006). *The Volatility Surface.* Wiley Finance.

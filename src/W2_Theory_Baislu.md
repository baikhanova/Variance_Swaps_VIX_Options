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

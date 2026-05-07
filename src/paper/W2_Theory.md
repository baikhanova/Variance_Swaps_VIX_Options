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

---

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

## 4. CIR Characteristic Function for Integrated Variance

### Motivation

To price VIX futures and VIX options via Fourier-based methods such as the COS method, we require the characteristic function of the integrated variance process under Heston.

Since \(\text{VIX}^2_t\) is affine in \(v_t\), pricing VIX derivatives reduces to computing the distribution of the integrated variance:

$$ I_{t,T} = \int_t^T v_s\,ds $$

The characteristic function of this quantity is:

$$ \phi(u;\,t,T) = \mathbb{E}^{\mathbb{Q}}\left[e^{iu \int_t^T v_s\,ds}\,\Big|\,v_t\right] $$

Under the Heston model, the variance process follows a CIR diffusion:

$$ dv_t = \kappa(\bar{v} - v_t)\,dt + \sigma\sqrt{v_t}\,dW_t^v $$

### Exponential-Affine Ansatz

Because the CIR process belongs to the affine class of stochastic processes, the characteristic function admits the exponential-affine representation:

$$ \phi(u;\,t,T) = \exp\!\Big(A(\tau,u) + B(\tau,u)\,v_t\Big) $$

where:

$$ \tau = T - t $$

with boundary conditions:

$$ A(0,u) = 0, \qquad B(0,u) = 0 $$

### Feynman–Kac PDE and Riccati System

Using the Feynman–Kac theorem, the characteristic function satisfies the partial differential equation:

$$ \frac{\partial \phi}{\partial t} + \kappa(\bar{v} - v)\frac{\partial \phi}{\partial v} + \frac{1}{2}\sigma^2 v \frac{\partial^2 \phi}{\partial v^2} + iuv\,\phi = 0 $$

Substituting the exponential-affine ansatz and collecting terms yields the Riccati system:

$$ \frac{dB}{d\tau} = -\kappa B + \frac{1}{2}\sigma^2 B^2 + iu $$

with:

$$ B(0,u) = 0 $$

and:

$$ \frac{dA}{d\tau} = \kappa\bar{v}\,B $$

with:

$$ A(0,u) = 0 $$

### Closed-Form Solution

Define:

$$ \gamma = \sqrt{\kappa^2 - 2\sigma^2(iu)} $$

The solution for \(B(\tau,u)\) is:

$$ B(\tau,u) = \frac{2iu(1 - e^{-\gamma\tau})}{(\gamma + \kappa)(e^{\gamma\tau} - 1) + 2\gamma} $$

Integrating the Riccati equation for \(A(\tau,u)\) gives:

$$ A(\tau,u) = \frac{\kappa\bar{v}}{\sigma^2}\left[(\kappa - \gamma)\tau - 2\ln\left(1-\frac{\kappa-\gamma}{2\gamma}(1-e^{-\gamma\tau})\right)\right] $$

Therefore, the complete characteristic function is:

$$ \boxed{\phi(u;\,t,T) = \exp\!\Big(A(\tau,u) + B(\tau,u)\,v_t\Big)} $$

### Verification

At \(u = 0\):

$$ \gamma = \kappa $$

$$ A(\tau,0)=0 $$

$$ B(\tau,0)=0 $$

which implies:

$$ \phi(0)=1 $$

as required for a characteristic function.

As the volatility-of-volatility parameter satisfies *σ → 0*,  the variance process becomes deterministic and the characteristic function converges to the transform of the deterministic integrated variance path.

### Financial Interpretation and Connection to W4

The affine structure of the CIR process is one of the main reasons why the Heston model remains computationally tractable.

The closed-form characteristic function allows volatility derivatives to be priced efficiently using Fourier inversion techniques and the COS method without requiring expensive Monte Carlo simulation.

This result is directly used in W4 for pricing VIX futures and VIX options.

---

## 5. The Joint Calibration Problem

### One-Factor Structure of Heston

The Heston model is a one-factor stochastic volatility model because a single latent variance process \(v_t\) drives both:

- the SPX implied volatility surface,
- and the VIX futures term structure.

From the affine representation:

$$ \text{VIX}_t^2 = \bar{v} + (v_t-\bar{v})\frac{1-e^{-\kappa\Delta}}{\kappa\Delta} $$

the entire VIX structure is determined by the same variance factor that governs SPX option prices.

### Calibration Conflict

In practice, market data shows that the parameter values required to fit SPX options are inconsistent with those required to fit VIX futures.

This occurs because:

- SPX options require strong short-term skew dynamics,
- VIX futures require smoother long-term variance dynamics,
- VIX options require realistic volatility-of-volatility behaviour.

As a result, calibrating Heston to SPX options alone generally produces a poor fit to the observed VIX futures curve.

### Economic Interpretation

This issue is known as the joint calibration problem.

The limitation arises because a single stochastic variance factor cannot fully capture the complexity of volatility markets.

Recent research shows that one-factor stochastic volatility models struggle to reproduce the joint dynamics of SPX and VIX markets simultaneously.

This motivates richer models such as:

- two-factor stochastic volatility models,
- forward variance models,
- and rough volatility models.

### Relation to This Project

The joint calibration problem is studied numerically in W6, where Heston is calibrated jointly to SPX implied volatilities and VIX futures.

The expected result is that one-factor Heston achieves a good fit for one market only at the expense of the other.

This demonstrates the structural limitation of the model rather than a numerical calibration issue.

---

## References

- Bergomi, L. (2005). *Smile Dynamics 2*. Risk Magazine.
- Carr, P., & Wu, L. (2009). Variance Risk Premiums. *Review of Financial Studies*, 22(3).
- Demeterfi, K., Derman, E., Kamal, M., & Zou, J. (1999). *More Than You Ever Wanted to Know About Volatility Swaps*. Goldman Sachs Quantitative Strategies Research Notes.
- Duffie, D., Pan, J., & Singleton, K. (2000). Transform Analysis and Asset Pricing for Affine Jump-Diffusions. *Econometrica*, 68(6), 1343–1376.
- Fang, F., & Oosterlee, C. W. (2008). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions. *SIAM Journal on Scientific Computing*, 31(2), 826–848.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.
- Heston, S. (1993). A Closed-Form Solution for Options with Stochastic Volatility. *Review of Financial Studies*, 6(2), 327–343.
- Hirsa, A. (2024). *Computational Methods in Finance*. Chapman & Hall/CRC.
- Shreve, S. (2004). *Stochastic Calculus for Finance II: Continuous-Time Models*. Springer.

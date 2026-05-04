# W1: Literature Review — Classical Foundations of Volatility as an Asset Class

## 1. Introduction

There are four main papers that the current theory of volatility as a tradeable asset class is built upon: the replication framework of Demeterfi, Derman, Kamal and Zou (1999), the variance risk premium (VRP) empirical measurement of Carr and Wu (2009), Bergomi's (2005) forward variance modelling, and the volatility surface comprehensive treatment by Gatheral (2006). Together, these books form the conceptual and mathematical framework that all subsequent works of this project — the Monte Carlo engine, the COS pricing engine, the joint calibration exercise, and the VRP backtest — rely on. This review looks at each source separately, points out its main contribution, places it among the rest of the literature, and indicates its function in the current project.

---

## 2. Demeterfi, Derman, Kamal and Zou (1999): Variance Swaps via Log-Contract Replication

### 2.1 Background and Achievements

Initially released as a Goldman Sachs Quantitative Strategies Research Note, Demeterfi et al. (1999) is generally recognized as the seminal paper that spurred the birth of the variance swap market. Before this paper, volatility was only traded indirectly, via option positions where volatility exposure was combined with delta and gamma risk. Demeterfi et al. proved that one can obtain a pure exposure to realised variance through a static, model-free replication approach.

### 2.2 The Log-Contract Replication Method

Demeterfi et al.'s main idea is that the payoff of a variance swap — the difference between the realised variance and the strike — is a portfolio of European puts and calls over the entire strike range that can be rebalanced continuously. More precisely, one needs $2/K^2$ units of each option at strike $K$, which combined together form a log-contract: a contract that pays out

$$\Pi = -2 \ln \left( \frac{S_T}{F_{0,T}} \right)$$

at expiry, where $F$ is the forward price. Using Itô's lemma, one can recover the variance of the log-price process from the value of this portfolio.

Thus, the variance strike that makes the value of the swap equal to zero is:

$$K_{\text{var}} = \frac{2}{T} \left[ \int_{0}^{F} \frac{P(K)}{K^{2}}\, dK + \int_{F}^{\infty} \frac{C(K)}{K^{2}}\, dK \right]$$

This result is **model-free**: it does not require any assumption about the dynamics of the underlying process except for the continuity of sample paths. The only inputs are the prices of liquidly traded options on the market.

### 2.3 Connection to the CBOE VIX Formula

The CBOE VIX index, which was made public in its present form in 2003, is a direct implementation of the Demeterfi et al. methodology. Equation (3) in the project specification restates the VIX formula exactly:

$$\text{VIX}_{t}^{2} = \frac{2}{T} \left[ \int_{0}^{F_{t}} \frac{P(K)}{K^{2}}\, dK + \int_{F_{t}}^{\infty} \frac{C(K)}{K^{2}}\, dK \right]$$

Understanding the log-contract argument is therefore a necessary step before any other derivation in the project — the VIX formula, the fair-strike calculation under Heston (Equation 2), and the model-free hedging strategy in W8.

### 2.4 Role in the Present Project

The replication result of Demeterfi et al. serves as a foundation for three workstreams directly. First, W2 (Theory Lead) uses the log-contract argument to derive Equation (3) from first principles, as required by the Week 4 deliverables. Second, W3 (MC Engineer) compares the Monte Carlo estimate of $\mathbb{E}^{\mathbb{Q}}[\text{RV}_{0,T}]$ with the closed-form solution for $K_{\text{var}}$ — a comparison that is only meaningful if one understands what the formula represents. Third, and most importantly, W8 (Risk Lead) depends on the model-free nature of the replication to construct a Vega hedge for the variance swap that does not rely on Heston parameters. The model-independence of the hedge is not simply a technical convenience but a fundamental economic point that originates in this paper.

---

## 3. Carr and Wu (2009): The Variance Risk Premium

### 3.1 Context and Contribution

Carr and Wu (2009), published in the *Review of Financial Studies*, made the first attempt to demonstrate empirically and on a grand scale, through rigorous methodology, the presence of the variance risk premium (VRP) in equity markets. Though it was suggested that a premium for taking on variance risk exists, Carr and Wu were pioneers in the systematic quantification of this premium across multiple asset classes and over long time periods. Their approach importantly disentangles the risk-neutral expectation of variance (implied from options) from the physical expectation (obtained using realised returns).

### 3.2 Definition and Measurement of the VRP

Carr and Wu define the VRP as:

$$\text{VRP}_t = \text{IV}_t^2 - \mathbb{E}_t^{\mathbb{P}}\left[\text{RV}_{t,\, t+\Delta}\right]$$

where $\text{IV}_{t}^{2}$ is the risk-neutral expected variance (approximated by the squared VIX) and $\mathbb{E}_t^{\mathbb{P}}[\text{RV}_{t,t+\Delta}]$ is the physical expectation of realised variance over the following month. The latter is estimated both *ex post* — using the actual realised variance of the subsequent period — and *ex ante* — using a GARCH(1,1) forecast.

The main empirical finding is that the VRP is on average positive, typically around two to four annualised volatility points, and highly stable over time and across different equity indices. This premium represents the compensation investors demand for taking on the risk of unexpected variance increases — a risk that is not diversifiable and that spikes sharply during market dislocations.

### 3.3 Economic Interpretation

The observation that the VRP is positive implies that investors are generally *overpaying* for variance exposure in the options markets. Equivalently, the variance swap seller — who receives a fixed strike $K_{\text{var}}$ and pays floating realised variance — earns a steady positive return. This explains the economic rationale for the Short Variance Swap strategy in W8 (Alpha section). Carr and Wu also show that the VRP is highly correlated with market conditions: it widens during periods of fear and uncertainty, and compresses or even turns negative during calm regimes. This regime-dependence is directly relevant to the VRP regime analysis required in Section 6.4 of the project specification.

### 3.4 Factor Decomposition

A major methodological contribution of Carr and Wu is their regression of VRP returns on standard risk factors, including equity returns, changes in the VIX level, and lagged VIX. This corresponds precisely to the beta decomposition required in Section 6.4 of the project:

$$R_{t}^{\text{VRP}} = \alpha + \beta_{1} R_{t}^{\text{SPX}} + \beta_{2} \Delta\text{VIX}_{t} + \beta_{3} \text{VIX}_{t-1} + \varepsilon_{t}$$

Carr and Wu's finding that a significant alpha remains after controlling for equity and volatility betas provides the theoretical basis for treating the VRP as a genuine risk premium rather than an equity beta in disguise — a question that W8 must address explicitly.

### 3.5 Role in the Present Project

The methodology of Carr and Wu (2009) is the direct template for W5 (Data Scientist), who reconstructs the monthly VRP time series using both ex-post and GARCH-based ex-ante estimates. The strategy logic of W8 — sizing, backtesting, crisis analysis, and factor decomposition — follows the empirical framework of this paper. The Sharpe ratio decomposition by VRP regime (high VRP vs. low VRP) is grounded in Carr and Wu's insight that the premium varies greatly across market states.

---

## 4. Bergomi (2005): Forward Variance Models and the Failure of One-Factor Specifications

### 4.1 Context and Contribution

Lorenzo Bergomi's "Smile Dynamics 2" (*Risk Magazine*, 2005) is a practitioner-oriented paper that identifies a serious structural flaw of the Heston model and proposes a broader conceptual framework based on **forward variance**. It is the intellectual origin of the joint calibration problem that W6 must demonstrate numerically.

### 4.2 The Joint Calibration Problem

The Heston model is a **one-factor stochastic volatility model**: a single latent process $v_t$ governs both the instantaneous variance of SPX returns and the level of the VIX. Under Heston, the squared VIX is an affine function of $v_t$ (Equation 4 of the project), which means that the VIX futures term structure is entirely determined by $\kappa$, $\bar{v}$, and $v_0$ — the same parameters that describe the SPX implied volatility surface.

Bergomi (2005) shows that this exclusive link between the variance process and the observable volatility surface leads to an inherent contradiction: the parameter values required to fit the SPX surface (particularly the skew term structure and the level of implied volatility across maturities) are generally inconsistent with those required to fit the observed VIX futures curve. In other words, a Heston model calibrated to SPX options will systematically misprice VIX futures, and vice versa.

### 4.3 The Forward Variance Framework

To resolve this tension, Bergomi proposes modelling the **forward variance curve**

$$\xi_u^t = \mathbb{E}_t^{\mathbb{Q}}[v_u]$$

directly as an infinite-dimensional object evolving under lognormal dynamics:

$$d\xi_u^t = \omega \cdot \xi_u^t \cdot dW_t$$

This framework decouples the instantaneous variance dynamics from the shape of the volatility surface, allowing the model to be calibrated to both SPX options and VIX futures simultaneously, subject to appropriate parameterisation. The forward variance model is the conceptual precursor to rough volatility models, which Participant 2 will discuss.

### 4.4 Role in the Present Project

Bergomi (2005) provides the theoretical motivation for the joint calibration exercise in W6. Steps 1–4 of Section 6.1 in the project specification are a direct operationalisation of the argument: calibrate Heston to SPX options (Step 1), compute the implied VIX futures curve (Step 2), observe the discrepancy (Step 3), and conclude that a two-factor or rough volatility model is necessary (Step 4). The bonus extension — implementing a two-factor CIR model $v_t = v^{(1)}_t + v^{(2)}_t$ — is explicitly motivated by the multi-factor framework that Bergomi advocates.

---

## 5. Gatheral (2006): The Volatility Surface — VIX Chapters

### 5.1 Context and Contribution

Jim Gatheral's *The Volatility Surface* (Wiley Finance, 2006) is the standard graduate-level reference for derivatives practitioners. The chapters relevant to this project address three topics: the affine structure of the VIX under Heston, the CIR characteristic function required for analytical pricing of VIX derivatives, and the shape and economic interpretation of the VIX implied volatility smile.

### 5.2 Affinity of VIX² under Heston

A central analytical result derived in Gatheral is that, under the Heston model, the squared VIX is an **affine function** of the instantaneous variance $v_t$:

$$\text{VIX}_{t}^{2} = \bar{v} + (v_{t} - \bar{v})\, \frac{1 - e^{-\kappa \Delta}}{\kappa \Delta}$$

This is Equation (4) of the project specification. The affinity property is critical because it establishes that $\text{VIX}_t$ is a direct, monotone transformation of $v_t$ under Heston — a feature which simultaneously enables tractable VIX option pricing *and* exposes the one-factor constraint that prevents joint calibration.

### 5.3 CIR Characteristic Function

Since $\text{VIX}_t^2$ is affine in $v_t$, and $v_t$ follows a CIR process under Heston, the distribution of a VIX option's payoff can be obtained analytically via the characteristic function of the integrated variance $\int_t^{t+\Delta} v_s\, ds$. Gatheral presents this characteristic function:

$$\varphi(u;\, t, T) = \mathbb{E}^{\mathbb{Q}}\!\left[ e^{iu \int_{t}^{T} v_s\, ds} \,\Big|\, v_t \right]$$

in closed form, expressed in terms of hyperbolic functions of $\kappa$, $\theta$, $\sigma_v$, and $u$. This is the input required by W4 (Fourier Engineer) to implement the COS method for pricing VIX futures and options. Without this result, the COS engine would require numerical Fourier inversion of a function with no closed form.

### 5.4 The VIX Implied Volatility Smile

Gatheral's treatment of the VIX smile provides the analytical basis for W7 (Vol Analyst). The key empirical observation — that the VIX smile is **positively skewed** (right-skewed), in contrast to the negatively skewed SPX smile — has a clear economic explanation: VIX spikes are asymmetric. A sudden jump in realised or implied volatility during a market crisis is far more probable than a symmetric upward drift, and this asymmetry is priced into the right tail of VIX option prices. The SPX smile, by contrast, reflects the negative correlation between equity returns and volatility (the leverage effect), which depresses the left tail of SPX returns and loads the left side of the smile.

### 5.5 Role in the Present Project

Gatheral (2006) serves as the primary analytical reference for W2, W4, and W7. W2 uses the book to prove that $\text{VIX}_t^2$ is affine in $v_t$ and to derive the CIR characteristic function (Week 4 deliverables 3 and 4). W4 implements the COS pricing engine using the characteristic function as input. W7 constructs the vol-of-vol surface and provides the economic explanation for the shape of the VIX smile — an explanation that is articulated most clearly in Gatheral's framework.

---

## 6. Synthesis: The Logical Chain of Classical Literature

The four sources reviewed above do not merely coexist in the literature — they form a coherent logical chain. Demeterfi et al. (1999) establish that variance is a replicable, tradeable quantity and define its fair price in a model-free manner. Carr and Wu (2009) demonstrate empirically that the market price of variance systematically exceeds its realised value, establishing the variance risk premium as a robust and economically significant phenomenon. Bergomi (2005) identifies the structural reason why the dominant one-factor model — Heston — cannot simultaneously account for the SPX surface and the VIX term structure, thereby motivating the forward variance and multi-factor extensions that define the current research frontier. Gatheral (2006) provides the mathematical tools — the affinity result, the CIR characteristic function, and the smile analysis — that enable tractable analytical work within the Heston framework while also exposing its limitations.

This logical progression — from replication, to empirical measurement, to model critique, to analytical tooling — mirrors the structure of the present project and provides the classical foundations upon which the contemporary literature reviewed by Participant 2 builds.

---

## References

Bergomi, L. (2005). Smile Dynamics 2. *Risk Magazine*.

Carr, P., & Wu, L. (2009). Variance risk premiums. *Review of Financial Studies*, 22(3), 1311–1341.

Demeterfi, K., Derman, E., Kamal, M., & Zou, J. (1999). *More than you ever wanted to know about volatility swaps*. Goldman Sachs Quantitative Strategies Research Notes.

Gatheral, J. (2006). *The volatility surface: A practitioner's guide*. Wiley Finance.

---

# W1: Literature Review — Contemporary Frontier Topics in Volatility Modelling

## Participant 2 Contribution

This section extends the classical literature review above with the contemporary research frontier relevant to variance swaps, VIX options, rough volatility, joint SPX--VIX calibration, and empirical realised-variance estimation. The classical sources explain why variance can be traded, how VIX is linked to option prices, and why one-factor stochastic volatility models are analytically convenient. The newer literature asks where those models break down and what methods are now used to repair the gap between theory and market data.

The main post-2015 developments are threefold. First, empirical studies show that volatility is much rougher than standard Markovian stochastic-volatility models imply, with a Hurst exponent around $H \approx 0.1$ rather than the Brownian value $H = 1/2$. Second, one-factor models such as Heston struggle to fit SPX options and VIX options simultaneously because VIX becomes too tightly linked to one state variable. Third, high-frequency realised variance is severely biased by market microstructure noise, so modern variance risk premium work needs robust estimators such as realised kernels rather than naive tick-by-tick realised variance.

---

## 7. Rough Volatility and the VIX Modelling Frontier

### 7.1 Empirical Motivation

Gatheral, Jaisson and Rosenbaum (2018) document that volatility is "rough": the regularity of log-volatility is far below that of a standard Brownian diffusion. A typical empirical scaling relation is

$$
\mathbb{E}\left[|\log \sigma_{t+\Delta} - \log \sigma_t|^q\right] \sim \Delta^{qH},
$$

with estimates of $H$ on equity-index data often around $0.05$ to $0.15$. This is a direct challenge to Heston-type models, where the instantaneous variance process is Markovian and Brownian-driven:

$$
dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dB_t.
$$

In such a model, the path regularity is Brownian-like. The data instead imply much more irregular volatility paths, which helps explain why classical models often understate short-maturity skew.

### 7.2 Short-Time Skew in Rough Models

Bayer, Friz and Gatheral (2016), together with related work on short-time asymptotics, show that rough volatility naturally explains the steep short-maturity SPX implied-volatility skew. Empirically, near-the-money skew behaves approximately as

$$
\left|\frac{\partial \sigma_{\mathrm{IV}}(K,T)}{\partial \log K}\right|_{K=S} \sim T^{H-1/2}.
$$

When $H \approx 0.1$, the skew explodes approximately like $T^{-0.4}$ as maturity approaches zero. In classical Heston, corresponding to Brownian regularity, the short-time skew does not reproduce this empirical explosion in the same way.

### 7.3 Rough Bergomi and VIX

The rough Bergomi model is a central benchmark in this literature. In simplified form,

$$
\frac{dS_t}{S_t} = \sqrt{v_t}\,dW_t,
$$

$$
v_t = \xi_0(t)\exp\left(\eta \widetilde{W}_t^H - \frac{1}{2}\eta^2t^{2H}\right),
$$

where

$$
\widetilde{W}_t^H = \sqrt{2H}\int_0^t (t-s)^{H-1/2}\,dZ_s.
$$

Here $Z$ is correlated with the Brownian motion driving the stock price, and $\xi_0(t)$ is the initial forward variance curve. The VIX is then linked to the conditional expectation of future integrated variance:

$$
\mathrm{VIX}_t^2 = \frac{1}{\Delta}\mathbb{E}_t\int_t^{t+\Delta} v_s\,ds,
\qquad \Delta = 30/365.
$$

The difficulty is that rough Bergomi is non-Markovian: $v_s$ depends on the past path of the fractional process, not just on a finite-dimensional state variable. This makes VIX option pricing, hedging, and calibration computationally demanding.

### 7.4 Deep Learning Volatility Calibration

Horvath, Muguruza and Tomas (2021) address the computational bottleneck in calibrating rough volatility models. Their main contribution is to use neural networks as fast approximators for the pricing and calibration maps. Conceptually, the workflow is

$$
\theta = (H,\eta,\rho,\xi_0)
\quad \longrightarrow \quad
\sigma_{\mathrm{IV}}(K,T)
\quad \longrightarrow \quad
\hat{\theta}.
$$

The first neural network approximates the map from model parameters to implied-volatility surfaces, using synthetic training data generated offline by expensive Monte Carlo simulation. The second network approximates the inverse calibration map from observed surfaces back to model parameters. This makes real-time calibration of rough models much more feasible and is especially relevant for testing whether such models can fit SPX and VIX smiles jointly.

---

## 8. Joint SPX--VIX Calibration and the Limits of One-Factor Models

### 8.1 The Structural Problem

The classical Heston framework has a single latent variance factor $v_t$. Under Heston, $\mathrm{VIX}_t^2$ is an affine function of $v_t$, so SPX options, VIX futures, and VIX options are all forced to depend on the same state variable. This gives tractability, but it is also the reason the model is too rigid.

Guyon (2020) formalises this failure through an inversion of convex ordering in the VIX market. For a broad class of one-factor diffusion models,

$$
\frac{dS_t}{S_t} = \sqrt{f(X_t)}\,dW_t,
\qquad
dX_t = \mu(X_t)\,dt + \sigma(X_t)\,dB_t,
$$

the model implies restrictions between the SPX surface and the VIX option surface. In market data, these restrictions are systematically violated. In practical terms, a one-factor model calibrated to SPX options tends to underprice VIX option volatility, especially the right tail of the VIX smile.

### 8.2 Proposed Solutions

The literature has proposed several ways to escape the one-factor restriction:

| Approach | Representative source | Main idea | Joint SPX--VIX fit |
| --- | --- | --- | --- |
| Two-factor Bergomi | Bergomi (2008) | Use multiple forward-variance factors | Partial improvement |
| Heston with jumps | Pacati et al. (2018) | Add jumps to variance dynamics | Better, but parameter-heavy |
| Quadratic rough Heston | Gatheral, Jusselin and Rosenbaum (2020) | Combine roughness with nonlinear variance | Strong joint fit |
| Path-dependent volatility | Guyon and Lekeufack (2023) | Make volatility a functional of past SPX returns | Strong empirical fit |

Gatheral, Jusselin and Rosenbaum (2020) propose the quadratic rough Heston model:

$$
v_t = a(Z_t - b)^2 + c,
$$

where

$$
Z_t = \int_0^t \frac{(t-s)^{H-1/2}}{\Gamma(H+1/2)}
\left(\theta(s)\,ds + \eta\,dW_s\right).
$$

The quadratic transformation of a rough Volterra process gives the model enough flexibility to generate asymmetric skew and stronger VIX option volatility without introducing a large number of separate factors.

Guyon and Lekeufack (2023) take a different route: they show that volatility can be modelled as largely path-dependent on past SPX returns. A schematic representation is

$$
v_t =
\beta_0
+ \beta_1\int_0^t K_1(t-s)\frac{dS_s}{S_s}
+ \beta_2\int_0^t K_2(t-s)\left(\frac{dS_s}{S_s}\right)^2.
$$

This approach suggests that much of volatility dynamics may be reconstructed from trend and realised-volatility features of the index itself, rather than from a separate hidden Markov factor.

---

## 9. Realised Kernels and High-Frequency Variance Estimation

### 9.1 Why Naive Realised Variance Fails

For the empirical part of the project, the target quantity is integrated variance:

$$
IV_T = \int_0^T \sigma_s^2\,ds.
$$

If efficient prices were observed without noise, realised variance would be consistent:

$$
RV_n = \sum_{i=1}^{n}
\left(\log P_{t_i} - \log P_{t_{i-1}}\right)^2
\xrightarrow{p} IV_T.
$$

In high-frequency data, however, observed prices include microstructure noise:

$$
\log P_{t_i} = \log P^*_{t_i} + \varepsilon_i.
$$

The noise term comes from bid-ask bounce, tick-size discreteness, asynchronous trading, and other market frictions. Under this observation equation,

$$
\mathbb{E}[RV_n] = IV_T + 2n\,\mathbb{E}[\varepsilon^2],
$$

so tick-by-tick realised variance diverges as the sampling frequency increases. This is the high-frequency variance estimation problem: more observations can make the naive estimator worse.

### 9.2 Realised Kernel Estimators

Barndorff-Nielsen, Hansen, Lunde and Shephard (2008, 2011) propose realised kernels as a robust solution. The estimator combines return autocovariances with kernel weights:

$$
RK = \gamma_0 + \sum_{h=1}^{H} k\left(\frac{h-1}{H}\right)(\gamma_h + \gamma_{-h}),
$$

where

$$
\gamma_h = \sum_{i=1}^{n-h} r_i r_{i+h},
\qquad
r_i = \log P_{t_i} - \log P_{t_{i-1}}.
$$

The term $\gamma_0$ is the usual realised variance, while the autocovariance terms offset the bias generated by market microstructure noise. Common choices include Parzen and Tukey-Hanning kernels. The bandwidth $H$ controls how many autocovariance lags are included, with optimal rates often of order $H \propto n^{3/5}$ under standard assumptions.

The key properties are consistency under noisy observations, robustness to some forms of autocorrelated noise, and the ability to construct positive semi-definite multivariate covariance estimates when appropriate kernels are used. This makes realised kernels more suitable than naive high-frequency realised variance for estimating the physical component of the variance risk premium.

### 9.3 Link to VIX and the Variance Risk Premium

The variance risk premium can be written as

$$
\mathrm{VRP}_t =
\underbrace{\mathbb{E}^{\mathbb{Q}}[IV_{t,t+\Delta}]}_{\text{approximately } \mathrm{VIX}^2}
-
\underbrace{\mathbb{E}^{\mathbb{P}}[IV_{t,t+\Delta}]}_{\text{estimated from realised variance measures}}.
$$

VIX gives a risk-neutral expectation of future variance, while realised kernels provide a cleaner empirical estimate of physical realised variance from high-frequency data. Therefore, the realised-kernel literature is directly connected to the Carr and Wu variance risk premium framework reviewed above.

For the project, the natural empirical workflow is:

1. Obtain high-frequency SPX or SPY data.
2. Estimate daily integrated variance using a realised kernel, such as the Parzen kernel.
3. Compare the realised-kernel estimate with naive 5-minute realised variance.
4. Annualise the estimates and compare them with $\mathrm{VIX}^2$.
5. Study VRP dynamics during stress periods such as COVID-19 in 2020, the inflation and rate-hike period of 2022, and the banking stress episode of 2023.

---

## 10. Synthesis: How the Classical and Frontier Literatures Connect

The classical literature establishes the project foundation: Demeterfi et al. explain why variance can be replicated from options; Carr and Wu turn the difference between implied and realised variance into an empirical risk premium; Bergomi shows why forward variance is the natural state variable for volatility derivatives; and Gatheral provides the analytical Heston and VIX machinery.

The contemporary literature explains why these foundations are not enough. Rough volatility improves the description of short-maturity skew and volatility path regularity. Joint SPX--VIX calibration results show that one-factor models are too restrictive for VIX options. Realised kernels solve the empirical problem that naive high-frequency realised variance is biased by market microstructure noise. Together, these frontier topics define the Participant 2 contribution to the group project: connecting modern rough-volatility models and robust realised-variance estimation to the pricing and empirical analysis of variance swaps and VIX options.

---

## Additional References for Participant 2

Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008). Designing realised kernels to measure the ex-post variation of equity prices in the presence of noise. *Econometrica*, 76(6), 1481--1536.

Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2011). Multivariate realised kernels: consistent positive semi-definite estimators of the covariation of equity prices with noise and non-synchronous trading. *Journal of Econometrics*, 162(2), 149--169.

Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*, 16(6), 887--904.

Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*, 18(6), 933--949.

Gatheral, J., Jusselin, P., & Rosenbaum, M. (2020). The quadratic rough Heston model. *Risk Magazine*.

Guyon, J. (2020). The joint SPX--VIX smile calibration puzzle solved. *Risk Magazine*.

Guyon, J., & Lekeufack, J. (2023). Volatility is mostly path-dependent. *Quantitative Finance*.

Horvath, B., Muguruza, A., & Tomas, M. (2021). Deep learning volatility. *Quantitative Finance*, 21(1), 11--27.

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

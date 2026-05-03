W1: Literature Review — Classical Foundations of Volatility as an Asset Class
1. Introduction

There are four main papers that the current theory of volatility as a tradeable asset class is built upon: the replication framework of Demeterfi, Derman, Kamal and Zou (1999), the variance risk premium (VRP) empirical measurement of Carr and Wu (2009), Bergomi's (2005) forward variance modelling, and the volatility surface comprehensive treatment by Gatheral (2006). Together, these books form the conceptual and mathematical framework that all subsequent works of this project the Monte Carlo engine, the COS pricing engine, the joint calibration exercise, and the VRP backtest rely on. This review looks at each source separately, points out its main contribution, places it among the rest of the literature, and indicates its function in the current project.

2. Demeterfi, Derman, Kamal and Zou (1999): Variance Swaps via Log-Contract Replication

2.1 Background and Achievements
Initially released as a Goldman Sachs Quantitative Strategies Research Note, Demeterfi et al. (1999) is generally recognized as the seminal paper that spurred the birth of the variance swap market. Before this paper, volatility was only traded indirectly, via option positions where volatility exposure was combined with delta and gamma risk. Demeterfi et al. proved that one can obtain a pure exposure to realised variance through a static, model-free replication approach.

2.2 The Log-Contract Replication Method
Demerfi et al.'s main idea is that the payoff of a variance swap, the difference between the realised variance and the strike, is a portfolio of European puts and calls over the entire strike range that can be rebalanced continuously. More precisely, one needs 2/K^2 units of each option at strike K, which combined together form a log-contract: a contract that pays out $$\Pi = 2 \ln \left( \frac{S_T}{F_{0,T}} \right)$$ at expiry, where F is the forward price. Using Itô's lemma, one can get the variance of the log-price process from the value of this portfolio.

Thus, the variance strike that makes the value of the swap equal to zero is:
$$VIX_{t}^{2} = \frac{2}{T} \left[ \int_{0}^{F_{t}} \frac{P(K)}{K^{2}} dK + \int_{F_{t}}^{\infty} \frac{C(K)}{K^{2}} dK \right]$$
This result is model-free, i.e., it does not require the assumption about the dynamics of the underlying process except for the continuity of the sample paths. Actually, the only inputs are the prices of liquidly traded options on the market.
2.3 Connection to the CBOE VIX Formula
The CBOE VIX index which was made public in its present form in 2003 is, in fact, a direct implementation of the Demeterfi et al. methodology. Below you will find that Equation (3) in the project specification restates the VIX formula exactly:
$$VIX_{t}^{2} = \frac{2}{T} \left[ \int_{0}^{F_{t}} \frac{P(K)}{K^{2}} dK + \int_{F_{t}}^{\infty} \frac{C(K)}{K^{2}} dK \right]$$

So, understanding the log-contract argument becomes a necessary step before any other derivations in the project i.e. the VIX formula, the fair-strike calculation under Heston (Equation 2), and the model-free hedging strategy in W8.
2.4 Role in the Present Project
The replication result by Demeterfi et al. serves as a foundation for three workstreams directly. Firstly, W2 (Theory Lead) uses the log-contract argument to obtain Equation (3) from the first principles, as per the Week 4 deliverables. Secondly, W3 (MC Engineer) compares the Monte Carlo simulation of E^Q[{RV}_{0, T}] with the closed-form solution for $K_{\text{var}}$, a comparison that is only meaningful if one understands what the formula stands for. Thirdly, and most importantly, W8 (Risk Lead) depends on the model-free nature of the replication to create a Vega hedge for the variance swap that is not reliant on the Heston parameters. The hedge being model-independent is not simply a technical facilitation but a major economic point that this paper is making.
3. Carr and Wu (2009): The Variance Risk Premium
3.1 Context and Contribution
Working in and publishing their results in the Review of Financial Studies, Carr and Wu (2009) made the first attempt to demonstrate empirically and on a grand scale, through rigorous methodology, the presence of variance risk premium (VRP) in equity markets. Though it was suggested that a premium for taking on variance risk exists, Carr and Wu were pioneers in the systematic quantification of this premium across multiple asset classes and over long time periods. Their approach importantly disentangles the risk-neutral expectation of variance (implied from options) from the physical expectation (obtained by using realised returns).
3.2 Definition and Measurement of the VRP
These two researchers focused on defining VRP as:
$$VRP_t = IV_t^2 - \mathbb{E}_t^{\mathbb{P}}[RV_{t,t+\Delta}]$$
where $IV_{t}^{2}$ is the risk-neutral expected variance (approximated by the squared VIX) and $\mathbb{E}^{\mathbb{Q}}[RV_{0,T}]$ is the physical expectation of realised variance over the following month. The latter is estimated both *ex post* — using the actual realised variance of the subsequent period — and *ex ante* — using a GARCH(1,1) forecast.
What the authors observe in their main data is that VRP is on average positive, usually about two to four annualised volatility points, and it is also very stable over time and very different equity indices. This premium represents the investors compensation for taking the risk of unexpected variance increases a risk which is not diversifiable and shoots up during market dislocations.
3.3 Economic Interpretation
The observation of the VRP being positive suggests that investors are generally *overpaying* for variance exposure in the options markets. On the other hand, the variance swap seller who receives a fixed strike Kvar and pays a floating realised variance is making a steady positive return. This explains the economic appeal of the Short Variance Swap strategy in W8 (Alpha section). Carr and Wu also show the VRP is highly correlated with market factors. It tends to increase in fear and uncertainty and decrease or even become negative in calm periods. This characteristic is important for the VRP regime analysis that has to be done in Section 6.4 according to the project specification.

3.4 Factor Decomposition

One major methodological highlight of Carr and Wu is that they not only identify which risk factors are most related to variations in VRP but also quantify their impacts with a regression. They pick up common factors such as the returns on the stock market, the change in the level of the VIX, and the VIX that is lagged. This is exactly the beta-nature breakdown that is mentioned in Section 6.4 of the project:
$$R_{t}^{VRP} = \alpha + \beta_{1} R_{t}^{SPX} + \beta_{2} \Delta VIX_{t} + \beta_{3} VIX_{t-1} + \epsilon_{t}$$

After accounting for equity and volatility betas, Carr and Wu's discovery that a great alpha still exists is the theoretical basis for considering the VRP as a real risk premium rather than an equity beta in disguise, this is an issue that W8 definitely needs to clarify.

3.5 Role in the Present Project
W5 (Data Scientist)'s work of recreating the monthly VRP time series using both clear-cut results and GARCH-based forward-looking estimates is directly based on Carr and Wu (2009) methodology. The W8's trading idea, including sizing, backtesting, crisis analysis, and factor decomposition, goes to the empirical framework of this work. The VRP regime (high VRP vs. low VRP) Sharpe ratio decomposition is grounded in Carr and Wu's insight that the premium varies greatly by states.

4. Bergomi (2005): Forward Variance Models and the Failure of One-Factor Specifications
4.1 Context and Contribution
Lorenzo Bergomi's "Smile Dynamics 2" (Risk Magazine, 2005) is a work directed towards professionals, which points out a serious flaw of the Heston model and at the same time, it lays down the foundation for a broader conceptual framework that uses forward variance. It is, in fact, the first piece of work that W6 must show through numerical experiments, the problem of joint calibration.
4.2 The Joint Calibration Problem
The Heston model is a one-factor stochastic volatility model: only one hidden process, $v_t$, determines the instantaneous variance of the SPX returns as well as the level of the VIX. According to Heston, the square of the VIX is an affine function of $v_t$ (see Equation 4 of the project), which implies that the VIX futures term structure is solely reliant on κ, v̄, and v₀, the very same set of parameters that describe the SPX implied volatility surface.
Bergomi (2005) reveals that this exclusive link between the variance process and the market volatility surface results in a paradox: the values needed to perfectly match the SPX surface (especially the skew term structure and the level of implied volatility across maturities) usually do not coincide with those needed for the VIX futures curve. Put simply, a Heston model that fits the SPX options will always be inaccurate in pricing VIX futures, and vice versa.
4.3 The Forward Variance Framework
Bergomi suggests modelling the forward variance curve $$\xi_u^t = \mathbb{E}_t^{\mathbb{Q}}[v_u]$$ directly as a function with infinite dimensions evolving according to a lognormal process:
$$d\xi_u^t = \omega \cdot \xi_u^t \cdot dW_t$$
This method enables the variance dynamics at any instant to be independent of the overall volatility surface which means that given the right parameterisation, the model can be adjusted to both the SPX options and VIX futures at the same time. The idea of a forward variance model was a stepping stone to the introduction of rough volatility models that Participant 2 will discuss.
4.4 Role in the Present Project
In Bergomi (2005), the theoretical idea behind the joint calibration exercise in W6 is explained. Steps 1, 4 of Section 6.1 in the project specification are a step-by-step implementation of the reasoning: first, calibrate Heston to SPX options (Step 1), second, get the implied VIX futures curve (Step 2), third, see the difference (Step 3), and finally, decide that a two-factor or rough volatility model is needed (Step 4). The extra part, the realization of a two-factor CIR model $$v_t = v^{(1)}_t + v^{(2)}_t$$ , is solely inspired by the multi-factor framework that Bergomi supports.
5. Gatheral (2006): The Volatility Surface, VIX Chapters
5.1 Context and Contribution
The Volatility Surface by Jim Gatheral (Wiley Finance, 2006) is the main reference at the graduate level for derivatives practitioners. The chapters that matter for this project cover three aspects: the affine structure of the VIX under Heston, the CIR characteristic function that one needs for the analytical pricing of VIX derivatives, and the VIX implied volatility smile, its shape and economic interpretation.
5.2 Affinity of VIX² under Heston
One important analytical conclusion from Gatheral is that in the Heston model, the squared VIX is a linear function of the instantaneous variance v_t:
$$VIX_{t}^{2} = \bar{v} + (v_{t} - \bar{v}) \frac{1 - e^{-\kappa \Delta}}{\kappa \Delta}$$
This is Equation (4) of the project specification. The property of affinity is very important because it shows that $VIX_t$ is in fact a direct, monotone transformation of the instantaneous variance $v_t$ in the Heston model, this is a feature which both allows us to carry out VIX option pricing *and* reveals the limitation of the one-factor model that makes it impossible to achieve joint calibration.
5.3 CIR Characteristic Function
Since $VIX_t^2$ is a linear function in $v_t$ and $v_t$ is a CIR process under Heston, one can get the distribution of a VIX option's payoff from the characteristic function of the integrated variance $\mathbb{E}_{t}^{\mathbb{Q}}[\frac{1}{\Delta}\int_{t}^{t+\Delta}v_{s}ds]$.
Gatheral gives this characteristic function:
$$\phi(u; t, T) = \mathbb{E}^{\mathbb{Q}} \left[ e^{iu \int_{t}^{T} v_s ds} \mid v_t \right]$$

which is closed-form, written using hyperbolic functions of $\kappa$ , $\theta$, $\sigma_v$, and u. This is what W4 (Fourier Engineer) needs as input in order to use the COS method for pricing VIX futures and options. If this result didn't exist, the COS engine would have to do a numerical Fourier inversion of a function that has no closed form.
5.4 The VIX Implied Volatility Smile
The analytical basis of W7 (Vol Analyst) is provided by Gatheral's manual on the VIX smile. The main empirical finding, that the VIX smile is positively skewed (right-skewed), as opposed to the negatively skewed SPX smile, can be explained economically: VIX raises are asymmetric. The chances of a volatility jump, realised or implied, during a crisis are much higher than those of a symmetric upward drift. This is the asymmetry that gets priced into the right tail of VIX options. The SPX smile, on the other hand, is due to the negative correlation between the return on equities and volatility (the leverage effect), which forces the left tail of SPX returns to be depressed and so enhances the left side of the smile.
5.5 Role in the Present Project
Gatheral (2006) is the main source of formal analysis for W2, W4, and W7. Producing the proof that the $VIX_t^2$ is an affine function of $v_t$ and extracting the CIR characteristic function is the application of the book by W2 (Week 4 deliverables 3 and 4). Once W4 got the characteristic function, it didn't take long to construct the COS pricing engine. For W7, the vol-of-vol surface was the main output, and the economic reasoning behind the VIX smile shape, which Gatheral expresses best of all, was also given there.
6. Synthesis: The Logical Chain of Classical Literature

The four papers discussed above are not just different voices in the literature but rather pieces of a logically connected whole. Starting with Demeterfi et al. (1999), they prove that variance is a measurable and tradable quantity and determine its fair price without any assumptions on the model. Then, Carr and Wu (2009) provide evidence that the market price of variance is on average greater than the realised one, thus they define the variance risk premium as a strong and economically relevant concept. Bergomi (2005) explains that the main one-factor model --- Heston --- structurally fails to explain both the SPX surface and the VIX term structure at the same time and that is the very reason why forward variance and multi-factor versions, which are at the research frontier today, have been motivated. Gatheral (2006) delivers the essential mathematical ingredients --- the affinity result, the CIR characteristic function, and smile analysis --- which allow working analytically within the Heston model but also make its drawbacks quite apparent.

This logical sequence --- starting with replication, moving on to empirical measurement, then to model critique and finally to the development of analytical tools --- is parallel to the structure of this paper and represents the classical bases, which contemporary literature reviewed by Participant 2, is built upon.
References
Bergomi, L. (2005). Smile Dynamics 2. Risk Magazine.
Carr, P., & Wu, L. (2009). Variance risk premiums. Review of Financial Studies, 22(3), 1311–1341.
Demeterfi, K., Derman, E., Kamal, M., & Zou, J. (1999). More than you ever wanted to know about volatility swaps. Goldman Sachs Quantitative Strategies Research Notes.
Gatheral, J. (2006). The volatility surface: A practitioner's guide. Wiley Finance.




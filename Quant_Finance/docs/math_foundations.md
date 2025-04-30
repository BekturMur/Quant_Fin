# Mathematical Foundations

This document provides a detailed explanation of the mathematical models and formulas used in the Portfolio Optimization and Risk Analysis Toolkit.

## 1. Modern Portfolio Theory (MPT)

### 1.1 Portfolio Return and Risk

The expected return of a portfolio is given by:

\[
E[R_p] = \sum_{i=1}^n w_i E[R_i]
\]

where:
- \(w_i\) is the weight of asset \(i\) in the portfolio
- \(E[R_i]\) is the expected return of asset \(i\)
- \(n\) is the number of assets

The portfolio variance is:

\[
\sigma_p^2 = \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij}
\]

where:
- \(\sigma_{ij}\) is the covariance between assets \(i\) and \(j\)

### 1.2 Sharpe Ratio Optimization

The Sharpe ratio is defined as:

\[
SR = \frac{E[R_p] - r_f}{\sigma_p}
\]

where:
- \(r_f\) is the risk-free rate
- \(\sigma_p\) is the portfolio standard deviation

The optimization problem is:

\[
\max_w \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}
\]

subject to:
\[
\sum_{i=1}^n w_i = 1
\]
\[
w_i \geq 0 \quad \forall i
\]

## 2. Black-Litterman Model

### 2.1 Equilibrium Returns

The equilibrium returns are calculated using the CAPM:

\[
\Pi = \delta \Sigma w_{mkt}
\]

where:
- \(\Pi\) is the vector of equilibrium returns
- \(\delta\) is the risk aversion coefficient
- \(\Sigma\) is the covariance matrix
- \(w_{mkt}\) is the market capitalization weights

### 2.2 Investor Views

The investor's views are expressed as:

\[
P \cdot R = Q + \epsilon
\]

where:
- \(P\) is the matrix of views
- \(Q\) is the vector of expected returns for the views
- \(\epsilon\) is the error term with covariance \(\Omega\)

### 2.3 Posterior Returns

The Black-Litterman posterior returns are:

\[
E[R] = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q]
\]

where:
- \(\tau\) is the scaling factor for the covariance matrix

## 3. Risk Parity

### 3.1 Risk Contribution

The risk contribution of asset \(i\) is:

\[
RC_i = w_i \frac{\partial \sigma_p}{\partial w_i} = w_i \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}}
\]

where:
- \(\sigma_p\) is the portfolio volatility
- \((\Sigma w)_i\) is the \(i\)-th element of the vector \(\Sigma w\)

### 3.2 Optimization Problem

The risk parity optimization minimizes:

\[
\sum_{i=1}^n \sum_{j=1}^n (RC_i - RC_j)^2
\]

subject to:
\[
\sum_{i=1}^n w_i = 1
\]
\[
w_i \geq 0 \quad \forall i
\]

## 4. Machine Learning for Return Prediction

### 4.1 Feature Engineering

For each asset, we create lagged returns as features:

\[
X_t = [r_{t-1}, r_{t-2}, \ldots, r_{t-p}]
\]

where:
- \(r_t\) is the return at time \(t\)
- \(p\) is the number of lags

### 4.2 Random Forest Model

The Random Forest model predicts returns as:

\[
\hat{r}_t = \frac{1}{B} \sum_{b=1}^B f_b(X_t)
\]

where:
- \(B\) is the number of trees
- \(f_b\) is the prediction of the \(b\)-th tree
- \(X_t\) is the feature vector at time \(t\)

## 5. Performance Metrics

### 5.1 Annualized Return

\[
R_{annual} = (1 + R_{daily})^{252} - 1
\]

### 5.2 Annualized Volatility

\[
\sigma_{annual} = \sigma_{daily} \sqrt{252}
\]

### 5.3 Maximum Drawdown

\[
MDD = \max_{t \in [0,T]} \left( \max_{s \in [0,t]} \frac{P_s - P_t}{P_s} \right)
\]

where:
- \(P_t\) is the portfolio value at time \(t\)
- \(T\) is the total time period

## References

1. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
2. Black, F., & Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal*, 48(5), 28-43.
3. Qian, E. (2005). Risk Parity Portfolios: Efficient Portfolios Through True Diversification. *PanAgora Asset Management*.
4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32. 
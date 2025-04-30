# Portfolio Optimization and Risk Analysis Toolkit

A professional Python toolkit for quantitative finance that implements various portfolio optimization techniques and risk analysis tools. This project showcases modern portfolio theory, risk management, and machine learning applications in finance.

![Efficient Frontier](efficient_frontier.png)

## Features

- **Modern Portfolio Theory (MPT)**: Optimize portfolios for maximum Sharpe ratio
- **Black-Litterman Model**: Incorporate investor views into expected returns
- **Risk Parity**: Allocate weights for equal risk contributions
- **Machine Learning**: Predict asset returns using Random Forest
- **Data Visualization**: Generate efficient frontier plots and portfolio weight charts
- **Real-time Data**: Fetch financial data using Yahoo Finance API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/portfolio-optimization-toolkit.git
cd portfolio-optimization-toolkit
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from portfolio_toolkit import PortfolioOptimizer

# Initialize with your assets and date range
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
start_date = '2020-01-01'
end_date = '2023-12-31'

optimizer = PortfolioOptimizer(tickers, start_date, end_date)

# Optimize for maximum Sharpe ratio
weights, max_sharpe = optimizer.optimize_sharpe_ratio()
print(f"Maximum Sharpe Ratio: {max_sharpe:.4f}")

# Implement Black-Litterman model
views = {'AAPL': 0.15, 'MSFT': 0.12}
view_confidences = {'AAPL': 0.5, 'MSFT': 0.5}
posterior_returns = optimizer.black_litterman(views, view_confidences)

# Calculate Risk Parity weights
weights_rp = optimizer.risk_parity()

# Predict returns using ML
predicted_returns = optimizer.predict_returns()

# Generate plots
optimizer.plot_efficient_frontier()
optimizer.plot_portfolio_weights(weights, 'Sharpe Ratio Optimal Weights')
```

### Jupyter Notebook Example

See `notebooks/example.ipynb` for a detailed walkthrough of the toolkit's features.

## Mathematical Foundations

### Modern Portfolio Theory

The Sharpe ratio optimization problem is formulated as:

\[
\max_w \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}
\]

where:
- \(w\) is the vector of portfolio weights
- \(\mu\) is the vector of expected returns
- \(\Sigma\) is the covariance matrix
- \(r_f\) is the risk-free rate

### Black-Litterman Model

The Black-Litterman model combines market equilibrium returns with investor views:

\[
E[R] = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q]
\]

where:
- \(\Pi\) is the vector of equilibrium returns
- \(P\) is the matrix of views
- \(Q\) is the vector of view returns
- \(\Omega\) is the uncertainty matrix
- \(\tau\) is the scaling factor

### Risk Parity

Risk parity aims to equalize risk contributions:

\[
RC_i = w_i \frac{\partial \sigma_p}{\partial w_i} = w_i \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}}
\]

where:
- \(RC_i\) is the risk contribution of asset \(i\)
- \(\sigma_p\) is the portfolio volatility

## Project Structure

```
portfolio-optimization-toolkit/
├── portfolio_toolkit.py      # Main implementation
├── requirements.txt          # Dependencies
├── notebooks/
│   └── example.ipynb        # Usage examples
├── tests/
│   └── test_portfolio.py    # Unit tests
├── docs/
│   └── math_foundations.md  # Mathematical documentation
└── README.md                # This file
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Future Enhancements

- Value-at-Risk (VaR) calculations
- Monte Carlo simulations
- Factor models integration
- Transaction cost optimization
- Backtesting framework
- Web interface for interactive portfolio analysis

## Acknowledgments

- Harry Markowitz for Modern Portfolio Theory
- Fischer Black and Robert Litterman for the Black-Litterman model
- The open-source Python community for the excellent libraries used in this project 
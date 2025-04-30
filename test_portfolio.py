from portfolio_toolkit import PortfolioOptimizer
import matplotlib.pyplot as plt

def main():
    # Initialize with test assets
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    start_date = '2020-01-01'
    end_date = '2023-12-31'

    # Create optimizer
    optimizer = PortfolioOptimizer(tickers, start_date, end_date)

    # Optimize for Sharpe ratio
    weights_sharpe, max_sharpe = optimizer.optimize_sharpe_ratio()
    print("\nOptimal weights by Sharpe ratio:")
    for ticker, weight in zip(tickers, weights_sharpe):
        print(f"{ticker}: {weight:.2%}")
    print(f"Maximum Sharpe ratio: {max_sharpe:.4f}")

    # Visualize portfolio weights
    optimizer.plot_portfolio_weights(weights_sharpe, 'Optimal Sharpe Weights')

    # Risk parity optimization
    weights_rp = optimizer.risk_parity()
    print("\nRisk parity weights:")
    for ticker, weight in zip(tickers, weights_rp):
        print(f"{ticker}: {weight:.2%}")

    # Visualize portfolio weights
    optimizer.plot_portfolio_weights(weights_rp, 'Risk Parity Weights')

    # Return prediction
    predicted_returns = optimizer.predict_returns()
    print("\nPredicted returns:")
    for ticker, ret in predicted_returns.items():
        print(f"{ticker}: {ret:.2%}")

    # Plot efficient frontier
    optimizer.plot_efficient_frontier()

    # Apply Black-Litterman model
    views = {'AAPL': 0.15, 'MSFT': 0.12}
    view_confidences = {'AAPL': 0.5, 'MSFT': 0.5}
    posterior_returns = optimizer.black_litterman(views, view_confidences)
    print("\nExpected returns from Black-Litterman model:")
    for ticker, ret in zip(tickers, posterior_returns):
        print(f"{ticker}: {ret:.2%}")

if __name__ == "__main__":
    main() 
"""
Portfolio Optimization and Risk Analysis Toolkit

This module implements various portfolio optimization techniques and risk analysis tools
for quantitative finance applications. It includes Modern Portfolio Theory, Black-Litterman
model, Risk Parity allocation, and machine learning-based return predictions.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Main class for portfolio optimization and risk analysis."""
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        Initialize the PortfolioOptimizer with historical data.
        
        Args:
            tickers: List of asset tickers
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._fetch_data()
        self.returns = self._calculate_returns()
        self.cov_matrix = self.returns.cov()
        self.mean_returns = self.returns.mean()
        
    def _fetch_data(self) -> pd.DataFrame:
        """Fetch historical price data from Yahoo Finance."""
        data = {}
        for ticker in self.tickers:
            try:
                data[ticker] = yf.download(ticker, start=self.start_date, 
                                         end=self.end_date)['Adj Close']
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                raise
        return pd.DataFrame(data)
    
    def _calculate_returns(self) -> pd.DataFrame:
        """Calculate daily returns from price data."""
        return self.data.pct_change().dropna()
    
    def optimize_sharpe_ratio(self, risk_free_rate: float = 0.02) -> Tuple[np.ndarray, float]:
        """
        Optimize portfolio weights to maximize Sharpe ratio using Modern Portfolio Theory.
        
        Args:
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Tuple of (optimal weights, maximum Sharpe ratio)
        """
        n_assets = len(self.tickers)
        args = (self.mean_returns, self.cov_matrix, risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1/n_assets] * n_assets)
        
        def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate/252) / portfolio_volatility
            return -sharpe_ratio
        
        result = minimize(negative_sharpe_ratio, initial_weights, args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x, -result.fun
    
    def black_litterman(self, views: Dict[str, float], 
                       view_confidences: Dict[str, float],
                       tau: float = 0.05) -> np.ndarray:
        """
        Implement Black-Litterman model to incorporate investor views.
        
        Args:
            views: Dictionary of expected returns for specific assets
            view_confidences: Dictionary of confidence levels for each view
            tau: Scaling factor for the covariance matrix
            
        Returns:
            Adjusted expected returns incorporating views
        """
        # Create view matrix P and view vector Q
        P = np.zeros((len(views), len(self.tickers)))
        Q = np.zeros(len(views))
        Omega = np.zeros((len(views), len(views)))
        
        for i, (ticker, view) in enumerate(views.items()):
            idx = self.tickers.index(ticker)
            P[i, idx] = 1
            Q[i] = view
            Omega[i, i] = 1 / view_confidences[ticker]
        
        # Calculate equilibrium returns (implied returns)
        market_weights = np.ones(len(self.tickers)) / len(self.tickers)
        implied_returns = np.dot(self.cov_matrix, market_weights)
        
        # Calculate posterior returns
        M1 = np.linalg.inv(np.linalg.inv(tau * self.cov_matrix) + np.dot(P.T, np.dot(np.linalg.inv(Omega), P)))
        M2 = np.dot(np.linalg.inv(tau * self.cov_matrix), implied_returns) + np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
        posterior_returns = np.dot(M1, M2)
        
        return posterior_returns
    
    def risk_parity(self) -> np.ndarray:
        """
        Implement Risk Parity portfolio allocation.
        
        Returns:
            Optimal weights that equalize risk contributions
        """
        n_assets = len(self.tickers)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        def risk_contribution(weights):
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contributions = np.dot(self.cov_matrix, weights) / portfolio_volatility
            risk_contributions = weights * marginal_contributions
            return risk_contributions
        
        def objective(weights):
            risk_contributions = risk_contribution(weights)
            target_contributions = np.ones(n_assets) / n_assets
            return np.sum((risk_contributions - target_contributions) ** 2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        result = minimize(objective, initial_weights, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        return result.x
    
    def predict_returns(self, n_lags: int = 5) -> pd.Series:
        """
        Predict asset returns using Random Forest.
        
        Args:
            n_lags: Number of lagged returns to use as features
            
        Returns:
            Predicted returns for each asset
        """
        predictions = {}
        
        for ticker in self.tickers:
            # Create lagged features
            X = pd.DataFrame()
            for lag in range(1, n_lags + 1):
                X[f'lag_{lag}'] = self.returns[ticker].shift(lag)
            
            # Remove rows with NaN values
            X = X.dropna()
            y = self.returns[ticker].iloc[n_lags:]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # Make prediction using most recent data
            latest_features = X.iloc[-1].values.reshape(1, -1)
            latest_features_scaled = scaler.transform(latest_features)
            predictions[ticker] = model.predict(latest_features_scaled)[0]
        
        return pd.Series(predictions)
    
    def plot_efficient_frontier(self, risk_free_rate: float = 0.02, 
                              n_points: int = 50) -> None:
        """
        Plot the efficient frontier with color-coded Sharpe ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate
            n_points: Number of points to plot on the frontier
        """
        target_returns = np.linspace(self.mean_returns.min(), 
                                   self.mean_returns.max(), n_points)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        def portfolio_return(weights):
            return np.sum(self.mean_returns * weights)
        
        def negative_sharpe_ratio(weights):
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -(ret - risk_free_rate/252) / vol
        
        n_assets = len(self.tickers)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        volatilities = []
        returns = []
        sharpe_ratios = []
        
        for target_return in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return}
            )
            
            result = minimize(portfolio_volatility, np.ones(n_assets)/n_assets,
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                vol = portfolio_volatility(weights)
                ret = portfolio_return(weights)
                sharpe = (ret - risk_free_rate/252) / vol
                
                volatilities.append(vol)
                returns.append(ret)
                sharpe_ratios.append(sharpe)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(volatilities, returns, c=sharpe_ratios, 
                            cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.grid(True)
        plt.savefig('efficient_frontier.png')
        plt.close()
    
    def plot_portfolio_weights(self, weights: np.ndarray, 
                             title: str = 'Portfolio Weights') -> None:
        """
        Plot portfolio weights as a bar chart.
        
        Args:
            weights: Portfolio weights
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.bar(self.tickers, weights)
        plt.xlabel('Assets')
        plt.ylabel('Weight')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{title.lower().replace(" ", "_")}.png')
        plt.close()

def main():
    """Example usage of the PortfolioOptimizer class."""
    # Example tickers and date range
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(tickers, start_date, end_date)
    
    # Optimize for maximum Sharpe ratio
    weights_sharpe, max_sharpe = optimizer.optimize_sharpe_ratio()
    print(f"Maximum Sharpe Ratio: {max_sharpe:.4f}")
    print("Optimal Weights (Sharpe):")
    for ticker, weight in zip(tickers, weights_sharpe):
        print(f"{ticker}: {weight:.4f}")
    
    # Implement Black-Litterman model
    views = {'AAPL': 0.15, 'MSFT': 0.12}  # Example views
    view_confidences = {'AAPL': 0.5, 'MSFT': 0.5}  # Example confidences
    posterior_returns = optimizer.black_litterman(views, view_confidences)
    
    # Calculate Risk Parity weights
    weights_rp = optimizer.risk_parity()
    print("\nRisk Parity Weights:")
    for ticker, weight in zip(tickers, weights_rp):
        print(f"{ticker}: {weight:.4f}")
    
    # Predict returns using ML
    predicted_returns = optimizer.predict_returns()
    print("\nPredicted Returns:")
    print(predicted_returns)
    
    # Generate plots
    optimizer.plot_efficient_frontier()
    optimizer.plot_portfolio_weights(weights_sharpe, 'Sharpe Ratio Optimal Weights')
    optimizer.plot_portfolio_weights(weights_rp, 'Risk Parity Weights')

if __name__ == "__main__":
    main() 
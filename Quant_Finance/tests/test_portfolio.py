import unittest
import numpy as np
import pandas as pd
from portfolio_toolkit import PortfolioOptimizer

class TestPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.tickers = ['AAPL', 'MSFT', 'GOOGL']
        self.start_date = '2020-01-01'
        self.end_date = '2020-12-31'
        self.optimizer = PortfolioOptimizer(self.tickers, self.start_date, self.end_date)
    
    def test_data_fetching(self):
        """Test if data is fetched correctly."""
        self.assertIsInstance(self.optimizer.data, pd.DataFrame)
        self.assertEqual(len(self.optimizer.data.columns), len(self.tickers))
        self.assertFalse(self.optimizer.data.empty)
    
    def test_returns_calculation(self):
        """Test if returns are calculated correctly."""
        self.assertIsInstance(self.optimizer.returns, pd.DataFrame)
        self.assertEqual(len(self.optimizer.returns.columns), len(self.tickers))
        self.assertFalse(self.optimizer.returns.empty)
        # Check if returns are within reasonable bounds
        self.assertTrue(self.optimizer.returns.max().max() < 1.0)
        self.assertTrue(self.optimizer.returns.min().min() > -1.0)
    
    def test_covariance_matrix(self):
        """Test if covariance matrix is calculated correctly."""
        self.assertIsInstance(self.optimizer.cov_matrix, pd.DataFrame)
        self.assertEqual(self.optimizer.cov_matrix.shape, (len(self.tickers), len(self.tickers)))
        # Check if matrix is symmetric
        self.assertTrue(np.allclose(self.optimizer.cov_matrix, self.optimizer.cov_matrix.T))
        # Check if diagonal elements are positive
        self.assertTrue(np.all(np.diag(self.optimizer.cov_matrix) > 0))
    
    def test_sharpe_optimization(self):
        """Test Sharpe ratio optimization."""
        weights, max_sharpe = self.optimizer.optimize_sharpe_ratio()
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(len(weights), len(self.tickers))
        # Check if weights sum to 1
        self.assertAlmostEqual(np.sum(weights), 1.0)
        # Check if weights are between 0 and 1
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 1))
        # Check if Sharpe ratio is reasonable
        self.assertGreater(max_sharpe, 0)
    
    def test_black_litterman(self):
        """Test Black-Litterman model."""
        views = {'AAPL': 0.15, 'MSFT': 0.12}
        view_confidences = {'AAPL': 0.5, 'MSFT': 0.5}
        posterior_returns = self.optimizer.black_litterman(views, view_confidences)
        self.assertIsInstance(posterior_returns, np.ndarray)
        self.assertEqual(len(posterior_returns), len(self.tickers))
        # Check if returns are reasonable
        self.assertTrue(np.all(posterior_returns > -1.0))
        self.assertTrue(np.all(posterior_returns < 1.0))
    
    def test_risk_parity(self):
        """Test Risk Parity allocation."""
        weights = self.optimizer.risk_parity()
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(len(weights), len(self.tickers))
        # Check if weights sum to 1
        self.assertAlmostEqual(np.sum(weights), 1.0)
        # Check if weights are between 0 and 1
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 1))
    
    def test_return_prediction(self):
        """Test return prediction using Random Forest."""
        predicted_returns = self.optimizer.predict_returns()
        self.assertIsInstance(predicted_returns, pd.Series)
        self.assertEqual(len(predicted_returns), len(self.tickers))
        # Check if predictions are reasonable
        self.assertTrue(np.all(predicted_returns > -1.0))
        self.assertTrue(np.all(predicted_returns < 1.0))

if __name__ == '__main__':
    unittest.main() 
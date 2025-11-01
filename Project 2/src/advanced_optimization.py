"""
Advanced Portfolio Optimization Module for Questions 2(f-g)
Implements utility maximization with log-normal assets and non-PSD covariance handling
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, lognorm
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class UtilityOptimizer:
    """Utility maximization for portfolio optimization (Question 2f)"""

    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                 risk_free_rate: float = 0.02):
        """
        Initialize utility optimizer

        Args:
            expected_returns: Expected annual returns
            cov_matrix: Annualized covariance matrix
            risk_free_rate: Annual risk-free rate
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        self.asset_names = expected_returns.index.tolist()

    def exponential_utility(self, wealth: float, gamma: float = 1) -> float:
        """
        Exponential utility function U(x) = -e^(-γx)

        Args:
            wealth: Terminal wealth
            gamma: Risk aversion parameter

        Returns:
            Utility value
        """
        return -np.exp(-gamma * wealth)

    def optimize_exponential_utility_lognormal(self, gamma: float = 1,
                                              constraints: Optional[Dict] = None) -> Dict:
        """
        Maximize expected exponential utility assuming log-normal returns (Question 2f)

        For log-normal returns and exponential utility, we have an analytical solution:
        E[U(W)] = E[-exp(-γ * w'μ_T)] where returns are log-normal

        Args:
            gamma: Risk aversion parameter
            constraints: Optional constraints dictionary

        Returns:
            Optimization results
        """
        # Convert to log-space parameters
        # If R ~ LogN(μ_log, σ²_log), then E[R] = exp(μ_log + σ²_log/2)
        # and Var[R] = (exp(σ²_log) - 1) * exp(2*μ_log + σ²_log)

        # Approximate log-normal parameters from moments (simplified approach)
        # This assumes returns are approximately log-normal
        annual_returns = self.expected_returns.values
        annual_vols = np.sqrt(np.diag(self.cov_matrix.values))

        # Convert to log-normal parameters (approximation)
        log_means = np.log(1 + annual_returns) - 0.5 * annual_vols**2
        log_vols = np.sqrt(np.log(1 + (annual_vols / (1 + annual_returns))**2))

        # For exponential utility with log-normal returns, the expected utility is:
        # E[U] = -exp(-γ * w'μ + 0.5 * γ² * w'Σw)

        def objective(weights):
            """Negative expected utility (to minimize)"""
            portfolio_log_mean = np.dot(weights, log_means)
            portfolio_log_var = np.dot(weights, np.dot(self.cov_matrix.values, weights))

            # Expected utility for exponential utility with log-normal returns
            exp_utility = -np.exp(-gamma * portfolio_log_mean + 0.5 * gamma**2 * portfolio_log_var)

            return -exp_utility  # Negative because we're minimizing

        # Constraints
        constraints_list = []

        # Sum to 1
        constraints_list.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        })

        # Additional constraints if provided
        if constraints:
            if 'min_return' in constraints:
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda w: np.dot(w, annual_returns) - constraints['min_return']
                })

        # Bounds (no short selling)
        bounds = [(0, 0.4) for _ in range(self.n_assets)]

        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )

        # Calculate portfolio metrics
        optimal_weights = result.x
        portfolio_return = np.dot(optimal_weights, annual_returns)
        portfolio_vol = np.sqrt(np.dot(optimal_weights, np.dot(self.cov_matrix.values, optimal_weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol

        # Calculate utility
        portfolio_log_mean = np.dot(optimal_weights, log_means)
        portfolio_log_var = np.dot(optimal_weights, np.dot(self.cov_matrix.values, optimal_weights))
        expected_utility = -np.exp(-gamma * portfolio_log_mean + 0.5 * gamma**2 * portfolio_log_var)

        return {
            'success': result.success,
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'expected_utility': expected_utility,
            'gamma': gamma,
            'method': 'log-normal_exponential_utility'
        }

    def compare_with_mean_variance(self, mv_weights: np.ndarray, gamma: float = 1) -> Dict:
        """
        Compare utility-optimized portfolio with mean-variance portfolio

        Args:
            mv_weights: Mean-variance optimal weights
            gamma: Risk aversion parameter

        Returns:
            Comparison results
        """
        # Get utility-optimal portfolio
        utility_result = self.optimize_exponential_utility_lognormal(gamma)

        # Calculate metrics for both portfolios
        comparison = {}

        for name, weights in [('Utility-Optimal', utility_result['weights']),
                             ('Mean-Variance', mv_weights)]:
            portfolio_return = np.dot(weights, self.expected_returns.values)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix.values, weights)))

            # Calculate utility (simplified)
            utility_value = self.exponential_utility(portfolio_return, gamma)

            comparison[name] = {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_vol,
                'utility': utility_value,
                'max_weight': np.max(weights),
                'min_weight': np.min(weights[weights > 0.001]),
                'n_assets': np.sum(weights > 0.001)
            }

        return comparison


class CovarianceMatrixCorrector:
    """Handle non-positive semidefinite covariance matrices (Question 2g)"""

    def __init__(self, cov_matrix: pd.DataFrame):
        """
        Initialize covariance matrix corrector

        Args:
            cov_matrix: Original covariance matrix
        """
        self.original_cov = cov_matrix.copy()
        self.asset_names = cov_matrix.index.tolist()
        self.n_assets = len(self.asset_names)

    def create_non_psd_matrix(self, asset1: str, asset2: str,
                             new_correlation: float) -> pd.DataFrame:
        """
        Create a non-PSD covariance matrix by modifying correlations

        Args:
            asset1: First asset name
            asset2: Second asset name
            new_correlation: New correlation value (potentially inconsistent)

        Returns:
            Modified covariance matrix (potentially non-PSD)
        """
        # Convert covariance to correlation
        std_devs = np.sqrt(np.diag(self.original_cov.values))
        corr_matrix = self.original_cov.values / np.outer(std_devs, std_devs)

        # Modify correlation
        idx1 = self.asset_names.index(asset1)
        idx2 = self.asset_names.index(asset2)

        corr_matrix[idx1, idx2] = new_correlation
        corr_matrix[idx2, idx1] = new_correlation

        # Create an inconsistent correlation structure
        # For example, if A highly correlated with B, B highly correlated with C,
        # but A negatively correlated with C (violating transitivity)
        if self.n_assets >= 3:
            # Find a third asset
            idx3 = (idx1 + 2) % self.n_assets
            if idx3 == idx2:
                idx3 = (idx1 + 3) % self.n_assets

            # Set inconsistent correlations
            corr_matrix[idx1, idx3] = -0.8  # Strong negative
            corr_matrix[idx3, idx1] = -0.8
            corr_matrix[idx2, idx3] = 0.9   # Strong positive
            corr_matrix[idx3, idx2] = 0.9

        # Convert back to covariance
        modified_cov = pd.DataFrame(
            corr_matrix * np.outer(std_devs, std_devs),
            index=self.asset_names,
            columns=self.asset_names
        )

        return modified_cov

    def check_positive_semidefinite(self, matrix: pd.DataFrame) -> Tuple[bool, np.ndarray]:
        """
        Check if matrix is positive semidefinite

        Args:
            matrix: Covariance matrix to check

        Returns:
            Tuple of (is_psd, eigenvalues)
        """
        eigenvalues = np.linalg.eigvalsh(matrix.values)
        is_psd = np.all(eigenvalues >= -1e-10)  # Small tolerance for numerical errors

        return is_psd, eigenvalues

    def eigenvalue_clipping(self, matrix: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
        """
        Fix non-PSD matrix using eigenvalue clipping

        Args:
            matrix: Non-PSD covariance matrix
            epsilon: Minimum eigenvalue threshold

        Returns:
            Corrected PSD matrix
        """
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix.values)

        # Clip negative eigenvalues
        eigenvalues_clipped = np.maximum(eigenvalues, epsilon)

        # Reconstruct matrix
        matrix_psd = eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T

        # Ensure symmetry
        matrix_psd = (matrix_psd + matrix_psd.T) / 2

        return pd.DataFrame(matrix_psd, index=matrix.index, columns=matrix.columns)

    def nearest_correlation_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Find nearest correlation matrix using Higham's algorithm (simplified)

        Args:
            matrix: Non-PSD covariance matrix

        Returns:
            Nearest PSD correlation matrix converted back to covariance
        """
        # Extract standard deviations
        std_devs = np.sqrt(np.diag(matrix.values))

        # Convert to correlation
        corr_matrix = matrix.values / np.outer(std_devs, std_devs)

        # Simplified Higham algorithm (iterative projection)
        max_iterations = 100
        tolerance = 1e-6

        X = corr_matrix.copy()
        for _ in range(max_iterations):
            # Project onto PSD matrices
            eigenvalues, eigenvectors = np.linalg.eigh(X)
            eigenvalues = np.maximum(eigenvalues, 0)
            X_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Project onto matrices with unit diagonal
            np.fill_diagonal(X_psd, 1.0)

            # Check convergence
            if np.linalg.norm(X_psd - X) < tolerance:
                break

            X = X_psd

        # Convert back to covariance
        cov_psd = pd.DataFrame(
            X_psd * np.outer(std_devs, std_devs),
            index=matrix.index,
            columns=matrix.columns
        )

        return cov_psd

    def shrinkage_to_diagonal(self, matrix: pd.DataFrame, shrinkage: float = 0.1) -> pd.DataFrame:
        """
        Shrink matrix toward diagonal to ensure PSD

        Args:
            matrix: Potentially non-PSD covariance matrix
            shrinkage: Shrinkage intensity (0 to 1)

        Returns:
            Shrunk PSD matrix
        """
        # Target is diagonal matrix with original variances
        target = np.diag(np.diag(matrix.values))

        # Shrink toward target
        shrunk_matrix = shrinkage * target + (1 - shrinkage) * matrix.values

        return pd.DataFrame(shrunk_matrix, index=matrix.index, columns=matrix.columns)

    def compare_correction_methods(self, non_psd_matrix: pd.DataFrame) -> Dict:
        """
        Compare different correction methods

        Args:
            non_psd_matrix: Non-PSD covariance matrix

        Returns:
            Comparison of correction methods
        """
        methods = {
            'eigenvalue_clipping': self.eigenvalue_clipping(non_psd_matrix),
            'nearest_correlation': self.nearest_correlation_matrix(non_psd_matrix),
            'shrinkage_0.1': self.shrinkage_to_diagonal(non_psd_matrix, 0.1),
            'shrinkage_0.2': self.shrinkage_to_diagonal(non_psd_matrix, 0.2)
        }

        comparison = {}
        original_is_psd, original_eigenvalues = self.check_positive_semidefinite(non_psd_matrix)

        comparison['original'] = {
            'is_psd': original_is_psd,
            'min_eigenvalue': np.min(original_eigenvalues),
            'max_eigenvalue': np.max(original_eigenvalues),
            'condition_number': np.max(np.abs(original_eigenvalues)) / np.max([np.min(np.abs(original_eigenvalues)), 1e-10]),
            'frobenius_norm': np.linalg.norm(non_psd_matrix.values, 'fro')
        }

        for method_name, corrected_matrix in methods.items():
            is_psd, eigenvalues = self.check_positive_semidefinite(corrected_matrix)

            # Calculate distance from original
            distance = np.linalg.norm(corrected_matrix.values - non_psd_matrix.values, 'fro')

            comparison[method_name] = {
                'is_psd': is_psd,
                'min_eigenvalue': np.min(eigenvalues),
                'max_eigenvalue': np.max(eigenvalues),
                'condition_number': np.max(eigenvalues) / np.max([np.min(eigenvalues), 1e-10]),
                'distance_from_original': distance,
                'matrix': corrected_matrix
            }

        return comparison


def run_advanced_optimization():
    """Run advanced optimization for Questions 2(f-g)"""
    from data_loader import AssetDataLoader
    from parameter_estimation import ParameterEstimator
    from static_optimization import StaticPortfolioOptimizer

    print("="*60)
    print("ADVANCED PORTFOLIO OPTIMIZATION (Questions 2f-g)")
    print("="*60)

    # Load data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data',
        'BBG Data (2000-2025).xlsx'
    )
    loader = AssetDataLoader(data_path)
    returns_data = loader.load_data()

    # Estimate parameters
    estimator = ParameterEstimator(returns_data)
    expected_returns = estimator.estimate_expected_returns('combined')
    cov_matrix = estimator.estimate_covariance_matrix('shrinkage')

    # Question 2(f): Utility Maximization with Log-Normal Assets
    print("\n" + "="*60)
    print("QUESTION 2(f): UTILITY MAXIMIZATION WITH LOG-NORMAL ASSETS")
    print("="*60)

    utility_optimizer = UtilityOptimizer(expected_returns, cov_matrix)

    # Get mean-variance optimal portfolio for comparison
    static_optimizer = StaticPortfolioOptimizer(
        expected_returns,
        cov_matrix,
        returns_data=returns_data
    )
    mv_result = static_optimizer.optimize_portfolio(target_return=0.05594, growth_allocation=0.7)

    # Compare utility-optimal with mean-variance
    comparison = utility_optimizer.compare_with_mean_variance(mv_result['weights'], gamma=1)

    print("\nPortfolio Comparison:")
    print("-"*60)
    for portfolio_name, metrics in comparison.items():
        print(f"\n{portfolio_name}:")
        print(f"  Expected Return: {metrics['expected_return']:.2%}")
        print(f"  Volatility: {metrics['volatility']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Utility: {metrics['utility']:.6f}")
        print(f"  Max Weight: {metrics['max_weight']:.2%}")
        print(f"  Active Assets: {metrics['n_assets']}")

    # Question 2(g): Non-PSD Covariance Matrix Handling
    print("\n" + "="*60)
    print("QUESTION 2(g): NON-PSD COVARIANCE MATRIX HANDLING")
    print("="*60)

    corrector = CovarianceMatrixCorrector(cov_matrix)

    # Create a non-PSD matrix
    asset1 = expected_returns.index[0]
    asset2 = expected_returns.index[1]
    non_psd_cov = corrector.create_non_psd_matrix(asset1, asset2, 0.95)

    # Check if it's non-PSD
    is_psd, eigenvalues = corrector.check_positive_semidefinite(non_psd_cov)
    print(f"\nCreated matrix is PSD: {is_psd}")
    print(f"Minimum eigenvalue: {np.min(eigenvalues):.6f}")

    if not is_psd:
        print("\nComparing correction methods:")
        print("-"*60)

        comparison = corrector.compare_correction_methods(non_psd_cov)

        for method, metrics in comparison.items():
            print(f"\n{method}:")
            print(f"  Is PSD: {metrics['is_psd']}")
            print(f"  Min eigenvalue: {metrics['min_eigenvalue']:.6f}")
            print(f"  Condition number: {metrics['condition_number']:.2f}")
            if method != 'original':
                print(f"  Distance from original: {metrics.get('distance_from_original', 0):.6f}")

        # Test portfolio optimization with corrected matrix
        print("\nPortfolio Impact Analysis:")
        print("-"*60)

        # Original (if PSD) or best corrected
        best_method = 'eigenvalue_clipping'
        corrected_cov = comparison[best_method]['matrix']

        # Optimize with corrected matrix
        corrected_optimizer = StaticPortfolioOptimizer(
            expected_returns,
            corrected_cov,
            returns_data=returns_data
        )
        corrected_result = corrected_optimizer.optimize_portfolio(
            target_return=0.05594,
            growth_allocation=0.7
        )

        print(f"\nUsing {best_method} correction:")
        print(f"  Expected Return: {corrected_result['metrics']['return']:.2%}")
        print(f"  Volatility: {corrected_result['metrics']['volatility']:.2%}")
        print(f"  Sharpe Ratio: {corrected_result['metrics']['sharpe_ratio']:.3f}")

    return {
        'utility_comparison': comparison,
        'covariance_correction': comparison if not is_psd else None
    }


if __name__ == "__main__":
    run_advanced_optimization()

"""
Parameter Estimation Module for Question 1
Implements various methods for estimating expected returns, volatilities, and covariances
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class ParameterEstimator:
    """Advanced parameter estimation for portfolio optimization"""

    def __init__(self, returns_data: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize parameter estimator

        Args:
            returns_data: DataFrame of monthly returns
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns_data.columns)
        self.asset_names = returns_data.columns.tolist()

        # CPI and target returns from requirements
        self.cpi = 0.02594  # 2.594% from notes
        self.target_return = self.cpi + 0.03  # CPI + 3%

    def estimate_expected_returns(self, method: str = 'combined',
                                 lookback_periods: Optional[int] = None) -> pd.Series:
        """
        Estimate expected returns using various methods

        Args:
            method: Estimation method ('historical', 'ewma', 'shrinkage', 'combined')
            lookback_periods: Number of months to use for estimation

        Returns:
            Series of annualized expected returns
        """
        data = self.returns_data
        if lookback_periods:
            data = self.returns_data.iloc[-lookback_periods:]

        if method == 'historical':
            # Arithmetic mean (for use in mean-variance optimization)
            monthly_returns = data.mean()
            annual_returns = monthly_returns * 12

        elif method == 'ewma':
            # Exponentially weighted moving average
            lambda_param = 0.94
            annual_returns = self._ewma_returns(data, lambda_param)

        elif method == 'shrinkage':
            # James-Stein shrinkage estimator
            annual_returns = self._shrinkage_returns(data)

        elif method == 'combined':
            # Weighted combination of methods
            hist = self.estimate_expected_returns('historical', lookback_periods)
            ewma = self.estimate_expected_returns('ewma', lookback_periods)
            shrink = self.estimate_expected_returns('shrinkage', lookback_periods)

            # Weights based on historical performance
            weights = [0.3, 0.4, 0.3]  # hist, ewma, shrink
            annual_returns = weights[0] * hist + weights[1] * ewma + weights[2] * shrink

        elif method == 'capm_adjusted':
            # Adjust historical returns using CAPM framework
            annual_returns = self._capm_adjusted_returns(data)

        else:
            raise ValueError(f"Unknown method: {method}")

        return annual_returns

    def _ewma_returns(self, data: pd.DataFrame, lambda_param: float) -> pd.Series:
        """Calculate EWMA returns"""
        n_periods = len(data)
        weights = np.array([(1-lambda_param) * lambda_param**i
                          for i in range(n_periods-1, -1, -1)])
        weights = weights / weights.sum()

        annual_returns = pd.Series(index=data.columns, dtype=float)
        for asset in data.columns:
            monthly_return = np.sum(data[asset].values * weights)
            annual_returns[asset] = monthly_return * 12

        return annual_returns

    def _shrinkage_returns(self, data: pd.DataFrame) -> pd.Series:
        """James-Stein shrinkage estimator"""
        monthly_means = data.mean()
        grand_mean = monthly_means.mean()
        n = len(data)

        # Calculate optimal shrinkage intensity
        variance_of_means = monthly_means.var()
        mean_of_variances = data.var().mean()

        if variance_of_means > 0:
            shrinkage_intensity = min(1, mean_of_variances / (n * variance_of_means))
        else:
            shrinkage_intensity = 1

        # Shrink toward grand mean
        shrunk_means = shrinkage_intensity * grand_mean + (1 - shrinkage_intensity) * monthly_means

        return shrunk_means * 12

    def _capm_adjusted_returns(self, data: pd.DataFrame) -> pd.Series:
        """Adjust returns using CAPM framework"""
        # Use Australian Equity as market proxy
        market_asset = 'Australian Listed Equity [G]'
        if market_asset not in data.columns:
            # Fallback to simple historical
            return data.mean() * 12

        market_returns = data[market_asset]
        market_premium = market_returns.mean() * 12 - self.risk_free_rate

        annual_returns = pd.Series(index=data.columns, dtype=float)

        for asset in data.columns:
            if asset == market_asset:
                annual_returns[asset] = market_returns.mean() * 12
            else:
                # Calculate beta
                covariance = np.cov(data[asset].values, market_returns.values)[0, 1]
                market_var = market_returns.var()
                beta = covariance / market_var if market_var > 0 else 1

                # CAPM expected return
                annual_returns[asset] = self.risk_free_rate + beta * market_premium

        return annual_returns

    def estimate_covariance_matrix(self, method: str = 'shrinkage',
                                  lookback_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Estimate covariance matrix using various methods

        Args:
            method: Estimation method ('sample', 'shrinkage', 'ewma')
            lookback_periods: Number of months to use

        Returns:
            Annualized covariance matrix
        """
        data = self.returns_data
        if lookback_periods:
            data = self.returns_data.iloc[-lookback_periods:]

        if method == 'sample':
            # Simple sample covariance
            cov_matrix = data.cov() * 12

        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage
            cov_matrix = self._ledoit_wolf_covariance(data)

        elif method == 'ewma':
            # Exponentially weighted covariance
            cov_matrix = self._ewma_covariance(data)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Ensure positive semi-definite
        cov_matrix = self._ensure_psd(cov_matrix)

        return cov_matrix

    def _ledoit_wolf_covariance(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage estimator for covariance matrix
        Shrinks sample covariance toward diagonal matrix
        """
        n, p = data.shape
        data_array = data.values

        # De-mean the data
        mean_vec = np.mean(data_array, axis=0, keepdims=True)
        data_centered = data_array - mean_vec

        # Sample covariance
        sample_cov = np.dot(data_centered.T, data_centered) / (n - 1)

        # Shrinkage target (diagonal with sample variances)
        target = np.diag(np.diag(sample_cov))

        # Calculate optimal shrinkage intensity (simplified Ledoit-Wolf)
        # This is a simplified version - full implementation would be more complex
        shrinkage = 0.1  # Fixed shrinkage for simplicity

        # Shrink covariance
        shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov

        # Convert to DataFrame and annualize
        cov_df = pd.DataFrame(shrunk_cov * 12,
                            index=data.columns,
                            columns=data.columns)

        return cov_df

    def _ewma_covariance(self, data: pd.DataFrame, lambda_param: float = 0.94) -> pd.DataFrame:
        """Exponentially weighted moving average covariance"""
        n_periods = len(data)

        # Calculate weights
        weights = np.array([lambda_param**i for i in range(n_periods-1, -1, -1)])
        weights = weights / weights.sum()

        # Weighted mean
        weighted_mean = np.average(data.values, axis=0, weights=weights)

        # Weighted covariance
        data_centered = data.values - weighted_mean
        weighted_cov = np.zeros((self.n_assets, self.n_assets))

        for i in range(n_periods):
            deviation = data_centered[i:i+1].T
            weighted_cov += weights[i] * np.dot(deviation, deviation.T)

        # Convert to DataFrame and annualize
        cov_df = pd.DataFrame(weighted_cov * 12,
                            index=data.columns,
                            columns=data.columns)

        return cov_df

    def _ensure_psd(self, matrix: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
        """
        Ensure matrix is positive semi-definite
        Uses eigenvalue clipping method
        """
        # Convert to numpy array
        mat = matrix.values

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(mat)

        # Clip negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, epsilon)

        # Reconstruct matrix
        mat_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Ensure symmetry
        mat_psd = (mat_psd + mat_psd.T) / 2

        return pd.DataFrame(mat_psd, index=matrix.index, columns=matrix.columns)

    def calculate_risk_metrics(self, returns: pd.Series, cov_matrix: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive risk metrics for the assets

        Args:
            returns: Expected returns
            cov_matrix: Covariance matrix

        Returns:
            Dictionary of risk metrics
        """
        volatilities = np.sqrt(np.diag(cov_matrix.values))
        correlations = cov_matrix.values / np.outer(volatilities, volatilities)

        metrics = {
            'expected_returns': returns.to_dict(),
            'volatilities': dict(zip(cov_matrix.columns, volatilities)),
            'sharpe_ratios': dict(zip(cov_matrix.columns,
                                     (returns.values - self.risk_free_rate) / volatilities)),
            'avg_correlation': np.mean(correlations[np.triu_indices_from(correlations, k=1)]),
            'max_correlation': np.max(correlations[np.triu_indices_from(correlations, k=1)]),
            'min_correlation': np.min(correlations[np.triu_indices_from(correlations, k=1)])
        }

        return metrics

    def perform_sensitivity_analysis(self, base_returns: pd.Series,
                                    base_cov: pd.DataFrame) -> Dict:
        """
        Perform sensitivity analysis on parameter estimates

        Args:
            base_returns: Baseline expected returns
            base_cov: Baseline covariance matrix

        Returns:
            Dictionary with sensitivity results
        """
        results = {}

        # Test different lookback periods
        lookback_tests = [36, 60, 120, None]  # 3, 5, 10 years, all data
        for period in lookback_tests:
            period_label = f"{period}_months" if period else "all_data"
            results[period_label] = {
                'returns': self.estimate_expected_returns('combined', period),
                'volatility': np.sqrt(np.diag(
                    self.estimate_covariance_matrix('shrinkage', period).values))
            }

        # Test different estimation methods
        methods = ['historical', 'ewma', 'shrinkage']
        for method in methods:
            results[f"method_{method}"] = {
                'returns': self.estimate_expected_returns(method),
                'sharpe': (self.estimate_expected_returns(method).values - self.risk_free_rate) /
                         np.sqrt(np.diag(base_cov.values))
            }

        return results

    def generate_parameter_report(self) -> Dict:
        """
        Generate comprehensive parameter estimation report for Question 1

        Returns:
            Dictionary containing all parameter estimates and analysis
        """
        report = {}

        # Expected returns using different methods
        report['expected_returns'] = {
            'historical': self.estimate_expected_returns('historical'),
            'ewma': self.estimate_expected_returns('ewma'),
            'shrinkage': self.estimate_expected_returns('shrinkage'),
            'combined': self.estimate_expected_returns('combined'),
            'recommended': self.estimate_expected_returns('combined')  # Our choice
        }

        # Covariance matrices
        report['covariance_matrices'] = {
            'sample': self.estimate_covariance_matrix('sample'),
            'shrinkage': self.estimate_covariance_matrix('shrinkage'),
            'recommended': self.estimate_covariance_matrix('shrinkage')  # Our choice
        }

        # Risk metrics
        recommended_returns = report['expected_returns']['recommended']
        recommended_cov = report['covariance_matrices']['recommended']
        report['risk_metrics'] = self.calculate_risk_metrics(recommended_returns, recommended_cov)

        # Sensitivity analysis
        report['sensitivity'] = self.perform_sensitivity_analysis(
            recommended_returns, recommended_cov)

        # Target return feasibility
        report['target_return_analysis'] = {
            'target_return': self.target_return,
            'cpi': self.cpi,
            'risk_free_rate': self.risk_free_rate,
            'max_asset_return': recommended_returns.max(),
            'min_asset_return': recommended_returns.min(),
            'feasible': recommended_returns.max() >= self.target_return
        }

        return report


def run_parameter_estimation():
    """Run complete parameter estimation for Question 1"""
    from data_loader import AssetDataLoader

    print("="*60)
    print("QUESTION 1: PARAMETER ESTIMATION")
    print("="*60)

    # Load data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data',
        'BBG Data (2000-2025).xlsx'
    )
    loader = AssetDataLoader(data_path)
    returns_data = loader.load_data()

    # Initialize estimator
    estimator = ParameterEstimator(returns_data)

    # Generate full report
    report = estimator.generate_parameter_report()

    # Print results
    print("\n1. EXPECTED RETURNS COMPARISON")
    print("-"*60)
    methods = ['historical', 'ewma', 'shrinkage', 'combined']
    print(f"{'Asset':40} " + " ".join([f"{m:>10}" for m in methods]))

    for asset in report['expected_returns']['historical'].index:
        values = [report['expected_returns'][m][asset] for m in methods]
        print(f"{asset[:40]:40} " + " ".join([f"{v:10.2%}" for v in values]))

    print("\n2. RECOMMENDED PARAMETERS (Combined Method with Shrinkage Covariance)")
    print("-"*60)
    rec_returns = report['expected_returns']['recommended']
    rec_cov = report['covariance_matrices']['recommended']

    print("\nExpected Annual Returns:")
    for asset, ret in rec_returns.items():
        vol = np.sqrt(rec_cov.loc[asset, asset])
        sharpe = (ret - estimator.risk_free_rate) / vol
        print(f"{asset[:40]:40} Return: {ret:7.2%}  Vol: {vol:7.2%}  Sharpe: {sharpe:6.3f}")

    print("\n3. CORRELATION MATRIX")
    print("-"*60)
    corr_matrix = loader.get_correlation_matrix()
    print("\nAverage Correlation:", report['risk_metrics']['avg_correlation'])
    print("Max Correlation:", report['risk_metrics']['max_correlation'])
    print("Min Correlation:", report['risk_metrics']['min_correlation'])

    print("\n4. TARGET RETURN FEASIBILITY")
    print("-"*60)
    target_info = report['target_return_analysis']
    print(f"CPI: {target_info['cpi']:.2%}")
    print(f"Target Return (CPI + 3%): {target_info['target_return']:.2%}")
    print(f"Maximum Asset Return: {target_info['max_asset_return']:.2%}")
    print(f"Minimum Asset Return: {target_info['min_asset_return']:.2%}")
    print(f"Target Feasible: {'Yes' if target_info['feasible'] else 'No'}")

    return report


if __name__ == "__main__":
    run_parameter_estimation()

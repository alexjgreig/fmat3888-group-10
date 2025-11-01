"""
Dynamic Portfolio Optimization Module for Questions 2(h-k)
Implements dynamic rebalancing, risk attribution, and multi-period optimization
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class RiskManager:
    """Advanced risk management and attribution (Question 2h)"""

    def __init__(self, weights: np.ndarray, expected_returns: pd.Series,
                 cov_matrix: pd.DataFrame):
        """
        Initialize risk manager

        Args:
            weights: Portfolio weights
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
        """
        self.weights = weights
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.asset_names = expected_returns.index.tolist()
        self.n_assets = len(weights)

    def calculate_risk_attribution(self) -> pd.DataFrame:
        """
        Calculate risk contribution from each asset

        Returns:
            DataFrame with risk attribution metrics
        """
        portfolio_variance = np.dot(self.weights, np.dot(self.cov_matrix.values, self.weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Marginal contributions to risk
        marginal_risk = np.dot(self.cov_matrix.values, self.weights) / portfolio_std

        # Component contributions to risk (marginal * weight)
        component_risk = marginal_risk * self.weights

        # Percentage contributions
        pct_risk_contribution = component_risk / portfolio_std * 100

        # Create attribution dataframe
        attribution_df = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': self.weights,
            'Marginal Risk': marginal_risk,
            'Component Risk': component_risk,
            'Risk Contribution (%)': pct_risk_contribution,
            'Expected Return': self.expected_returns.values
        })

        return attribution_df

    def calculate_var_cvar(self, confidence_level: float = 0.95,
                           n_simulations: int = 10000) -> Dict:
        """
        Calculate Value at Risk and Conditional Value at Risk

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary with VaR and CVaR metrics
        """
        # Portfolio parameters
        portfolio_return = np.dot(self.weights, self.expected_returns.values)
        portfolio_variance = np.dot(self.weights, np.dot(self.cov_matrix.values, self.weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Monte Carlo simulation (assuming normal distribution)
        np.random.seed(42)
        simulated_returns = np.random.normal(portfolio_return, portfolio_std, n_simulations)

        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var = -np.percentile(simulated_returns, var_percentile)

        # Calculate CVaR (expected shortfall)
        losses = -simulated_returns
        cvar = np.mean(losses[losses > var])

        return {
            'VaR': var,
            'CVaR': cvar,
            'confidence_level': confidence_level,
            'expected_return': portfolio_return,
            'volatility': portfolio_std
        }

    def implement_risk_budgeting(self, risk_budget: Dict[str, float]) -> np.ndarray:
        """
        Implement risk budgeting strategy

        Args:
            risk_budget: Dictionary mapping asset names to risk budgets (sum to 1)

        Returns:
            Optimized weights matching risk budget
        """
        risk_budgets = np.array([risk_budget.get(asset, 0) for asset in self.asset_names])

        def objective(weights):
            """Minimize deviation from risk budget"""
            portfolio_std = np.sqrt(np.dot(weights, np.dot(self.cov_matrix.values, weights)))
            marginal_risk = np.dot(self.cov_matrix.values, weights) / portfolio_std
            risk_contributions = weights * marginal_risk / portfolio_std

            # Squared deviation from target risk budget
            deviation = np.sum((risk_contributions - risk_budgets)**2)
            return deviation

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
        ]

        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(self.n_assets)]

        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        return result.x


class DynamicPortfolioOptimizer:
    """Dynamic portfolio optimization with rebalancing (Questions 2i-k)"""

    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                 returns_data: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize dynamic optimizer

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            returns_data: Historical returns data for backtesting
            risk_free_rate: Risk-free rate
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate
        self.asset_names = expected_returns.index.tolist()
        self.n_assets = len(expected_returns)
        self.periods_per_year = 4  # Quarterly modelling by default

    def simulate_multiperiod_returns(self, n_periods: int = 12,
                                    n_scenarios: int = 1000) -> np.ndarray:
        """
        Simulate multi-period returns for dynamic optimization

        Args:
            n_periods: Number of periods (e.g., quarters)
            n_scenarios: Number of scenarios to simulate

        Returns:
            Array of simulated returns (scenarios x periods x assets)
        """
        # Quarterly parameters (assuming periods are quarters)
        quarterly_returns = self.expected_returns.values / 4
        quarterly_cov = self.cov_matrix.values / 4

        # Cholesky decomposition for correlated returns
        try:
            L = np.linalg.cholesky(quarterly_cov)
        except:
            # If not positive definite, use eigenvalue method
            eigenvalues, eigenvectors = np.linalg.eigh(quarterly_cov)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            L = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        # Simulate returns
        np.random.seed(42)
        simulated_returns = np.zeros((n_scenarios, n_periods, self.n_assets))

        for scenario in range(n_scenarios):
            for period in range(n_periods):
                z = np.random.normal(0, 1, self.n_assets)
                simulated_returns[scenario, period, :] = quarterly_returns + L @ z

        return simulated_returns

    def optimize_dynamic_portfolio(self, n_periods: int = 12,
                                  rebalance_frequency: int = 3,
                                  transaction_cost: float = 0.001,
                                  gamma: float = 1) -> Dict:
        """
        Optimize dynamic portfolio with periodic rebalancing (Questions 2i-j)

        Args:
            n_periods: Total number of periods
            rebalance_frequency: Rebalance every N periods
            transaction_cost: Transaction cost rate
            gamma: Risk aversion parameter

        Returns:
            Optimization results
        """
        # Simulate scenarios
        scenarios = self.simulate_multiperiod_returns(n_periods, n_scenarios=1000)
        n_scenarios = scenarios.shape[0]

        # Decision variables: weights at each rebalancing point
        n_rebalance_points = n_periods // rebalance_frequency + 1

        def simulate_portfolio_value(weights_sequence):
            """Simulate portfolio value over all periods"""
            total_values = []

            for scenario in range(n_scenarios):
                portfolio_value = 1.0  # Start with unit value
                current_weights = weights_sequence[0]

                for period in range(n_periods):
                    # Check if rebalancing period
                    rebalance_idx = period // rebalance_frequency
                    if period % rebalance_frequency == 0 and rebalance_idx > 0:
                        # Rebalance
                        new_weights = weights_sequence[min(rebalance_idx, len(weights_sequence)-1)]
                        # Calculate turnover
                        turnover = np.sum(np.abs(new_weights - current_weights))
                        # Apply transaction costs
                        portfolio_value *= (1 - transaction_cost * turnover)
                        current_weights = new_weights

                    # Apply returns
                    period_returns = scenarios[scenario, period, :]
                    portfolio_return = np.dot(current_weights, 1 + period_returns)
                    portfolio_value *= portfolio_return

                    # Update weights due to price changes (no rebalancing)
                    if period < n_periods - 1:
                        asset_values = current_weights * (1 + period_returns) * portfolio_value
                        current_weights = asset_values / np.sum(asset_values)

                total_values.append(portfolio_value)

            return np.array(total_values)

        # Optimize using simplified approach (static weights for now)
        def objective(weights):
            """Maximize expected utility"""
            weights = weights.reshape(1, -1)  # Single weight vector for simplification
            final_values = simulate_portfolio_value(weights)

            annualised = self._final_values_to_annualised(final_values, n_periods)
            utilities = -np.exp(-gamma * annualised)
            expected_utility = np.mean(utilities)

            return -expected_utility  # Negative for minimization

        # Constraints and bounds
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 0.4) for _ in range(self.n_assets)]

        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100}
        )

        # Calculate performance metrics
        optimal_weights = result.x
        final_values = simulate_portfolio_value(optimal_weights.reshape(1, -1))
        annualised = self._final_values_to_annualised(final_values, n_periods)
        expected_return = np.mean(annualised)
        volatility = np.std(annualised)
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        return {
            'optimal_weights': optimal_weights,
            'expected_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            'annualised_returns': annualised,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'percentile_5': np.percentile(annualised, 5),
            'percentile_95': np.percentile(annualised, 95)
        }

    def compare_static_vs_dynamic(self, static_weights: np.ndarray,
                                 n_periods: int = 12,
                                 rebalance_frequencies: List[int] = [1, 3, 12]) -> pd.DataFrame:
        """
        Compare static allocation vs dynamic rebalancing strategies (Question 2k)

        Args:
            static_weights: Static portfolio weights
            n_periods: Number of periods
            rebalance_frequencies: List of rebalancing frequencies to test

        Returns:
            DataFrame comparing strategies
        """
        results = []

        # Simulate scenarios once
        scenarios = self.simulate_multiperiod_returns(n_periods, n_scenarios=1000)

        # Static strategy (buy and hold)
        static_values = self._backtest_strategy(
            static_weights, scenarios, n_periods,
            rebalance_frequency=n_periods+1,  # Never rebalance
            transaction_cost=0
        )
        static_annualised = self._final_values_to_annualised(static_values, n_periods)

        results.append({
            'Strategy': 'Static (Buy & Hold)',
            'Rebalance Frequency': 'Never',
            'Expected Return': np.mean(static_annualised),
            'Volatility': np.std(static_annualised),
            'Sharpe Ratio': (np.mean(static_annualised) - self.risk_free_rate) / np.std(static_annualised)
            if np.std(static_annualised) > 0 else 0,
            '5th Percentile': np.percentile(static_annualised, 5),
            '95th Percentile': np.percentile(static_annualised, 95),
            'Transaction Costs': 0
        })

        # Dynamic strategies with different rebalancing frequencies
        for freq in rebalance_frequencies:
            dynamic_result = self.optimize_dynamic_portfolio(
                n_periods=n_periods,
                rebalance_frequency=freq,
                transaction_cost=0.001
            )

            freq_label = f"Every {freq} periods" if freq > 1 else "Every period"

            # Calculate total transaction costs
            n_rebalances = n_periods // freq
            avg_turnover = 0.2  # Assumed average turnover per rebalancing
            total_tc = n_rebalances * avg_turnover * 0.001

            results.append({
                'Strategy': f'Dynamic (Rebalance {freq_label})',
                'Rebalance Frequency': freq_label,
                'Expected Return': dynamic_result['expected_return'],
                'Volatility': dynamic_result['volatility'],
                'Sharpe Ratio': dynamic_result['sharpe_ratio'],
                '5th Percentile': dynamic_result['percentile_5'],
                '95th Percentile': dynamic_result['percentile_95'],
                'Transaction Costs': total_tc
            })

        return pd.DataFrame(results)

    def _backtest_strategy(self, weights: np.ndarray, scenarios: np.ndarray,
                          n_periods: int, rebalance_frequency: int,
                          transaction_cost: float) -> np.ndarray:
        """
        Backtest a strategy on simulated scenarios

        Args:
            weights: Portfolio weights
            scenarios: Simulated return scenarios
            n_periods: Number of periods
            rebalance_frequency: How often to rebalance
            transaction_cost: Transaction cost rate

        Returns:
            Array of final portfolio values
        """
        n_scenarios = scenarios.shape[0]
        final_values = []

        for scenario in range(n_scenarios):
            portfolio_value = 1.0
            current_weights = weights.copy()

            for period in range(n_periods):
                # Apply returns
                period_returns = scenarios[scenario, period, :]
                portfolio_return = np.dot(current_weights, 1 + period_returns)
                portfolio_value *= portfolio_return

                # Update weights due to price changes
                if period < n_periods - 1:
                    asset_values = current_weights * (1 + period_returns)
                    current_weights = asset_values / np.sum(asset_values)

                    # Rebalance if needed
                    if (period + 1) % rebalance_frequency == 0:
                        turnover = np.sum(np.abs(weights - current_weights))
                        portfolio_value *= (1 - transaction_cost * turnover)
                        current_weights = weights.copy()

            final_values.append(portfolio_value)

        return np.array(final_values)

    def _final_values_to_annualised(self, final_values: np.ndarray, n_periods: int) -> np.ndarray:
        """Convert terminal wealth values into annualised returns."""
        years = n_periods / self.periods_per_year
        years = max(years, 1e-9)
        final_values = np.maximum(final_values, 1e-9)
        return final_values ** (1 / years) - 1


def run_dynamic_optimization():
    """Run dynamic optimization for Questions 2(h-k)"""
    from pathlib import Path
    from data_loader import AssetDataLoader
    from parameter_estimation import ParameterEstimator
    from static_optimization import StaticPortfolioOptimizer

    print("="*60)
    print("DYNAMIC PORTFOLIO OPTIMIZATION (Questions 2h-k)")
    print("="*60)

    # Load data
<<<<<<< HEAD
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / 'data' / 'HistoricalData(2012-2024).xlsm'
    loader = AssetDataLoader(str(data_path))
=======
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data',
        'BBG Data (2000-2025).xlsx'
    )
    loader = AssetDataLoader(data_path)
>>>>>>> ab13223e224ad082ffed9bbf5757bb99c78c5e69
    returns_data = loader.load_data()

    # Estimate parameters
    estimator = ParameterEstimator(returns_data)
    expected_returns = estimator.estimate_expected_returns('combined')
    cov_matrix = estimator.estimate_covariance_matrix('shrinkage')

    # Get static optimal weights for comparison
    static_optimizer = StaticPortfolioOptimizer(
        expected_returns,
        cov_matrix,
        returns_data=returns_data
    )
    static_result = static_optimizer.optimize_portfolio(
        target_return=0.05594,
        growth_allocation=0.73
    )
    static_weights = static_result['weights']

    # Question 2(h): Risk Attribution
    print("\n" + "="*60)
    print("QUESTION 2(h): RISK ATTRIBUTION")
    print("="*60)

    risk_manager = RiskManager(static_weights, expected_returns, cov_matrix)
    risk_attribution = risk_manager.calculate_risk_attribution()

    print("\nRisk Attribution Analysis:")
    print(risk_attribution.to_string(index=False))

    # Calculate VaR and CVaR
    risk_metrics = risk_manager.calculate_var_cvar(confidence_level=0.95)
    print(f"\nValue at Risk (95%): {risk_metrics['VaR']:.2%}")
    print(f"Conditional VaR (95%): {risk_metrics['CVaR']:.2%}")

    # Questions 2(i-k): Dynamic Optimization
    print("\n" + "="*60)
    print("QUESTIONS 2(i-k): DYNAMIC PORTFOLIO OPTIMIZATION")
    print("="*60)

    dynamic_optimizer = DynamicPortfolioOptimizer(
        expected_returns, cov_matrix, returns_data
    )

    # Compare static vs dynamic strategies
    comparison = dynamic_optimizer.compare_static_vs_dynamic(
        static_weights,
        n_periods=12,  # 12 quarters (3 years)
        rebalance_frequencies=[1, 3, 6, 12]
    )

    print("\nStatic vs Dynamic Strategy Comparison:")
    print(comparison.to_string(index=False))

    # Additional analysis
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)

    best_strategy = comparison.loc[comparison['Sharpe Ratio'].idxmax()]
    print(f"\nBest Strategy (by Sharpe Ratio): {best_strategy['Strategy']}")
    print(f"  Expected Return: {best_strategy['Expected Return']:.2%}")
    print(f"  Sharpe Ratio: {best_strategy['Sharpe Ratio']:.3f}")
    print(f"  Transaction Costs: {best_strategy['Transaction Costs']:.3%}")

    return {
        'risk_attribution': risk_attribution,
        'risk_metrics': risk_metrics,
        'strategy_comparison': comparison
    }


if __name__ == "__main__":
    run_dynamic_optimization()

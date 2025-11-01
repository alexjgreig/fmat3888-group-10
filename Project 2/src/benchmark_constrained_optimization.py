"""
Benchmark-Constrained Portfolio Optimization
Keeps the portfolio close to the MySuper benchmark while permitting qualitative tilts.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple
import warnings

try:
    from .constraints_config import (
        BENCHMARK_WEIGHTS,
        TARGET_CONSTRAINTS,
        get_asset_bounds,
        get_growth_target,
        get_growth_tolerance,
        get_tracking_error_limit,
    )
except ImportError:  # pragma: no cover - allow standalone execution
    from constraints_config import (
        BENCHMARK_WEIGHTS,
        TARGET_CONSTRAINTS,
        get_asset_bounds,
        get_growth_target,
        get_growth_tolerance,
        get_tracking_error_limit,
    )

warnings.filterwarnings("ignore")


class BenchmarkConstrainedOptimizer:
    """Portfolio optimisation with benchmark tracking constraints."""

    def __init__(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
    ):
        self.expected_returns = expected_returns.sort_index()
        aligned_cov = cov_matrix.loc[self.expected_returns.index, self.expected_returns.index]
        self.cov_matrix = aligned_cov
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        self.asset_names = expected_returns.index.tolist()

        self.growth_indices = [i for i, name in enumerate(self.asset_names) if "[G]" in name]
        self.defensive_indices = [i for i, name in enumerate(self.asset_names) if "[D]" in name]

        self.target_return = 0.05594  # CPI + 3%
        self.cpi = 0.02594

        # Define benchmark allocations based on notes.md guidance
        self.benchmark_weights = self._get_benchmark_weights()

    def _get_benchmark_weights(self) -> np.ndarray:
        """
        Get benchmark weights based on notes.md guidance and typical MySuper allocations

        Returns:
            Array of benchmark weights
        """
        # Initialize weights dictionary
        weights_dict = {}

        # Based on notes.md guidance:
        # Growth assets (73% total):
        weights_dict['Australian Listed Equity [G]'] = 0.25  # 25% as per notes
        weights_dict["Int'l Listed Equity (Hedged) [G]"] = 0.13  # ~34.2% of 38% int'l
        weights_dict["Int'l Listed Equity (Unhedged) [G]"] = 0.25  # ~65.8% of 38% int'l
        weights_dict['Australian Listed Property [G]'] = 0.06  # 6% A-REITs
        weights_dict["Int'l Listed Property [G]"] = 0.02  # Small allocation
        weights_dict["Int'l Listed Infrastructure [G]"] = 0.02  # Small allocation

        # Defensive assets (27% total):
        weights_dict['Australian Fixed Income [D]'] = 0.16  # 16% as per notes
        weights_dict["Int'l Fixed Income (Hedged) [D]"] = 0.07  # 7% as per notes
        weights_dict['Cash [D]'] = 0.04  # 4% as per notes

        # Convert to array in correct order
        benchmark = np.zeros(self.n_assets)
        for i, asset in enumerate(self.asset_names):
            benchmark[i] = weights_dict.get(asset, 0)

        # Normalize to ensure sum to 1
        benchmark = benchmark / benchmark.sum()

        return benchmark

    def optimize_with_tracking_error(self, max_tracking_error: float = 0.02,
                                    max_active_weight: float = 0.05,
                                    target_return: Optional[float] = None) -> Dict:
        """
        Optimize portfolio with tracking error constraint

        Args:
            max_tracking_error: Maximum annualized tracking error (e.g., 2%)
            max_active_weight: Maximum deviation from benchmark for any asset
            target_return: Minimum required return

        Returns:
            Optimization results
        """
        if target_return is None:
            target_return = self.target_return

        # Objective: Minimize variance while staying close to benchmark
        def objective(weights):
            portfolio_var = np.dot(weights, np.dot(self.cov_matrix.values, weights))
            # Add penalty for deviation from benchmark
            active_weights = weights - self.benchmark_weights
            tracking_var = np.dot(active_weights, np.dot(self.cov_matrix.values, active_weights))
            # Combined objective: minimize variance + penalty for tracking error
            return portfolio_var + 0.5 * tracking_var

        # Constraints
        constraints = []

        # Sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        })

        # Target return constraint
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: np.dot(w, self.expected_returns.values) - target_return
        })

        # Tracking error constraint
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: max_tracking_error**2 - np.dot(
                w - self.benchmark_weights,
                np.dot(self.cov_matrix.values, w - self.benchmark_weights)
            )
        })

        # Growth/Defensive allocation constraint (70-76% growth as per notes)
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: sum(w[i] for i in self.growth_indices) - 0.70
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: 0.76 - sum(w[i] for i in self.growth_indices)
        })

        # Bounds: limited deviation from benchmark
        bounds = []
        for i in range(self.n_assets):
            lower = max(0, self.benchmark_weights[i] - max_active_weight)
            upper = min(0.4, self.benchmark_weights[i] + max_active_weight)
            bounds.append((lower, upper))

        # Initial guess: start with benchmark
        x0 = self.benchmark_weights.copy()

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        # Calculate metrics
        optimal_weights = result.x
        portfolio_return = np.dot(optimal_weights, self.expected_returns.values)
        portfolio_variance = np.dot(optimal_weights, np.dot(self.cov_matrix.values, optimal_weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Calculate tracking error
        active_weights = optimal_weights - self.benchmark_weights
        tracking_variance = np.dot(active_weights, np.dot(self.cov_matrix.values, active_weights))
        tracking_error = np.sqrt(tracking_variance)

        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std

        # Growth/Defensive split
        growth_weight = sum(optimal_weights[i] for i in self.growth_indices)
        defensive_weight = sum(optimal_weights[i] for i in self.defensive_indices)

        return {
            'success': result.success,
            'weights': optimal_weights,
            'benchmark_weights': self.benchmark_weights,
            'active_weights': active_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio,
            'tracking_error': tracking_error,
            'growth_weight': growth_weight,
            'defensive_weight': defensive_weight,
            'optimization_result': result
        }

    def get_benchmark_portfolio_metrics(self) -> Dict:
        """
        Calculate metrics for the benchmark portfolio

        Returns:
            Dictionary of benchmark metrics
        """
        benchmark_return = np.dot(self.benchmark_weights, self.expected_returns.values)
        benchmark_variance = np.dot(self.benchmark_weights,
                                   np.dot(self.cov_matrix.values, self.benchmark_weights))
        benchmark_std = np.sqrt(benchmark_variance)
        benchmark_sharpe = (benchmark_return - self.risk_free_rate) / benchmark_std

        growth_weight = sum(self.benchmark_weights[i] for i in self.growth_indices)
        defensive_weight = sum(self.benchmark_weights[i] for i in self.defensive_indices)

        return {
            'weights': self.benchmark_weights,
            'expected_return': benchmark_return,
            'volatility': benchmark_std,
            'sharpe_ratio': benchmark_sharpe,
            'growth_weight': growth_weight,
            'defensive_weight': defensive_weight
        }

    def generate_constrained_efficient_frontier(
        self,
        n_points: int = 30,
        max_tracking_error: Optional[float] = None,
    ) -> pd.DataFrame:
        """Generate an efficient frontier under the benchmark constraints."""
        if max_tracking_error is None:
            max_tracking_error = self.tracking_error_limit

        min_ret, max_ret = self._compute_extreme_returns(max_tracking_error)
        target_returns = np.linspace(min_ret, max_ret, n_points)
        points = []

        for target in target_returns:
            result = self.optimise_with_tracking_error(
                max_tracking_error=max_tracking_error,
                target_return=target,
            )
            if result["success"]:
                points.append(
                    {
                        "return": result["expected_return"],
                        "volatility": result["volatility"],
                        "sharpe_ratio": result["sharpe_ratio"],
                        "tracking_error": result["tracking_error"],
                        "growth_weight": result["growth_weight"],
                        "target_return": target,
                    }
                )

        return pd.DataFrame(points)

    def compare_portfolios(self) -> pd.DataFrame:
        """
        Compare different portfolio configurations

        Returns:
            DataFrame comparing portfolios
        """
        portfolios = []

        # 1. Benchmark portfolio
        benchmark_metrics = self.get_benchmark_portfolio_metrics()
        portfolios.append({
            'Portfolio': 'Benchmark (MySuper Typical)',
            'Expected Return': benchmark_metrics['expected_return'],
            'Volatility': benchmark_metrics['volatility'],
            'Sharpe Ratio': benchmark_metrics['sharpe_ratio'],
            'Growth %': benchmark_metrics['growth_weight'] * 100,
            'Tracking Error': 0.0
        })

        # 2. Optimized with tight tracking (1% TE)
        tight_result = self.optimize_with_tracking_error(
            max_tracking_error=0.01,
            max_active_weight=0.03
        )
        if tight_result['success']:
            portfolios.append({
                'Portfolio': 'Optimized (1% Tracking Error)',
                'Expected Return': tight_result['expected_return'],
                'Volatility': tight_result['volatility'],
                'Sharpe Ratio': tight_result['sharpe_ratio'],
                'Growth %': tight_result['growth_weight'] * 100,
                'Tracking Error': tight_result['tracking_error']
            })

        # 3. Optimized with moderate tracking (2% TE)
        moderate_result = self.optimize_with_tracking_error(
            max_tracking_error=0.02,
            max_active_weight=0.05
        )
        if moderate_result['success']:
            portfolios.append({
                'Portfolio': 'Optimized (2% Tracking Error)',
                'Expected Return': moderate_result['expected_return'],
                'Volatility': moderate_result['volatility'],
                'Sharpe Ratio': moderate_result['sharpe_ratio'],
                'Growth %': moderate_result['growth_weight'] * 100,
                'Tracking Error': moderate_result['tracking_error']
            })

        # 4. Optimized with loose tracking (3% TE)
        loose_result = self.optimize_with_tracking_error(
            max_tracking_error=0.03,
            max_active_weight=0.07
        )
        if loose_result['success']:
            portfolios.append({
                'Portfolio': 'Optimized (3% Tracking Error)',
                'Expected Return': loose_result['expected_return'],
                'Volatility': loose_result['volatility'],
                'Sharpe Ratio': loose_result['sharpe_ratio'],
                'Growth %': loose_result['growth_weight'] * 100,
                'Tracking Error': loose_result['tracking_error']
            })

        return pd.DataFrame(rows)

    def get_recommended_portfolio(self) -> Dict:
        """
        Get the recommended portfolio with 2% tracking error constraint

        Returns:
            Recommended portfolio details
        """
        # Use 2% tracking error as reasonable balance
        result = self.optimize_with_tracking_error(
            max_tracking_error=0.02,
            max_active_weight=0.05,
            target_return=self.target_return
        )

        if result['success']:
            print("\n" + "="*60)
            print("RECOMMENDED PORTFOLIO (2% Tracking Error Constraint)")
            print("="*60)

            print(f"\nPortfolio Metrics:")
            print(f"  Expected Return: {result['expected_return']:.2%}")
            print(f"  Volatility: {result['volatility']:.2%}")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print(f"  Tracking Error: {result['tracking_error']:.2%}")
            print(f"  Growth Allocation: {result['growth_weight']:.1%}")
            print(f"  Defensive Allocation: {result['defensive_weight']:.1%}")

            print(f"\nAsset Allocation:")
            print(f"{'Asset':40} {'Benchmark':>10} {'Optimal':>10} {'Active':>10}")
            print("-"*70)

            for i, asset in enumerate(self.asset_names):
                if result['weights'][i] > 0.001 or self.benchmark_weights[i] > 0.001:
                    print(f"{asset[:40]:40} {self.benchmark_weights[i]:10.2%} "
                          f"{result['weights'][i]:10.2%} {result['active_weights'][i]:+10.2%}")

            # Calculate tilts
            print(f"\nKey Portfolio Tilts vs Benchmark:")
            tilts = []
            for i, asset in enumerate(self.asset_names):
                if abs(result['active_weights'][i]) > 0.01:
                    tilts.append((asset, result['active_weights'][i]))

            tilts.sort(key=lambda x: abs(x[1]), reverse=True)
            for asset, tilt in tilts[:5]:
                direction = "Overweight" if tilt > 0 else "Underweight"
                print(f"  {direction:12} {asset[:35]:35} by {abs(tilt):5.2%}")

        return result


def run_benchmark_constrained_optimization():
    """Command-line hook for standalone execution."""
    from pathlib import Path
    from .data_loader import AssetDataLoader
    from .parameter_estimation import ParameterEstimator

    print("=" * 70)
    print("BENCHMARK-CONSTRAINED PORTFOLIO OPTIMISATION")
    print("=" * 70)

    # Load data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data',
        'BBG Data (2000-2025).xlsx'
    )
    loader = AssetDataLoader(data_path)
    returns_data = loader.load_data()
    estimator = ParameterEstimator(returns_data)
    exp_returns = estimator.estimate_expected_returns("combined")
    cov = estimator.estimate_covariance_matrix("shrinkage")

    # Initialize benchmark-constrained optimizer
    optimizer = BenchmarkConstrainedOptimizer(expected_returns, cov_matrix)

    # Show benchmark portfolio
    print("\n" + "="*60)
    print("BENCHMARK PORTFOLIO (Based on MySuper Industry Standards)")
    print("="*60)
    benchmark_metrics = optimizer.get_benchmark_portfolio_metrics()
    print(f"\nBenchmark Metrics:")
    print(f"  Expected Return: {benchmark_metrics['expected_return']:.2%}")
    print(f"  Volatility: {benchmark_metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:.3f}")
    print(f"  Growth Allocation: {benchmark_metrics['growth_weight']:.1%}")

    print(f"\nBenchmark Weights:")
    for i, asset in enumerate(optimizer.asset_names):
        if optimizer.benchmark_weights[i] > 0.001:
            print(f"  {asset[:40]:40} {optimizer.benchmark_weights[i]:7.2%}")

    # Get recommended portfolio
    recommended = optimizer.get_recommended_portfolio()

    # Compare different tracking error levels
    print("\n" + "="*60)
    print("PORTFOLIO COMPARISON (Different Tracking Error Constraints)")
    print("="*60)
    comparison = optimizer.compare_portfolios()
    print(comparison.to_string(index=False))

    # Generate constrained efficient frontier
    print("\n" + "="*60)
    print("GENERATING CONSTRAINED EFFICIENT FRONTIER")
    print("="*60)
    frontier = optimizer.generate_constrained_efficient_frontier(
        n_points=30,
        max_tracking_error=0.02
    )
    print(f"Generated {len(frontier)} frontier points with tracking error ≤ 2%")

    return {
        "benchmark": benchmark,
        "recommended": recommended,
        "comparison": comparison,
        "frontier": frontier,
    }


if __name__ == "__main__":
    run_benchmark_constrained_optimization()

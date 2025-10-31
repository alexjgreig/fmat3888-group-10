"""
Static Portfolio Optimization Module for Question 2(a-e)
Implements efficient frontier, minimum variance portfolio, and portfolio comparisons
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

try:
    from .constraints_config import (
        TARGET_CONSTRAINTS,
        get_asset_bounds,
        get_growth_target,
        get_growth_tolerance,
    )
except ImportError:  # pragma: no cover - allow standalone execution
    from constraints_config import (
        TARGET_CONSTRAINTS,
        get_asset_bounds,
        get_growth_target,
        get_growth_tolerance,
    )

warnings.filterwarnings('ignore')


class StaticPortfolioOptimizer:
    """Static portfolio optimization using Markowitz framework"""

    def __init__(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize portfolio optimizer

        Args:
            expected_returns: Expected annual returns for each asset
            cov_matrix: Annualized covariance matrix
            risk_free_rate: Annual risk-free rate
        """
        self.expected_returns = expected_returns.sort_index()
        self.cov_matrix = cov_matrix.loc[self.expected_returns.index, self.expected_returns.index]
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(self.expected_returns)
        self.asset_names = self.expected_returns.index.tolist()

        # Identify growth and defensive assets
        self.growth_indices = [i for i, name in enumerate(self.asset_names) if "[G]" in name]
        self.defensive_indices = [i for i, name in enumerate(self.asset_names) if "[D]" in name]

        # Target parameters from requirements
        self.target_return = 0.05594  # CPI + 3%
        bounds_lookup = get_asset_bounds(tuple(self.asset_names))
        self.asset_bounds = np.array([bounds_lookup[name] for name in self.asset_names])
        self.lower_bounds = np.array([b[0] for b in self.asset_bounds])
        self.upper_bounds = np.array([b[1] for b in self.asset_bounds])
        self.growth_target = get_growth_target()
        self.growth_tolerance = get_growth_tolerance()

    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict:
        """
        Calculate portfolio metrics given weights

        Args:
            weights: Portfolio weights

        Returns:
            Dictionary of portfolio metrics
        """
        # Expected return
        portfolio_return = np.dot(weights, self.expected_returns.values)

        # Portfolio variance and standard deviation
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix.values, weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0

        # Growth/Defensive split
        growth_weight = sum(weights[i] for i in self.growth_indices)
        defensive_weight = sum(weights[i] for i in self.defensive_indices)

        return {
            'return': portfolio_return,
            'volatility': portfolio_std,
            'variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio,
            'growth_weight': growth_weight,
            'defensive_weight': defensive_weight,
            'weights': weights
        }

    def optimize_portfolio(self,
                          target_return: Optional[float] = None,
                          growth_allocation: Optional[float] = None,
                          min_weight: float = 0.0,
                          max_weight: float = 0.4,
                          allow_short: bool = False,
                          enforce_exact_return: bool = False) -> Dict:
        """
        Optimize portfolio with various constraints

        Args:
            target_return: Minimum required return
            growth_allocation: Target growth asset allocation (e.g., 0.7 for 70%)
            min_weight: Minimum weight for any asset
            max_weight: Maximum weight for any asset
            allow_short: Whether to allow short selling
            enforce_exact_return: If True, force portfolio return to equal target_return

        Returns:
            Optimization results dictionary
        """
        if growth_allocation is None:
            growth_allocation = self.growth_target
        if growth_tolerance is None:
            growth_tolerance = self.growth_tolerance
        if bounds is None:
            bounds = [tuple(b) for b in self.asset_bounds]

        # Objective function (minimize variance)
        def objective(weights: np.ndarray) -> float:
            return weights @ self.cov_matrix.values @ weights

        constraints = self._build_constraints(
            target_return=target_return,
            growth_allocation=growth_allocation,
            enforce_exact_return=enforce_exact_return
        )
        bounds = self._build_bounds(
            min_weight=min_weight,
            max_weight=max_weight,
            allow_short=allow_short
        )

        # Initial guess (equal weights adjusted for constraints)
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            print(f"Optimization warning: {result.message}")

        # Calculate metrics for optimal portfolio
        optimal_weights = result.x
        metrics = self.calculate_portfolio_metrics(optimal_weights)
        metrics['constraint_checks'] = self._summarise_constraint_checks(
            optimal_weights,
            bounds=bounds,
            growth_allocation=growth_allocation,
            growth_tolerance=growth_tolerance,
        )

        return {
            'success': result.success,
            'weights': optimal_weights,
            'metrics': metrics,
            'optimization_result': result
        }

    def _build_constraints(self,
                           target_return: Optional[float] = None,
                           growth_allocation: Optional[float] = None,
                           enforce_exact_return: bool = False) -> List[Dict]:
        """Create constraint list shared across optimizations."""
        constraints: List[Dict] = [{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        }]

        if target_return is not None:
            if enforce_exact_return:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda w, tr=target_return: np.dot(w, self.expected_returns.values) - tr
                })
            else:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, tr=target_return: np.dot(w, self.expected_returns.values) - tr
                })

        if growth_allocation is not None:
            tolerance = 0.06  # Allow ±6% deviation as per requirements
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, target=growth_allocation, tol=tolerance:
                    np.sum(w[self.growth_indices]) - (target - tol)
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, target=growth_allocation, tol=tolerance:
                    (target + tol) - np.sum(w[self.growth_indices])
            })

        return constraints

    def _build_bounds(self,
                      min_weight: float,
                      max_weight: float,
                      allow_short: bool) -> List[Tuple[float, float]]:
        """Create bounds for portfolio weights."""
        if allow_short:
            return [(-1, max_weight) for _ in range(self.n_assets)]
        return [(min_weight, max_weight) for _ in range(self.n_assets)]

    def _optimize_expected_return(self,
                                  maximize: bool,
                                  growth_allocation: Optional[float],
                                  min_weight: float,
                                  max_weight: float,
                                  allow_short: bool = False) -> Dict:
        """
        Optimize purely for expected return subject to feasibility constraints.

        Args:
            maximize: Whether to maximize (True) or minimize (False) return
            growth_allocation: Optional growth proportion constraint
            min_weight: Lower bound for asset weights
            max_weight: Upper bound for asset weights
            allow_short: Whether short positions are allowed

        Returns:
            Optimization result dictionary mirroring optimize_portfolio
        """
        sign = -1 if maximize else 1

        def objective(weights):
            return sign * np.dot(weights, self.expected_returns.values)

        constraints = self._build_constraints(
            target_return=None,
            growth_allocation=growth_allocation
        )
        bounds = self._build_bounds(min_weight, max_weight, allow_short)
        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            print(f"Return optimization warning: {result.message}")

        metrics = self.calculate_portfolio_metrics(result.x)
        return {
            'success': result.success,
            'weights': result.x,
            'metrics': metrics,
            'optimization_result': result
        }

    def generate_efficient_frontier(self, n_points: int = 100000,
                                   growth_allocation: Optional[float] = None,
                                   min_weight: float = 0.0,
                                   max_weight: float = 0.4) -> pd.DataFrame:
        """
        Generate efficient frontier points under the qualitative asset bounds.
        """
        min_variance_result = self.optimize_portfolio(
            target_return=None,
            growth_allocation=growth_allocation,
            min_weight=min_weight,
            max_weight=max_weight
        )
        max_return_result = self._optimize_expected_return(
            maximize=True,
            growth_allocation=growth_allocation,
            min_weight=min_weight,
            max_weight=max_weight
        )

        if not (min_variance_result['success'] and max_return_result['success']):
            raise ValueError("Failed to determine feasible bounds for the efficient frontier.")

        min_return = min_variance_result['metrics']['return']
        max_return = max_return_result['metrics']['return']

        return_span = max_return - min_return
        if return_span < 1e-8:
            # Degenerate case: all portfolios yield same return
            metrics = min_variance_result['metrics']
            return pd.DataFrame([{
                'return': metrics['return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'growth_weight': metrics['growth_weight']
            }])

        eps = max(1e-6, return_span * 1e-3)
        interior_points = max(n_points - 2, 0)

        frontier_points = []

        def _append_metrics(result_dict):
            metrics = result_dict['metrics']
            frontier_points.append({
                'return': metrics['return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'growth_weight': metrics['growth_weight']
            })

        _append_metrics(min_variance_result)

        if interior_points > 0:
            raw_targets = np.linspace(min_return, max_return, interior_points + 2)[1:-1]
            for target_return in raw_targets:
                adjusted_target = min(max(target_return, min_return + eps), max_return - eps)
                if adjusted_target <= min_return or adjusted_target >= max_return:
                    continue
                result = self.optimize_portfolio(
                    target_return=adjusted_target,
                    growth_allocation=growth_allocation,
                    min_weight=min_weight,
                    max_weight=max_weight,
                    enforce_exact_return=True
                )

                if result['success']:
                    _append_metrics(result)

        if max_return_result['success']:
            _append_metrics(max_return_result)

        if not frontier_points:
            return pd.DataFrame()

        frontier_df = pd.DataFrame(frontier_points)
        frontier_df = frontier_df.sort_values('volatility').drop_duplicates(subset=['volatility', 'return'])
        frontier_df.reset_index(drop=True, inplace=True)

        return frontier_df

    def find_minimum_variance_portfolio(self, target_return: float,
                                       growth_allocation: float = 0.7) -> Dict:
        """
        Find minimum variance portfolio with target return constraint (Question 2b)

        Args:
            target_return: Minimum required return
            growth_allocation: Target growth asset allocation

        Returns:
            Optimization results
        """
        result = self.optimize_portfolio(
            target_return=target_return,
            growth_allocation=growth_allocation,
        )

        if result['success']:
            print(f"\nMinimum Variance Portfolio (Return ≥ {target_return:.2%}):")
            print(f"Expected Return: {result['metrics']['return']:.2%}")
            print(f"Volatility: {result['metrics']['volatility']:.2%}")
            print(f"Sharpe Ratio: {result['metrics']['sharpe_ratio']:.3f}")
            print(f"Growth Allocation: {result['metrics']['growth_weight']:.1%}")

            print("\nOptimal Weights:")
            for i, asset in enumerate(self.asset_names):
                if result['weights'][i] > 0.001:  # Only show non-zero weights
                    print(f"  {asset[:40]:40} {result['weights'][i]:7.2%}")

        return result

    def compare_risk_profiles(self) -> pd.DataFrame:
        """
        Compare different portfolio risk profiles (Question 2e)

        Returns:
            DataFrame comparing defensive, balanced, and aggressive portfolios
        """
        profiles = {
            'Defensive': {'growth': 0.3, 'defensive': 0.7},
            'Balanced': {'growth': 0.7, 'defensive': 0.3},
            'Aggressive': {'growth': 0.9, 'defensive': 0.1}
        }

        results = []

        for profile_name, allocation in profiles.items():
            # Optimize portfolio for this profile
            if profile_name == 'Balanced':
                profile_bounds = [tuple(b) for b in self.asset_bounds]
            elif profile_name == 'Defensive':
                profile_bounds = [
                    (0.0, min(0.5, b[1] + 0.1)) if '[G]' in self.asset_names[idx]
                    else (max(0.0, b[0]), min(0.6, b[1] + 0.1))
                    for idx, b in enumerate(self.asset_bounds)
                ]
            else:  # Aggressive
                profile_bounds = [
                    (max(0.0, b[0]), min(0.5, b[1] + 0.1)) if '[G]' in self.asset_names[idx]
                    else (0.0, min(0.3, b[1]))
                    for idx, b in enumerate(self.asset_bounds)
                ]
            result = self.optimize_portfolio(
                target_return=self.target_return,
                growth_allocation=allocation['growth'],
                bounds=profile_bounds,
            )

            if not result['success']:
                fallback_bounds = [
                    (0.0, 0.6) if '[G]' in self.asset_names[idx] else (0.0, 1.0)
                    for idx in range(self.n_assets)
                ]
                result = self.optimize_portfolio(
                    target_return=self.target_return,
                    growth_allocation=allocation['growth'],
                    bounds=fallback_bounds,
                )

            if result['success']:
                metrics = result['metrics']

                # Calculate exponential utility
                gamma = 1  # Risk aversion parameter
                expected_utility = -np.exp(-gamma * metrics['return'])

                # Calculate downside risk metrics
                downside_vol = self._calculate_downside_volatility(result['weights'])

                results.append({
                    'Profile': profile_name,
                    'Growth %': allocation['growth'] * 100,
                    'Defensive %': allocation['defensive'] * 100,
                    'Expected Return': metrics['return'],
                    'Volatility': metrics['volatility'],
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Exponential Utility': expected_utility,
                    'Downside Volatility': downside_vol,
                    'Sortino Ratio': (metrics['return'] - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
                })

        return pd.DataFrame(results)

    def _calculate_downside_volatility(self, weights: np.ndarray,
                                      threshold: Optional[float] = None) -> float:
        """
        Calculate downside volatility (semi-deviation)

        Args:
            weights: Portfolio weights
            threshold: Return threshold (default is risk-free rate)

        Returns:
            Annualized downside volatility
        """
        if threshold is None:
            threshold = self.risk_free_rate / 12  # Monthly threshold

        # This is simplified - ideally would use actual return distribution
        # For now, approximate using normal distribution
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix.values, weights)))

        # Approximate downside volatility (simplified)
        downside_vol = portfolio_vol * np.sqrt(2/np.pi)  # Rough approximation

        return downside_vol

    def _compute_extreme_returns(
        self,
        growth_allocation: float,
        bounds: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """
        Solve for the minimum and maximum expected returns under constraints.
        """
        def optimise(direction: int) -> float:
            def obj(weights: np.ndarray) -> float:
                return -direction * (weights @ self.expected_returns.values)

            initial = np.clip(
                self.benchmark_initial_guess(bounds),
                [b[0] for b in bounds],
                [b[1] for b in bounds],
            )

            res = minimize(
                obj,
                initial,
                method="SLSQP",
                bounds=bounds,
                constraints=[
                    {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                    {
                        "type": "ineq",
                        "fun": lambda w: sum(w[i] for i in self.growth_indices)
                        - (growth_allocation - self.growth_tolerance),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda w: (growth_allocation + self.growth_tolerance)
                        - sum(w[i] for i in self.growth_indices),
                    },
                ],
                options={"maxiter": 1000, "ftol": 1e-9},
            )
            return res.x @ self.expected_returns.values

        min_ret = optimise(direction=-1)
        max_ret = optimise(direction=1)
        if min_ret > max_ret:
            min_ret, max_ret = max_ret, min_ret
        return min_ret, max_ret

    def benchmark_initial_guess(self, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """
        Produce a feasible starting point anchored to target weights.
        """
        target = np.array([TARGET_CONSTRAINTS[name].target for name in self.asset_names])
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        guess = np.clip(target, lower, upper)
        total = guess.sum()
        if total == 0:
            guess = np.full_like(guess, 1 / len(guess))
        else:
            guess = guess / total
        return guess

    def _summarise_constraint_checks(
        self,
        weights: np.ndarray,
        bounds: List[Tuple[float, float]],
        growth_allocation: float,
        growth_tolerance: float,
    ) -> Dict[str, float]:
        """Diagnostics for constraint tightness."""
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        checks = {
            "growth_gap": sum(weights[i] for i in self.growth_indices) - growth_allocation,
            "growth_lower_buffer": sum(weights[i] for i in self.growth_indices)
            - (growth_allocation - growth_tolerance),
            "growth_upper_buffer": (growth_allocation + growth_tolerance)
            - sum(weights[i] for i in self.growth_indices),
        }
        for idx, name in enumerate(self.asset_names):
            checks[f"{name}::lower_buffer"] = weights[idx] - lower[idx]
            checks[f"{name}::upper_buffer"] = upper[idx] - weights[idx]
        return checks

    def plot_efficient_frontier(self, frontier_df: pd.DataFrame,
                               special_portfolios: Optional[List[Dict]] = None,
                               overlay_frontiers: Optional[List[Dict]] = None,
                               save_path: Optional[str] = None):
        """
        Plot the efficient frontier

        Args:
            frontier_df: DataFrame with frontier points
            special_portfolios: List of special portfolios to highlight
            overlay_frontiers: Additional frontier curves (dict with 'data' and optional style)
            save_path: Path to save the plot
        """
        if frontier_df.empty:
            raise ValueError("Efficient frontier data is empty. Generate frontier before plotting.")

        frontier_sorted = frontier_df.sort_values('volatility')

        plt.figure(figsize=(12, 8))

        # Plot efficient frontier
        plt.plot(frontier_sorted['volatility'] * 100, frontier_sorted['return'] * 100,
                'b-', linewidth=2, label='Efficient Frontier')

        if overlay_frontiers:
            for frontier in overlay_frontiers:
                data = frontier.get('data')
                if data is None or data.empty:
                    continue
                sorted_data = data.sort_values('volatility')
                plot_kwargs = {
                    'linewidth': frontier.get('linewidth', 2),
                    'linestyle': frontier.get('linestyle', '-'),
                    'color': frontier.get('color', 'b'),
                    'alpha': frontier.get('alpha', 0.3),
                    'label': frontier.get('label', 'Additional Frontier')
                }
                plt.plot(sorted_data['volatility'] * 100, sorted_data['return'] * 100,
                        **plot_kwargs)

        # Plot individual assets
        asset_vols = np.sqrt(np.diag(self.cov_matrix.values)) * 100
        asset_returns = self.expected_returns.values * 100

        for i, asset in enumerate(self.asset_names):
            marker = 'o' if i in self.growth_indices else 's'
            color = 'green' if i in self.growth_indices else 'red'
            plt.scatter(asset_vols[i], asset_returns[i],
                       marker=marker, s=100, c=color, alpha=0.6)
            plt.annotate(asset.split('[')[0][:10], (asset_vols[i], asset_returns[i]),
                        fontsize=8, ha='right')

        # Plot special portfolios if provided
        if special_portfolios:
            for portfolio in special_portfolios:
                plt.scatter(portfolio['volatility'] * 100, portfolio['return'] * 100,
                          marker='x', s=200, c='gold', edgecolor='black',
                          label=portfolio.get('name', 'Special'))

        # Add target return line
        plt.axhline(y=self.target_return * 100, color='r', linestyle='--',
                   alpha=0.5, label=f'Target Return ({self.target_return:.2%})')

        plt.xlabel('Volatility (%)', fontsize=12)
        plt.ylabel('Expected Return (%)', fontsize=12)
        plt.title('Efficient Frontier with Asset Constraints', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_optimization_report(self) -> Dict:
        """
        Generate comprehensive optimization report for Question 2(a-e)

        Returns:
            Dictionary containing all optimization results
        """
        report = {}

        # Question 2a: Efficient Frontier
        print("\n" + "="*60)
        print("QUESTION 2a: EFFICIENT FRONTIER")
        print("="*60)

        # Generate frontiers with different constraints
        frontier_unconstrained = self.generate_efficient_frontier(
            n_points=500,
            min_weight=0.0,
            max_weight=1.0
        )

        frontier_constrained = self.generate_efficient_frontier(
            n_points=500,
            min_weight=0.0,
            max_weight=0.4
        )

        frontier_balanced = self.generate_efficient_frontier(
            n_points=500,
            growth_allocation=0.7,
            min_weight=0.0,
            max_weight=0.4
        )

        report['efficient_frontiers'] = {
            'unconstrained': frontier_unconstrained,
            'constrained': frontier_constrained,
            'balanced_70_30': frontier_balanced
        }

        # Question 2b: Minimum Variance Portfolio
        print("\n" + "="*60)
        print("QUESTION 2b: MINIMUM VARIANCE PORTFOLIO")
        print("="*60)

        min_var_result = self.find_minimum_variance_portfolio(
            target_return=self.target_return,
            growth_allocation=0.7
        )
        report['minimum_variance_portfolio'] = min_var_result

        # Question 2e: Risk Profile Comparison
        print("\n" + "="*60)
        print("QUESTION 2e: PORTFOLIO RISK PROFILES COMPARISON")
        print("="*60)

        comparison_df = self.compare_risk_profiles()
        print("\n", comparison_df.to_string(index=False))

        report['risk_profile_comparison'] = comparison_df

        # Additional analysis
        report['analysis'] = {
            'target_return': self.target_return,
            'risk_free_rate': self.risk_free_rate,
            'n_assets': self.n_assets,
            'growth_assets': [self.asset_names[i] for i in self.growth_indices],
            'defensive_assets': [self.asset_names[i] for i in self.defensive_indices]
        }

        return report


def run_static_optimization():
    """Run complete static portfolio optimization for Question 2(a-e)"""
    from data_loader import AssetDataLoader
    from parameter_estimation import ParameterEstimator

    print("="*60)
    print("STATIC PORTFOLIO OPTIMIZATION (Questions 2a-e)")
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

    # Initialize optimizer
    optimizer = StaticPortfolioOptimizer(expected_returns, cov_matrix)

    # Generate full report
    report = optimizer.generate_optimization_report()

    # Plot efficient frontier
    frontier_df = report['efficient_frontiers']['balanced_70_30']
    min_var_portfolio = report['minimum_variance_portfolio']['metrics']

    special_portfolios = [{
        'name': 'Min Variance (TR≥5.594%)',
        'return': min_var_portfolio['return'],
        'volatility': min_var_portfolio['volatility']
    }]

    overlay_frontiers = [{
        'data': report['efficient_frontiers']['unconstrained'],
        'label': 'Efficient Frontier (Unconstrained)',
        'color': 'blue',
        'alpha': 0.3,
        'linewidth': 2
    }]

    optimizer.plot_efficient_frontier(
        frontier_df,
        special_portfolios=special_portfolios,
        overlay_frontiers=overlay_frontiers,
        save_path='../outputs/figures/efficient_frontier.png'
    )

    return report


if __name__ == "__main__":
    run_static_optimization()

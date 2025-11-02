"""
Static Portfolio Optimization Module for Question 2(a-e)
Implements efficient frontier, minimum variance portfolio, and portfolio comparisons
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')


class StaticPortfolioOptimizer:
    """Static portfolio optimization using Markowitz framework"""

    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                 risk_free_rate: float = 0.02, returns_data: Optional[pd.DataFrame] = None):
        """
        Initialize portfolio optimizer

        Args:
            expected_returns: Expected annual returns for each asset
            cov_matrix: Annualized covariance matrix
            risk_free_rate: Annual risk-free rate
            returns_data: Optional historical returns (monthly) for additional analytics
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        self.asset_names = expected_returns.index.tolist()
        self.returns_data = returns_data
        self.asset_weight_ranges = self._get_asset_weight_ranges()
        self.asset_fees = self._get_asset_fees()
        self.asset_bounds = [
            self.asset_weight_ranges.get(name, (0.0, 0.4))
            for name in self.asset_names
        ]

        # Identify growth and defensive assets
        self.growth_indices = [i for i, name in enumerate(self.asset_names) if '[G]' in name]
        self.defensive_indices = [i for i, name in enumerate(self.asset_names) if '[D]' in name]

        # Target parameters from requirements
        self.target_return = 0.05594  # CPI + 3%
        self.growth_target = 0.73  # Balanced corridor midpoint
        self.growth_bounds = (0.60, 0.76)

    @staticmethod
    def _get_asset_weight_ranges() -> Dict[str, Tuple[float, float]]:
        """
        Asset-level minimum and maximum weights aligned with MySuper guidance.
        Australian Listed Equity minimum updated to 15% per latest specification.
        """
        return {
            'Australian Listed Equity [G]': (0.15, 0.45),
            "Int'l Listed Equity (Hedged) [G]": (0.0, 0.35),
            "Int'l Listed Equity (Unhedged) [G]": (0.0, 0.35),
            'Australian Listed Property [G]': (0.0, 0.25),
            "Int'l Listed Property [G]": (0.0, 0.25),
            "Int'l Listed Infrastructure [G]": (0.0, 0.30),
            'Australian Fixed Income [D]': (0.05, 0.40),
            "Int'l Fixed Income (Hedged) [D]": (0.0, 0.35),
            'Cash [D]': (0.0, 0.15)
        }

    @staticmethod
    def _get_asset_fees() -> Dict[str, float]:
        """
        Annual fee assumptions (in decimal form) for each asset class.
        """
        return {
            'Australian Listed Equity [G]': 0.0005,
            "Int'l Listed Equity (Hedged) [G]": 0.0011,
            "Int'l Listed Equity (Unhedged) [G]": 0.0009,
            'Australian Listed Property [G]': 0.0012,
            "Int'l Listed Property [G]": 0.0022,
            "Int'l Listed Infrastructure [G]": 0.0026,
            'Australian Fixed Income [D]': 0.0010,
            "Int'l Fixed Income (Hedged) [D]": 0.0010,
            'Cash [D]': 0.0004
        }

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

        # Fee drag and net return
        weighted_fee = sum(
            weights[i] * self.asset_fees.get(self.asset_names[i], 0.0)
            for i in range(self.n_assets)
        )
        net_return = portfolio_return - weighted_fee

        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0

        # Growth/Defensive split
        growth_weight = sum(weights[i] for i in self.growth_indices)
        defensive_weight = sum(weights[i] for i in self.defensive_indices)

        return {
            'return': portfolio_return,
            'net_return': net_return,
            'volatility': portfolio_std,
            'variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio,
            'growth_weight': growth_weight,
            'defensive_weight': defensive_weight,
            'weighted_fee': weighted_fee,
            'weights': weights
        }

    def optimize_portfolio(self,
                          target_return: Optional[float] = None,
                          growth_allocation: Optional[float] = None,
                          min_weight: float = 0.0,
                          max_weight: float = 0.4,
                          allow_short: bool = False,
                          enforce_exact_return: bool = False,
                          enforce_asset_ranges: bool = True,
                          custom_bounds: Optional[List[Tuple[float, float]]] = None) -> Dict:
        """
        Optimize portfolio with various constraints

        Args:
            target_return: Minimum required return
            growth_allocation: Target growth asset allocation (e.g., 0.7 for 70%)
            min_weight: Minimum weight for any asset
            max_weight: Maximum weight for any asset
            allow_short: Whether to allow short selling
            enforce_exact_return: If True, force portfolio return to equal target_return
            enforce_asset_ranges: Whether to apply asset-level min/max bands

        Returns:
            Optimization results dictionary
        """
        # Objective function (minimize variance)
        def objective(weights):
            return np.dot(weights, np.dot(self.cov_matrix.values, weights))

        constraints = self._build_constraints(
            target_return=target_return,
            growth_allocation=growth_allocation,
            enforce_exact_return=enforce_exact_return
        )
        bounds = self._build_bounds(
            min_weight=min_weight,
            max_weight=max_weight,
            allow_short=allow_short,
            enforce_asset_ranges=enforce_asset_ranges,
            custom_bounds=custom_bounds
        )

        # Initial guess (equal weights adjusted for constraints)
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            print(f"Optimization warning: {result.message}")

        # Calculate metrics for optimal portfolio
        optimal_weights = result.x
        metrics = self.calculate_portfolio_metrics(optimal_weights)

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
                      min_weight: Optional[float],
                      max_weight: Optional[float],
                      allow_short: bool,
                      enforce_asset_ranges: bool = True,
                      custom_bounds: Optional[List[Tuple[float, float]]] = None) -> List[Tuple[float, float]]:
        """Create bounds for portfolio weights."""
        if custom_bounds is not None:
            if len(custom_bounds) != self.n_assets:
                raise ValueError("Custom bounds length must match number of assets.")
            return custom_bounds

        if not enforce_asset_ranges:
            lower = min_weight if min_weight is not None else (-1.0 if allow_short else 0.0)
            upper = max_weight if max_weight is not None else (1.0 if allow_short else 1.0)
            if allow_short and min_weight is None:
                lower = -1.0
            if allow_short and max_weight is None:
                upper = 1.0
            return [(lower, upper) for _ in range(self.n_assets)]

        bounds: List[Tuple[float, float]] = []

        for asset in self.asset_names:
            range_lower, range_upper = self.asset_weight_ranges.get(asset, (min_weight, max_weight))

            lower = range_lower if min_weight is None else max(range_lower, min_weight)
            upper = range_upper if max_weight is None else min(range_upper, max_weight)

            if allow_short:
                lower = min(lower, -1.0 if min_weight is None else min(min_weight, -1.0))
                upper = range_upper if max_weight is None else max(range_upper, max_weight)

            if lower > upper:
                lower, upper = range_lower, range_upper

            bounds.append((lower, upper))

        return bounds

    def _optimize_expected_return(self,
                                  maximize: bool,
                                  growth_allocation: Optional[float],
                                  min_weight: float,
                                  max_weight: float,
                                  allow_short: bool = False,
                                  enforce_asset_ranges: bool = True,
                                  custom_bounds: Optional[List[Tuple[float, float]]] = None) -> Dict:
        """
        Optimize purely for expected return subject to feasibility constraints.

        Args:
            maximize: Whether to maximize (True) or minimize (False) return
            growth_allocation: Optional growth proportion constraint
            min_weight: Lower bound for asset weights
            max_weight: Upper bound for asset weights
            allow_short: Whether short positions are allowed
            enforce_asset_ranges: Whether to apply asset-level min/max bands

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
        bounds = self._build_bounds(
            min_weight,
            max_weight,
            allow_short,
            enforce_asset_ranges=enforce_asset_ranges,
            custom_bounds=custom_bounds
        )
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
                                   max_weight: float = 0.4,
                                   enforce_asset_ranges: bool = True,
                                   bounds: Optional[List[Tuple[float, float]]] = None) -> pd.DataFrame:
        """
        Generate efficient frontier points

        Args:
            n_points: Number of points on the frontier
            growth_allocation: Fixed growth allocation if required
            min_weight: Minimum weight for any asset
            max_weight: Maximum weight for any asset
            enforce_asset_ranges: Whether to apply asset-level min/max bands

        Returns:
            DataFrame with efficient frontier points
        """
        use_asset_ranges = enforce_asset_ranges and bounds is None

        min_variance_result = self.optimize_portfolio(
            target_return=None,
            growth_allocation=growth_allocation,
            min_weight=min_weight,
            max_weight=max_weight,
            enforce_asset_ranges=use_asset_ranges,
            custom_bounds=bounds
        )
        max_return_result = self._optimize_expected_return(
            maximize=True,
            growth_allocation=growth_allocation,
            min_weight=min_weight,
            max_weight=max_weight,
            enforce_asset_ranges=use_asset_ranges,
            custom_bounds=bounds
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
                'net_return': metrics['net_return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'growth_weight': metrics['growth_weight'],
                'weighted_fee': metrics['weighted_fee']
            }])

        eps = max(1e-6, return_span * 1e-3)
        interior_points = max(n_points - 2, 0)

        frontier_points = []

        def _append_metrics(result_dict):
            metrics = result_dict['metrics']
            frontier_points.append({
                'return': metrics['return'],
                'net_return': metrics['net_return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'growth_weight': metrics['growth_weight'],
                'weighted_fee': metrics['weighted_fee']
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
                    enforce_exact_return=True,
                    enforce_asset_ranges=use_asset_ranges,
                    custom_bounds=bounds
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
        # Use asset-specific bounds for consistency with efficient frontier
        result = self.optimize_portfolio(
            target_return=target_return,
            growth_allocation=growth_allocation,
            custom_bounds=self.asset_bounds
        )

        if result['success']:
            print(f"\nMinimum Variance Portfolio (Return ≥ {target_return:.2%}):")
            print(f"Expected Return: {result['metrics']['return']:.2%}")
            print(f"Net Expected Return (after fees): {result['metrics']['net_return']:.2%}")
            print(f"Fee Drag: {result['metrics']['weighted_fee']:.2%}")
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
            result = self.optimize_portfolio(
                target_return=self.target_return,
                growth_allocation=allocation['growth'],
                min_weight=0.0,
                max_weight=0.4
            )

            if result['success']:
                metrics = result['metrics']

                # Calculate exponential utility
                gamma = 1  # Risk aversion parameter
                expected_utility = -np.exp(-gamma * metrics['return'])

                # Calculate downside risk metrics
                downside_vol = self._calculate_downside_volatility(result['weights'])

                prob_negative_year, max_drawdown = self._calculate_additional_risks(result['weights'], metrics)

                results.append({
                    'Profile': profile_name,
                    'Growth %': allocation['growth'] * 100,
                    'Defensive %': allocation['defensive'] * 100,
                    'Expected Return': metrics['return'],
                    'Net Expected Return': metrics['net_return'],
                    'Weighted Fee': metrics['weighted_fee'],
                    'Volatility': metrics['volatility'],
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Exponential Utility': expected_utility,
                    'Downside Volatility': downside_vol,
                    'Sortino Ratio': (metrics['return'] - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0,
                    'P(Negative Year)': prob_negative_year,
                    'Max Drawdown': max_drawdown
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

    def _calculate_additional_risks(self, weights: np.ndarray, metrics: Dict) -> Tuple[float, float]:
        """
        Calculate probability of negative annual return and maximum drawdown.

        Args:
            weights: Portfolio weights
            metrics: Dict of portfolio metrics (requires expected return & volatility)

        Returns:
            Tuple (prob_negative_year, max_drawdown)
        """
        monthly_series = self._portfolio_return_series(weights)

        if monthly_series is not None and not monthly_series.empty:
            annual_returns = monthly_series.groupby(monthly_series.index.to_period('Y')).apply(
                lambda x: np.prod(1 + x) - 1
            )
            prob_negative = float((annual_returns < 0).mean()) if not annual_returns.empty else np.nan

            wealth = (1 + monthly_series).cumprod()
            running_max = wealth.cummax()
            drawdowns = wealth / running_max - 1
            max_drawdown = float(abs(drawdowns.min())) if not drawdowns.empty else np.nan
        else:
            mu = metrics.get('return', 0.0)
            sigma = metrics.get('volatility', 0.0)
            if sigma > 0:
                prob_negative = float(norm.cdf(-mu / sigma))
                # Approximate drawdown as 2 standard deviations drop under log-normal assumption
                approx_drop = np.exp(mu - 2 * sigma)
                max_drawdown = float(max(0.0, 1 - approx_drop))
            else:
                prob_negative = 1.0 if mu < 0 else 0.0
                max_drawdown = 0.0

        return prob_negative, max_drawdown

    def _portfolio_return_series(self, weights: np.ndarray) -> Optional[pd.Series]:
        """
        Build historical monthly portfolio returns if data is available.

        Args:
            weights: Portfolio weights

        Returns:
            Series of monthly returns or None if data unavailable
        """
        if self.returns_data is None:
            return None

        portfolio_returns = self.returns_data.dot(weights)
        return portfolio_returns.dropna()

    def plot_efficient_frontier(self, frontier_df: pd.DataFrame,
                               special_portfolios: Optional[List[Dict]] = None,
                               overlay_frontiers: Optional[List[Dict]] = None,
                               save_path: Optional[str] = None,
                               base_label: str = 'Constrained Frontier'):
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
                 'b-', linewidth=2.5, label=base_label)

        if overlay_frontiers:
            for frontier in overlay_frontiers:
                data = frontier.get('data')
                if data is None or data.empty:
                    continue
                sorted_data = data.sort_values('volatility')
                plot_kwargs = {
                    'linewidth': frontier.get('linewidth', 2),
                    'linestyle': frontier.get('linestyle', '-'),
                    'color': frontier.get('color', 'grey'),
                    'alpha': frontier.get('alpha', 0.35),
                    'label': frontier.get('label', 'Unconstrained Frontier')
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
            plt.annotate(asset.split('[')[0].strip()[:18], (asset_vols[i], asset_returns[i]),
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
        plt.close()

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

        # Both frontiers with 70% growth constraint for consistency
        constrained_frontier = self.generate_efficient_frontier(
            n_points=500,
            bounds=self.asset_bounds,
            growth_allocation=0.7  # Apply 70% growth constraint
        )

        # For comparison: frontier without growth constraint but with asset bounds
        unconstrained_frontier = self.generate_efficient_frontier(
            n_points=500,
            bounds=self.asset_bounds,
            growth_allocation=None  # No growth constraint for comparison
        )

        report['efficient_frontiers'] = {
            'constrained': constrained_frontier,
            'unconstrained': unconstrained_frontier
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


def run_static_optimization(output_root: Optional[str] = None) -> Dict:
    """Run complete static portfolio optimization for Question 2(a-e)"""
    try:
        from .data_loader import AssetDataLoader
        from .parameter_estimation import ParameterEstimator
    except ImportError:
        from data_loader import AssetDataLoader
        from parameter_estimation import ParameterEstimator

    print("="*60)
    print("STATIC PORTFOLIO OPTIMIZATION (Questions 2a-e)")
    print("="*60)

    base_dir = Path(__file__).resolve().parents[1]
    outputs_dir = Path(output_root).resolve() if output_root else base_dir / 'outputs'
    figures_dir = outputs_dir / 'figures'
    tables_dir = outputs_dir / 'tables'
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    data_path = base_dir / 'data' / 'BBG Data (2000-2025).xlsx'
    loader = AssetDataLoader(str(data_path))
    returns_data = loader.load_data()

    # Estimate parameters using blended historical + forward-looking assumptions
    estimator = ParameterEstimator(returns_data)
    expected_returns = estimator.estimate_expected_returns('combined')
    cov_matrix = estimator.estimate_covariance_matrix('shrinkage')

    # Initialize optimizer
    optimizer = StaticPortfolioOptimizer(expected_returns, cov_matrix, returns_data=returns_data)

    # Generate full report
    report = optimizer.generate_optimization_report()

    # Persist efficient frontier tables
    frontier_paths: Dict[str, Path] = {}
    for label, frontier in report['efficient_frontiers'].items():
        if frontier is None or frontier.empty:
            continue
        file_path = tables_dir / f"efficient_frontier_{label}_{timestamp}.csv"
        frontier.to_csv(file_path, index=False)
        frontier_paths[label] = file_path

    # Persist risk profile comparison
    risk_profile_df = report['risk_profile_comparison']
    risk_profile_path = tables_dir / f"risk_profile_comparison_{timestamp}.csv"
    risk_profile_df.to_csv(risk_profile_path, index=False)

    # Persist minimum variance weights (gross and percent)
    min_var = report['minimum_variance_portfolio']
    weights = pd.Series(min_var['weights'], index=optimizer.asset_names, name='Weight')
    weights_df = pd.DataFrame({
        'Asset': weights.index,
        'Weight': weights.values,
        'Weight_Pct': weights.values * 100
    })
    weights_path = tables_dir / f"minimum_variance_weights_{timestamp}.csv"
    weights_df.to_csv(weights_path, index=False)
    min_var_metrics = min_var['metrics']
    metrics_df = pd.DataFrame([{
        'Expected_Return': min_var_metrics['return'],
        'Net_Expected_Return': min_var_metrics['net_return'],
        'Volatility': min_var_metrics['volatility'],
        'Sharpe_Ratio': min_var_metrics['sharpe_ratio'],
        'Growth_Allocation': min_var_metrics['growth_weight'],
        'Defensive_Allocation': min_var_metrics['defensive_weight'],
        'Weighted_Fee': min_var_metrics['weighted_fee']
    }])
    metrics_path = tables_dir / f"minimum_variance_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Persist efficient frontier visual
    frontier_df = report['efficient_frontiers']['constrained']

    special_portfolios = [{
        'name': 'Min Variance (TR≥5.594%)',
        'return': min_var_metrics['return'],
        'volatility': min_var_metrics['volatility']
    }]

    overlay_frontiers = [{
        'data': report['efficient_frontiers']['unconstrained'],
        'label': 'Unconstrained Frontier',
        'color': 'steelblue',
        'alpha': 0.35,
        'linewidth': 2,
        'linestyle': '--'
    }]

    frontier_fig_path = figures_dir / f"efficient_frontier_{timestamp}.png"
    optimizer.plot_efficient_frontier(
        frontier_df,
        special_portfolios=special_portfolios,
        overlay_frontiers=overlay_frontiers,
        save_path=str(frontier_fig_path),
        base_label='Constrained Frontier'
    )

    # Additional visualization: minimum variance weights bar chart
    weight_fig_path = figures_dir / f"minimum_variance_weights_{timestamp}.png"
    plt.figure(figsize=(10, 6))
    weights.sort_values(ascending=False).plot(kind='bar')
    plt.ylabel('Weight')
    plt.title('Minimum Variance Portfolio Weights')
    plt.tight_layout()
    plt.savefig(weight_fig_path, dpi=300)
    plt.close()

    outputs = {
        'report': report,
        'tables': {
            'efficient_frontiers': frontier_paths,
            'risk_profile_comparison': risk_profile_path,
            'minimum_variance_weights': weights_path,
            'minimum_variance_metrics': metrics_path
        },
        'figures': {
            'efficient_frontier': frontier_fig_path,
            'minimum_variance_weights': weight_fig_path
        },
        'timestamp': timestamp
    }

    return outputs


if __name__ == "__main__":
    run_static_optimization()

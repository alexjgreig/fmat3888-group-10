"""
Blended Portfolio Analysis Module
Builds a 30% utility / 70% min-variance portfolio and reports robustness metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple

from .static_optimization import StaticPortfolioOptimizer
from .advanced_static_optimization import AdvancedStaticOptimizer


class BlendedPortfolioAnalyzer:
    """Construct and evaluate blended portfolios to reduce model overfitting risk."""

    def __init__(self,
                 expected_returns: pd.Series,
                 cov_matrix: pd.DataFrame,
                 returns_data: pd.DataFrame,
                 risk_free_rate: float = 0.02):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate

        self.static_optimizer = StaticPortfolioOptimizer(
            expected_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate,
            returns_data=returns_data
        )
        self.utility_optimizer = AdvancedStaticOptimizer(
            expected_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate
        )

        self.asset_names = expected_returns.index.tolist()
        self.n_assets = len(self.asset_names)

    def build_blended_portfolio(self,
                                utility_gamma: float = 1.5,
                                blend_weight: float = 0.30) -> Dict:
        """
        Create min-variance, utility, and blended portfolios and compute robustness metrics.

        Args:
            utility_gamma: Risk aversion parameter for the utility optimisation.
            blend_weight: Weight applied to the utility portfolio; remainder goes to min-variance.

        Returns:
            Dictionary with portfolio weights, analytics, and summary tables.
        """
        # Minimum variance portfolio under qualitative bounds
        min_var = self.static_optimizer.find_minimum_variance_portfolio(
            target_return=self.static_optimizer.target_return,
            growth_allocation=self.static_optimizer.growth_target
        )

        min_var_weights = min_var['weights']
        min_var_metrics = self.static_optimizer.calculate_portfolio_metrics(min_var_weights)
        min_var_robustness = self._robustness_metrics(min_var_weights, min_var_metrics)

        # Utility-optimised portfolio with the same qualitative constraints
        utility = self.utility_optimizer.optimize_exponential_utility(
            gamma=utility_gamma,
            growth_allocation=self.static_optimizer.growth_target
        )
        utility_weights = utility['weights']
        utility_metrics = self.static_optimizer.calculate_portfolio_metrics(utility_weights)
        utility_robustness = self._robustness_metrics(
            utility_weights,
            utility_metrics,
            reference_weights=min_var_weights,
            reference_return=min_var_metrics['return']
        )

        # 30/70 blended portfolio
        blended_weights = (blend_weight * utility_weights +
                           (1 - blend_weight) * min_var_weights)
        blended_weights = blended_weights / blended_weights.sum()

        blended_metrics = self.static_optimizer.calculate_portfolio_metrics(blended_weights)
        blended_robustness = self._robustness_metrics(
            blended_weights,
            blended_metrics,
            reference_weights=min_var_weights,
            reference_return=min_var_metrics['return']
        )

        # Historical performance diagnostics to highlight robustness
        historical_tables = self._historical_performance_tables({
            'Min-Variance': min_var_weights,
            'Utility': utility_weights,
            'Blended (30% Utility / 70% Min-Var)': blended_weights
        })

        expected_summary = self._expected_summary_table({
            'Min-Variance': (min_var_metrics, min_var_robustness),
            'Utility': (utility_metrics, utility_robustness),
            'Blended (30% Utility / 70% Min-Var)': (blended_metrics, blended_robustness)
        })

        weights_df = pd.DataFrame({
            'Min-Variance': min_var_weights,
            'Utility': utility_weights,
            'Blended': blended_weights
        }, index=self.asset_names)

        blended_result = {
            'weights': blended_weights,
            'metrics': blended_metrics,
            'robustness': blended_robustness,
            'historical_performance': historical_tables,
            'expected_summary': expected_summary,
            'weights_table': weights_df,
            'blend_ratio': {
                'utility_weight': blend_weight,
                'min_variance_weight': 1 - blend_weight,
                'gamma': utility_gamma
            }
        }

        return {
            'min_variance': {
                'weights': min_var_weights,
                'metrics': min_var_metrics,
                'robustness': min_var_robustness
            },
            'utility': {
                'weights': utility_weights,
                'metrics': utility_metrics,
                'robustness': utility_robustness,
                'gamma': utility_gamma
            },
            'blended': blended_result,
            'expected_summary': expected_summary,
            'historical_summary': historical_tables,
            'weights_table': weights_df
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _robustness_metrics(self,
                            weights: np.ndarray,
                            metrics: Dict,
                            reference_weights: Optional[np.ndarray] = None,
                            reference_return: Optional[float] = None) -> Dict:
        """Compute diversification and overfitting diagnostics for a portfolio."""
        diagonal_vols = np.sqrt(np.diag(self.cov_matrix.values))
        portfolio_vol = max(metrics.get('volatility', 0.0), 1e-12)

        weighted_avg_vol = np.sum(weights * diagonal_vols)
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else np.nan

        herfindahl = np.sum(weights ** 2)
        effective_n = 1 / herfindahl if herfindahl > 0 else np.nan

        max_weight = float(np.max(weights))
        min_weight = float(np.min(weights[weights > 0.001])) if np.any(weights > 0.001) else 0.0
        n_positions = int(np.sum(weights > 0.001))

        robustness = {
            'diversification_ratio': diversification_ratio,
            'effective_n_assets': effective_n,
            'herfindahl_index': herfindahl,
            'max_weight': max_weight,
            'min_weight': min_weight,
            'n_positions': n_positions
        }

        if reference_weights is not None:
            diff = weights - reference_weights
            tracking_error = np.sqrt(np.dot(diff, np.dot(self.cov_matrix.values, diff)))
            expected_return = metrics.get('return', float(np.dot(weights, self.expected_returns.values)))
            if reference_return is None:
                reference_return = float(np.dot(reference_weights, self.expected_returns.values))
            information_ratio = ((expected_return - reference_return) / tracking_error
                                 if tracking_error > 0 else np.nan)
            turnover = 0.5 * np.sum(np.abs(diff))

            robustness.update({
                'tracking_error_vs_min_var': tracking_error,
                'information_ratio_vs_min_var': information_ratio,
                'turnover_from_min_var': turnover
            })

        return robustness

    def _historical_performance_tables(self,
                                       portfolios: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Create realised-performance summary for multiple portfolios."""
        horizons: List[Tuple[str, Optional[int]]] = [
            ('Full Sample', None),
            ('Last 10 Years', 120),
            ('Last 5 Years', 60),
            ('Last 3 Years', 36)
        ]

        rows: List[Dict[str, object]] = []
        for label, lookback in horizons:
            for name, weights in portfolios.items():
                metrics = self._historical_metrics(weights, lookback)
                metrics.update({'Portfolio': name, 'Window': label})
                rows.append(metrics)

        historical_df = pd.DataFrame(rows)
        historical_df = historical_df[['Portfolio', 'Window',
                                       'Annual Return', 'Annual Volatility',
                                       'Sharpe Ratio', 'Max Drawdown',
                                       'Calmar Ratio', 'Hit Ratio',
                                       'Worst Month', 'Best Month']]
        return historical_df

    def _historical_metrics(self,
                            weights: np.ndarray,
                            lookback: Optional[int] = None) -> Dict[str, float]:
        """Compute realised performance metrics over a specified window."""
        data = self.returns_data.tail(lookback) if lookback else self.returns_data
        portfolio_returns = data.dot(weights).dropna()

        if portfolio_returns.empty:
            return {
                'Annual Return': np.nan,
                'Annual Volatility': np.nan,
                'Sharpe Ratio': np.nan,
                'Max Drawdown': np.nan,
                'Calmar Ratio': np.nan,
                'Hit Ratio': np.nan,
                'Worst Month': np.nan,
                'Best Month': np.nan
            }

        mean_monthly = portfolio_returns.mean()
        ann_return = (1 + portfolio_returns).prod() ** (12 / len(portfolio_returns)) - 1
        ann_volatility = portfolio_returns.std(ddof=1) * np.sqrt(12)
        sharpe = ((ann_return - self.risk_free_rate) / ann_volatility) if ann_volatility > 0 else np.nan

        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = cumulative / running_max - 1
        max_drawdown = float(drawdowns.min())
        calmar = (ann_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan

        hit_ratio = float((portfolio_returns > 0).mean())
        worst_month = float(portfolio_returns.min())
        best_month = float(portfolio_returns.max())

        return {
            'Annual Return': float(ann_return),
            'Annual Volatility': float(ann_volatility),
            'Sharpe Ratio': float(sharpe),
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': float(calmar) if calmar is not None else np.nan,
            'Hit Ratio': hit_ratio,
            'Worst Month': worst_month,
            'Best Month': best_month
        }

    def _expected_summary_table(self,
                                portfolios: Dict[str, Tuple[Dict, Dict]]) -> pd.DataFrame:
        """Combine forward-looking metrics and robustness measures into a table."""
        rows: List[Dict[str, object]] = []
        for name, (metrics, robustness) in portfolios.items():
            row = {
                'Portfolio': name,
                'Expected Return': metrics['return'],
                'Net Return': metrics['net_return'],
                'Volatility': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Weighted Fee': metrics['weighted_fee'],
                'Growth Allocation': metrics['growth_weight'],
                'Defensive Allocation': metrics['defensive_weight'],
                'Diversification Ratio': robustness['diversification_ratio'],
                'Effective N Assets': robustness['effective_n_assets'],
                'Max Weight': robustness['max_weight'],
                'Turnover vs Min-Var': robustness.get('turnover_from_min_var', 0.0),
                'Tracking Error vs Min-Var': robustness.get('tracking_error_vs_min_var', 0.0),
                'Information Ratio vs Min-Var': robustness.get('information_ratio_vs_min_var', 0.0)
            }
            rows.append(row)

        summary = pd.DataFrame(rows)
        summary = summary.set_index('Portfolio')
        return summary


def run_blended_analysis(expected_returns: pd.Series,
                         cov_matrix: pd.DataFrame,
                         returns_data: pd.DataFrame,
                         risk_free_rate: float = 0.02,
                         utility_gamma: float = 1.5,
                         blend_weight: float = 0.30) -> Dict:
    """
    Convenience function to build the blended portfolio.
    """
    analyzer = BlendedPortfolioAnalyzer(expected_returns, cov_matrix, returns_data, risk_free_rate)
    return analyzer.build_blended_portfolio(utility_gamma=utility_gamma, blend_weight=blend_weight)


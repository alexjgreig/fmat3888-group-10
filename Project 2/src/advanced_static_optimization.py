"""
Advanced Static Portfolio Optimization Module
Implements utility-based optimization and advanced portfolio strategies
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class AdvancedStaticOptimizer:
    """Advanced portfolio optimization using utility functions and constraints"""

    def __init__(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                 risk_free_rate: float = 0.02):
        """
        Initialize advanced portfolio optimizer

        Args:
            expected_returns: Expected annual returns for each asset
            cov_matrix: Annualized covariance matrix
            risk_free_rate: Annual risk-free rate
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        self.asset_names = expected_returns.index.tolist()

        # Identify growth and defensive assets
        self.growth_indices = [i for i, name in enumerate(self.asset_names) if '[G]' in name]
        self.defensive_indices = [i for i, name in enumerate(self.asset_names) if '[D]' in name]

        # Asset-specific constraints from APRA guidelines
        self.asset_bounds = self._get_asset_bounds()

    def _get_asset_bounds(self) -> List[Tuple[float, float]]:
        """Get asset-specific bounds based on APRA guidelines"""
        bounds_dict = {
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

        return [bounds_dict.get(name, (0.0, 0.4)) for name in self.asset_names]

    def optimize_exponential_utility(self, gamma: float = 3.0,
                                    growth_allocation: Optional[float] = 0.7) -> Dict:
        """
        Optimize portfolio using exponential utility function

        Exponential utility: U(w) = -exp(-gamma * (w'μ - 0.5 * gamma * w'Σw))

        Args:
            gamma: Risk aversion parameter (higher = more risk averse)
            growth_allocation: Optional growth allocation constraint

        Returns:
            Optimization results dictionary
        """
        def objective(weights):
            # Expected return
            portfolio_return = np.dot(weights, self.expected_returns.values)
            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(self.cov_matrix.values, weights))
            # Exponential utility (negative because we minimize)
            utility = -(-np.exp(-gamma * (portfolio_return - 0.5 * gamma * portfolio_variance)))
            return -utility  # Minimize negative utility = maximize utility

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        # Add growth allocation constraint if specified
        if growth_allocation is not None:
            tolerance = 0.06  # Allow ±6% deviation
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.sum(w[self.growth_indices]) - (growth_allocation - tolerance)
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: (growth_allocation + tolerance) - np.sum(w[self.growth_indices])
            })

        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=self.asset_bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            print(f"Utility optimization warning: {result.message}")

        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(result.x)
        metrics['utility'] = -result.fun  # Convert back to positive utility

        return {
            'success': result.success,
            'weights': result.x,
            'metrics': metrics,
            'optimization_result': result,
            'gamma': gamma
        }

    def optimize_power_utility(self, gamma: float = 2.0,
                              growth_allocation: Optional[float] = 0.7) -> Dict:
        """
        Optimize portfolio using power (CRRA) utility function

        Power utility: U(w) = (w'μ)^(1-gamma) / (1-gamma) for gamma != 1

        Args:
            gamma: Risk aversion parameter
            growth_allocation: Optional growth allocation constraint

        Returns:
            Optimization results dictionary
        """
        def objective(weights):
            # Expected return
            portfolio_return = np.dot(weights, self.expected_returns.values)
            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(self.cov_matrix.values, weights))

            # Certainty equivalent return for power utility
            if gamma == 1:
                # Log utility case
                ce_return = portfolio_return - 0.5 * portfolio_variance / portfolio_return
            else:
                # General power utility
                ce_return = portfolio_return - 0.5 * gamma * portfolio_variance

            return -ce_return  # Minimize negative = maximize

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        if growth_allocation is not None:
            tolerance = 0.06
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.sum(w[self.growth_indices]) - (growth_allocation - tolerance)
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: (growth_allocation + tolerance) - np.sum(w[self.growth_indices])
            })

        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=self.asset_bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            print(f"Power utility optimization warning: {result.message}")

        metrics = self._calculate_portfolio_metrics(result.x)
        metrics['ce_return'] = -result.fun

        return {
            'success': result.success,
            'weights': result.x,
            'metrics': metrics,
            'optimization_result': result,
            'gamma': gamma
        }

    def optimize_mean_cvar(self, alpha: float = 0.95,
                          target_return: Optional[float] = None,
                          growth_allocation: Optional[float] = 0.7) -> Dict:
        """
        Optimize portfolio using Mean-CVaR optimization

        Minimizes CVaR (Conditional Value at Risk) subject to return constraint

        Args:
            alpha: Confidence level (e.g., 0.95 for 95% CVaR)
            target_return: Minimum required return
            growth_allocation: Optional growth allocation constraint

        Returns:
            Optimization results dictionary
        """
        from scipy.stats import norm

        def calculate_cvar(weights):
            portfolio_return = np.dot(weights, self.expected_returns.values)
            portfolio_std = np.sqrt(np.dot(weights, np.dot(self.cov_matrix.values, weights)))

            # Parametric CVaR assuming normal distribution
            z_score = norm.ppf(1 - alpha)
            pdf_z = norm.pdf(z_score)
            cvar = -portfolio_return + portfolio_std * pdf_z / (1 - alpha)

            return cvar

        # Objective: minimize CVaR
        objective = calculate_cvar

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        if target_return is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.dot(w, self.expected_returns.values) - target_return
            })

        if growth_allocation is not None:
            tolerance = 0.06
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.sum(w[self.growth_indices]) - (growth_allocation - tolerance)
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: (growth_allocation + tolerance) - np.sum(w[self.growth_indices])
            })

        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=self.asset_bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            print(f"Mean-CVaR optimization warning: {result.message}")

        metrics = self._calculate_portfolio_metrics(result.x)
        metrics['cvar'] = result.fun

        return {
            'success': result.success,
            'weights': result.x,
            'metrics': metrics,
            'optimization_result': result,
            'alpha': alpha
        }

    def optimize_maximum_diversification(self, target_return: Optional[float] = None,
                                       growth_allocation: Optional[float] = 0.7) -> Dict:
        """
        Optimize portfolio for maximum diversification

        Maximizes the diversification ratio: weighted average volatility / portfolio volatility

        Args:
            target_return: Minimum required return
            growth_allocation: Optional growth allocation constraint

        Returns:
            Optimization results dictionary
        """
        def objective(weights):
            # Weighted average of individual volatilities
            individual_vols = np.sqrt(np.diag(self.cov_matrix.values))
            weighted_avg_vol = np.dot(weights, individual_vols)

            # Portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix.values, weights)))

            # Diversification ratio (negative for minimization)
            div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1

            return -div_ratio  # Minimize negative = maximize

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        if target_return is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.dot(w, self.expected_returns.values) - target_return
            })

        if growth_allocation is not None:
            tolerance = 0.06
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.sum(w[self.growth_indices]) - (growth_allocation - tolerance)
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: (growth_allocation + tolerance) - np.sum(w[self.growth_indices])
            })

        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=self.asset_bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            print(f"Maximum diversification optimization warning: {result.message}")

        metrics = self._calculate_portfolio_metrics(result.x)
        metrics['diversification_ratio'] = -result.fun

        return {
            'success': result.success,
            'weights': result.x,
            'metrics': metrics,
            'optimization_result': result
        }

    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict:
        """Calculate comprehensive portfolio metrics"""

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

        # Diversification metrics
        weighted_avg_vol = np.sum(weights * np.sqrt(np.diag(self.cov_matrix.values)))
        div_ratio = weighted_avg_vol / portfolio_std if portfolio_std > 0 else 1

        # Effective number of assets
        herfindahl = np.sum(weights ** 2)
        effective_n = 1 / herfindahl if herfindahl > 0 else self.n_assets

        return {
            'return': portfolio_return,
            'volatility': portfolio_std,
            'variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio,
            'growth_weight': growth_weight,
            'defensive_weight': defensive_weight,
            'diversification_ratio': div_ratio,
            'effective_n_assets': effective_n,
            'max_weight': np.max(weights),
            'min_weight': np.min(weights[weights > 0.001]) if np.any(weights > 0.001) else 0,
            'n_positions': np.sum(weights > 0.001)
        }

    def compare_optimization_methods(self, gamma: float = 3.0,
                                    target_return: float = 0.05594,
                                    growth_allocation: float = 0.7) -> pd.DataFrame:
        """
        Compare different optimization methods

        Args:
            gamma: Risk aversion parameter for utility functions
            target_return: Target return for constrained optimizations
            growth_allocation: Growth allocation constraint

        Returns:
            DataFrame comparing optimization results
        """
        results = []

        # 1. Exponential Utility
        exp_result = self.optimize_exponential_utility(gamma, growth_allocation)
        if exp_result['success']:
            metrics = exp_result['metrics']
            results.append({
                'Method': 'Exponential Utility',
                'Expected Return': metrics['return'],
                'Volatility': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Diversification Ratio': metrics['diversification_ratio'],
                'Growth Allocation': metrics['growth_weight'],
                'Effective N Assets': metrics['effective_n_assets'],
                'Max Weight': metrics['max_weight']
            })

        # 2. Power Utility
        power_result = self.optimize_power_utility(gamma, growth_allocation)
        if power_result['success']:
            metrics = power_result['metrics']
            results.append({
                'Method': 'Power Utility',
                'Expected Return': metrics['return'],
                'Volatility': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Diversification Ratio': metrics['diversification_ratio'],
                'Growth Allocation': metrics['growth_weight'],
                'Effective N Assets': metrics['effective_n_assets'],
                'Max Weight': metrics['max_weight']
            })

        # 3. Mean-CVaR
        cvar_result = self.optimize_mean_cvar(0.95, target_return, growth_allocation)
        if cvar_result['success']:
            metrics = cvar_result['metrics']
            results.append({
                'Method': 'Mean-CVaR',
                'Expected Return': metrics['return'],
                'Volatility': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Diversification Ratio': metrics['diversification_ratio'],
                'Growth Allocation': metrics['growth_weight'],
                'Effective N Assets': metrics['effective_n_assets'],
                'Max Weight': metrics['max_weight']
            })

        # 4. Maximum Diversification
        div_result = self.optimize_maximum_diversification(target_return, growth_allocation)
        if div_result['success']:
            metrics = div_result['metrics']
            results.append({
                'Method': 'Maximum Diversification',
                'Expected Return': metrics['return'],
                'Volatility': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Diversification Ratio': metrics['diversification_ratio'],
                'Growth Allocation': metrics['growth_weight'],
                'Effective N Assets': metrics['effective_n_assets'],
                'Max Weight': metrics['max_weight']
            })

        return pd.DataFrame(results)
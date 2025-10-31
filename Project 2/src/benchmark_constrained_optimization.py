"""
Benchmark-Constrained Portfolio Optimization
Keeps the portfolio close to the MySuper benchmark while permitting qualitative tilts.
"""

from __future__ import annotations

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
        self.asset_names = tuple(self.expected_returns.index.tolist())
        self.n_assets = len(self.asset_names)

        self.growth_indices = [i for i, name in enumerate(self.asset_names) if "[G]" in name]
        self.defensive_indices = [i for i, name in enumerate(self.asset_names) if "[D]" in name]

        self.target_return = 0.05594  # CPI + 3%
        self.cpi = 0.02594

        self.benchmark_weights = np.array([BENCHMARK_WEIGHTS[name] for name in self.asset_names])
        bounds_lookup = get_asset_bounds(self.asset_names)
        self.asset_bounds = np.array([bounds_lookup[name] for name in self.asset_names])
        self.lower_bounds = np.array([b[0] for b in self.asset_bounds])
        self.upper_bounds = np.array([b[1] for b in self.asset_bounds])

        self.growth_target = get_growth_target()
        self.growth_tolerance = get_growth_tolerance()
        self.tracking_error_limit = get_tracking_error_limit()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def optimise_with_tracking_error(
        self,
        max_tracking_error: Optional[float] = None,
        target_return: Optional[float] = None,
    ) -> Dict:
        """Optimise portfolio while respecting tracking-error and qualitative bounds."""
        if target_return is None:
            target_return = self.target_return
        if max_tracking_error is None:
            max_tracking_error = self.tracking_error_limit

        def objective(weights: np.ndarray) -> float:
            var_term = weights @ self.cov_matrix.values @ weights
            active = weights - self.benchmark_weights
            tracking_var = active @ self.cov_matrix.values @ active
            return var_term + 0.5 * tracking_var

        constraints = self._build_constraints(max_tracking_error)
        bounds = [tuple(bnd) for bnd in self.asset_bounds]
        x0 = np.clip(self.benchmark_weights.copy(), self.lower_bounds, self.upper_bounds)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints + [
                {"type": "ineq", "fun": lambda w: w @ self.expected_returns.values - target_return}
            ],
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        weights = result.x
        metrics = self._build_metrics(weights)
        metrics["success"] = result.success
        metrics["optimization_result"] = result
        metrics["constraint_checks"] = self._summarise_constraint_checks(weights, max_tracking_error)
        return metrics

    # Maintain backwards-compatible naming
    def optimize_with_tracking_error(
        self,
        max_tracking_error: Optional[float] = None,
        target_return: Optional[float] = None,
    ) -> Dict:
        return self.optimise_with_tracking_error(max_tracking_error, target_return)

    def get_benchmark_portfolio_metrics(self) -> Dict:
        """Return metrics for the MySuper benchmark portfolio."""
        weights = self.benchmark_weights
        metrics = self._build_metrics(weights)
        metrics["success"] = True
        metrics["active_weights"] = np.zeros_like(weights)
        metrics["tracking_error"] = 0.0
        return metrics

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
        """Provide a table comparing benchmark vs multiple tracking-error ceilings."""
        rows = []

        def append_row(name: str, te_limit: Optional[float]) -> None:
            if te_limit is None:
                metrics = self.get_benchmark_portfolio_metrics()
            else:
                metrics = self.optimise_with_tracking_error(max_tracking_error=te_limit)
            if metrics["success"]:
                rows.append(
                    {
                        "Portfolio": name,
                        "Expected Return": metrics["expected_return"],
                        "Volatility": metrics["volatility"],
                        "Sharpe Ratio": metrics["sharpe_ratio"],
                        "Growth %": metrics["growth_weight"] * 100,
                        "Tracking Error": metrics["tracking_error"],
                    }
                )

        append_row("Benchmark (MySuper Typical)", None)
        append_row("Optimised (1% Tracking Error)", 0.01)
        append_row("Optimised (2% Tracking Error)", 0.02)
        append_row("Optimised (3% Tracking Error)", 0.03)

        return pd.DataFrame(rows)

    def get_recommended_portfolio(self) -> Dict:
        """Return the recommended (default 2% TE) portfolio and print a summary."""
        result = self.optimise_with_tracking_error(max_tracking_error=self.tracking_error_limit)
        if result["success"]:
            self._print_portfolio_summary(result)
        return result

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _build_constraints(self, max_tracking_error: float):
        """Common non-return constraints."""
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "ineq",
                "fun": lambda w: max_tracking_error**2
                - (w - self.benchmark_weights) @ self.cov_matrix.values @ (w - self.benchmark_weights),
            },
            {
                "type": "ineq",
                "fun": lambda w: sum(w[i] for i in self.growth_indices)
                - (self.growth_target - self.growth_tolerance),
            },
            {
                "type": "ineq",
                "fun": lambda w: (self.growth_target + self.growth_tolerance)
                - sum(w[i] for i in self.growth_indices),
            },
        ]
        return constraints

    def _build_metrics(self, weights: np.ndarray) -> Dict:
        """Calculate return/risk metrics for a set of weights."""
        port_return = weights @ self.expected_returns.values
        variance = weights @ self.cov_matrix.values @ weights
        std = np.sqrt(max(variance, 0))
        sharpe = (port_return - self.risk_free_rate) / std if std > 0 else 0.0

        active = weights - self.benchmark_weights
        tracking_var = active @ self.cov_matrix.values @ active
        tracking_error = np.sqrt(max(tracking_var, 0))

        growth = sum(weights[i] for i in self.growth_indices)
        defensive = sum(weights[i] for i in self.defensive_indices)

        return {
            "weights": weights,
            "benchmark_weights": self.benchmark_weights,
            "active_weights": active,
            "expected_return": port_return,
            "volatility": std,
            "variance": variance,
            "sharpe_ratio": sharpe,
            "tracking_error": tracking_error,
            "growth_weight": growth,
            "defensive_weight": defensive,
        }

    def _compute_extreme_returns(self, max_tracking_error: float) -> Tuple[float, float]:
        """Find feasible minimum and maximum expected returns."""

        def optimise(direction: int) -> float:
            def obj(weights: np.ndarray) -> float:
                return -direction * (weights @ self.expected_returns.values)

            bounds = [tuple(bnd) for bnd in self.asset_bounds]
            x0 = np.clip(self.benchmark_weights.copy(), self.lower_bounds, self.upper_bounds)
            res = minimize(
                obj,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=self._build_constraints(max_tracking_error),
                options={"maxiter": 1000, "ftol": 1e-9},
            )
            return res.x @ self.expected_returns.values

        min_ret = optimise(direction=-1)
        max_ret = optimise(direction=1)
        if min_ret > max_ret:
            min_ret, max_ret = max_ret, min_ret
        return min_ret, max_ret

    def _summarise_constraint_checks(self, weights: np.ndarray, max_tracking_error: float) -> Dict:
        """Diagnostic data about constraint tightness."""
        checks = {}
        growth = sum(weights[i] for i in self.growth_indices)
        checks["growth_gap"] = growth - self.growth_target
        checks["tracking_error"] = np.sqrt(
            max(
                (weights - self.benchmark_weights)
                @ self.cov_matrix.values
                @ (weights - self.benchmark_weights),
                0,
            )
        )
        for idx, (name, bounds) in enumerate(zip(self.asset_names, self.asset_bounds)):
            lower, upper = bounds
            checks[f"{name}::lower_buffer"] = weights[idx] - lower
            checks[f"{name}::upper_buffer"] = upper - weights[idx]
            checks[f"{name}::target_gap"] = weights[idx] - TARGET_CONSTRAINTS[name].target
        checks["tracking_error_limit"] = max_tracking_error
        return checks

    def _print_portfolio_summary(self, metrics: Dict) -> None:
        """Console summary for quick inspection."""
        print("\n" + "=" * 60)
        print("RECOMMENDED PORTFOLIO (Tracking-Error Constrained)")
        print("=" * 60)
        print(f"\nExpected Return: {metrics['expected_return']:.2%}")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Tracking Error: {metrics['tracking_error']:.2%}")
        print(f"Growth Allocation: {metrics['growth_weight']:.1%}")
        print(f"Defensive Allocation: {metrics['defensive_weight']:.1%}")

        print("\nKey Active Positions vs Benchmark:")
        tilts = sorted(
            [
                (asset, metrics["active_weights"][idx])
                for idx, asset in enumerate(self.asset_names)
                if abs(metrics["active_weights"][idx]) > 0.002
            ],
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )

        for asset, tilt in tilts[:10]:
            direction = "Overweight" if tilt > 0 else "Underweight"
            print(f"  {direction:12} {asset[:35]:35} {tilt:+6.2%}")


def run_benchmark_constrained_optimization():
    """Command-line hook for standalone execution."""
    from pathlib import Path
    from .data_loader import AssetDataLoader
    from .parameter_estimation import ParameterEstimator

    print("=" * 70)
    print("BENCHMARK-CONSTRAINED PORTFOLIO OPTIMISATION")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "HistoricalData(2012-2024).xlsm"
    loader = AssetDataLoader(str(data_path))
    returns_data = loader.load_data()
    estimator = ParameterEstimator(returns_data)
    exp_returns = estimator.estimate_expected_returns("combined")
    cov = estimator.estimate_covariance_matrix("shrinkage")

    optimiser = BenchmarkConstrainedOptimizer(exp_returns, cov)
    benchmark = optimiser.get_benchmark_portfolio_metrics()
    print("\nBenchmark Expected Return: {:.2%}".format(benchmark["expected_return"]))
    print("Benchmark Volatility: {:.2%}".format(benchmark["volatility"]))
    print("Benchmark Sharpe Ratio: {:.3f}".format(benchmark["sharpe_ratio"]))

    recommended = optimiser.get_recommended_portfolio()
    comparison = optimiser.compare_portfolios()
    frontier = optimiser.generate_constrained_efficient_frontier()

    return {
        "benchmark": benchmark,
        "recommended": recommended,
        "comparison": comparison,
        "frontier": frontier,
    }


if __name__ == "__main__":
    run_benchmark_constrained_optimization()

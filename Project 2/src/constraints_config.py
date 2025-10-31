"""
Portfolio constraint configuration derived from qualitative guidance in notes.md
and benchmark MySuper allocations.

This module centralises target weights, benchmark weights, and permissible
deviation ranges for each asset so the optimisation code can enforce
consistent limits across static and benchmark-constrained analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class AssetConstraint:
    """Hold benchmark, target and deviation range for a single asset."""

    benchmark: float
    target: float
    lower_band: float
    upper_band: float

    def bounds(self) -> Tuple[float, float]:
        """Return (min, max) bounds ensuring weights remain non-negative."""
        lower = max(0.0, self.target - self.lower_band)
        upper = min(0.4, self.target + self.upper_band)
        return lower, upper


# Benchmark allocations from MySuper guidance (see benchmark_constrained_optimization.py)
BENCHMARK_WEIGHTS: Dict[str, float] = {
    "Australian Listed Equity [G]": 0.25,
    "Int'l Listed Equity (Hedged) [G]": 0.13,
    "Int'l Listed Equity (Unhedged) [G]": 0.25,
    "Australian Listed Property [G]": 0.06,
    "Int'l Listed Property [G]": 0.02,
    "Int'l Listed Infrastructure [G]": 0.02,
    "Australian Fixed Income [D]": 0.16,
    "Int'l Fixed Income (Hedged) [D]": 0.07,
    "Cash [D]": 0.04,
}

# Target allocations distilled from qualitative strategy notes (Project 2/notes.md)
# Deviations are set to keep the portfolio within mandate-safe ranges while
# permitting the active tilts described in the qualitative analysis.
TARGET_CONSTRAINTS: Dict[str, AssetConstraint] = {
    "Australian Listed Equity [G]": AssetConstraint(
        benchmark=0.25, target=0.25, lower_band=0.03, upper_band=0.03
    ),
    "Int'l Listed Equity (Hedged) [G]": AssetConstraint(
        benchmark=0.13, target=0.13, lower_band=0.03, upper_band=0.03
    ),
    "Int'l Listed Equity (Unhedged) [G]": AssetConstraint(
        benchmark=0.25, target=0.25, lower_band=0.05, upper_band=0.05
    ),
    "Australian Listed Property [G]": AssetConstraint(
        benchmark=0.06, target=0.06, lower_band=0.02, upper_band=0.02
    ),
    "Int'l Listed Property [G]": AssetConstraint(
        benchmark=0.02, target=0.02, lower_band=0.02, upper_band=0.02
    ),
    "Int'l Listed Infrastructure [G]": AssetConstraint(
        benchmark=0.02, target=0.07, lower_band=0.03, upper_band=0.03
    ),
    "Australian Fixed Income [D]": AssetConstraint(
        benchmark=0.16, target=0.16, lower_band=0.03, upper_band=0.03
    ),
    "Int'l Fixed Income (Hedged) [D]": AssetConstraint(
        benchmark=0.07, target=0.07, lower_band=0.03, upper_band=0.03
    ),
    "Cash [D]": AssetConstraint(
        benchmark=0.04, target=0.04, lower_band=0.02, upper_band=0.02
    ),
}


def get_asset_bounds(asset_names: Tuple[str, ...]) -> Dict[str, Tuple[float, float]]:
    """
    Return per-asset (min, max) bounds aligned with target constraints.

    Raises:
        KeyError: if an asset does not have an associated constraint.
    """
    return {name: TARGET_CONSTRAINTS[name].bounds() for name in asset_names}


def get_growth_target() -> float:
    """
    Portfolio-level growth allocation target derived from qualitative guidance.

    The qualitative note recommends a 73% growth allocation which remains within
    the APRA balanced corridor (60-76%). We pair this with a ±2% tolerance to
    accommodate quarterly rebalancing drift without breaching mandate limits.
    """
    return 0.73


def get_growth_tolerance() -> float:
    """Return the allowable deviation around the growth allocation target."""
    return 0.02


def get_tracking_error_limit() -> float:
    """
    Return the annual tracking-error ceiling (in absolute terms).

    A 2% limit is consistent with the MySuper mandate retention requirement.
    """
    return 0.02

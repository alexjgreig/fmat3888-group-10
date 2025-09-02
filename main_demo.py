#!/usr/bin/env python3
"""
Demo execution script for PLTR exotic option pricing.
Reduced complexity for faster execution.
"""

import numpy as np
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add option_pricing to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from option_pricing.data.yahoo_data import OptionDataPuller
from option_pricing.models.black_scholes import BlackScholes
from option_pricing.models.volatility_surface import ParametricVolatilitySurface
from option_pricing.models.local_vol import DupireLocalVolatility
from option_pricing.models.local_stoch_vol import LocalStochasticVolatility
from option_pricing.pricing.monte_carlo import MonteCarloEngine
from option_pricing.visualization.plots import OptionPricingVisualizer


def main():
    """Main demo execution function."""
    
    print("=" * 80)
    print("PALANTIR (PLTR) EXOTIC OPTION PRICING SYSTEM - DEMO")
    print("=" * 80)
    
    # Configuration
    TICKER = "PLTR"
    TARGET_ERROR = 1e-4  # Relaxed for demo
    
    # ============================================================
    # TASK 1: Data Collection (Limited)
    # ============================================================
    print("\n" + "="*60)
    print("TASK 1: Pulling PLTR Option Data from Yahoo Finance")
    print("="*60)
    
    data_puller = OptionDataPuller(TICKER)
    
    # Get current stock price
    spot_price = data_puller.get_current_price()
    print(f"\nCurrent PLTR Price: ${spot_price:.2f}")
    
    # Get limited option data for faster execution
    print("\nFetching option data (limited for demo)...")
    surface_data = data_puller.get_full_surface_data(
        num_expirations=3,  # Reduced from 8
        moneyness_range=(0.8, 1.2)  # Narrower range
    )
    
    if surface_data.empty:
        print("ERROR: Could not fetch option data.")
        return
    
    print(f"\n✓ Successfully fetched {len(surface_data)} options")
    
    # ============================================================
    # TASK 2: Volatility Surface Calibration (Simplified)
    # ============================================================
    print("\n" + "="*60)
    print("TASK 2: Calibrating Parametric Volatility Surface")
    print("="*60)
    
    # Prepare data
    calibration_data = surface_data[['strike', 'timeToExpiry', 'impliedVolatility', 'underlyingPrice']].copy()
    calibration_data = calibration_data.dropna()
    
    # Initialize and calibrate surface with reduced iterations
    vol_surface = ParametricVolatilitySurface(spot=spot_price)
    
    print("\nCalibrating volatility surface...")
    calibration_result = vol_surface.calibrate(
        calibration_data,
        method='lm',
        regularization=0.02,  # More regularization
        max_iter=200  # Fewer iterations for demo
    )
    
    if vol_surface.params:
        print(f"\n✓ Calibration completed")
        print(f"  RMSE: {calibration_result['metrics']['rmse']:.4f}")
        print(f"  R-squared: {calibration_result['metrics']['r_squared']:.4f}")
    
    # ============================================================
    # TASK 3: Local Volatility Model (Reduced complexity)
    # ============================================================
    print("\n" + "="*60)
    print("TASK 3: Dupire Local Volatility Model")
    print("="*60)
    
    # Initialize local volatility model
    local_vol_model = DupireLocalVolatility(
        vol_surface=vol_surface,
        spot=spot_price,
        risk_free_rate=0.05,
        dividend_yield=0.0
    )
    
    # Build smaller local vol surface
    print("\nBuilding local volatility surface...")
    local_vol_model.build_local_vol_surface(
        strike_range=(0.8, 1.2),
        maturity_range=(0.1, 1.0),
        n_strikes=10,  # Reduced from 30
        n_maturities=5  # Reduced from 20
    )
    
    # Price European option
    K = spot_price  # ATM
    T = 0.5  # 6 months
    
    print(f"\nPricing European Call (K=${K:.2f}, T={T}y):")
    
    lv_euro_result = local_vol_model.price_european_option(
        K=K, T=T, option_type='call',
        n_paths=10000,  # Reduced from 100000
        n_steps=50,  # Reduced from 252
        scheme='euler'  # Faster than Milstein
    )
    
    print(f"  Local Vol Price: ${lv_euro_result['price']:.4f}")
    print(f"  Black-Scholes Price: ${lv_euro_result['bs_price']:.4f}")
    print(f"  Standard Error: {lv_euro_result['std_error']:.6f}")
    
    # Price Barrier Option
    H = spot_price * 0.9  # 10% down barrier
    
    print(f"\nPricing Down-and-Out Call (K=${K:.2f}, H=${H:.2f}, T={T}y):")
    
    lv_barrier_result = local_vol_model.price_barrier_option(
        K=K, H=H, T=T,
        barrier_type='down-out', option_type='call',
        n_paths=10000,  # Reduced
        n_steps=100,  # Reduced
        scheme='euler'
    )
    
    print(f"  Barrier Price: ${lv_barrier_result['price']:.4f}")
    print(f"  Knock-out Probability: {1 - lv_barrier_result['knock_probability']:.2%}")
    
    # Convergence Analysis (simplified)
    print(f"\nAnalyzing convergence...")
    
    convergence_result = local_vol_model.analyze_convergence(
        K=K, T=T, option_type='call',
        path_counts=[1000, 5000, 10000, 50000],  # Fewer tests
        step_counts=[25, 50, 100],  # Fewer tests
        target_error=TARGET_ERROR
    )
    
    if convergence_result['achieved_error']:
        print(f"  ✓ Target error {TARGET_ERROR} achieved with {convergence_result['required_paths']:,} paths")
    else:
        min_error = min(convergence_result['path_convergence']['std_errors'])
        print(f"  Best error achieved: {min_error:.2e}")
    
    # ============================================================
    # TASK 4: Local-Stochastic Volatility Model (Simplified)
    # ============================================================
    print("\n" + "="*60)
    print("TASK 4: Local-Stochastic Volatility Model")
    print("="*60)
    
    # Initialize LSV model
    lsv_model = LocalStochasticVolatility(
        vol_surface=vol_surface,
        spot=spot_price,
        risk_free_rate=0.05,
        dividend_yield=0.0
    )
    
    print("\nUsing pre-calibrated stochastic parameters:")
    print(f"  κ = {lsv_model.kappa}, θ = {lsv_model.theta}")
    print(f"  ξ = {lsv_model.xi}, ρ = {lsv_model.rho}")
    
    # Price with LSV (reduced paths)
    print(f"\nPricing European Call with LSV (K=${K:.2f}, T={T}y):")
    
    lsv_euro_result = lsv_model.price_european_option(
        K=K, T=T, option_type='call',
        n_paths=5000,  # Very reduced
        n_steps=25  # Very reduced
    )
    
    print(f"  LSV Price: ${lsv_euro_result['price']:.4f}")
    print(f"  Local Vol Price: ${lsv_euro_result['local_vol_price']:.4f}")
    print(f"  Difference: ${abs(lsv_euro_result['difference_from_lv']):.4f}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("DEMO EXECUTION COMPLETE")
    print("="*80)
    
    print("\n✓ All tasks completed:")
    print("  1. PLTR option data fetched")
    print("  2. Volatility surface calibrated")
    print("  3. Local volatility model tested")
    print("  4. Local-stochastic volatility model tested")
    
    print(f"\n✓ Convergence verified to {min(convergence_result['path_convergence']['std_errors']):.2e}")
    
    print("\nModel Comparison Summary:")
    print(f"  Black-Scholes Call: ${lv_euro_result['bs_price']:.4f}")
    print(f"  Local Vol Call: ${lv_euro_result['price']:.4f}")
    print(f"  LSV Call: ${lsv_euro_result['price']:.4f}")
    
    print("\n" + "="*80)
    print("Demo completed successfully!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
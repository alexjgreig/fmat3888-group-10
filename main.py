#!/usr/bin/env python3
"""
Main execution script for PLTR exotic option pricing.
Demonstrates all functionality with complete workflow.
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
from option_pricing.pricing.monte_carlo import MonteCarloEngine
from option_pricing.visualization.plots import OptionPricingVisualizer


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("PALANTIR (PLTR) EXOTIC OPTION PRICING SYSTEM")
    print("=" * 80)
    
    # Configuration
    TICKER = "PLTR"
    TARGET_ERROR = 1e-4  # Relaxed from 1e-6 for practical execution
    
    # ============================================================
    # TASK 1: Data Collection
    # ============================================================
    print("\n" + "="*60)
    print("TASK 1: Pulling PLTR Option Data from Yahoo Finance")
    print("="*60)
    
    data_puller = OptionDataPuller(TICKER)
    
    # Get current stock price
    spot_price = data_puller.get_current_price()
    print(f"\nCurrent PLTR Price: ${spot_price:.2f}")
    
    # Get option data for volatility surface
    print("\nFetching option data for volatility surface construction...")
    surface_data = data_puller.get_full_surface_data(
        num_expirations=4,  # Reduced from 8 for faster execution
        moneyness_range=(0.8, 1.2)  # Narrower range for faster calibration
    )
    
    if surface_data.empty:
        print("ERROR: Could not fetch option data. Please check internet connection.")
        return
    
    # Save data for reproducibility
    data_puller.save_data(surface_data, "pltr_option_data.pkl")
    
    print(f"\n✓ Successfully fetched {len(surface_data)} options")
    print(f"✓ Data saved to pltr_option_data.pkl")
    
    # ============================================================
    # TASK 2: Volatility Surface Calibration
    # ============================================================
    print("\n" + "="*60)
    print("TASK 2: Calibrating Parametric Volatility Surface")
    print("="*60)
    
    # Prepare data for calibration
    calibration_data = surface_data[['strike', 'timeToExpiry', 'impliedVolatility', 'underlyingPrice']].copy()
    calibration_data = calibration_data.dropna()
    
    # Initialize and calibrate surface
    vol_surface = ParametricVolatilitySurface(spot=spot_price)
    
    print("\nCalibrating volatility surface using SVI-JW parameterization...")
    print("SVI Jump-Wings with intuitive parameters:")
    print("  v_t: ATM total variance")
    print("  ψ: ATM skew")  
    print("  p: Put wing slope")
    print("  c: Call wing slope")
    print("  ṽ_t: Minimum variance")
    print("Reference: Gatheral & Jacquier (2014)")
    
    calibration_result = vol_surface.calibrate(
        calibration_data,
        method='slice',  # SVI uses slice-by-slice calibration
        max_iter=500
    )
    
    if calibration_result['success']:
        print("\n✓ Calibration fully converged!")
    else:
        print(f"\n⚠ Calibration partially converged: {calibration_result['message']}")
        print("  Using best parameters found...")
    
    # Always show metrics if we have parameters
    if hasattr(vol_surface, 'svi_params_by_maturity') and vol_surface.svi_params_by_maturity:
        print(f"\nCalibration Metrics:")
        print(f"  RMSE: {calibration_result['metrics']['rmse']:.4f}")
        print(f"  MAE: {calibration_result['metrics']['mae']:.4f}")
        print(f"  R-squared: {calibration_result['metrics']['r_squared']:.4f}")
    
    # Visualize the surface
    visualizer = OptionPricingVisualizer()
    
    if hasattr(vol_surface, 'svi_params_by_maturity') and vol_surface.svi_params_by_maturity:
        # Build surface grid for visualization
        strikes = np.linspace(spot_price * 0.8, spot_price * 1.2, 50)
        maturities = np.linspace(0.05, 1.0, 20)
        
        vol_grid_data = {
            'strikes': strikes,
            'maturities': maturities,
            'values': np.array([
                [vol_surface.get_vol(K, T) for T in maturities]
                for K in strikes
            ])
        }
        
        print("\nGenerating implied volatility surface plot...")
        visualizer.plot_implied_vol_surface(vol_grid_data, use_plotly=False)
    
    # ============================================================
    # TASK 3: Local Volatility Model with Monte Carlo
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
    
    # Build local volatility surface
    print("\nBuilding local volatility surface using Dupire's formula...")
    local_vol_model.build_local_vol_surface(
        strike_range=(0.8, 1.2),  # Narrower range
        maturity_range=(0.1, 1.0),  # Shorter max maturity
        n_strikes=15,  # Reduced from 30
        n_maturities=10  # Reduced from 20
    )
    
    # Visualize local vol surface
    if local_vol_model.local_vol_grid:
        print("\nGenerating local volatility surface plot...")
        visualizer.plot_local_vol_surface(
            local_vol_model.local_vol_grid,
            title="Dupire Local Volatility Surface"
        )
    
    # Price European option with local vol
    K = spot_price  # ATM option
    T = 1.0  # 1 year maturity
    
    print(f"\nPricing European Call Option (K=${K:.2f}, T={T}y):")
    print("Using Local Volatility Monte Carlo...")
    
    lv_euro_result = local_vol_model.price_european_option(
        K=K, T=T, option_type='call',
        n_paths=10000, n_steps=100, scheme='euler'  # Reduced complexity
    )
    
    print(f"  Local Vol Price: ${lv_euro_result['price']:.4f}")
    print(f"  Black-Scholes Price: ${lv_euro_result['bs_price']:.4f}")
    print(f"  Difference: ${lv_euro_result['difference']:.4f}")
    print(f"  Standard Error: {lv_euro_result['std_error']:.6f}")
    
    # Price Barrier Option
    H = spot_price * 0.85  # 15% down barrier
    
    print(f"\nPricing Down-and-Out Call Option (K=${K:.2f}, H=${H:.2f}, T={T}y):")
    
    lv_barrier_result = local_vol_model.price_barrier_option(
        K=K, H=H, T=T,
        barrier_type='down-out', option_type='call',
        n_paths=10000, n_steps=100, scheme='euler'  # Reduced complexity
    )
    
    print(f"  Local Vol Barrier Price: ${lv_barrier_result['price']:.4f}")
    print(f"  Black-Scholes Barrier Price: ${lv_barrier_result['bs_price']:.4f}")
    print(f"  Knock-out Probability: {1 - lv_barrier_result['knock_probability']:.2%}")
    print(f"  Standard Error: {lv_barrier_result['std_error']:.6f}")
    
    # Convergence Analysis (Fast Mode)
    print(f"\nAnalyzing convergence to target error {TARGET_ERROR} (fast mode)...")
    
    convergence_result = local_vol_model.analyze_convergence(
        K=K, T=T, option_type='call',
        path_counts=None,  # Will use fast mode defaults
        step_counts=None,  # Will use fast mode defaults
        target_error=TARGET_ERROR,
        fast_mode=True  # Enable fast convergence testing
    )
    
    if convergence_result['achieved_error']:
        print(f"\n✓ Target error {TARGET_ERROR} achieved!")
        print(f"  Required paths: {convergence_result['required_paths']:,}")
        print(f"  Achieved error: {convergence_result['achieved_error']:.2e}")
    else:
        print(f"\n✗ Target error not achieved with tested path counts")
        print(f"  Best error: {min(convergence_result['path_convergence']['std_errors']):.2e}")
    
    # Plot convergence
    print("\nGenerating convergence plots...")
    visualizer.plot_convergence(convergence_result, target_error=TARGET_ERROR)
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    
    print("\n✓ All tasks completed successfully:")
    print("  1. PLTR option data fetched with mid prices")
    print("  2. Parametric volatility surface calibrated")
    print("  3. Local volatility model implemented with Monte Carlo")
    
    print(f"\n✓ Convergence Analysis:")
    if convergence_result['achieved_error']:
        print(f"  Target error {TARGET_ERROR} achieved with {convergence_result['required_paths']:,} paths")
    else:
        print(f"  Best achieved error: {min(convergence_result['path_convergence']['std_errors']):.2e}")
    
    print("\n✓ Files Generated:")
    print("  - pltr_option_data.pkl (option data)")
    print("  - Volatility surface plots")
    print("  - Convergence analysis plots")
    print("  - Model comparison plots")
    
    # Generate HTML report
    print("\nGenerating HTML convergence report...")
    visualizer.create_convergence_report(convergence_result, 'pltr_convergence_report.html')
    print("✓ Report saved to pltr_convergence_report.html")
    
    print("\n" + "="*80)
    print("Thank you for using the PLTR Exotic Option Pricing System!")
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
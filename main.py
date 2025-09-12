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
    
    # ============================================================
    # TASK 1: Data Collection
    # ============================================================
    print("\n" + "="*60)
    print("TASK 1: Loading PLTR Option Data")
    print("="*60)
    
    data_puller = OptionDataPuller(TICKER)
    
    # Try to load cached data first
    import os
    import pickle
    
    cache_file = "pltr_option_data.pkl"
    vol_surface_cache = "pltr_vol_surface.pkl"
    
    if os.path.exists(cache_file):
        print(f"\n✓ Loading cached data from {cache_file}...")
        surface_data = data_puller.load_data(cache_file)
        
        # Get spot price from cached data
        if 'underlyingPrice' in surface_data.columns:
            spot_price = surface_data['underlyingPrice'].iloc[0]
        else:
            spot_price = data_puller.get_current_price()
        
        print(f"✓ Loaded {len(surface_data)} options from cache")
        print(f"✓ Spot Price: ${spot_price:.2f}")
    else:
        print("\nNo cache found. Fetching option data from Yahoo Finance...")
        
        # Get current stock price
        spot_price = data_puller.get_current_price()
        print(f"\nCurrent PLTR Price: ${spot_price:.2f}")
        
        # Get option data for volatility surface
        print("\nFetching option data for volatility surface construction...")
        surface_data = data_puller.get_full_surface_data(
            num_expirations=15,  # Reduced for stability
            moneyness_range=(0.7, 1.3),  # Reasonable range
            min_price=0.01,  # Filter out penny options
            max_iv=1.5  # Filter extreme IVs
        )
        
        if surface_data.empty:
            print("ERROR: Could not fetch option data. Please check internet connection.")
            return
        
        # Save data for reproducibility
        data_puller.save_data(surface_data, cache_file)
        
        print(f"\n✓ Successfully fetched {len(surface_data)} options")
        print(f"✓ Data saved to {cache_file}")
    
    # ============================================================
    # TASK 2: Volatility Surface Calibration
    # ============================================================
    print("\n" + "="*60)
    print("TASK 2: Loading/Calibrating Parametric Volatility Surface")
    print("="*60)
    
    # Try to load cached volatility surface
    if os.path.exists(vol_surface_cache):
        print(f"\n✓ Loading cached volatility surface from {vol_surface_cache}...")
        with open(vol_surface_cache, 'rb') as f:
            vol_surface = pickle.load(f)
        print("✓ Volatility surface loaded from cache")
        
        # Show cached surface metrics if available
        if hasattr(vol_surface, 'svi_params_by_maturity') and vol_surface.svi_params_by_maturity:
            print(f"\nCached Surface Info:")
            print(f"  Number of maturity slices: {len(vol_surface.svi_params_by_maturity)}")
            print(f"  Spot price: ${vol_surface.spot:.2f}")
    else:
        print("\nNo cached surface found. Calibrating new surface...")
        
        # Prepare data for calibration
        calibration_data = surface_data[['strike', 'timeToExpiry', 'impliedVolatility', 'underlyingPrice']].copy()
        calibration_data = calibration_data.dropna()
        
        print(f"\nUsing {len(calibration_data)} options for calibration")
        
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
            max_iter=10000
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
            
            # Save calibrated surface for future use
            with open(vol_surface_cache, 'wb') as f:
                pickle.dump(vol_surface, f)
            print(f"\n✓ Volatility surface saved to {vol_surface_cache}")
    
    # Visualize the surface
    visualizer = OptionPricingVisualizer()
    
    if hasattr(vol_surface, 'svi_params_by_maturity') and vol_surface.svi_params_by_maturity:
        # Build surface grid for visualization
        strikes = np.linspace(spot_price * 0.7, spot_price * 1.3, 50)
        maturities = np.linspace(0.01, 1.5, 30)
        
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
    
    local_vol_cache = "pltr_local_vol.pkl"
    
    # Try to load cached local volatility model
    if os.path.exists(local_vol_cache):
        print(f"\n✓ Loading cached local volatility model from {local_vol_cache}...")
        with open(local_vol_cache, 'rb') as f:
            local_vol_model = pickle.load(f)
        print("✓ Local volatility model loaded from cache")
    else:
        print("\nNo cached local vol model found. Building new model...")
        
        # Initialize local volatility model
        local_vol_model = DupireLocalVolatility(
            vol_surface=vol_surface,
            spot=spot_price,
            risk_free_rate=0.035,
            dividend_yield=0.0
        )
        
        # Build local volatility surface
        print("\nBuilding local volatility surface using Dupire's formula...")
        local_vol_model.build_local_vol_surface(
            strike_range=(0.7, 1.3),  # Wider range matching the vol surface
            maturity_range=(0.01, 1.5),  # Full range from 0.2 years to 2 years
            n_strikes=50,  # More strikes for better interpolation
            n_maturities=15  # More maturities for better interpolation
        )
        
        # Save the model
        with open(local_vol_cache, 'wb') as f:
            pickle.dump(local_vol_model, f)
        print(f"✓ Local volatility model saved to {local_vol_cache}")
    
    # Visualize local vol surface
    if local_vol_model.local_vol_grid is not None:
        print("\nGenerating local volatility surface plot...")
        visualizer.plot_local_vol_surface(
            {
                'strikes': local_vol_model.strikes,
                'maturities': local_vol_model.maturities,
                'values': local_vol_model.local_vol_grid
            },
            title="Dupire Local Volatility Surface"
        )
    
    # Price European option with local vol
    K = spot_price  # ATM option
    T = 1.0  # 1 year maturity
    
    print(f"\nPricing European Call Option (K=${K:.2f}, T={T}y):")
    print("Using Local Volatility Monte Carlo...")
    
    lv_euro_result = local_vol_model.price_european_option(
        K=K, T=T, option_type='call',
        n_paths=10000, n_steps=100  # Reduced complexity
    )
    
    print(f"  Local Vol Price: ${lv_euro_result['price']:.4f}")
    print(f"  Black-Scholes Price: ${lv_euro_result['bs_price']:.4f}")
    print(f"  Difference: ${lv_euro_result['difference']:.4f}")
    print(f"  Standard Error: {lv_euro_result['std_error']:.6f}")
    
    # Price Barrier Option
    H = spot_price * 0.80  # 15% down barrier
    
    print(f"\nPricing Down-and-Out Put Option (K=${K:.2f}, H=${H:.2f}, T={T}y):")
    
    lv_barrier_result = local_vol_model.price_barrier_option(
        K=K, H=H, T=T,
        barrier_type='down-out', option_type='put',
        n_paths=10000, n_steps=100  # Reduced complexity
    )
    
    print(f"  Local Vol Barrier Price: ${lv_barrier_result['price']:.4f}")
    print(f"  Black-Scholes Barrier Price: ${lv_barrier_result['bs_price']:.4f}")
    print(f"  Knock-out Probability: {1 - lv_barrier_result['knock_probability']:.2%}")
    print(f"  Standard Error: {lv_barrier_result['std_error']:.6f}")
    
    # Convergence Analysis with specified path counts
    print(f"\nAnalyzing convergence with path and step variations...")
    
    '''
    # Use the requested path counts
    path_counts = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,100000]
    # Add step counts for convergence analysis
    step_counts = [2,4,8,16,32,64,128,256,512,1024,2048]
    
    convergence_result = local_vol_model.analyze_convergence(
        K=K, T=T, option_type='call',
        path_counts=path_counts,
        step_counts=step_counts,
        target_error=TARGET_ERROR,
        fast_mode=False  # Enable fast mode for quicker execution
    )
    
    # Display convergence results with enhanced error analysis
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS RESULTS")
    print("="*60)
    
    # Display Black-Scholes benchmark price
    if 'bs_price' in convergence_result:
        print(f"\nBlack-Scholes Benchmark Price: ${convergence_result['bs_price']:.4f}")
        print(f"ATM Volatility Used: {convergence_result['sigma_atm']:.2%}")
    
    # Display path convergence results
    print("\nPath Convergence Results:")
    print("-" * 50)
    path_data = convergence_result['path_convergence']
    print(f"{'Paths':>10} | {'MC Price':>10} | {'Abs Error':>10} | {'Std Error':>10} | {'95% CI Width':>12}")
    print("-" * 65)
    
    for i, n_paths in enumerate(path_data['path_counts']):
        mc_price = path_data['prices'][i]
        abs_err = path_data['absolute_errors'][i]
        std_err = path_data['std_errors'][i]
        ci_width = path_data['confidence_intervals'][i]['precision']
        
        print(f"{n_paths:>10,} | ${mc_price:>9.4f} | {abs_err:>10.5f} | {std_err:>10.5f} | {ci_width:>12.5f}")
    
    # Display step convergence results if available
    if 'step_convergence' in convergence_result and convergence_result['step_convergence']:
        print("\nStep Convergence Results (with 10,000 paths):")
        print("-" * 50)
        step_data = convergence_result['step_convergence']
        print(f"{'Steps':>8} | {'MC Price':>10} | {'Abs Error':>10} | {'Std Error':>10}")
        print("-" * 50)
        
        for i, n_steps in enumerate(step_data['step_counts']):
            mc_price = step_data['prices'][i]
            abs_err = step_data['absolute_errors'][i]
            std_err = step_data['std_errors'][i]
            
            print(f"{n_steps:>8} | ${mc_price:>9.4f} | {abs_err:>10.5f} | {std_err:>10.5f}")
    
    if convergence_result['achieved_error']:
        print(f"\n✓ Target error {TARGET_ERROR} achieved!")
        print(f"  Required paths: {convergence_result['required_paths']:,}")
        print(f"  Achieved error: {convergence_result['achieved_error']:.2e}")
    else:
        print(f"\n✗ Target error not achieved with tested path counts")
        print(f"  Best error: {min(convergence_result['path_convergence']['std_errors']):.2e}")
    
    # Plot convergence
    print("\nGenerating enhanced convergence plots...")
    visualizer.plot_convergence(convergence_result, target_error=TARGET_ERROR)
    
    # 2D Convergence Analysis with both paths and steps varying
    print("\n" + "="*60)
    print("2D CONVERGENCE ANALYSIS (Paths × Steps)")
    print("="*60)
    
    print("\nRunning 2D convergence analysis with both paths and steps varying...")
    convergence_2d_result = local_vol_model.analyze_convergence_2d(
        K=K, T=T, option_type='call',
        path_counts = [2,4,8,16,32,64,128,256,512,1024,2048],
        step_counts = [2,4,8,16,32,64,128,256,512,1024,2048],
    )
    
    # Display 2D convergence results
    print("\n2D Convergence Results:")
    print("-" * 50)
    print(f"Path counts tested: {convergence_2d_result['path_counts']}")
    print(f"Step counts tested: {convergence_2d_result['step_counts']}")
    print(f"\nMinimum absolute error achieved: {np.min(convergence_2d_result['error_grid']):.6f}")
    print(f"Maximum absolute error: {np.max(convergence_2d_result['error_grid']):.6f}")
    
    # Find best combination
    min_idx = np.unravel_index(np.argmin(convergence_2d_result['error_grid']), 
                               convergence_2d_result['error_grid'].shape)
    best_paths = convergence_2d_result['path_counts'][min_idx[0]]
    best_steps = convergence_2d_result['step_counts'][min_idx[1]]
    print(f"\nBest combination: {best_paths:,} paths × {best_steps:,} steps")
    print(f"Achieved error: {convergence_2d_result['error_grid'][min_idx]:.6f}")
    
    # Plot 2D convergence
    print("\nGenerating 2D convergence plots...")
    visualizer.plot_convergence_2d(convergence_2d_result)
    
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
'''

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
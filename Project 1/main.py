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
    T = 0.5  # 1 year maturity
    
    print(f"\nPricing European Call Option (K=${K:.2f}, T={T}y):")
    print("Using Local Volatility Monte Carlo...")
    
    lv_euro_result = local_vol_model.price_european_option(
        K=K, T=T, option_type='call',
        n_paths=100000, n_steps=252  # Reduced complexity
    )
    
    print(f"  Local Vol Price: ${lv_euro_result['price']:.4f}")
    print(f"  Black-Scholes Price: ${lv_euro_result['bs_price']:.4f}")
    print(f"  Difference: ${lv_euro_result['difference']:.4f}")
    print(f"  Standard Error: {lv_euro_result['std_error']:.6f}")
    
    # Price Barrier Option
    H = spot_price * 0.85  # 15% down barrier
    
    print(f"\nPricing Down-and-Out Put Option (K=${K:.2f}, H=${H:.2f}, T={T}y):")
    
    lv_barrier_result = local_vol_model.price_barrier_option(
        K=K, H=H, T=T,
        barrier_type='down-out', option_type='put',
        n_paths=100000, n_steps=252  # Reduced complexity
    )
    
    print(f"  Local Vol Barrier Price: ${lv_barrier_result['price']:.4f}")
    print(f"  Black-Scholes Barrier Price: ${lv_barrier_result['bs_price']:.4f}")
    print(f"  Knock-out Probability: {1 - lv_barrier_result['knock_probability']:.2%}")
    print(f"  Standard Error: {lv_barrier_result['std_error']:.6f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
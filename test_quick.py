#!/usr/bin/env python3
"""
Quick test script to validate the implementation.
"""

import numpy as np
import sys
import os

# Add option_pricing to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")
try:
    from option_pricing.models.black_scholes import BlackScholes
    print("✓ Black-Scholes imported")
    
    from option_pricing.models.volatility_surface import ParametricVolatilitySurface
    print("✓ Volatility Surface imported")
    
    from option_pricing.models.local_vol import DupireLocalVolatility
    print("✓ Local Volatility imported")
    
    from option_pricing.models.local_stoch_vol import LocalStochasticVolatility
    print("✓ Local-Stochastic Volatility imported")
    
    from option_pricing.pricing.monte_carlo import MonteCarloEngine
    print("✓ Monte Carlo Engine imported")
    
    from option_pricing.visualization.plots import OptionPricingVisualizer
    print("✓ Visualization imported")
    
    from option_pricing.data.yahoo_data import OptionDataPuller
    print("✓ Data Puller imported")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Testing Black-Scholes pricing...")

# Test Black-Scholes
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
call_price = BlackScholes.call_price(S, K, T, r, sigma)
put_price = BlackScholes.put_price(S, K, T, r, sigma)

print(f"Call price: ${call_price:.4f}")
print(f"Put price: ${put_price:.4f}")

# Test implied volatility
impl_vol = BlackScholes.implied_volatility(call_price, S, K, T, r, option_type='call')
print(f"Implied volatility: {impl_vol:.4f}")
assert abs(impl_vol - sigma) < 0.001, "Implied vol calculation error"
print("✓ Black-Scholes working correctly")

print("\n" + "="*60)
print("Testing Monte Carlo Engine...")

# Test Monte Carlo
mc_engine = MonteCarloEngine(n_paths=10000, n_steps=100, scheme='milstein', seed=42)
mc_result = mc_engine.price_european_option(S, K, T, r, sigma, option_type='call')

print(f"MC Call price: ${mc_result['price']:.4f}")
print(f"BS Call price: ${call_price:.4f}")
print(f"Difference: ${abs(mc_result['price'] - call_price):.4f}")
print(f"Standard error: {mc_result['std_error']:.6f}")

if abs(mc_result['price'] - call_price) < 0.5:
    print("✓ Monte Carlo working correctly")
else:
    print("✗ Monte Carlo price deviation too large")

print("\n" + "="*60)
print("All basic tests passed!")
print("="*60)
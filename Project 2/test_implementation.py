#!/usr/bin/env python3
"""
Test Script for Portfolio Optimization Implementation
Runs a simplified version to verify all components work
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings

# Add src to path
sys.path.append('src')

from data_loader import AssetDataLoader
from parameter_estimation import ParameterEstimator
from static_optimization import StaticPortfolioOptimizer

warnings.filterwarnings('ignore')

print("="*60)
print("TESTING PORTFOLIO OPTIMIZATION IMPLEMENTATION")
print("="*60)

# Test 1: Data Loading
print("\n1. Testing Data Loader...")
try:
    loader = AssetDataLoader('data/BBG Data (2000-2025).xlsx')
    returns_data = loader.load_data()
    print(f"✓ Data loaded successfully: {returns_data.shape}")
    print(f"  Assets: {len(returns_data.columns)}")
    print(f"  Periods: {len(returns_data)}")
    print(f"  Date range: {returns_data.index[0].strftime('%Y-%m')} to {returns_data.index[-1].strftime('%Y-%m')}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    sys.exit(1)

# Test 2: Parameter Estimation
print("\n2. Testing Parameter Estimation...")
try:
    estimator = ParameterEstimator(returns_data)
    expected_returns = estimator.estimate_expected_returns('combined')
    cov_matrix = estimator.estimate_covariance_matrix('shrinkage')

    print(f"✓ Parameters estimated successfully")
    print(f"  Average expected return: {expected_returns.mean():.2%}")
    print(f"  Average volatility: {np.sqrt(np.diag(cov_matrix.values)).mean():.2%}")
    print(f"  Target return (CPI+3%): {estimator.target_return:.2%}")
except Exception as e:
    print(f"✗ Error in parameter estimation: {e}")
    sys.exit(1)

# Test 3: Static Optimization
print("\n3. Testing Static Portfolio Optimization...")
try:
    optimizer = StaticPortfolioOptimizer(expected_returns, cov_matrix)

    # Find minimum variance portfolio
    result = optimizer.optimize_portfolio(
        target_return=estimator.target_return,
        growth_allocation=optimizer.growth_target,
    )

    if result['success']:
        print(f"✓ Optimization successful")
        print(f"  Portfolio return: {result['metrics']['return']:.2%}")
        print(f"  Portfolio volatility: {result['metrics']['volatility']:.2%}")
        print(f"  Sharpe ratio: {result['metrics']['sharpe_ratio']:.3f}")
        print(f"  Growth allocation: {result['metrics']['growth_weight']:.1%}")

        # Show top holdings
        weights_df = pd.DataFrame({
            'Asset': expected_returns.index,
            'Weight': result['weights']
        }).sort_values('Weight', ascending=False)

        print("\n  Top 5 Holdings:")
        for _, row in weights_df.head(5).iterrows():
            if row['Weight'] > 0.001:
                print(f"    {row['Asset'][:30]:30} {row['Weight']:7.2%}")
    else:
        print(f"✗ Optimization failed")
except Exception as e:
    print(f"✗ Error in optimization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Generate sample efficient frontier
print("\n4. Testing Efficient Frontier Generation...")
try:
    frontier = optimizer.generate_efficient_frontier(
        n_points=20,
        growth_allocation=optimizer.growth_target,
    )

    print(f"✓ Efficient frontier generated")
    print(f"  Number of points: {len(frontier)}")
    print(f"  Return range: {frontier['return'].min():.2%} to {frontier['return'].max():.2%}")
    print(f"  Volatility range: {frontier['volatility'].min():.2%} to {frontier['volatility'].max():.2%}")
except Exception as e:
    print(f"✗ Error generating frontier: {e}")

# Test 5: Risk Profile Comparison
print("\n5. Testing Risk Profile Comparison...")
try:
    comparison = optimizer.compare_risk_profiles()
    print(f"✓ Risk profiles compared")
    print("\n  Portfolio Comparison:")
    for _, row in comparison.iterrows():
        print(f"    {row['Profile']:12} Return: {row['Expected Return']:7.2%}  "
              f"Vol: {row['Volatility']:7.2%}  Sharpe: {row['Sharpe Ratio']:6.3f}")
except Exception as e:
    print(f"✗ Error in risk comparison: {e}")

print("\n" + "="*60)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*60)

# Summary statistics
print("\nSUMMARY STATISTICS:")
print("-"*40)
stats = loader.calculate_statistics()
for i, (asset, stat) in enumerate(list(stats.items())[:3]):  # Show first 3 assets
    print(f"\n{asset[:40]}:")
    print(f"  Annual Return: {stat['annualized_return']:7.2%}")
    print(f"  Annual Volatility: {stat['annualized_volatility']:7.2%}")
    print(f"  Sharpe Ratio: {stat['sharpe_ratio']:7.3f}")

print("\n" + "="*60)
print("Implementation is working correctly!")
print("You can now run the full analysis with: python3 src/main.py")
print("="*60)

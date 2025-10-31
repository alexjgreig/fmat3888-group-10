# FMAT3888 Project 2: Portfolio Construction and Optimization

## Overview
This project implements a comprehensive Strategic Asset Allocation (SAA) model for a MySuper balanced fund using Markowitz Modern Portfolio Theory and advanced optimization techniques. The solution addresses all questions in the assignment, including both compulsory and advanced topics.

## Project Structure
```
Project 2/
├── src/                          # Source code modules
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── parameter_estimation.py   # Question 1: Parameter estimation
│   ├── static_optimization.py    # Questions 2(a-e): Static optimization
│   ├── advanced_optimization.py  # Questions 2(f-g): Advanced techniques
│   ├── dynamic_optimization.py   # Questions 2(h-k): Dynamic strategies
│   ├── visualization.py          # Plotting and visualization
│   ├── constraints_config.py     # Benchmark weights and qualitative bands
│   └── main.py                   # Main orchestrator script
├── data/                         # Input data
│   └── HistoricalData(2012-2024).xlsm
├── outputs/                      # Generated outputs
│   ├── figures/                 # Visualizations
│   └── tables/                  # CSV results
├── test_implementation.py        # Test script
└── README.md                    # This file
```

## Features Implemented

### Question 1: Parameter Estimation
- **Multiple estimation methods**: Historical, EWMA, Shrinkage, Combined
- **Covariance estimation**: Sample, Ledoit-Wolf shrinkage, EWMA
- **Risk metrics**: Volatility, correlations, Sharpe ratios
- **Target return**: CPI + 3% = 5.594%

### Question 2: Static Portfolio Optimization

#### 2(a): Efficient Frontier
- Qualitative frontier using asset-specific bounds from `constraints_config.py`
- Relaxed and wide frontiers for sensitivity testing
- Growth corridor fixed at 73% with a ±2% tolerance

#### 2(b): Minimum Variance Portfolio
- CPI + 3% return floor enforced
- Growth/defensive split constrained to the qualitative range (71–75%)
- Per-asset minimum/maximum weights imported from `constraints_config.py`

#### 2(c-d): Methodology Documentation
- Comprehensive parameter estimation choices
- Optimization technique selection
- Risk metrics analysis

#### 2(e): Portfolio Risk Profiles
- Defensive (30/70), balanced (71/29), and aggressive (90/10) allocations
- Sharpe and Sortino diagnostics with relaxed bounds for feasibility
- Exponential utility comparison consistent with question requirements

### Advanced Topics

#### 2(f): Utility Maximization with Log-Normal Assets
- Log-normal return assumption
- Exponential utility maximization
- Comparison with mean-variance optimization

#### 2(g): Non-PSD Covariance Matrix Handling
- Detection of non-positive semidefinite matrices
- Multiple correction methods:
  - Eigenvalue clipping
  - Nearest correlation matrix (Higham algorithm)
  - Shrinkage to diagonal
- Impact analysis on portfolio optimization

#### 2(h): Risk Management
- Risk attribution (marginal and component contributions)
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Hooks for risk budgeting and drift monitoring

#### 2(i-k): Dynamic Portfolio Optimization
- Multi-period utility maximisation with transaction costs
- Rebalancing frequency sweep (quarterly, semi-annual, annual)
- Static versus dynamic comparison with annualised reporting
- Monte Carlo simulation (1,000 scenarios)

## Key Results

### Optimised Qualitative Portfolio (71/29)
- **Expected Return**: 8.31%
- **Volatility**: 7.64%
- **Sharpe Ratio**: 0.825
- **Growth Allocation**: 71%
- **Defensive Allocation**: 29%
- **Tracking Error vs Benchmark**: 0.85%

### Key Active Positions vs Benchmark
1. Int'l Listed Infrastructure [G]: +6.6%
2. Australian Fixed Income [D]: +3.0%
3. Cash [D]: +2.0%
4. Australian Listed Equity [G]: -3.0%
5. Int'l Listed Equity (Hedged) [G]: -3.0%

### Risk Metrics (Balanced Qualitative Portfolio)
- **Value at Risk (95%)**: 6.11%
- **Conditional VaR (95%)**: 9.48%
- **Largest Risk Contributors**: Int'l Listed Equity (Hedged) 31%, Australian Listed Equity 28%, Int'l Listed Infrastructure 11%

### Dynamic Strategy Comparison (Annualised)
- Static buy-and-hold: return 7.29%, volatility 5.23%, Sharpe 1.01
- Rebalance every 12 months: return 11.40%, volatility 7.31%, Sharpe 1.29, turnover cost 0.02%
- Rebalance quarterly: similar Sharpe uplift (1.28) with higher cost (0.24%)

## How to Run

### Prerequisites
```bash
# Required Python packages
pip install pandas numpy scipy matplotlib seaborn openpyxl
```

### Quick Test
```bash
cd "Project 2"
python3 test_implementation.py
```

### Full Analysis
```bash
cd "Project 2"
python3 src/main.py
```

### Individual Components
```python
# Run specific analyses
cd "Project 2/src"
python3 parameter_estimation.py     # Question 1
python3 static_optimization.py      # Questions 2(a-e)
python3 advanced_optimization.py    # Questions 2(f-g)
python3 dynamic_optimization.py     # Questions 2(h-k)
```

## Output Files

### Visualizations (`outputs/figures/`)
- `efficient_frontier_*.png`: Efficient frontier with constraints
- `correlation_heatmap_*.png`: Asset correlation matrix
- `portfolio_weights_*.png`: Optimal portfolio allocation
- `risk_attribution_*.png`: Risk contribution analysis
- `strategy_comparison_*.png`: Static vs. dynamic performance

### Data Tables (`outputs/tables/`)
- `optimal_weights_*.csv`: Portfolio weights
- `efficient_frontier_*.csv`: Frontier points
- `risk_profile_comparison_*.csv`: Portfolio comparisons
- `risk_attribution_*.csv`: Risk contributions
- `strategy_comparison_*.csv`: Strategy performance metrics

## Technical Notes

### Data Processing
- Monthly returns from Jan 2012 to Jan 2024 (145 months)
- 9 asset classes: 6 growth [G] and 3 defensive [D]
- Missing values handled with forward/backward fill

### Optimization Methods
- **Solver**: Sequential Quadratic Programming (SLSQP)
- **Constraints**: Linear and nonlinear constraints supported
- **Bounds**: No short-selling (weights ≥ 0)

### Parameter Estimation
- **Combined Method**: Weighted average of historical (30%), EWMA (40%), and shrinkage (30%)
- **Shrinkage**: Ledoit-Wolf covariance estimator
- **EWMA**: λ = 0.94 decay factor

### Risk Measures
- **Volatility**: Annualized standard deviation
- **Sharpe Ratio**: (Return - Risk-free) / Volatility
- **VaR/CVaR**: 95% confidence level, 1,000 Monte Carlo simulations (quarterly horizon)

## Key Insights

1. **Qualitative alignment**: Asset-level bounds and the 73% growth corridor from `notes.md` are enforced directly through `constraints_config.py`.

2. **Infrastructure tilt**: Increasing listed infrastructure exposure provides meaningful inflation participation without breaching tracking error limits.

3. **Mandate safety**: The optimised portfolio delivers CPI+3% with only 0.85% tracking error, reducing risk of YFYS penalties.

4. **Dynamic uplift**: Annual rebalancing materially lifts the Sharpe ratio (1.29 vs 1.01 static) for minimal turnover.

5. **Robust covariance handling**: Eigenvalue clipping restores PSD structure when correlations are stressed, keeping optimal weights stable.

## Authors
Group 10 - FMAT3888 Projects in Financial Mathematics

## License
Academic use only - University of Sydney

## Acknowledgments
- APRA Quarterly MySuper Statistics for benchmark data
- Course instructors for project guidance

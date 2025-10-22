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
- Unconstrained frontier
- Constrained frontier (max 40% per asset)
- Balanced frontier (70% growth, 30% defensive)

#### 2(b): Minimum Variance Portfolio
- Optimization with target return constraint (≥ 5.594%)
- Growth/defensive allocation constraints (70/30 ± 6%)
- Individual asset bounds (0-40%)

#### 2(c-d): Methodology Documentation
- Comprehensive parameter estimation choices
- Optimization technique selection
- Risk metrics analysis

#### 2(e): Portfolio Risk Profiles
- Comparison of Defensive (30/70), Balanced (70/30), and Aggressive (90/10) portfolios
- Sharpe ratio analysis
- Exponential utility function evaluation (U(x) = -e^(-x))

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
- Risk attribution analysis
- Value at Risk (VaR) calculation
- Conditional VaR (CVaR) calculation
- Risk budgeting implementation

#### 2(i-k): Dynamic Portfolio Optimization
- Multi-period optimization
- Quarterly rebalancing strategies
- Transaction cost analysis
- Static vs. dynamic strategy comparison
- Monte Carlo simulation (1000 scenarios)

## Key Results

### Optimal Portfolio (Balanced 70/30)
- **Expected Return**: 7.52%
- **Volatility**: 5.99%
- **Sharpe Ratio**: 0.922
- **Growth Allocation**: 64.0%
- **Defensive Allocation**: 36.0%

### Top Holdings
1. Int'l Listed Equity (Unhedged) [G]: 37.35%
2. Cash [D]: 36.00%
3. Int'l Listed Infrastructure [G]: 23.56%
4. Australian Listed Equity [G]: 3.09%

### Risk Metrics
- **Value at Risk (95%)**: Based on Monte Carlo simulation
- **Conditional VaR (95%)**: Expected shortfall calculation

### Strategy Comparison
The analysis compares static buy-and-hold with dynamic rebalancing strategies at different frequencies (quarterly, semi-annual, annual).

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
- **VaR/CVaR**: 95% confidence level, 10,000 Monte Carlo simulations

## Key Insights

1. **Diversification Benefits**: The optimal portfolio achieves higher Sharpe ratio than individual assets through diversification.

2. **International Exposure**: Significant allocation to international equities (unhedged) provides better risk-adjusted returns.

3. **Growth/Defensive Balance**: The 70/30 split constraint is binding, with optimization pushing toward the growth limit.

4. **Dynamic Rebalancing**: Quarterly rebalancing provides marginal improvement over static allocation after accounting for transaction costs.

5. **Non-PSD Handling**: Eigenvalue clipping provides the most stable correction method with minimal distortion.

## Authors
Group 10 - FMAT3888 Projects in Financial Mathematics

## License
Academic use only - University of Sydney

## Acknowledgments
- APRA Quarterly MySuper Statistics for benchmark data
- Course instructors for project guidance
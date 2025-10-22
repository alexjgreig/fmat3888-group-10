# FMAT3888 Project 2 - Report Deliverables

## 📄 LaTeX Report Created

### File Location
`Project 2/report/report.tex` (12 pages, 365KB PDF)

### Report Structure (Following Assignment Guidelines)

1. **Executive Summary** (½ page)
   - Non-technical overview for finance professionals
   - Key findings and recommendations
   - Portfolio performance metrics

2. **Introduction**
   - Background on MySuper products and APRA requirements
   - Portfolio optimization theory overview
   - Report objectives and structure

3. **Mathematical Setup**
   - Technical formulation of optimization problems
   - Tracking error constraints
   - Utility maximization framework
   - Dynamic optimization formulation

4. **Theoretical Results**
   - Parameter estimation methodology
   - Efficient frontier derivation with constraints
   - Non-PSD covariance matrix correction methods
   - KKT conditions for constrained optimization

5. **Computational Results**
   - **Question 1**: Parameter estimation results
     - Combined estimation methods (historical, EWMA, shrinkage)
     - Covariance matrix with Ledoit-Wolf shrinkage
     - Risk metrics and correlations

   - **Question 2(a-e)**: Static Portfolio Optimization
     - Benchmark portfolio based on MySuper standards
     - Constrained efficient frontier with 2% tracking error
     - Minimum variance portfolio achieving target return
     - Risk profile comparison (Defensive/Balanced/Aggressive)

   - **Question 2(f)**: Utility Maximization
     - Log-normal asset dynamics
     - Exponential utility optimization
     - Comparison with mean-variance

   - **Question 2(g)**: Non-PSD Covariance Handling
     - Creation of inconsistent correlation structure
     - Comparison of correction methods
     - Impact on portfolio optimization

   - **Question 2(h-k)**: Dynamic Optimization
     - Risk attribution analysis
     - VaR and CVaR calculations
     - Static vs dynamic strategy comparison
     - Transaction cost analysis

6. **Conclusions**
   - Key findings summary
   - Implementation recommendations
   - Limitations and future work
   - Final remarks on mandate safety

7. **References**
   - 15 academic references (Markowitz, Ledoit-Wolf, etc.)
   - APRA regulatory documents
   - Industry benchmarks

8. **Appendix**
   - Program structure diagram
   - Algorithm implementation details
   - Computational complexity analysis

## 📊 Key Results Presented

### Portfolio Metrics
- **Benchmark Portfolio**: 8.64% return, 8.20% volatility, 0.809 Sharpe
- **Optimized Portfolio**: 8.34% return, 7.36% volatility, 0.862 Sharpe
- **Tracking Error**: 1.31% (well within 2% constraint)
- **VaR (95%)**: 3.83% (improved from 4.86% benchmark)

### Asset Allocation
| Asset Class | Benchmark | Optimized | Active Position |
|------------|-----------|-----------|-----------------|
| Australian Equity | 25% | 20% | -5% |
| Int'l Equity (Unhedged) | 25% | 30% | +5% |
| Int'l Infrastructure | 2% | 7% | +5% |
| Cash | 4% | 9% | +5% |

### Strategic Insights
- Maintains close alignment with industry benchmarks (low tracking error)
- Improves Sharpe ratio while reducing volatility
- Aligns with late-cycle macro view from notes.md
- Implementation cost of 7 basis points

## 📈 Visualizations Included

1. **Efficient Frontier Plot** (ASCII art in report)
   - Shows benchmark vs optimized portfolio
   - Illustrates tracking error constraint impact

2. **Tables** (9 total)
   - Parameter estimates
   - Portfolio allocations
   - Performance comparisons
   - Risk attribution
   - Strategy comparisons

## ✅ Assignment Requirements Met

### Compulsory Questions
- [x] Question 1: Parameter Estimation ✓
- [x] Question 2(a): Efficient Frontier ✓
- [x] Question 2(b): Minimum Variance Portfolio ✓
- [x] Question 2(c-d): Methodology Documentation ✓
- [x] Question 2(e): Risk Profile Comparison ✓

### Advanced Topics
- [x] Question 2(f): Utility Maximization ✓
- [x] Question 2(g): Non-PSD Matrix Handling ✓
- [x] Question 2(h): Risk Attribution ✓
- [x] Question 2(i-k): Dynamic Optimization ✓

## 📝 Report Quality Features

- **Professional formatting**: 12pt font, proper margins, APA7 citations
- **Mathematical rigor**: All formulas properly derived and explained
- **Non-technical summary**: Executive summary accessible to finance professionals
- **Implementation focus**: Practical recommendations for fund managers
- **Code documentation**: Appendix with program structure

## 🔧 How to Compile

```bash
cd "Project 2/report"
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

The report is ready for submission and includes all required elements from the assignment specification.
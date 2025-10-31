# Portfolio Optimization Results Summary

## Key Achievement: Benchmark-Aligned Optimization

The refreshed optimisation delivers the qualitative 73% growth posture from `notes.md` while keeping tracking error comfortably below the mandate ceiling. The resulting portfolio lifts risk-adjusted returns without diverging materially from the MySuper benchmark.

## Portfolio Comparison

### Benchmark Portfolio (Industry Standard MySuper)
Based on typical MySuper allocations and the qualitative target:
- **Expected Return**: 8.64%
- **Volatility**: 8.20%
- **Sharpe Ratio**: 0.809
- **Growth/Defensive**: 73% / 27%

### Optimised Qualitative Portfolio (TE ≤ 2%)
- **Expected Return**: 8.31% (-33 bps)
- **Volatility**: 7.64% (-56 bps)
- **Sharpe Ratio**: 0.825 (+0.016)
- **Growth/Defensive**: 71% / 29%
- **Tracking Error**: 0.85% (vs 2% limit)

## Asset Allocation

| Asset Class | Benchmark | Optimised | Active | Comment |
|-------------|-----------|-----------|--------|---------|
| **Growth (71%)** |||||
| Australian Listed Equity | 25.0% | 22.0% | -3.0% | Trim domestic cyclicality |
| Int'l Listed Equity (Hedged) | 13.0% | 10.0% | -3.0% | Moderately higher FX exposure |
| Int'l Listed Equity (Unhedged) | 25.0% | 26.4% | +1.4% | Preserve offshore growth bias |
| Australian Listed Property | 6.0% | 4.0% | -2.0% | Avoid doubling up on housing beta |
| Int'l Listed Property | 2.0% | 0.0% | -2.0% | Redeploy to infrastructure |
| Int'l Listed Infrastructure | 2.0% | 8.6% | +6.6% | Inflation-linked cash flows |
| **Defensive (29%)** |||||
| Australian Fixed Income | 16.0% | 19.0% | +3.0% | Anchor duration locally |
| Int'l Fixed Income (Hedged) | 7.0% | 4.0% | -3.0% | Simplify hedged rates exposure |
| Cash | 4.0% | 6.0% | +2.0% | Liquidity buffer for member flows |

## Risk Metrics

- **Tracking error**: 0.85% relative to the benchmark weights
- **Portfolio VaR (95%, annualised)**: 6.11%
- **Portfolio CVaR (95%, annualised)**: 9.48%
- **Top risk contributors**: Int'l Listed Equity (Hedged) 31.3%, Australian Listed Equity 28.3%, Int'l Listed Infrastructure 11.5%, Australian Listed Property 11.5%

## Dynamic Strategy Comparison (Quarterly Simulation)

| Strategy | Expected Return | Volatility | Sharpe | 5th Pctl | 95th Pctl | Costs |
|----------|----------------|------------|--------|----------|-----------|-------|
| Static buy & hold | 7.29% | 5.23% | 1.01 | -1.31% | 15.70% | 0.00% |
| Rebalance every quarter | 11.38% | 7.34% | 1.28 | -1.10% | 23.20% | 0.24% |
| Rebalance every 12 months | **11.40%** | **7.31%** | **1.29** | -0.82% | 23.34% | 0.02% |

Quarterly-to-annual rebalance frequencies dominate the static policy on a Sharpe basis even after incorporating turnover penalties, with annual rebalancing providing the best risk/return trade-off.

## Compliance with Requirements

- ✅ **Target return**: 8.31% exceeds CPI + 3% = 5.59%
- ✅ **Growth share**: 71% sits inside the 60–76% APRA corridor and within ±2% of the qualitative 73% target
- ✅ **Tracking error**: 0.85% << 2% mandate ceiling
- ✅ **No position breaches**: All assets respect qualitative min/max bands derived from `notes.md`

## Strategic Rationale (from `notes.md`)

- Late-cycle macro view supports a mild global equity bias while tilting toward real assets (infrastructure) for inflation participation
- Domestic duration retained for defensive ballast; international rates exposure pared back to simplify hedging
- Cash lifted to 6% to service expected member withdrawals without forced selling

## Implementation Considerations

- **One-off turnover**: ~7 bps assuming 0.25% round-trip cost per asset
- **Ongoing costs**: Additional 2 bps p.a. to maintain FX overlays and infrastructure mandates
- **Monitoring**: Rebalance trigger of ±2% on the growth allocation keeps the portfolio inside mandate while allowing tactical drift

## Summary

The optimisation embeds the qualitative target, keeps the portfolio inside MySuper risk limits, and demonstrates that modest tilts (international infrastructure, slightly higher cash) can lift the Sharpe ratio without sacrificing mandate safety. Dynamic rebalancing analysis suggests an annual policy offers the best cost-adjusted uplift if the investment committee elects to move beyond static weights.

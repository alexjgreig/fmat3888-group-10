#!/usr/bin/env python3
"""
Main Script for FMAT3888 Project 2
Portfolio Construction and Optimization using Market Data and APRA Guidelines
This script runs all analyses for Questions 1 and 2 (including advanced topics)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.data_loader import AssetDataLoader
from src.parameter_estimation import ParameterEstimator
from src.static_optimization import StaticPortfolioOptimizer
from src.advanced_optimization import UtilityOptimizer, CovarianceMatrixCorrector
from src.dynamic_optimization import RiskManager, DynamicPortfolioOptimizer
from src.visualization_enhanced import EnhancedPortfolioVisualizer
from src.blended_portfolio_analysis import BlendedPortfolioAnalyzer

warnings.filterwarnings('ignore')


class PortfolioAnalysisRunner:
    """Main runner class for portfolio analysis"""

    def __init__(self, data_path: str):
        """
        Initialize the analysis runner

        Args:
            data_path: Path to the Excel data file
        """
        self.data_path = Path(data_path)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(__file__).resolve().parents[1]
        self.output_root = self.base_dir / 'outputs'

    def run_complete_analysis(self) -> Dict:
        """
        Run the complete portfolio analysis for all questions

        Returns:
            Dictionary containing all results
        """
        print("="*70)
        print(" FMAT3888 PROJECT 2: PORTFOLIO OPTIMIZATION ANALYSIS")
        print("="*70)
        print(f"\nAnalysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Step 1: Load and prepare data
        print("\n" + "="*70)
        print(" STEP 1: DATA LOADING AND PREPARATION")
        print("="*70)
        loader = AssetDataLoader(str(self.data_path))
        returns_data = loader.load_data()
        print(f"✓ Data loaded: {returns_data.shape[0]} months, {returns_data.shape[1]} assets")
        print(f"✓ Date range: {returns_data.index[0].strftime('%Y-%m')} to {returns_data.index[-1].strftime('%Y-%m')}")

        # Store basic data info
        self.results['data_info'] = {
            'n_periods': len(returns_data),
            'n_assets': len(returns_data.columns),
            'asset_names': returns_data.columns.tolist(),
            'date_range': f"{returns_data.index[0]} to {returns_data.index[-1]}"
        }

        # Step 2: Question 1 - Parameter Estimation
        print("\n" + "="*70)
        print(" QUESTION 1: PARAMETER ESTIMATION")
        print("="*70)
        estimator = ParameterEstimator(returns_data)
        param_report = estimator.generate_parameter_report()

        # Extract key parameters
        expected_returns = param_report['expected_returns']['recommended']
        cov_matrix = param_report['covariance_matrices']['recommended']

        print("\n📊 Parameter Estimation Results:")
        print(f"✓ Target Return (CPI + 3%): {estimator.target_return:.2%}")
        print(f"✓ Average Expected Return: {expected_returns.mean():.2%}")
        print(f"✓ Average Volatility: {np.sqrt(np.diag(cov_matrix.values)).mean():.2%}")
        print(f"✓ Average Correlation: {param_report['risk_metrics']['avg_correlation']:.3f}")

        self.results['question_1'] = param_report

        # Step 3: Question 2(a-e) - Static Portfolio Optimization
        print("\n" + "="*70)
        print(" QUESTION 2(a-e): STATIC PORTFOLIO OPTIMIZATION")
        print("="*70)
        static_optimizer = StaticPortfolioOptimizer(
            expected_returns,
            cov_matrix,
            returns_data=returns_data
        )

        static_report = static_optimizer.generate_optimization_report()
        min_var_result = static_report['minimum_variance_portfolio']
        risk_comparison = static_report['risk_profile_comparison']

        self.results['question_2_static'] = static_report

        # Step 4: Question 2(f) - Utility Maximization
        print("\n" + "="*70)
        print(" QUESTION 2(f): UTILITY MAXIMIZATION WITH LOG-NORMAL ASSETS")
        print("="*70)
        utility_optimizer = UtilityOptimizer(expected_returns, cov_matrix)
        utility_result = utility_optimizer.optimize_exponential_utility_lognormal(gamma=1)

        print(f"\n✓ Utility-Optimal Expected Return: {utility_result['expected_return']:.2%}")
        print(f"✓ Utility-Optimal Volatility: {utility_result['volatility']:.2%}")
        print(f"✓ Utility-Optimal Sharpe Ratio: {utility_result['sharpe_ratio']:.3f}")

        # Compare with mean-variance
        utility_comparison = utility_optimizer.compare_with_mean_variance(
            min_var_result['weights'], gamma=1
        )

        self.results['question_2f_utility'] = {
            'utility_optimization': utility_result,
            'comparison': utility_comparison
        }

        # Step 5: Question 2(g) - Non-PSD Covariance Matrix
        print("\n" + "="*70)
        print(" QUESTION 2(g): NON-PSD COVARIANCE MATRIX HANDLING")
        print("="*70)
        corrector = CovarianceMatrixCorrector(cov_matrix)

        # Create non-PSD matrix
        asset1 = expected_returns.index[0]
        asset2 = expected_returns.index[1]
        non_psd_cov = corrector.create_non_psd_matrix(asset1, asset2, 0.98)

        is_psd, eigenvalues = corrector.check_positive_semidefinite(non_psd_cov)
        print(f"\n✓ Created matrix is PSD: {is_psd}")
        print(f"✓ Minimum eigenvalue: {np.min(eigenvalues):.6f}")

        if not is_psd:
            correction_comparison = corrector.compare_correction_methods(non_psd_cov)
            print("\n📊 Correction Methods Comparison:")
            for method, metrics in correction_comparison.items():
                if method != 'original':
                    print(f"  {method}: PSD={metrics['is_psd']}, "
                         f"Min λ={metrics['min_eigenvalue']:.6f}")

        self.results['question_2g_covariance'] = {
            'non_psd_created': not is_psd,
            'min_eigenvalue_original': np.min(eigenvalues),
            'correction_comparison': correction_comparison if not is_psd else None
        }

        # Step 6: Questions 2(h-k) - Dynamic Portfolio Optimization
        print("\n" + "="*70)
        print(" QUESTIONS 2(h-k): DYNAMIC PORTFOLIO OPTIMIZATION")
        print("="*70)

        # Risk attribution
        print("\n📊 Risk Attribution Analysis...")
        risk_manager = RiskManager(min_var_result['weights'], expected_returns, cov_matrix)
        risk_attribution = risk_manager.calculate_risk_attribution()
        risk_metrics = risk_manager.calculate_var_cvar(confidence_level=0.95)

        print(f"✓ VaR (95%): {risk_metrics['VaR']:.2%}")
        print(f"✓ CVaR (95%): {risk_metrics['CVaR']:.2%}")

        # Dynamic optimization
        print("\n🔄 Dynamic Portfolio Optimization...")
        dynamic_optimizer = DynamicPortfolioOptimizer(
            expected_returns, cov_matrix, returns_data
        )

        # Compare strategies
        strategy_comparison = dynamic_optimizer.compare_static_vs_dynamic(
            min_var_result['weights'],
            n_periods=12,
            rebalance_frequencies=[1, 3, 6, 12]
        )

        best_strategy = strategy_comparison.loc[strategy_comparison['Sharpe Ratio'].idxmax()]
        print(f"\n✓ Best Strategy: {best_strategy['Strategy']}")
        print(f"✓ Best Sharpe Ratio: {best_strategy['Sharpe Ratio']:.3f}")

        self.results['question_2_dynamic'] = {
            'risk_attribution': risk_attribution,
            'risk_metrics': risk_metrics,
            'strategy_comparison': strategy_comparison
        }

        # Step 7: Blended Portfolio (30% Utility / 70% Min-Variance)
        print("\n" + "="*70)
        print(" BLENDED PORTFOLIO: 30% UTILITY / 70% MIN-VARIANCE")
        print("="*70)
        blended_analyzer = BlendedPortfolioAnalyzer(
            expected_returns,
            cov_matrix,
            returns_data,
            risk_free_rate=estimator.risk_free_rate
        )
        blended_results = blended_analyzer.build_blended_portfolio(
            utility_gamma=1.5,
            blend_weight=0.30
        )

        blended_metrics = blended_results['blended']['metrics']
        blended_robustness = blended_results['blended']['robustness']
        historical_summary = blended_results['historical_summary']
        latest_window = historical_summary[
            (historical_summary['Portfolio'] == 'Blended (30% Utility / 70% Min-Var)') &
            (historical_summary['Window'] == 'Last 5 Years')
        ]

        print(f"\n🔀 Blended Portfolio Metrics (Forward-Looking):")
        print(f"  • Expected Return: {blended_metrics['return']:.2%}")
        print(f"  • Net Expected Return: {blended_metrics['net_return']:.2%}")
        print(f"  • Volatility: {blended_metrics['volatility']:.2%}")
        print(f"  • Sharpe Ratio: {blended_metrics['sharpe_ratio']:.3f}")
        print(f"  • Diversification Ratio: {blended_robustness['diversification_ratio']:.3f}")
        print(f"  • Effective Number of Assets: {blended_robustness['effective_n_assets']:.2f}")
        if not latest_window.empty:
            last5 = latest_window.iloc[0]
            print(f"\n📈 Realised 5Y Performance:")
            print(f"  • Annual Return: {last5['Annual Return']:.2%}")
            print(f"  • Volatility: {last5['Annual Volatility']:.2%}")
            print(f"  • Sharpe Ratio: {last5['Sharpe Ratio']:.3f}")
            print(f"  • Max Drawdown: {last5['Max Drawdown']:.2%}")

        self.results['blended_portfolio'] = blended_results

        # Step 8: Generate Visualizations
        print("\n" + "="*70)
        print(" GENERATING VISUALIZATIONS")
        print("="*70)
        self._create_visualizations()

        # Step 9: Save Results
        print("\n" + "="*70)
        print(" SAVING RESULTS")
        print("="*70)
        self._save_results()

        print("\n" + "="*70)
        print(" ANALYSIS COMPLETE")
        print("="*70)
        print(f"\n✅ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return self.results

    def _create_visualizations(self):
        """Create all visualizations with enhanced styling"""
        visualizer = EnhancedPortfolioVisualizer()
        output_dir = self.output_root / 'figures'

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        print("\n📊 Creating enhanced visualizations...")

        # 1. Efficient Frontier
        if 'question_2_static' in self.results:
            frontiers = self.results['question_2_static']['efficient_frontiers']
            constrained_frontier = frontiers.get('constrained')
            min_var = self.results['question_2_static']['minimum_variance_portfolio']['metrics']

            if constrained_frontier is not None and not constrained_frontier.empty:
                special_portfolios = [{
                    'name': 'Min Variance (TR≥5.594%)',
                    'return': min_var['return'],
                    'volatility': min_var['volatility']
                }]

                overlay_frontiers = []
                unconstrained = frontiers.get('unconstrained')
                if unconstrained is not None and not unconstrained.empty:
                    overlay_frontiers.append({
                        'data': unconstrained,
                        'label': 'Frontier (No Growth Constraint)',
                        'color': 'steelblue',
                        'alpha': 0.35,
                        'linewidth': 2,
                        'linestyle': '--'
                    })

                expected_returns = self.results['question_1']['expected_returns']['recommended']
                cov_matrix = self.results['question_1']['covariance_matrices']['recommended']
                asset_stats = {
                    asset: {
                        'return': expected_returns[asset],
                        'volatility': np.sqrt(cov_matrix.loc[asset, asset])
                    }
                    for asset in expected_returns.index
                }

                visualizer.plot_efficient_frontier(
                    constrained_frontier,
                    special_portfolios=special_portfolios,
                    overlay_frontiers=overlay_frontiers,
                    individual_assets=asset_stats,
                    title='Portfolio Efficient Frontier Analysis',
                    save_path=str(output_dir / f"efficient_frontier_{self.timestamp}.png"),
                    base_label='Efficient Frontier (70% Growth, APRA Bounds)'
                )
                print("✓ Enhanced efficient frontier plot created")

        # 2. Correlation Heatmap
        if 'question_1' in self.results:
            cov_matrix = self.results['question_1']['covariance_matrices']['recommended']
            # Convert to correlation
            std_devs = np.sqrt(np.diag(cov_matrix.values))
            corr_matrix = pd.DataFrame(
                cov_matrix.values / np.outer(std_devs, std_devs),
                index=cov_matrix.index,
                columns=cov_matrix.columns
            )

            visualizer.plot_correlation_heatmap(
                corr_matrix,
                title='Asset Correlation Matrix Analysis',
                save_path=str(output_dir / f"correlation_heatmap_{self.timestamp}.png")
            )
            print("✓ Enhanced correlation heatmap created")

        # 3. Portfolio Weights
        if 'question_2_static' in self.results:
            weights = self.results['question_2_static']['minimum_variance_portfolio']['weights']
            weights_series = pd.Series(
                weights,
                index=self.results['question_1']['expected_returns']['recommended'].index
            )

            visualizer.plot_portfolio_weights(
                weights_series,
                title='Optimal Portfolio Allocation Strategy',
                save_path=str(output_dir / f"portfolio_weights_{self.timestamp}.png")
            )
            print("✓ Enhanced portfolio weights visualization created")

        if 'blended_portfolio' in self.results:
            blended_weights = self.results['blended_portfolio']['blended']['weights']
            blended_series = pd.Series(
                blended_weights,
                index=self.results['question_1']['expected_returns']['recommended'].index
            )
            visualizer.plot_portfolio_weights(
                blended_series,
                title='Blended Portfolio Allocation (30% Utility / 70% Min-Var)',
                save_path=str(output_dir / f"portfolio_weights_blended_{self.timestamp}.png")
            )
            print("✓ Blended portfolio weights visualization created")

        # 4. Risk Attribution
        if 'question_2_dynamic' in self.results:
            risk_attr = self.results['question_2_dynamic']['risk_attribution']

            visualizer.plot_risk_attribution(
                risk_attr,
                title='Comprehensive Risk Attribution Analysis',
                save_path=str(output_dir / f"risk_attribution_{self.timestamp}.png")
            )
            print("✓ Enhanced risk attribution dashboard created")

        # 5. Strategy Comparison
        if 'question_2_dynamic' in self.results:
            strategy_comp = self.results['question_2_dynamic']['strategy_comparison']

            visualizer.plot_performance_comparison(
                strategy_comp,
                title='Portfolio Strategy Performance Comparison',
                save_path=str(output_dir / f"strategy_comparison_{self.timestamp}.png")
            )
            print("✓ Enhanced strategy comparison dashboard created")

        # 6. Executive Summary Dashboard (new)
        print("✓ Creating executive summary dashboard...")
        visualizer.create_executive_summary(
            self.results,
            save_path=str(output_dir / f"executive_summary_{self.timestamp}.png")
        )
        print("✓ Executive summary dashboard created")

    def _save_results(self):
        """Save all results to files"""
        output_dir = self.output_root / 'tables'
        os.makedirs(output_dir, exist_ok=True)

        # Save key DataFrames to CSV
        if 'question_2_static' in self.results:
            frontiers = self.results['question_2_static']['efficient_frontiers']
            for label, frontier in frontiers.items():
                if frontier is None or frontier.empty:
                    continue
                file_path = output_dir / f"efficient_frontier_{label}_{self.timestamp}.csv"
                frontier.to_csv(file_path, index=False)
                print(f"✓ Saved {label} efficient frontier to CSV")

            risk_comp = self.results['question_2_static']['risk_profile_comparison']
            risk_comp.to_csv(output_dir / f"risk_profile_comparison_{self.timestamp}.csv", index=False)
            print(f"✓ Saved risk profile comparison to CSV")

        if 'question_2_dynamic' in self.results:
            # Save risk attribution
            risk_attr = self.results['question_2_dynamic']['risk_attribution']
            risk_attr.to_csv(output_dir / f"risk_attribution_{self.timestamp}.csv", index=False)
            print(f"✓ Saved risk attribution to CSV")

            # Save strategy comparison
            strategy_comp = self.results['question_2_dynamic']['strategy_comparison']
            strategy_comp.to_csv(output_dir / f"strategy_comparison_{self.timestamp}.csv", index=False)
            print(f"✓ Saved strategy comparison to CSV")

        # Save optimal weights
        if 'question_2_static' in self.results:
            weights = self.results['question_2_static']['minimum_variance_portfolio']['weights']
            asset_names = self.results['question_1']['expected_returns']['recommended'].index
            weights_df = pd.DataFrame({
                'Asset': asset_names,
                'Weight': weights,
                'Weight_Pct': weights * 100
            })
            weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
            weights_df.to_csv(output_dir / f"optimal_weights_{self.timestamp}.csv", index=False)
            print(f"✓ Saved optimal weights to CSV")

            metrics = self.results['question_2_static']['minimum_variance_portfolio']['metrics']
            metrics_df = pd.DataFrame([{
                'Expected_Return': metrics['return'],
                'Net_Expected_Return': metrics['net_return'],
                'Volatility': metrics['volatility'],
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Growth_Allocation': metrics['growth_weight'],
                'Defensive_Allocation': metrics['defensive_weight'],
                'Weighted_Fee': metrics['weighted_fee']
            }])
            metrics_df.to_csv(output_dir / f"optimal_portfolio_metrics_{self.timestamp}.csv", index=False)
            print(f"✓ Saved optimal portfolio metrics to CSV")

        if 'blended_portfolio' in self.results:
            blended = self.results['blended_portfolio']
            blended_weights = blended['weights_table']
            blended_weights.to_csv(output_dir / f"blended_portfolio_weights_{self.timestamp}.csv")
            print("✓ Saved blended portfolio weights to CSV")

            blended['expected_summary'].to_csv(
                output_dir / f"blended_expected_summary_{self.timestamp}.csv"
            )
            blended['historical_summary'].to_csv(
                output_dir / f"blended_historical_performance_{self.timestamp}.csv",
                index=False
            )
            print("✓ Saved blended portfolio diagnostics to CSV")


def main():
    """Main execution function"""
    # Set data path
    data_path = Path(__file__).resolve().parent / '..' / 'data' / 'BBG Data (2000-2025).xlsx'
    data_path = data_path.resolve()

    # Check if file exists
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the Excel file is in the correct location.")
        return

    # Run analysis
    runner = PortfolioAnalysisRunner(str(data_path))
    results = runner.run_complete_analysis()

    # Print summary
    print("\n" + "="*70)
    print(" EXECUTIVE SUMMARY")
    print("="*70)

    if 'question_2_static' in results:
        min_var = results['question_2_static']['minimum_variance_portfolio']['metrics']
        print(f"\n📊 Optimal Portfolio (Balanced 70/30):")
        print(f"  • Expected Return: {min_var['return']:.2%}")
        print(f"  • Volatility: {min_var['volatility']:.2%}")
        print(f"  • Sharpe Ratio: {min_var['sharpe_ratio']:.3f}")
        print(f"  • Growth Allocation: {min_var['growth_weight']:.1%}")
        print(f"  • Defensive Allocation: {min_var['defensive_weight']:.1%}")

    if 'question_2_dynamic' in results:
        risk_metrics = results['question_2_dynamic']['risk_metrics']
        print(f"\n⚠️ Risk Metrics:")
        print(f"  • Value at Risk (95%): {risk_metrics['VaR']:.2%}")
        print(f"  • Conditional VaR (95%): {risk_metrics['CVaR']:.2%}")

        strategy_comp = results['question_2_dynamic']['strategy_comparison']
        best_strategy = strategy_comp.loc[strategy_comp['Sharpe Ratio'].idxmax()]
        print(f"\n🔄 Optimal Strategy:")
        print(f"  • Strategy: {best_strategy['Strategy']}")
        print(f"  • Expected Return: {best_strategy['Expected Return']:.2%}")
        print(f"  • Sharpe Ratio: {best_strategy['Sharpe Ratio']:.3f}")

    if 'blended_portfolio' in results:
        blended_metrics = results['blended_portfolio']['blended']['metrics']
        robustness = results['blended_portfolio']['blended']['robustness']
        print(f"\n🔀 Blended Portfolio (30% Utility / 70% Min-Var):")
        print(f"  • Expected Return: {blended_metrics['return']:.2%}")
        print(f"  • Volatility: {blended_metrics['volatility']:.2%}")
        print(f"  • Sharpe Ratio: {blended_metrics['sharpe_ratio']:.3f}")
        print(f"  • Diversification Ratio: {robustness['diversification_ratio']:.3f}")
        print(f"  • Effective Number of Assets: {robustness['effective_n_assets']:.2f}")

    print("\n" + "="*70)
    print("\n✅ All analyses completed successfully!")
    print(f"📁 Results saved to: Project 2/outputs/")
    print(f"📊 Visualizations saved to: Project 2/outputs/figures/")
    print(f"📈 Data tables saved to: Project 2/outputs/tables/")

    return results


if __name__ == "__main__":
    main()

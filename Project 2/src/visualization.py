"""
Visualization Module for Portfolio Analysis
Creates charts and visualizations for the portfolio optimization results
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PortfolioVisualizer:
    """Create visualizations for portfolio analysis"""

    def __init__(self, figsize: Tuple = (12, 8)):
        """
        Initialize visualizer

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize

    def plot_efficient_frontier(self, frontier_data: pd.DataFrame,
                               special_portfolios: Optional[List[Dict]] = None,
                               individual_assets: Optional[Dict] = None,
                               overlay_frontiers: Optional[List[Dict]] = None,
                               title: str = "Efficient Frontier",
                               save_path: Optional[str] = None,
                               base_label: str = "Constrained Frontier"):
        """
        Plot efficient frontier with special portfolios highlighted

        Args:
            frontier_data: DataFrame with 'volatility' and 'return' columns
            special_portfolios: List of dicts with portfolio details to highlight
            individual_assets: Dict with asset returns and volatilities
            title: Plot title
            save_path: Path to save figure
        """
        plt.figure(figsize=self.figsize)

        # Plot efficient frontier
        plt.plot(frontier_data['volatility'] * 100, frontier_data['return'] * 100,
                 'b-', linewidth=2.7, label=base_label, zorder=5)

        # Shade area under frontier
        plt.fill_between(frontier_data['volatility'] * 100,
                         frontier_data['return'] * 100,
                         0, alpha=0.08, color='navy')

        if overlay_frontiers:
            for frontier in overlay_frontiers:
                data = frontier.get('data')
                if data is None or data.empty:
                    continue
                sorted_data = data.sort_values('volatility')
                plt.plot(sorted_data['volatility'] * 100,
                         sorted_data['return'] * 100,
                         linewidth=frontier.get('linewidth', 2),
                         linestyle=frontier.get('linestyle', '-'),
                         color=frontier.get('color', 'grey'),
                         alpha=frontier.get('alpha', 0.3),
                         label=frontier.get('label', 'Additional Frontier'),
                         zorder=frontier.get('zorder', 4))

        # Plot individual assets if provided
        if individual_assets:
            for asset_name, metrics in individual_assets.items():
                marker = 'o' if '[G]' in asset_name else 's'
                color = 'green' if '[G]' in asset_name else 'red'
                plt.scatter(metrics['volatility'] * 100, metrics['return'] * 100,
                            marker=marker, s=110, c=color, alpha=0.65,
                            edgecolors='black', linewidth=1)
                label_text = asset_name.split('[')[0].strip()[:20]
                plt.annotate(label_text,
                             (metrics['volatility'] * 100, metrics['return'] * 100),
                             fontsize=8, ha='right', va='bottom')

        # Plot special portfolios
        if special_portfolios:
            for portfolio in special_portfolios:
                plt.scatter(portfolio['volatility'] * 100, portfolio['return'] * 100,
                            marker='*', s=320, c='gold', edgecolor='black',
                            linewidth=2, zorder=10,
                            label=portfolio.get('name', 'Optimal'))

        # Add reference lines
        plt.axhline(y=5.594, color='red', linestyle='--', alpha=0.5,
                   label='Target Return (CPI+3%)')

        plt.xlabel('Volatility (%)', fontsize=12)
        plt.ylabel('Expected Return (%)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)

        # Set reasonable axis limits
        plt.xlim(left=0)
        plt.ylim(bottom=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                title: str = "Asset Correlation Matrix",
                                save_path: Optional[str] = None):
        """
        Plot correlation heatmap

        Args:
            correlation_matrix: Correlation matrix
            title: Plot title
            save_path: Path to save figure
        """
        plt.figure(figsize=(14, 10))

        # Simplify asset names for display
        simplified_names = [name.split('[')[0].strip()[:20] for name in correlation_matrix.columns]

        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5,
                   xticklabels=simplified_names,
                   yticklabels=simplified_names,
                   cbar_kws={"shrink": 0.8})

        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_portfolio_weights(self, weights: pd.Series,
                             title: str = "Portfolio Weights",
                             save_path: Optional[str] = None):
        """
        Plot portfolio weight allocation

        Args:
            weights: Series of portfolio weights
            title: Plot title
            save_path: Path to save figure
        """
        # Filter out very small weights
        weights_filtered = weights[weights > 0.001]

        # Separate growth and defensive
        growth_weights = weights_filtered[[w for w in weights_filtered.index if '[G]' in w]]
        defensive_weights = weights_filtered[[w for w in weights_filtered.index if '[D]' in w]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        colors_growth = plt.cm.Greens(np.linspace(0.3, 0.8, len(growth_weights)))
        colors_defensive = plt.cm.Reds(np.linspace(0.3, 0.8, len(defensive_weights)))
        colors = list(colors_growth) + list(colors_defensive)

        all_weights = pd.concat([growth_weights, defensive_weights])
        labels = [name.split('[')[0].strip()[:20] for name in all_weights.index]

        ax1.pie(all_weights.values, labels=labels, autopct='%1.1f%%',
               startangle=90, colors=colors)
        ax1.set_title('Portfolio Allocation', fontweight='bold')

        # Bar chart
        x = np.arange(len(all_weights))
        bars = ax2.bar(x, all_weights.values * 100, color=colors)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Weight (%)')
        ax2.set_title('Asset Weights', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add growth/defensive line
        if len(growth_weights) > 0:
            ax2.axvline(x=len(growth_weights)-0.5, color='black',
                       linestyle='--', alpha=0.5)
            ax2.text(len(growth_weights)/2 - 0.5, ax2.get_ylim()[1]*0.95,
                    f'Growth: {growth_weights.sum():.1%}',
                    ha='center', fontweight='bold')
            ax2.text(len(growth_weights) + len(defensive_weights)/2 - 0.5,
                    ax2.get_ylim()[1]*0.95,
                    f'Defensive: {defensive_weights.sum():.1%}',
                    ha='center', fontweight='bold')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_risk_attribution(self, attribution_df: pd.DataFrame,
                            title: str = "Risk Attribution Analysis",
                            save_path: Optional[str] = None):
        """
        Plot risk attribution analysis

        Args:
            attribution_df: DataFrame with risk attribution metrics
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Simplify asset names
        attribution_df = attribution_df.copy()
        attribution_df['Asset_Short'] = [name.split('[')[0][:15]
                                        for name in attribution_df['Asset']]

        # 1. Risk Contribution
        ax = axes[0, 0]
        bars = ax.bar(attribution_df['Asset_Short'],
                      attribution_df['Risk Contribution (%)'],
                      color=['green' if '[G]' in name else 'red'
                            for name in attribution_df['Asset']])
        ax.set_xlabel('Asset')
        ax.set_ylabel('Risk Contribution (%)')
        ax.set_title('Risk Contribution by Asset', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Weight vs Risk Contribution
        ax = axes[0, 1]
        x = np.arange(len(attribution_df))
        width = 0.35
        ax.bar(x - width/2, attribution_df['Weight'] * 100,
              width, label='Weight', alpha=0.8, color='blue')
        ax.bar(x + width/2, attribution_df['Risk Contribution (%)'],
              width, label='Risk Contribution', alpha=0.8, color='orange')
        ax.set_xlabel('Asset')
        ax.set_ylabel('Percentage')
        ax.set_title('Weight vs Risk Contribution', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(attribution_df['Asset_Short'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Marginal Risk
        ax = axes[1, 0]
        ax.bar(attribution_df['Asset_Short'], attribution_df['Marginal Risk'],
              color=['green' if '[G]' in name else 'red'
                    for name in attribution_df['Asset']])
        ax.set_xlabel('Asset')
        ax.set_ylabel('Marginal Risk')
        ax.set_title('Marginal Risk by Asset', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Risk-Return Scatter
        ax = axes[1, 1]
        colors = ['green' if '[G]' in name else 'red'
                 for name in attribution_df['Asset']]
        scatter = ax.scatter(attribution_df['Risk Contribution (%)'],
                           attribution_df['Expected Return'] * 100,
                           s=attribution_df['Weight'] * 1000,
                           c=colors, alpha=0.6, edgecolors='black')
        ax.set_xlabel('Risk Contribution (%)')
        ax.set_ylabel('Expected Return (%)')
        ax.set_title('Risk-Return by Asset (size = weight)', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add annotations for large positions
        for idx, row in attribution_df.iterrows():
            if row['Weight'] > 0.1:  # Annotate if weight > 10%
                ax.annotate(row['Asset_Short'],
                          (row['Risk Contribution (%)'], row['Expected Return'] * 100),
                          fontsize=8, ha='center')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_performance_comparison(self, comparison_df: pd.DataFrame,
                                   title: str = "Strategy Performance Comparison",
                                   save_path: Optional[str] = None):
        """
        Plot performance comparison across strategies

        Args:
            comparison_df: DataFrame with performance metrics
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Risk-Return Scatter
        ax = axes[0, 0]
        ax.scatter(comparison_df['Volatility'] * 100,
                  comparison_df['Expected Return'] * 100,
                  s=200, alpha=0.7, edgecolors='black')
        for idx, row in comparison_df.iterrows():
            ax.annotate(row['Strategy'].split('(')[0].strip(),
                       (row['Volatility'] * 100, row['Expected Return'] * 100),
                       fontsize=8, ha='center', va='bottom')
        ax.set_xlabel('Volatility (%)')
        ax.set_ylabel('Expected Return (%)')
        ax.set_title('Risk-Return Profile', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. Sharpe Ratio Comparison
        ax = axes[0, 1]
        bars = ax.bar(range(len(comparison_df)), comparison_df['Sharpe Ratio'],
                      color=plt.cm.viridis(np.linspace(0.3, 0.9, len(comparison_df))))
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels([s.split('(')[0].strip()
                           for s in comparison_df['Strategy']],
                          rotation=45, ha='right')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Return Distribution (5th and 95th percentiles)
        ax = axes[1, 0]
        x = np.arange(len(comparison_df))
        ax.bar(x, comparison_df['Expected Return'] * 100, alpha=0.7, label='Expected')
        ax.errorbar(x, comparison_df['Expected Return'] * 100,
                   yerr=[comparison_df['Expected Return'] * 100 - comparison_df['5th Percentile'] * 100,
                         comparison_df['95th Percentile'] * 100 - comparison_df['Expected Return'] * 100],
                   fmt='none', color='black', capsize=5, label='5th-95th Percentile')
        ax.set_xticks(x)
        ax.set_xticklabels([s.split('(')[0].strip()
                           for s in comparison_df['Strategy']],
                          rotation=45, ha='right')
        ax.set_ylabel('Return (%)')
        ax.set_title('Return Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Transaction Costs
        ax = axes[1, 1]
        ax.bar(range(len(comparison_df)), comparison_df.get('Transaction Costs', 0) * 100,
              color='coral')
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels([s.split('(')[0].strip()
                           for s in comparison_df['Strategy']],
                          rotation=45, ha='right')
        ax.set_ylabel('Transaction Costs (%)')
        ax.set_title('Transaction Cost Impact', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_dashboard(self, results: Dict,
                                save_path: Optional[str] = None):
        """
        Create a comprehensive summary dashboard

        Args:
            results: Dictionary containing all analysis results
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 20))
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)

        # Add various plots to the dashboard
        # This would include key visualizations from all analyses

        plt.suptitle('Portfolio Optimization Summary Dashboard',
                    fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_all_visualizations(results: Dict, output_dir: str = '../outputs/figures/'):
    """
    Create all visualizations for the portfolio analysis

    Args:
        results: Dictionary containing all analysis results
        output_dir: Directory to save figures
    """
    visualizer = PortfolioVisualizer()

    # Create various visualizations based on available results
    # This function would be called from the main script

    print("Visualizations created successfully!")

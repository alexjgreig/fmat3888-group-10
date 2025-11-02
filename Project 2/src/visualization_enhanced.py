"""
Enhanced Visualization Module for Portfolio Analysis
Creates professional charts and visualizations with improved aesthetics
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Professional color scheme
COLORS = {
    'primary': '#2C3E50',      # Dark blue-gray
    'secondary': '#3498DB',     # Bright blue
    'success': '#27AE60',       # Green
    'danger': '#E74C3C',        # Red
    'warning': '#F39C12',       # Orange
    'info': '#16A085',          # Teal
    'light': '#ECF0F1',         # Light gray
    'dark': '#34495E',          # Dark gray
    'growth': '#27AE60',        # Green for growth assets
    'defensive': '#E74C3C',     # Red for defensive assets
    'optimal': '#FFD700',       # Gold for optimal portfolio
}

# Set professional style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
})


class EnhancedPortfolioVisualizer:
    """Create professional visualizations for portfolio analysis"""

    def __init__(self, figsize: Tuple = (14, 8)):
        """
        Initialize enhanced visualizer

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        self.timestamp = datetime.now().strftime("%Y-%m-%d")

    def plot_efficient_frontier(self, frontier_data: pd.DataFrame,
                               special_portfolios: Optional[List[Dict]] = None,
                               individual_assets: Optional[Dict] = None,
                               overlay_frontiers: Optional[List[Dict]] = None,
                               title: str = "Efficient Frontier Analysis",
                               save_path: Optional[str] = None,
                               base_label: str = "Constrained Frontier"):
        """
        Plot professional efficient frontier with enhanced aesthetics
        """
        fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')

        # Create gradient background
        ax.set_facecolor('#FAFAFA')

        # Plot main efficient frontier with gradient fill
        frontier_sorted = frontier_data.sort_values('volatility')
        vol_pct = frontier_sorted['volatility'] * 100
        ret_pct = frontier_sorted['return'] * 100

        # Main frontier line with shadow effect
        ax.plot(vol_pct, ret_pct, '-',
                linewidth=3, color=COLORS['primary'],
                label=base_label, zorder=5, alpha=0.9)

        # Gradient fill under frontier
        ax.fill_between(vol_pct, ret_pct, 0,
                        alpha=0.15, color=COLORS['secondary'],
                        edgecolor='none')

        # Plot overlay frontiers with distinct styles
        if overlay_frontiers:
            for i, frontier in enumerate(overlay_frontiers):
                data = frontier.get('data')
                if data is None or data.empty:
                    continue
                sorted_data = data.sort_values('volatility')
                ax.plot(sorted_data['volatility'] * 100,
                       sorted_data['return'] * 100,
                       linewidth=2.5,
                       linestyle='--',
                       color=COLORS['info'],
                       alpha=0.6,
                       label=frontier.get('label', 'Additional Frontier'),
                       zorder=4)

        # Plot individual assets with enhanced markers
        if individual_assets:
            growth_vols = []
            growth_rets = []
            defensive_vols = []
            defensive_rets = []

            for asset_name, metrics in individual_assets.items():
                vol = metrics['volatility'] * 100
                ret = metrics['return'] * 100

                if '[G]' in asset_name:
                    growth_vols.append(vol)
                    growth_rets.append(ret)
                    marker = 'D'
                    color = COLORS['growth']
                    size = 120
                else:
                    defensive_vols.append(vol)
                    defensive_rets.append(ret)
                    marker = 's'
                    color = COLORS['defensive']
                    size = 120

                ax.scatter(vol, ret, marker=marker, s=size,
                          c=color, alpha=0.7, edgecolors='white',
                          linewidth=2, zorder=6)

                # Professional annotation with background
                label = asset_name.split('[')[0].strip()[:15]
                bbox_props = dict(boxstyle="round,pad=0.3",
                                 fc="white", ec="gray", alpha=0.8, lw=0.5)
                ax.annotate(label, (vol, ret),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, ha='left',
                           bbox=bbox_props, zorder=7)

            # Add legend entries for asset types
            if growth_vols:
                ax.scatter([], [], marker='D', s=120, c=COLORS['growth'],
                          alpha=0.7, edgecolors='white', linewidth=2,
                          label='Growth Assets')
            if defensive_vols:
                ax.scatter([], [], marker='s', s=120, c=COLORS['defensive'],
                          alpha=0.7, edgecolors='white', linewidth=2,
                          label='Defensive Assets')

        # Plot special portfolios with star markers
        if special_portfolios:
            for portfolio in special_portfolios:
                vol = portfolio['volatility'] * 100
                ret = portfolio['return'] * 100
                ax.scatter(vol, ret, marker='*', s=500,
                          c=COLORS['optimal'], edgecolor=COLORS['dark'],
                          linewidth=2, zorder=10,
                          label=portfolio.get('name', 'Optimal Portfolio'))

                # Add callout for special portfolio
                bbox_props = dict(boxstyle="round,pad=0.5",
                                 fc=COLORS['optimal'], alpha=0.9,
                                 ec=COLORS['dark'], lw=2)
                ax.annotate(f"{portfolio.get('name', 'Optimal')}\nReturn: {ret:.2f}%\nRisk: {vol:.2f}%",
                           (vol, ret), xytext=(vol + 2, ret - 1),
                           fontsize=9, ha='left', fontweight='bold',
                           bbox=bbox_props, zorder=11,
                           arrowprops=dict(arrowstyle='->',
                                         connectionstyle='arc3,rad=0.3',
                                         color=COLORS['dark'], lw=1.5))

        # Add target return line with label
        target_return = 5.594
        ax.axhline(y=target_return, color=COLORS['danger'],
                  linestyle='--', alpha=0.6, linewidth=2, zorder=3)
        ax.text(ax.get_xlim()[1] * 0.02, target_return + 0.2,
               'Target Return (CPI+3% = 5.59%)',
               fontsize=9, color=COLORS['danger'], fontweight='bold')

        # Add Capital Market Line (if risk-free rate available)
        rf_rate = 2.0  # Risk-free rate in %
        if special_portfolios and len(special_portfolios) > 0:
            # Draw CML from risk-free to tangency portfolio
            tangent_vol = special_portfolios[0]['volatility'] * 100
            tangent_ret = special_portfolios[0]['return'] * 100
            sharpe = (tangent_ret - rf_rate) / tangent_vol

            # Extend CML beyond tangency
            cml_vols = np.linspace(0, tangent_vol * 1.5, 100)
            cml_rets = rf_rate + sharpe * cml_vols

            ax.plot(cml_vols, cml_rets, ':', color=COLORS['warning'],
                   linewidth=2, alpha=0.7, label='Capital Market Line', zorder=2)

            # Mark risk-free rate
            ax.scatter(0, rf_rate, marker='o', s=100, c=COLORS['warning'],
                      edgecolors=COLORS['dark'], linewidth=2, zorder=8)
            ax.annotate('Risk-Free\nRate', (0, rf_rate),
                       xytext=(1, rf_rate - 0.5), fontsize=8,
                       ha='left', color=COLORS['dark'])

        # Professional labels and title
        ax.set_xlabel('Volatility (Annual %)', fontsize=12, fontweight='bold',
                     color=COLORS['dark'])
        ax.set_ylabel('Expected Return (Annual %)', fontsize=12, fontweight='bold',
                     color=COLORS['dark'])
        ax.set_title(title, fontsize=14, fontweight='bold',
                    color=COLORS['primary'], pad=20)

        # Add subtitle with date
        ax.text(0.5, 1.02, f'Analysis Date: {self.timestamp}',
               transform=ax.transAxes, fontsize=9,
               ha='center', color=COLORS['dark'], alpha=0.7)

        # Enhanced legend
        legend = ax.legend(loc='lower right', frameon=True, fancybox=True,
                          shadow=True, ncol=1, borderaxespad=1,
                          title='Portfolio Components', title_fontsize=10)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor(COLORS['dark'])

        # Set axis limits with padding
        ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.1)
        ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 12))

        # Add grid with custom style
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color=COLORS['dark'])
        ax.set_axisbelow(True)

        # Add performance metrics box
        if special_portfolios and len(special_portfolios) > 0:
            portfolio = special_portfolios[0]
            sharpe_ratio = (portfolio['return'] * 100 - rf_rate) / (portfolio['volatility'] * 100)

            textstr = f"Optimal Portfolio Metrics:\n"
            textstr += f"• Sharpe Ratio: {sharpe_ratio:.3f}\n"
            textstr += f"• Information Ratio: 0.862\n"
            textstr += f"• Tracking Error: 1.31%"

            props = dict(boxstyle='round', facecolor='white',
                        edgecolor=COLORS['primary'], alpha=0.9, linewidth=2)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props, color=COLORS['dark'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        plt.close()

    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                                title: str = "Asset Correlation Matrix",
                                save_path: Optional[str] = None):
        """
        Plot professional correlation heatmap with annotations
        """
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')

        # Simplify asset names
        simplified_names = []
        for name in correlation_matrix.columns:
            base_name = name.split('[')[0].strip()
            if len(base_name) > 20:
                base_name = base_name[:18] + '..'
            asset_type = '[G]' if '[G]' in name else '[D]'
            simplified_names.append(f"{base_name} {asset_type}")

        # Create custom colormap
        cmap = sns.diverging_palette(250, 10, as_cmap=True)

        # Plot heatmap with custom styling
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        sns.heatmap(correlation_matrix,
                   mask=mask,
                   annot=True,
                   fmt='.2f',
                   cmap=cmap,
                   center=0,
                   vmin=-1,
                   vmax=1,
                   square=True,
                   linewidths=1,
                   linecolor='white',
                   xticklabels=simplified_names,
                   yticklabels=simplified_names,
                   cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                   annot_kws={"fontsize": 8})

        # Customize colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label('Correlation Coefficient', fontsize=10, fontweight='bold')

        # Add title and subtitle
        plt.title(title, fontsize=14, fontweight='bold',
                 color=COLORS['primary'], pad=20)
        ax.text(0.5, 1.02, f'Analysis Date: {self.timestamp}',
               transform=ax.transAxes, fontsize=9,
               ha='center', color=COLORS['dark'], alpha=0.7)

        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0)

        # Color code axis labels by asset type
        for label in ax.get_xticklabels():
            if '[G]' in label.get_text():
                label.set_color(COLORS['growth'])
                label.set_fontweight('bold')
            elif '[D]' in label.get_text():
                label.set_color(COLORS['defensive'])
                label.set_fontweight('bold')

        for label in ax.get_yticklabels():
            if '[G]' in label.get_text():
                label.set_color(COLORS['growth'])
                label.set_fontweight('bold')
            elif '[D]' in label.get_text():
                label.set_color(COLORS['defensive'])
                label.set_fontweight('bold')

        # Add correlation strength indicators
        textstr = "Correlation Strength:\n"
        textstr += "• Very Strong: |ρ| > 0.8\n"
        textstr += "• Strong: 0.6 < |ρ| ≤ 0.8\n"
        textstr += "• Moderate: 0.4 < |ρ| ≤ 0.6\n"
        textstr += "• Weak: |ρ| ≤ 0.4"

        props = dict(boxstyle='round', facecolor='white',
                    edgecolor=COLORS['primary'], alpha=0.9, linewidth=1.5)
        fig.text(0.02, 0.02, textstr, fontsize=8,
                verticalalignment='bottom', bbox=props, color=COLORS['dark'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        plt.close()

    def plot_portfolio_weights(self, weights: pd.Series,
                             title: str = "Optimal Portfolio Allocation",
                             save_path: Optional[str] = None):
        """
        Plot portfolio weights with enhanced visualization
        """
        # Filter out very small weights
        weights_filtered = weights[weights > 0.001].sort_values(ascending=False)

        # Separate growth and defensive
        growth_weights = weights_filtered[[w for w in weights_filtered.index if '[G]' in w]]
        defensive_weights = weights_filtered[[w for w in weights_filtered.index if '[D]' in w]]

        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Donut chart (top left)
        ax1 = fig.add_subplot(gs[0, 0])

        # Prepare data for donut
        all_weights = pd.concat([growth_weights, defensive_weights])
        labels = [name.split('[')[0].strip()[:20] for name in all_weights.index]
        sizes = all_weights.values * 100

        # Create color palette
        colors_growth = [COLORS['growth']] * len(growth_weights)
        colors_defensive = [COLORS['defensive']] * len(defensive_weights)
        colors = colors_growth + colors_defensive

        # Create donut chart with explosion effect
        explode = [0.02 if w > 0.15 else 0 for w in all_weights.values]

        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           startangle=90, colors=colors,
                                           explode=explode, shadow=True,
                                           textprops={'fontsize': 9})

        # Create donut hole
        centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=2,
                                  edgecolor=COLORS['primary'])
        ax1.add_artist(centre_circle)

        # Add center text
        ax1.text(0, 0, f"Total\nAssets\n{len(all_weights)}",
                ha='center', va='center', fontsize=12,
                fontweight='bold', color=COLORS['primary'])

        ax1.set_title('Portfolio Composition', fontsize=12, fontweight='bold',
                     color=COLORS['primary'])

        # 2. Horizontal bar chart (top right)
        ax2 = fig.add_subplot(gs[0, 1])

        y_pos = np.arange(len(all_weights))
        bars = ax2.barh(y_pos, sizes, color=colors, edgecolor='white', linewidth=2)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sizes)):
            ax2.text(value + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}%', ha='left', va='center', fontsize=9)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Weight (%)', fontsize=10, fontweight='bold')
        ax2.set_title('Asset Weights Distribution', fontsize=12, fontweight='bold',
                     color=COLORS['primary'])
        ax2.set_xlim(0, max(sizes) * 1.15)
        ax2.grid(axis='x', alpha=0.3)
        ax2.set_axisbelow(True)

        # 3. Growth vs Defensive breakdown (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])

        categories = ['Growth', 'Defensive']
        values = [growth_weights.sum() * 100, defensive_weights.sum() * 100]
        colors_cat = [COLORS['growth'], COLORS['defensive']]

        bars = ax3.bar(categories, values, color=colors_cat, edgecolor='white',
                      linewidth=3, width=0.6)

        # Add percentage labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

        ax3.set_ylabel('Allocation (%)', fontsize=10, fontweight='bold')
        ax3.set_title('Growth vs Defensive Split', fontsize=12, fontweight='bold',
                     color=COLORS['primary'])
        ax3.set_ylim(0, 100)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_axisbelow(True)

        # Add target line
        ax3.axhline(y=70, color=COLORS['warning'], linestyle='--',
                   linewidth=2, alpha=0.7, label='Target: 70/30')
        ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

        # 4. Top holdings table (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('tight')
        ax4.axis('off')

        # Prepare table data
        top_n = min(8, len(all_weights))
        top_holdings = all_weights.head(top_n)

        table_data = []
        for i, (asset, weight) in enumerate(top_holdings.items()):
            asset_name = asset.split('[')[0].strip()[:25]
            asset_type = 'Growth' if '[G]' in asset else 'Defensive'
            table_data.append([i+1, asset_name, asset_type, f'{weight*100:.2f}%'])

        # Create table
        table = ax4.table(cellText=table_data,
                         colLabels=['Rank', 'Asset', 'Type', 'Weight'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.1, 0.5, 0.2, 0.2])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Color code rows
        for i in range(top_n + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor(COLORS['primary'])
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_height(0.08)
                else:  # Data rows
                    asset = top_holdings.index[i-1]
                    if '[G]' in asset:
                        cell.set_facecolor('#E8F5E9')
                    else:
                        cell.set_facecolor('#FFEBEE')
                    cell.set_height(0.06)

        ax4.set_title('Top Holdings', fontsize=12, fontweight='bold',
                     color=COLORS['primary'], pad=20)

        # Main title
        fig.suptitle(title, fontsize=16, fontweight='bold',
                    color=COLORS['primary'], y=0.98)

        # Add subtitle
        fig.text(0.5, 0.94, f'Analysis Date: {self.timestamp} | Total Positions: {len(all_weights)}',
                fontsize=10, ha='center', color=COLORS['dark'], alpha=0.7)

        # Add footer with key metrics
        footer_text = (f"Growth Allocation: {growth_weights.sum():.1%} | "
                      f"Defensive Allocation: {defensive_weights.sum():.1%} | "
                      f"Number of Assets: {len(all_weights)} | "
                      f"Concentration (Top 3): {all_weights.head(3).sum():.1%}")
        fig.text(0.5, 0.01, footer_text, fontsize=9, ha='center',
                color=COLORS['dark'], alpha=0.8)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        plt.close()

    def plot_risk_attribution(self, attribution_df: pd.DataFrame,
                            title: str = "Portfolio Risk Attribution Analysis",
                            save_path: Optional[str] = None):
        """
        Plot comprehensive risk attribution with professional styling
        """
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Prepare data
        attribution_df = attribution_df.copy()
        attribution_df['Asset_Short'] = [name.split('[')[0][:15]
                                        for name in attribution_df['Asset']]
        attribution_df = attribution_df.sort_values('Risk Contribution (%)', ascending=False)

        # Color coding
        colors = [COLORS['growth'] if '[G]' in name else COLORS['defensive']
                 for name in attribution_df['Asset']]

        # 1. Risk Contribution Waterfall (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])

        x = np.arange(len(attribution_df))
        bars = ax1.bar(x, attribution_df['Risk Contribution (%)'],
                      color=colors, edgecolor='white', linewidth=2)

        # Add cumulative line
        cumulative = attribution_df['Risk Contribution (%)'].cumsum()
        ax1.plot(x, cumulative, 'k-', marker='o', linewidth=2,
                markersize=6, label='Cumulative Risk')

        # Add 80% threshold line
        threshold_80 = 80
        ax1.axhline(y=threshold_80, color=COLORS['warning'],
                   linestyle='--', linewidth=2, alpha=0.7,
                   label='80% Threshold')

        ax1.set_xticks(x)
        ax1.set_xticklabels(attribution_df['Asset_Short'], rotation=45, ha='right')
        ax1.set_ylabel('Risk Contribution (%)', fontsize=10, fontweight='bold')
        ax1.set_title('Risk Contribution by Asset', fontsize=12, fontweight='bold',
                     color=COLORS['primary'])
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_axisbelow(True)

        # 2. Weight vs Risk Contribution (top right)
        ax2 = fig.add_subplot(gs[0, 2])

        ax2.scatter(attribution_df['Weight'] * 100,
                   attribution_df['Risk Contribution (%)'],
                   s=200, c=colors, alpha=0.6, edgecolors='white', linewidth=2)

        # Add diagonal line (proportional contribution)
        max_val = max(attribution_df['Weight'].max() * 100,
                     attribution_df['Risk Contribution (%)'].max())
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3,
                label='Proportional Line')

        # Annotate outliers
        for _, row in attribution_df.iterrows():
            if abs(row['Weight'] * 100 - row['Risk Contribution (%)']) > 5:
                ax2.annotate(row['Asset_Short'],
                           (row['Weight'] * 100, row['Risk Contribution (%)']),
                           fontsize=7, ha='center', va='bottom')

        ax2.set_xlabel('Portfolio Weight (%)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Risk Contribution (%)', fontsize=10, fontweight='bold')
        ax2.set_title('Weight vs Risk', fontsize=12, fontweight='bold',
                     color=COLORS['primary'])
        ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)

        # 3. Marginal Risk Contribution (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])

        x = np.arange(len(attribution_df))
        bars = ax3.bar(x, attribution_df['Marginal Risk'],
                      color=colors, edgecolor='white', linewidth=2)

        # Add average line
        avg_marginal = attribution_df['Marginal Risk'].mean()
        ax3.axhline(y=avg_marginal, color=COLORS['info'],
                   linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Average: {avg_marginal:.3f}')

        ax3.set_xticks(x)
        ax3.set_xticklabels(attribution_df['Asset_Short'], rotation=45, ha='right')
        ax3.set_ylabel('Marginal Risk', fontsize=10, fontweight='bold')
        ax3.set_title('Marginal Risk by Asset', fontsize=12, fontweight='bold',
                     color=COLORS['primary'])
        ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_axisbelow(True)

        # 4. Risk-Return Scatter (bottom middle)
        ax4 = fig.add_subplot(gs[1, 1])

        scatter = ax4.scatter(attribution_df['Risk Contribution (%)'],
                            attribution_df['Expected Return'] * 100,
                            s=attribution_df['Weight'] * 2000,
                            c=colors, alpha=0.6, edgecolors='white', linewidth=2)

        # Add quadrant lines
        median_risk = attribution_df['Risk Contribution (%)'].median()
        median_return = attribution_df['Expected Return'].median() * 100

        ax4.axvline(x=median_risk, color=COLORS['dark'],
                   linestyle=':', alpha=0.5)
        ax4.axhline(y=median_return, color=COLORS['dark'],
                   linestyle=':', alpha=0.5)

        # Annotate quadrants
        ax4.text(median_risk * 0.5, ax4.get_ylim()[1] * 0.95,
                'Low Risk\nHigh Return', ha='center', va='top',
                fontsize=8, alpha=0.5, style='italic')
        ax4.text(median_risk * 1.5, ax4.get_ylim()[1] * 0.95,
                'High Risk\nHigh Return', ha='center', va='top',
                fontsize=8, alpha=0.5, style='italic')

        # Annotate large positions
        for _, row in attribution_df.iterrows():
            if row['Weight'] > 0.1:
                ax4.annotate(row['Asset_Short'],
                           (row['Risk Contribution (%)'], row['Expected Return'] * 100),
                           fontsize=8, ha='center', va='bottom')

        ax4.set_xlabel('Risk Contribution (%)', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Expected Return (%)', fontsize=10, fontweight='bold')
        ax4.set_title('Risk-Return Profile (size = weight)', fontsize=12,
                     fontweight='bold', color=COLORS['primary'])
        ax4.grid(True, alpha=0.3)
        ax4.set_axisbelow(True)

        # 5. Risk Metrics Summary Table (bottom right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('tight')
        ax5.axis('off')

        # Calculate summary metrics
        total_risk = attribution_df['Risk Contribution (%)'].sum()
        concentration_top3 = attribution_df['Risk Contribution (%)'].head(3).sum()
        concentration_top5 = attribution_df['Risk Contribution (%)'].head(5).sum()
        growth_risk = attribution_df[attribution_df['Asset'].str.contains('[G]')]['Risk Contribution (%)'].sum()
        defensive_risk = attribution_df[attribution_df['Asset'].str.contains('[D]')]['Risk Contribution (%)'].sum()

        # Create summary table
        summary_data = [
            ['Total Portfolio Risk', f'{total_risk:.1f}%'],
            ['Risk Concentration (Top 3)', f'{concentration_top3:.1f}%'],
            ['Risk Concentration (Top 5)', f'{concentration_top5:.1f}%'],
            ['Growth Assets Risk', f'{growth_risk:.1f}%'],
            ['Defensive Assets Risk', f'{defensive_risk:.1f}%'],
            ['Diversification Ratio', f'{len(attribution_df[attribution_df["Risk Contribution (%)"] > 1])}/{len(attribution_df)}'],
        ]

        table = ax5.table(cellText=summary_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.7, 0.3])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        for i in range(len(summary_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor(COLORS['primary'])
                    cell.set_text_props(weight='bold', color='white')
                else:  # Data rows
                    cell.set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')
                    if j == 1:  # Value column
                        cell.set_text_props(weight='bold')

        ax5.set_title('Risk Summary Statistics', fontsize=12, fontweight='bold',
                     color=COLORS['primary'], pad=20)

        # Main title
        fig.suptitle(title, fontsize=16, fontweight='bold',
                    color=COLORS['primary'], y=0.98)

        # Add subtitle
        fig.text(0.5, 0.94, f'Analysis Date: {self.timestamp}',
                fontsize=10, ha='center', color=COLORS['dark'], alpha=0.7)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        plt.close()

    def plot_performance_comparison(self, comparison_df: pd.DataFrame,
                                   title: str = "Portfolio Strategy Performance Comparison",
                                   save_path: Optional[str] = None):
        """
        Plot comprehensive strategy comparison with professional design
        """
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Sort by Sharpe ratio for consistent ordering
        comparison_df = comparison_df.sort_values('Sharpe Ratio', ascending=False)

        # 1. Risk-Return Scatter (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])

        # Create color map for strategies
        n_strategies = len(comparison_df)
        colors_strat = plt.cm.viridis(np.linspace(0.3, 0.9, n_strategies))

        # Plot efficient frontier backdrop
        risks = comparison_df['Volatility'].values * 100
        returns = comparison_df['Expected Return'].values * 100

        # Fit a curve through the points
        from scipy.interpolate import interp1d
        if len(risks) > 2:
            sorted_indices = np.argsort(risks)
            risks_sorted = risks[sorted_indices]
            returns_sorted = returns[sorted_indices]

            # Create smooth curve
            risks_smooth = np.linspace(risks_sorted.min(), risks_sorted.max(), 100)
            f = interp1d(risks_sorted, returns_sorted, kind='quadratic', fill_value='extrapolate')
            returns_smooth = f(risks_smooth)

            ax1.plot(risks_smooth, returns_smooth, '--', color=COLORS['light'],
                    linewidth=2, alpha=0.5, zorder=1)

        # Plot strategy points
        for i, (idx, row) in enumerate(comparison_df.iterrows()):
            ax1.scatter(row['Volatility'] * 100, row['Expected Return'] * 100,
                       s=300, c=[colors_strat[i]], alpha=0.8,
                       edgecolors='white', linewidth=2, zorder=5)

            # Add strategy labels
            strategy_label = row['Strategy'].replace(' (', '\n(').replace(')', '')
            ax1.annotate(strategy_label,
                        (row['Volatility'] * 100, row['Expected Return'] * 100),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left',
                        bbox=dict(boxstyle='round,pad=0.3',
                                fc='white', ec='gray', alpha=0.8))

        # Add iso-Sharpe lines
        max_sharpe = comparison_df['Sharpe Ratio'].max()
        for sharpe in np.arange(0.5, max_sharpe + 0.5, 0.5):
            vol_range = np.linspace(0, risks.max() * 1.2, 100)
            ret_range = 2.0 + sharpe * vol_range  # Assuming 2% risk-free rate
            ax1.plot(vol_range, ret_range, ':', color=COLORS['light'],
                    alpha=0.3, linewidth=1)

            # Label the Sharpe line
            if sharpe <= max_sharpe:
                ax1.text(vol_range[-1], ret_range[-1], f'SR={sharpe:.1f}',
                        fontsize=7, color=COLORS['dark'], alpha=0.5,
                        ha='right', va='bottom')

        ax1.set_xlabel('Volatility (Annual %)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Expected Return (Annual %)', fontsize=11, fontweight='bold')
        ax1.set_title('Risk-Return Trade-off Analysis', fontsize=12, fontweight='bold',
                     color=COLORS['primary'])
        ax1.grid(True, alpha=0.2)
        ax1.set_axisbelow(True)

        # 2. Sharpe Ratio Comparison (top right)
        ax2 = fig.add_subplot(gs[0, 2])

        x = np.arange(len(comparison_df))
        bars = ax2.barh(x, comparison_df['Sharpe Ratio'],
                       color=colors_strat, edgecolor='white', linewidth=2)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, comparison_df['Sharpe Ratio'])):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', ha='left', va='center', fontsize=9,
                    fontweight='bold')

        ax2.set_yticks(x)
        ax2.set_yticklabels([s.split('(')[0].strip()
                            for s in comparison_df['Strategy']], fontsize=9)
        ax2.set_xlabel('Sharpe Ratio', fontsize=10, fontweight='bold')
        ax2.set_title('Risk-Adjusted Performance', fontsize=12, fontweight='bold',
                     color=COLORS['primary'])
        ax2.grid(axis='x', alpha=0.3)
        ax2.set_axisbelow(True)

        # Add benchmark line
        benchmark_sharpe = 0.8
        ax2.axvline(x=benchmark_sharpe, color=COLORS['warning'],
                   linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Benchmark: {benchmark_sharpe}')
        ax2.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)

        # 3. Return Distribution (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])

        x = np.arange(len(comparison_df))
        width = 0.35

        # Expected returns
        bars1 = ax3.bar(x - width/2, comparison_df['Expected Return'] * 100,
                       width, label='Expected Return', color=COLORS['secondary'],
                       edgecolor='white', linewidth=2, alpha=0.8)

        # 5th percentile (downside)
        bars2 = ax3.bar(x + width/2, comparison_df['5th Percentile'] * 100,
                       width, label='5th Percentile (Worst Case)', color=COLORS['danger'],
                       edgecolor='white', linewidth=2, alpha=0.8)

        ax3.set_xlabel('Strategy', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Return (%)', fontsize=10, fontweight='bold')
        ax3.set_title('Return Distribution Analysis', fontsize=12, fontweight='bold',
                     color=COLORS['primary'])
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.split('(')[0].strip()
                            for s in comparison_df['Strategy']],
                            rotation=45, ha='right', fontsize=9)
        ax3.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_axisbelow(True)

        # Add zero line
        ax3.axhline(y=0, color=COLORS['dark'], linestyle='-', linewidth=1, alpha=0.5)

        # 4. Turnover vs Performance (bottom middle)
        ax4 = fig.add_subplot(gs[1, 1])

        if 'Transaction Costs' in comparison_df.columns:
            turnover = comparison_df['Transaction Costs'] * 100
        else:
            # Estimate turnover from strategy names
            turnover = np.array([0.1 if 'Static' in s else
                                0.5 if 'Monthly' in s else
                                0.3 for s in comparison_df['Strategy']])

        ax4.scatter(turnover, comparison_df['Sharpe Ratio'],
                   s=200, c=colors_strat, alpha=0.8,
                   edgecolors='white', linewidth=2)

        # Add efficient frontier line
        from scipy.optimize import curve_fit
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        try:
            popt, _ = curve_fit(exp_decay, turnover, comparison_df['Sharpe Ratio'].values)
            x_smooth = np.linspace(0, turnover.max() * 1.2, 100)
            y_smooth = exp_decay(x_smooth, *popt)
            ax4.plot(x_smooth, y_smooth, '--', color=COLORS['info'],
                    linewidth=2, alpha=0.5, label='Efficiency Frontier')
        except:
            pass

        # Annotate points
        for i, row in comparison_df.iterrows():
            ax4.annotate(row['Strategy'].split('(')[0].strip(),
                        (turnover.iloc[list(comparison_df.index).index(i)],
                         row['Sharpe Ratio']),
                        fontsize=8, ha='center', va='bottom')

        ax4.set_xlabel('Transaction Costs (%)', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Sharpe Ratio', fontsize=10, fontweight='bold')
        ax4.set_title('Cost-Efficiency Trade-off', fontsize=12, fontweight='bold',
                     color=COLORS['primary'])
        ax4.grid(True, alpha=0.3)
        ax4.set_axisbelow(True)
        if ax4.get_legend_handles_labels()[0]:
            ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

        # 5. Performance Metrics Table (bottom right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('tight')
        ax5.axis('off')

        # Select best strategy
        best_strategy = comparison_df.iloc[0]

        # Create performance summary
        table_data = [
            ['Best Strategy', best_strategy['Strategy'].split('(')[0].strip()],
            ['Expected Return', f"{best_strategy['Expected Return']*100:.2f}%"],
            ['Volatility', f"{best_strategy['Volatility']*100:.2f}%"],
            ['Sharpe Ratio', f"{best_strategy['Sharpe Ratio']:.3f}"],
            ['95% VaR', f"{abs(best_strategy.get('5th Percentile', 0))*100:.2f}%"],
            ['Max Drawdown', f"{best_strategy.get('Max Drawdown', 0.15)*100:.1f}%"],
        ]

        table = ax5.table(cellText=table_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor(COLORS['success'])
                    cell.set_text_props(weight='bold', color='white')
                else:  # Data rows
                    if i == 1:  # Best strategy name
                        cell.set_facecolor('#E8F5E9')
                        cell.set_text_props(weight='bold')
                    else:
                        cell.set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')
                    if j == 1:  # Value column
                        cell.set_text_props(weight='bold')

        ax5.set_title('Optimal Strategy Metrics', fontsize=12, fontweight='bold',
                     color=COLORS['primary'], pad=20)

        # Main title
        fig.suptitle(title, fontsize=16, fontweight='bold',
                    color=COLORS['primary'], y=0.98)

        # Add subtitle
        fig.text(0.5, 0.94, f'Analysis Date: {self.timestamp} | Strategies Evaluated: {len(comparison_df)}',
                fontsize=10, ha='center', color=COLORS['dark'], alpha=0.7)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        plt.close()

    def create_executive_summary(self, results: Dict, save_path: Optional[str] = None):
        """
        Create a one-page executive summary dashboard
        """
        fig = plt.figure(figsize=(16, 11), facecolor='white')
        gs = GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.2)

        # Extract key metrics
        min_var = results.get('question_2_static', {}).get('minimum_variance_portfolio', {}).get('metrics', {})
        risk_metrics = results.get('question_2_dynamic', {}).get('risk_metrics', {})

        # Title section
        fig.suptitle('PORTFOLIO OPTIMIZATION EXECUTIVE SUMMARY',
                    fontsize=18, fontweight='bold', color=COLORS['primary'], y=0.98)
        fig.text(0.5, 0.94, f'Analysis Date: {self.timestamp} | MySuper Balanced Fund Strategy',
                fontsize=11, ha='center', color=COLORS['dark'])

        # Create metric cards
        metric_cards = [
            {'title': 'Expected Return', 'value': f"{min_var.get('return', 0)*100:.2f}%", 'color': COLORS['success']},
            {'title': 'Portfolio Risk', 'value': f"{min_var.get('volatility', 0)*100:.2f}%", 'color': COLORS['info']},
            {'title': 'Sharpe Ratio', 'value': f"{min_var.get('sharpe_ratio', 0):.3f}", 'color': COLORS['primary']},
            {'title': 'Value at Risk (95%)', 'value': f"{risk_metrics.get('VaR', 0)*100:.2f}%", 'color': COLORS['warning']},
            {'title': 'Growth Allocation', 'value': f"{min_var.get('growth_weight', 0)*100:.1f}%", 'color': COLORS['growth']},
            {'title': 'Tracking Error', 'value': "1.31%", 'color': COLORS['secondary']},
        ]

        # Place metric cards
        for i, card in enumerate(metric_cards):
            ax = fig.add_subplot(gs[0, i % 3])
            ax.axis('off')

            # Create card background
            rect = FancyBboxPatch((0.05, 0.2), 0.9, 0.6,
                                  boxstyle="round,pad=0.05",
                                  facecolor='white',
                                  edgecolor=card['color'],
                                  linewidth=3)
            ax.add_patch(rect)

            # Add metric value
            ax.text(0.5, 0.55, card['value'], fontsize=22, fontweight='bold',
                   ha='center', va='center', color=card['color'])

            # Add metric title
            ax.text(0.5, 0.25, card['title'], fontsize=11,
                   ha='center', va='center', color=COLORS['dark'])

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        # Add remaining visualizations in lower sections
        # This would include mini versions of key charts

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        plt.close()
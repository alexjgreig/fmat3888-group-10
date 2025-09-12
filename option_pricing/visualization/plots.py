"""
Visualization module for option pricing results.
Includes volatility surfaces, convergence plots, and price comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple


class OptionPricingVisualizer:
    """Visualization tools for option pricing analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            # Use default style if seaborn not available
            plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_implied_vol_surface(self, vol_surface_data: Dict, 
                                 title: str = "Implied Volatility Surface",
                                 use_plotly: bool = True) -> None:
        """
        Plot 3D implied volatility surface.
        
        Args:
            vol_surface_data: Dictionary with 'strikes', 'maturities', 'values'
            title: Plot title
            use_plotly: Use plotly for interactive plot
        """
        strikes = vol_surface_data['strikes']
        maturities = vol_surface_data['maturities']
        vols = vol_surface_data['values']
        
        if use_plotly:
            # Create interactive plotly surface
            fig = go.Figure(data=[go.Surface(
                x=maturities,
                y=strikes,
                z=vols,
                colorscale='Viridis',
                name='Implied Vol'
            )])
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='Time to Maturity (years)',
                    yaxis_title='Strike Price',
                    zaxis_title='Implied Volatility',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                width=900,
                height=700
            )
            
            fig.show()
        else:
            # Create matplotlib 3D surface
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            T_grid, K_grid = np.meshgrid(maturities, strikes)
            
            surf = ax.plot_surface(T_grid, K_grid, vols, cmap='viridis',
                                  linewidth=0, antialiased=True, alpha=0.9)
            
            ax.set_xlabel('Time to Maturity (years)')
            ax.set_ylabel('Strike Price')
            ax.set_zlabel('Implied Volatility')
            ax.set_title(title)
            
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.show()
    
    def plot_local_vol_surface(self, local_vol_data: Dict,
                              title: str = "Local Volatility Surface") -> None:
        """
        Plot local volatility surface.
        
        Args:
            local_vol_data: Dictionary with local vol grid data
            title: Plot title
        """
        self.plot_implied_vol_surface(local_vol_data, title)
    
    def plot_convergence(self, convergence_data: Dict,
                        target_error: float = 1e-6,
                        log_scale: bool = True) -> None:
        """
        Plot Monte Carlo convergence analysis with absolute errors and confidence intervals.
        
        Args:
            convergence_data: Dictionary with convergence results
            target_error: Target error tolerance line
            log_scale: Use log scale for axes
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        
        # Path convergence - Prices with BS comparison
        ax = axes[0, 0]
        path_counts = convergence_data['path_convergence']['path_counts']
        prices = convergence_data['path_convergence']['prices']
        bs_price = convergence_data.get('bs_price', prices[-1])
        
        ax.plot(path_counts, prices, 'o-', label='MC Price', linewidth=2)
        ax.axhline(y=bs_price, color='r', linestyle='--', alpha=0.5, 
                  label=f'BS Price: {bs_price:.4f}')
        
        if log_scale:
            ax.set_xscale('log')
        ax.set_xlabel('Number of Paths')
        ax.set_ylabel('Option Price')
        ax.set_title('Price Convergence vs Number of Paths')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Absolute Error (MC - BS)
        ax = axes[0, 1]
        absolute_errors = convergence_data['path_convergence'].get('absolute_errors', 
                                                                   [abs(p - bs_price) for p in prices])
        
        ax.loglog(path_counts, absolute_errors, 'o-', label='|MC - BS|', linewidth=2, color='red')
        
        ax.set_xlabel('Number of Paths')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Absolute Error: |MC Price - BS Price|')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Standard Errors
        ax = axes[1, 0]
        std_errors = convergence_data['path_convergence']['std_errors']
        
        ax.loglog(path_counts, std_errors, 'o-', label='Standard Error', linewidth=2)
        
        # Theoretical 1/sqrt(n) convergence
        theoretical = std_errors[0] * np.sqrt(path_counts[0] / np.array(path_counts))
        ax.loglog(path_counts, theoretical, 'r--', label='Theoretical O(1/√n)', alpha=0.7)
        
        # Target error line
        ax.axhline(y=target_error, color='g', linestyle=':', linewidth=2,
                  label=f'Target Error: {target_error}')
        
        ax.set_xlabel('Number of Paths')
        ax.set_ylabel('Standard Error')
        ax.set_title('Standard Error Convergence (Log-Log Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Confidence Interval Precision
        ax = axes[1, 1]
        if 'confidence_intervals' in convergence_data['path_convergence']:
            ci_data = convergence_data['path_convergence']['confidence_intervals']
            ci_precisions = [ci['precision'] for ci in ci_data]
            
            ax.loglog(path_counts, ci_precisions, 'o-', label='95% CI Width', linewidth=2, color='purple')
            ax.set_xlabel('Number of Paths')
            ax.set_ylabel('Confidence Interval Width')
            ax.set_title('95% Confidence Interval Precision')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Step convergence - Prices
        if 'step_convergence' in convergence_data and convergence_data['step_convergence']:
            ax = axes[2, 0]
            step_counts = convergence_data['step_convergence']['step_counts']
            step_prices = convergence_data['step_convergence']['prices']
            
            ax.plot(step_counts, step_prices, 's-', color='green', linewidth=2, label='MC Price')
            ax.axhline(y=bs_price, color='r', linestyle='--', alpha=0.5, 
                      label=f'BS Price: {bs_price:.4f}')
            ax.set_xlabel('Number of Time Steps')
            ax.set_ylabel('Option Price')
            ax.set_title('Price Convergence vs Time Steps (10k paths)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Step convergence - Absolute Errors
            ax = axes[2, 1]
            if 'absolute_errors' in convergence_data['step_convergence']:
                step_abs_errors = convergence_data['step_convergence']['absolute_errors']
                ax.semilogy(step_counts, step_abs_errors, 's-', color='red', linewidth=2)
                ax.set_xlabel('Number of Time Steps')
                ax.set_ylabel('Absolute Error')
                ax.set_title('Step Discretization Error: |MC - BS|')
                ax.grid(True, alpha=0.3)
        
        # Convergence Summary
        ax = axes[2, 1] if 'step_convergence' not in convergence_data else plt.subplot2grid((3, 2), (2, 1))
        
        # Calculate convergence rate
        log_n = np.log(path_counts)
        log_err = np.log(std_errors)
        
        # Fit linear regression to get convergence rate
        coeffs = np.polyfit(log_n, log_err, 1)
        convergence_rate = coeffs[0]
        
        summary_text = f'Path Convergence Summary:\n'
        summary_text += f'━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        summary_text += f'Convergence Rate: {convergence_rate:.3f}\n'
        summary_text += f'Theoretical Rate: -0.5\n'
        summary_text += f'BS Price: ${bs_price:.4f}\n'
        
        if 'sigma_atm' in convergence_data:
            summary_text += f'ATM Volatility: {convergence_data["sigma_atm"]:.2%}\n'
        
        summary_text += f'\nError Analysis:\n'
        summary_text += f'━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        
        # Show errors for each path count
        for i, n_paths in enumerate(path_counts[:min(5, len(path_counts))]):
            summary_text += f'{n_paths:,} paths:\n'
            summary_text += f'  Abs Error: {absolute_errors[i]:.5f}\n'
            summary_text += f'  Std Error: {std_errors[i]:.5f}\n'
            if 'confidence_intervals' in convergence_data['path_convergence']:
                ci = convergence_data['path_convergence']['confidence_intervals'][i]
                summary_text += f'  95% CI: [{ci["lower"]:.4f}, {ci["upper"]:.4f}]\n'
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace')
        ax.set_title('Convergence Summary')
        ax.axis('off')
        
        plt.suptitle('Monte Carlo Convergence Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_convergence_2d(self, convergence_data: Dict) -> None:
        """
        Plot 2D convergence analysis with both paths and steps varying.
        
        Args:
            convergence_data: Dictionary with 2D convergence results
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        path_counts = convergence_data['path_counts']
        step_counts = convergence_data['step_counts']
        error_grid = convergence_data['error_grid']
        bs_price = convergence_data['bs_price']
        
        fig = plt.figure(figsize=(16, 10))
        
        # 3D surface plot of absolute errors
        ax1 = fig.add_subplot(221, projection='3d')
        P, S = np.meshgrid(path_counts, step_counts)
        surf = ax1.plot_surface(np.log10(P.T), np.log10(S.T), error_grid, 
                                cmap='viridis', alpha=0.8)
        ax1.set_xlabel('log10(Paths)')
        ax1.set_ylabel('log10(Steps)')
        ax1.set_zlabel('Absolute Error')
        ax1.set_title('Convergence Surface: |MC - BS|')
        fig.colorbar(surf, ax=ax1, shrink=0.5)
        
        # Heatmap of absolute errors
        ax2 = fig.add_subplot(222)
        im = ax2.imshow(error_grid, cmap='RdYlGn_r', aspect='auto', 
                       interpolation='nearest')
        ax2.set_xticks(range(len(step_counts)))
        ax2.set_xticklabels([str(s) for s in step_counts])
        ax2.set_yticks(range(len(path_counts)))
        ax2.set_yticklabels([str(p) for p in path_counts])
        ax2.set_xlabel('Number of Steps')
        ax2.set_ylabel('Number of Paths')
        ax2.set_title('Absolute Error Heatmap')
        
        # Add text annotations
        for i in range(len(path_counts)):
            for j in range(len(step_counts)):
                text = ax2.text(j, i, f'{error_grid[i, j]:.4f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        fig.colorbar(im, ax=ax2)
        
        # Line plots for fixed steps
        ax3 = fig.add_subplot(223)
        for j, n_steps in enumerate(step_counts):
            ax3.loglog(path_counts, error_grid[:, j], 'o-', 
                      label=f'{n_steps} steps', alpha=0.7)
        ax3.set_xlabel('Number of Paths')
        ax3.set_ylabel('Absolute Error')
        ax3.set_title('Convergence vs Paths (fixed steps)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Line plots for fixed paths
        ax4 = fig.add_subplot(224)
        for i, n_paths in enumerate(path_counts):
            ax4.loglog(step_counts, error_grid[i, :], 's-', 
                      label=f'{n_paths} paths', alpha=0.7)
        ax4.set_xlabel('Number of Steps')
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Convergence vs Steps (fixed paths)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'2D Monte Carlo Convergence Analysis\nBS Price: ${bs_price:.4f}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_price_comparison(self, comparison_data: Dict) -> None:
        """
        Plot price comparison between different models.
        
        Args:
            comparison_data: Dictionary with model comparison results
        """
        strikes = comparison_data['strikes']
        maturities = comparison_data['maturities']
        
        # Create subplots for each maturity
        n_maturities = len(maturities)
        fig, axes = plt.subplots(1, n_maturities, figsize=(5*n_maturities, 5))
        
        if n_maturities == 1:
            axes = [axes]
        
        for i, T in enumerate(maturities):
            ax = axes[i]
            
            # Extract prices for this maturity
            lsv_prices = comparison_data['lsv_prices'][i]
            lv_prices = comparison_data['lv_prices'][i]
            bs_prices = comparison_data['bs_prices'][i]
            
            # Plot prices
            ax.plot(strikes, lsv_prices, 'o-', label='LSV Model', linewidth=2)
            ax.plot(strikes, lv_prices, 's-', label='Local Vol', linewidth=2)
            ax.plot(strikes, bs_prices, '^-', label='Black-Scholes', linewidth=2)
            
            ax.set_xlabel('Strike Price')
            ax.set_ylabel('Option Price')
            ax.set_title(f'T = {T} years')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Model Price Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Plot differences
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, T in enumerate(maturities):
            lsv_prices = np.array(comparison_data['lsv_prices'][i])
            lv_prices = np.array(comparison_data['lv_prices'][i])
            
            diff = lsv_prices - lv_prices
            ax.plot(strikes, diff, 'o-', label=f'T = {T}y', linewidth=2)
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Price Difference (LSV - Local Vol)')
        ax.set_title('LSV vs Local Volatility Model Differences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
    
    def plot_volatility_smile(self, option_data: pd.DataFrame, 
                            maturity: Optional[float] = None) -> None:
        """
        Plot implied volatility smile.
        
        Args:
            option_data: DataFrame with option data
            maturity: Specific maturity to plot (or closest)
        """
        # Filter for specific maturity if provided
        if maturity is not None:
            # Find closest maturity
            unique_maturities = option_data['timeToExpiry'].unique()
            closest_maturity = min(unique_maturities, 
                                  key=lambda x: abs(x - maturity))
            data = option_data[option_data['timeToExpiry'] == closest_maturity]
            title_suffix = f" (T = {closest_maturity:.2f} years)"
        else:
            data = option_data
            title_suffix = ""
        
        # Group by maturity
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calls vs Puts
        ax = axes[0]
        for option_type in ['CALL', 'PUT']:
            type_data = data[data['optionType'] == option_type]
            if not type_data.empty:
                ax.scatter(type_data['moneyness'], type_data['impliedVolatility'],
                          alpha=0.6, label=option_type, s=50)
        
        ax.set_xlabel('Moneyness (K/S)')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'Volatility Smile by Option Type{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # By maturity
        ax = axes[1]
        unique_maturities = data['timeToExpiry'].unique()[:5]  # Limit to 5
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_maturities)))
        
        for i, T in enumerate(unique_maturities):
            maturity_data = data[data['timeToExpiry'] == T]
            ax.scatter(maturity_data['moneyness'], maturity_data['impliedVolatility'],
                      alpha=0.6, label=f'T = {T:.2f}y', s=50, color=colors[i])
        
        ax.set_xlabel('Moneyness (K/S)')
        ax.set_ylabel('Implied Volatility')
        ax.set_title('Volatility Smile Term Structure')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_barrier_analysis(self, barrier_results: List[Dict],
                            barrier_levels: List[float]) -> None:
        """
        Plot barrier option analysis.
        
        Args:
            barrier_results: List of barrier pricing results
            barrier_levels: List of barrier levels tested
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        prices = [r['price'] for r in barrier_results]
        knock_probs = [r['knock_probability'] for r in barrier_results]
        std_errors = [r['std_error'] for r in barrier_results]
        
        # Barrier price vs level
        ax = axes[0, 0]
        ax.plot(barrier_levels, prices, 'o-', linewidth=2)
        ax.set_xlabel('Barrier Level')
        ax.set_ylabel('Option Price')
        ax.set_title('Barrier Option Price vs Barrier Level')
        ax.grid(True, alpha=0.3)
        
        # Knock probability
        ax = axes[0, 1]
        ax.plot(barrier_levels, knock_probs, 's-', color='red', linewidth=2)
        ax.set_xlabel('Barrier Level')
        ax.set_ylabel('Knock Probability')
        ax.set_title('Barrier Hit Probability')
        ax.grid(True, alpha=0.3)
        
        # Standard errors
        ax = axes[1, 0]
        ax.plot(barrier_levels, std_errors, '^-', color='green', linewidth=2)
        ax.set_xlabel('Barrier Level')
        ax.set_ylabel('Standard Error')
        ax.set_title('Pricing Standard Error')
        ax.grid(True, alpha=0.3)
        
        # Price sensitivity
        ax = axes[1, 1]
        if len(prices) > 1:
            sensitivities = np.diff(prices) / np.diff(barrier_levels)
            barrier_mid = (np.array(barrier_levels[:-1]) + np.array(barrier_levels[1:])) / 2
            ax.plot(barrier_mid, sensitivities, 'd-', color='purple', linewidth=2)
            ax.set_xlabel('Barrier Level')
            ax.set_ylabel('Price Sensitivity (∂Price/∂Barrier)')
            ax.set_title('Barrier Price Sensitivity')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Barrier Option Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_convergence_report(self, convergence_data: Dict,
                                 output_file: str = 'convergence_report.html') -> None:
        """
        Create an HTML report of convergence analysis with enhanced error metrics.
        
        Args:
            convergence_data: Convergence analysis results
            output_file: Output HTML file path
        """
        import plotly.subplots as sp
        
        # Create subplots - adjusted for more plots
        fig = sp.make_subplots(
            rows=3, cols=2,
            subplot_titles=('Price Convergence', 'Absolute Error (|MC - BS|)',
                          'Standard Error', 'Confidence Interval Width',
                          'Step Convergence', 'Summary Statistics'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}],
                  [{'type': 'scatter'}, {'type': 'table'}]]
        )
        
        # Path convergence - prices with BS comparison
        path_counts = convergence_data['path_convergence']['path_counts']
        prices = convergence_data['path_convergence']['prices']
        bs_price = convergence_data.get('bs_price', prices[-1])
        
        fig.add_trace(
            go.Scatter(x=path_counts, y=prices, mode='lines+markers',
                      name='MC Price', marker=dict(size=8)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=path_counts, y=[bs_price]*len(path_counts), 
                      mode='lines', name='BS Price',
                      line=dict(dash='dash', color='red')),
            row=1, col=1
        )
        
        # Absolute Error convergence
        if 'absolute_errors' in convergence_data['path_convergence']:
            absolute_errors = convergence_data['path_convergence']['absolute_errors']
        else:
            absolute_errors = [abs(p - bs_price) for p in prices]
            
        fig.add_trace(
            go.Scatter(x=path_counts, y=absolute_errors, mode='lines+markers',
                      name='|MC - BS|', marker=dict(size=8, color='red')),
            row=1, col=2
        )
        
        # Standard Error convergence
        std_errors = convergence_data['path_convergence']['std_errors']
        
        fig.add_trace(
            go.Scatter(x=path_counts, y=std_errors, mode='lines+markers',
                      name='Std Error', marker=dict(size=8)),
            row=2, col=1
        )
        
        # Confidence Interval Width
        if 'confidence_intervals' in convergence_data['path_convergence']:
            ci_widths = [ci['precision'] for ci in convergence_data['path_convergence']['confidence_intervals']]
            fig.add_trace(
                go.Scatter(x=path_counts, y=ci_widths, mode='lines+markers',
                          name='95% CI Width', marker=dict(size=8, color='purple')),
                row=2, col=2
            )
        
        # Step convergence
        if 'step_convergence' in convergence_data and convergence_data['step_convergence']:
            step_counts = convergence_data['step_convergence']['step_counts']
            step_prices = convergence_data['step_convergence']['prices']
            
            fig.add_trace(
                go.Scatter(x=step_counts, y=step_prices, mode='lines+markers',
                          name='Price vs Steps', marker=dict(size=8, color='green')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=step_counts, y=[bs_price]*len(step_counts), 
                          mode='lines', name='BS Price',
                          line=dict(dash='dash', color='red')),
                row=3, col=1
            )
        
        # Enhanced Summary table with error analysis
        summary_data = [
            ['BS Price', f'${bs_price:.4f}'],
            ['ATM Volatility', f'{convergence_data.get("sigma_atm", 0):.2%}'],
            ['Final MC Price', f'${prices[-1]:.4f}'],
            ['Final Abs Error', f'{absolute_errors[-1]:.5f}'],
            ['Final Std Error', f'{std_errors[-1]:.5f}'],
            ['Convergence Rate', f'{np.polyfit(np.log(path_counts), np.log(std_errors), 1)[0]:.3f}'],
            ['Target Error Achieved', 'Yes' if convergence_data.get('achieved_error') else 'No'],
            ['Required Paths', f'{convergence_data.get("required_paths", "N/A")}']
        ]
        
        # Add confidence interval information
        if 'confidence_intervals' in convergence_data['path_convergence']:
            final_ci = convergence_data['path_convergence']['confidence_intervals'][-1]
            summary_data.append(['Final 95% CI', f'[{final_ci["lower"]:.4f}, {final_ci["upper"]:.4f}]'])
            summary_data.append(['CI Precision', f'{final_ci["precision"]:.5f}'])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                          fill_color='lightgray',
                          align='left'),
                cells=dict(values=list(zip(*summary_data)),
                         align='left')
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Number of Paths", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Number of Paths", type="log", row=1, col=2)
        fig.update_xaxes(title_text="Number of Paths", type="log", row=2, col=1)
        fig.update_xaxes(title_text="Number of Paths", type="log", row=2, col=2)
        fig.update_xaxes(title_text="Number of Steps", row=3, col=1)
        
        fig.update_yaxes(title_text="Option Price", row=1, col=1)
        fig.update_yaxes(title_text="Absolute Error", type="log", row=1, col=2)
        fig.update_yaxes(title_text="Standard Error", type="log", row=2, col=1)
        fig.update_yaxes(title_text="CI Width", type="log", row=2, col=2)
        fig.update_yaxes(title_text="Option Price", row=3, col=1)
        
        fig.update_layout(
            title_text="Monte Carlo Convergence Analysis Report",
            showlegend=True,
            height=1200,
            width=1400
        )
        
        # Save to HTML
        fig.write_html(output_file)
        print(f"Convergence report saved to {output_file}")
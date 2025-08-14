"""
Higuchi Fractal Dimension Visualization Module

This module provides comprehensive visualization tools for Higuchi fractal dimension analysis:
- L(k) vs k plots
- Log-log plots for fractal dimension estimation
- Residual analysis plots
- Quality assessment plots
- Comparison plots for multiple time series
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import pandas as pd

from src.analysis.higuchi_analysis import HiguchiSummary


def plot_higuchi_analysis(k_values: np.ndarray,
                         l_values: np.ndarray,
                         summary: HiguchiSummary,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create a comprehensive plot of Higuchi fractal dimension analysis.
    
    Parameters:
    -----------
    k_values : np.ndarray
        Array of k values used in the analysis
    l_values : np.ndarray
        Array of corresponding L(k) values
    summary : HiguchiSummary
        Higuchi analysis results
    save_path : Optional[str]
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        The created figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Higuchi Fractal Dimension Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: L(k) vs k (original scale)
    ax1.plot(k_values, l_values, 'bo-', linewidth=2, markersize=6, label='L(k) values')
    ax1.set_xlabel('k (time interval)')
    ax1.set_ylabel('L(k)')
    ax1.set_title('L(k) vs k')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-log plot for fractal dimension estimation
    log_k = np.log(k_values)
    log_l = np.log(l_values)
    
    ax2.scatter(log_k, log_l, color='red', s=50, alpha=0.7, label='Data points')
    
    # Plot regression line
    predicted = summary.slope * log_k + summary.intercept
    ax2.plot(log_k, predicted, 'b--', linewidth=2, 
             label=f'Regression (D = {summary.fractal_dimension:.3f})')
    
    ax2.set_xlabel('log(k)')
    ax2.set_ylabel('log(L(k))')
    ax2.set_title(f'Log-Log Plot (R² = {summary.r_squared:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add equation text
    equation_text = f'y = {summary.slope:.3f}x + {summary.intercept:.3f}'
    ax2.text(0.05, 0.95, equation_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # Plot 3: Residuals analysis
    residuals = summary.residuals
    ax3.scatter(log_k, residuals, color='green', alpha=0.7, s=40)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('log(k)')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals vs log(k)')
    ax3.grid(True, alpha=0.3)
    
    # Add residual statistics
    residual_std = np.std(residuals)
    ax3.text(0.05, 0.95, f'Std: {residual_std:.4f}', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # Plot 4: Summary statistics
    summary_text = f"""
    Fractal Dimension: {summary.fractal_dimension:.4f}
    Standard Error: {summary.std_error:.4f}
    R²: {summary.r_squared:.4f}
    P-value: {summary.p_value:.4f}
    
    k range: [{summary.k_range[0]}, {summary.k_range[1]}]
    n_k values: {len(k_values)}
    n_points: {summary.n_points}
    
    Method: {summary.additional_info.get('optimization_method', 'linear')}
    k optimized: {summary.additional_info.get('k_optimized', False)}
    """
    
    ax4.text(0.05, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_higuchi_comparison(summaries: Dict[str, HiguchiSummary],
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Create comparison plots for multiple Higuchi analyses.
    
    Parameters:
    -----------
    summaries : Dict[str, HiguchiSummary]
        Dictionary mapping names to HiguchiSummary objects
    save_path : Optional[str]
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        The created figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Higuchi Fractal Dimension Comparison', fontsize=18, fontweight='bold')
    
    names = list(summaries.keys())
    fractal_dims = [summaries[name].fractal_dimension for name in names]
    r_squared_values = [summaries[name].r_squared for name in names]
    std_errors = [summaries[name].std_error for name in names]
    
    # Plot 1: Fractal dimensions comparison
    bars1 = axes[0, 0].bar(names, fractal_dims, color='skyblue', alpha=0.7)
    axes[0, 0].set_ylabel('Fractal Dimension')
    axes[0, 0].set_title('Fractal Dimensions Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, fractal_dims):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: R² values comparison
    bars2 = axes[0, 1].bar(names, r_squared_values, color='lightgreen', alpha=0.7)
    axes[0, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Threshold (0.8)')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_title('R² Values Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.1)
    
    # Plot 3: Standard errors comparison
    bars3 = axes[0, 2].bar(names, std_errors, color='lightcoral', alpha=0.7)
    axes[0, 2].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold (0.1)')
    axes[0, 2].set_ylabel('Standard Error')
    axes[0, 2].set_title('Standard Errors Comparison')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Log-log plots overlay
    for i, name in enumerate(names):
        summary = summaries[name]
        log_k = summary.log_k
        log_l = summary.log_l
        
        # Plot data points
        axes[1, 0].scatter(log_k, log_l, alpha=0.6, s=30, label=f'{name} (D={summary.fractal_dimension:.3f})')
        
        # Plot regression line
        predicted = summary.slope * log_k + summary.intercept
        axes[1, 0].plot(log_k, predicted, '--', alpha=0.8, linewidth=1.5)
    
    axes[1, 0].set_xlabel('log(k)')
    axes[1, 0].set_ylabel('log(L(k))')
    axes[1, 0].set_title('Log-Log Plots Overlay')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Quality score heatmap
    quality_data = []
    for name in names:
        summary = summaries[name]
        quality_data.append([
            summary.fractal_dimension,
            summary.r_squared,
            1.0 / (1.0 + summary.std_error),  # Inverse of std error (higher is better)
            len(summary.k_values) / 20.0  # Normalized number of k values
        ])
    
    quality_df = pd.DataFrame(quality_data, 
                             index=names,
                             columns=['Fractal Dim', 'R²', 'Quality', 'k Coverage'])
    
    sns.heatmap(quality_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=axes[1, 1], cbar_kws={'label': 'Value'})
    axes[1, 1].set_title('Quality Metrics Heatmap')
    
    # Plot 6: Summary statistics table
    summary_text = f"""
    Number of Series: {len(names)}
    
    Average Fractal Dimension: {np.mean(fractal_dims):.3f}
    Std Fractal Dimension: {np.std(fractal_dims):.3f}
    
    Average R²: {np.mean(r_squared_values):.3f}
    Average Std Error: {np.mean(std_errors):.3f}
    
    Best Fit: {names[np.argmax(r_squared_values)]}
    Highest Dimension: {names[np.argmax(fractal_dims)]}
    """
    
    axes[1, 2].text(0.05, 0.5, summary_text, transform=axes[1, 2].transAxes, fontsize=10,
                     verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_higuchi_quality_assessment(summary: HiguchiSummary,
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create quality assessment plots for Higuchi analysis.
    
    Parameters:
    -----------
    summary : HiguchiSummary
        Higuchi analysis results
    save_path : Optional[str]
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        The created figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Higuchi Analysis Quality Assessment', fontsize=16, fontweight='bold')
    
    # Plot 1: R² and standard error
    metrics = ['R²', 'Std Error']
    values = [summary.r_squared, summary.std_error]
    colors = ['green' if summary.r_squared > 0.8 else 'orange',
              'green' if summary.std_error < 0.1 else 'red']
    
    bars1 = ax1.bar(metrics, values, color=colors, alpha=0.7)
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='R² threshold')
    ax1.set_ylabel('Value')
    ax1.set_title('Quality Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 2: k values distribution
    k_values = summary.k_values
    ax2.hist(k_values, bins=min(10, len(k_values)), alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(k_values), color='red', linestyle='--', label=f'Mean: {np.mean(k_values):.1f}')
    ax2.set_xlabel('k values')
    ax2.set_ylabel('Frequency')
    ax2.set_title('k Values Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residuals normality check
    residuals = summary.residuals
    ax3.hist(residuals, bins=min(15, len(residuals)), alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(np.mean(residuals), color='red', linestyle='--', label=f'Mean: {np.mean(residuals):.4f}')
    ax3.axvline(np.std(residuals), color='orange', linestyle='--', label=f'Std: {np.std(residuals):.4f}')
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Residuals Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Quality score radar chart (simplified as bar chart)
    quality_metrics = ['R²', 'Std Error', 'k Coverage', 'Residuals']
    quality_scores = [
        min(1.0, summary.r_squared / 0.8),  # R² score (0.8 is target)
        max(0.0, 1.0 - summary.std_error / 0.1),  # Std error score (0.1 is target)
        min(1.0, len(summary.k_values) / 10.0),  # k coverage score (10 is target)
        max(0.0, 1.0 - np.std(residuals) / 0.05)  # Residuals score (0.05 is target)
    ]
    
    bars4 = ax4.bar(quality_metrics, quality_scores, color='lightcoral', alpha=0.7)
    ax4.set_ylabel('Quality Score')
    ax4.set_title('Overall Quality Assessment')
    ax4.set_ylim(0, 1.1)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars4, quality_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_higuchi_report(summaries: Dict[str, HiguchiSummary],
                         save_dir: str = "results/higuchi") -> str:
    """
    Create a comprehensive Higuchi analysis report.
    
    Parameters:
    -----------
    summaries : Dict[str, HiguchiSummary]
        Dictionary mapping names to HiguchiSummary objects
    save_dir : str
        Directory to save the report
        
    Returns:
    --------
    str
        Path to the saved report
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create individual plots for each series
    for name, summary in summaries.items():
        # Get k_values and l_values from summary
        k_values = summary.k_values
        l_values = summary.l_values
        
        # Create individual analysis plot
        plot_higuchi_analysis(
            k_values, l_values, summary,
            save_path=str(save_path / f"{name}_analysis.png")
        )
        
        # Create quality assessment plot
        plot_higuchi_quality_assessment(
            summary,
            save_path=str(save_path / f"{name}_quality.png")
        )
    
    # Create comparison plot
    if len(summaries) > 1:
        plot_higuchi_comparison(
            summaries,
            save_path=str(save_path / "comparison.png")
        )
    
    return str(save_path)


# Convenience functions
def plot_higuchi_result(summary: HiguchiSummary,
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Convenience function to plot Higuchi analysis results.
    
    Parameters:
    -----------
    summary : HiguchiSummary
        Higuchi analysis results
    save_path : Optional[str]
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        The created figure
    """
    return plot_higuchi_analysis(
        summary.k_values, summary.l_values, summary, save_path, figsize
    )

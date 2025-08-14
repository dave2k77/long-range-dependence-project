"""
Validation Visualization Module for Long-Range Dependence Analysis

This module provides comprehensive visualization tools for statistical validation results:
- Hypothesis test visualizations
- Bootstrap distribution plots
- Monte Carlo significance test plots
- Cross-validation result plots
- Robustness test visualizations
- Comprehensive validation summary plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from src.analysis.statistical_validation import (
    HypothesisTestResult,
    CrossValidationResult,
    BootstrapResult,
    MonteCarloResult
)


def plot_hypothesis_test_result(result: HypothesisTestResult,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot hypothesis test results.
    
    Parameters:
    -----------
    result : HypothesisTestResult
        Hypothesis test result
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
    fig.suptitle(f'Hypothesis Test Results: {result.test_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Test statistic and critical value
    ax1.axvline(result.test_statistic, color='red', linestyle='--', linewidth=2, 
                label=f'Test Statistic: {result.test_statistic:.3f}')
    ax1.axvline(result.critical_value, color='blue', linestyle='--', linewidth=2,
                label=f'Critical Value: {result.critical_value:.3f}')
    ax1.axvline(-result.critical_value, color='blue', linestyle='--', linewidth=2,
                label=f'Critical Value: {-result.critical_value:.3f}')
    
    # Shade rejection region
    x = np.linspace(-4, 4, 1000)
    y = stats.t.pdf(x, df=result.additional_info['sample_size'] - 1)
    ax1.plot(x, y, 'k-', linewidth=1.5, label='t-distribution')
    
    if result.decision == 'reject':
        if result.test_statistic > 0:
            ax1.fill_between(x, y, where=(x > result.critical_value), alpha=0.3, color='red')
        else:
            ax1.fill_between(x, y, where=(x < -result.critical_value), alpha=0.3, color='red')
    
    ax1.set_xlabel('Test Statistic')
    ax1.set_ylabel('Density')
    ax1.set_title('Test Statistic vs Critical Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence interval
    ci_lower, ci_upper = result.confidence_interval
    hurst_est = result.additional_info['hurst_estimate']
    h0_hurst = 0.5  # Default null hypothesis
    
    ax2.axhline(h0_hurst, color='red', linestyle='--', linewidth=2, 
                label=f'Null Hypothesis: H = {h0_hurst}')
    ax2.axhline(hurst_est, color='blue', linewidth=2, 
                label=f'Estimate: H = {hurst_est:.3f}')
    ax2.axhspan(ci_lower, ci_upper, alpha=0.3, color='blue', 
                label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    
    ax2.set_ylabel('Hurst Exponent')
    ax2.set_title('Confidence Interval')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Effect size
    effect_size = result.effect_size
    ax3.bar(['Effect Size'], [effect_size], color='green' if abs(effect_size) > 0.5 else 'orange')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_ylabel('Cohen\'s d')
    ax3.set_title(f'Effect Size: {effect_size:.3f}')
    ax3.grid(True, alpha=0.3)
    
    # Add effect size interpretation
    if abs(effect_size) < 0.2:
        interpretation = 'Small'
    elif abs(effect_size) < 0.5:
        interpretation = 'Medium'
    else:
        interpretation = 'Large'
    ax3.text(0, effect_size + 0.05, interpretation, ha='center', fontweight='bold')
    
    # Plot 4: Summary statistics
    summary_text = f"""
    Decision: {result.decision.upper()}
    p-value: {result.p_value:.4f}
    Significance Level: {result.significance_level}
    Sample Size: {result.additional_info['sample_size']}
    Method: {result.additional_info['method'].upper()}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_bootstrap_result(result: BootstrapResult,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot bootstrap analysis results.
    
    Parameters:
    -----------
    result : BootstrapResult
        Bootstrap analysis result
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
    fig.suptitle('Bootstrap Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Bootstrap distribution
    ax1.hist(result.bootstrap_estimates, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(result.original_estimate, color='red', linestyle='--', linewidth=2,
                label=f'Original: {result.original_estimate:.3f}')
    ax1.axvline(result.mean_estimate, color='blue', linestyle='--', linewidth=2,
                label=f'Bootstrap Mean: {result.mean_estimate:.3f}')
    ax1.axvline(result.confidence_interval[0], color='green', linestyle=':', linewidth=2,
                label=f'CI Lower: {result.confidence_interval[0]:.3f}')
    ax1.axvline(result.confidence_interval[1], color='green', linestyle=':', linewidth=2,
                label=f'CI Upper: {result.confidence_interval[1]:.3f}')
    
    ax1.set_xlabel('Hurst Exponent')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Bootstrap Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence interval
    ci_lower, ci_upper = result.confidence_interval
    ax2.axhline(result.original_estimate, color='red', linewidth=2, 
                label=f'Original: {result.original_estimate:.3f}')
    ax2.axhspan(ci_lower, ci_upper, alpha=0.3, color='blue',
                label=f'{int(result.confidence_level*100)}% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    
    ax2.set_ylabel('Hurst Exponent')
    ax2.set_title('Confidence Interval')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Bias analysis
    bias = result.bias
    ax3.bar(['Bias'], [bias], color='red' if abs(bias) > 0.05 else 'green')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_ylabel('Bias')
    ax3.set_title(f'Bootstrap Bias: {bias:.4f}')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    summary_text = f"""
    Original Estimate: {result.original_estimate:.3f}
    Bootstrap Mean: {result.mean_estimate:.3f}
    Bootstrap Std: {result.std_estimate:.3f}
    Bias: {result.bias:.4f}
    Standard Error: {result.standard_error:.4f}
    Confidence Level: {int(result.confidence_level*100)}%
    Successful Bootstraps: {result.additional_stats['n_successful_bootstrap']}
    Method: {result.additional_stats['bootstrap_method']}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_monte_carlo_result(result: MonteCarloResult,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot Monte Carlo significance test results.
    
    Parameters:
    -----------
    result : MonteCarloResult
        Monte Carlo test result
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
    fig.suptitle('Monte Carlo Significance Test Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Null distribution with observed statistic
    ax1.hist(result.null_distribution, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.axvline(result.test_statistic, color='red', linestyle='--', linewidth=2,
                label=f'Observed: {result.test_statistic:.3f}')
    ax1.axvline(result.additional_stats['null_mean'], color='blue', linestyle='--', linewidth=2,
                label=f'Null Mean: {result.additional_stats["null_mean"]:.3f}')
    
    # Shade p-value region
    if result.test_statistic > result.additional_stats['null_median']:
        ax1.fill_between([result.test_statistic, np.max(result.null_distribution)], 
                        [0, 0], [np.max(ax1.get_ylim()), np.max(ax1.get_ylim())], 
                        alpha=0.3, color='red')
    else:
        ax1.fill_between([np.min(result.null_distribution), result.test_statistic], 
                        [0, 0], [np.max(ax1.get_ylim()), np.max(ax1.get_ylim())], 
                        alpha=0.3, color='red')
    
    ax1.set_xlabel('Hurst Exponent')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Null Distribution (p-value: {result.p_value:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Effect size
    effect_size = result.effect_size
    ax2.bar(['Effect Size'], [effect_size], color='green' if abs(effect_size) > 1 else 'orange')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('Standard Deviations')
    ax2.set_title(f'Effect Size: {effect_size:.3f}')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Decision visualization
    decision_color = 'red' if result.decision == 'reject' else 'green'
    ax3.bar(['Decision'], [1], color=decision_color)
    ax3.set_ylabel('Decision')
    ax3.set_title(f'Decision: {result.decision.upper()}')
    ax3.set_ylim(0, 1.2)
    ax3.text(0, 0.6, result.decision.upper(), ha='center', va='center', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    summary_text = f"""
    Observed Statistic: {result.test_statistic:.3f}
    Null Model: {result.additional_stats['null_model']}
    Null Mean: {result.additional_stats['null_mean']:.3f}
    Null Std: {result.additional_stats['null_std']:.3f}
    p-value: {result.p_value:.4f}
    Significance Level: {result.significance_level}
    Effect Size: {result.effect_size:.3f}
    Simulations: {result.n_simulations}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cross_validation_result(result: CrossValidationResult,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot cross-validation results.
    
    Parameters:
    -----------
    result : CrossValidationResult
        Cross-validation result
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
    fig.suptitle(f'Cross-Validation Results: {result.method}', fontsize=16, fontweight='bold')
    
    # Plot 1: Hurst estimates across folds
    fold_numbers = list(range(1, len(result.hurst_estimates) + 1))
    ax1.plot(fold_numbers, result.hurst_estimates, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(result.mean_hurst, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {result.mean_hurst:.3f}')
    ax1.fill_between(fold_numbers, 
                     [result.mean_hurst - result.std_hurst] * len(fold_numbers),
                     [result.mean_hurst + result.std_hurst] * len(fold_numbers),
                     alpha=0.3, color='red', label=f'±1 Std: {result.std_hurst:.3f}')
    
    ax1.set_xlabel('Fold Number')
    ax1.set_ylabel('Hurst Exponent')
    ax1.set_title('Hurst Estimates Across Folds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence intervals
    ci_lower = [ci[0] for ci in result.confidence_intervals]
    ci_upper = [ci[1] for ci in result.confidence_intervals]
    
    ax2.fill_between(fold_numbers, ci_lower, ci_upper, alpha=0.3, color='green',
                     label='95% Confidence Intervals')
    ax2.plot(fold_numbers, result.hurst_estimates, 'bo-', linewidth=2, markersize=8)
    ax2.axhline(result.mean_hurst, color='red', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Fold Number')
    ax2.set_ylabel('Hurst Exponent')
    ax2.set_title('Confidence Intervals Across Folds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stability metrics
    metrics = ['CV Score', 'Stability Score']
    values = [result.cv_score, result.stability_score]
    colors = ['red' if result.cv_score > 0.1 else 'green', 
              'green' if result.stability_score > 0.8 else 'orange']
    
    bars = ax3.bar(metrics, values, color=colors)
    ax3.set_ylabel('Score')
    ax3.set_title('Stability Metrics')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 4: Summary statistics
    summary_text = f"""
    Number of Folds: {result.n_folds}
    Mean Hurst: {result.mean_hurst:.3f}
    Std Hurst: {result.std_hurst:.3f}
    CV Score: {result.cv_score:.3f}
    Stability Score: {result.stability_score:.3f}
    Min Estimate: {result.additional_metrics['min_estimate']:.3f}
    Max Estimate: {result.additional_metrics['max_estimate']:.3f}
    Range: {result.additional_metrics['range']:.3f}
    Test Size: {result.additional_metrics['test_size']}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comprehensive_validation_summary(validation_results: Dict[str, Any],
                                         save_path: Optional[str] = None,
                                         figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Plot comprehensive validation summary across all methods.
    
    Parameters:
    -----------
    validation_results : Dict[str, Any]
        Comprehensive validation results
    save_path : Optional[str]
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        The created figure
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle('Comprehensive Statistical Validation Summary', fontsize=18, fontweight='bold')
    
    methods = list(validation_results['hypothesis_tests'].keys())
    
    # Plot 1: Hypothesis test p-values
    p_values = [validation_results['hypothesis_tests'][method].p_value for method in methods]
    decisions = [validation_results['hypothesis_tests'][method].decision for method in methods]
    colors = ['red' if decision == 'reject' else 'green' for decision in decisions]
    
    bars1 = axes[0, 0].bar(methods, p_values, color=colors)
    axes[0, 0].axhline(0.05, color='black', linestyle='--', alpha=0.7)
    axes[0, 0].set_ylabel('p-value')
    axes[0, 0].set_title('Hypothesis Test p-values')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add significance labels
    for bar, p_val in zip(bars1, p_values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{p_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Bootstrap confidence intervals
    ci_lower = []
    ci_upper = []
    original_estimates = []
    
    for method in methods:
        bootstrap_result = validation_results['bootstrap_analyses'][method]
        ci_lower.append(bootstrap_result.confidence_interval[0])
        ci_upper.append(bootstrap_result.confidence_interval[1])
        original_estimates.append(bootstrap_result.original_estimate)
    
    x_pos = np.arange(len(methods))
    axes[0, 1].errorbar(x_pos, original_estimates, 
                       yerr=[np.array(original_estimates) - np.array(ci_lower),
                             np.array(ci_upper) - np.array(original_estimates)],
                       fmt='o', capsize=5, capthick=2)
    axes[0, 1].set_ylabel('Hurst Exponent')
    axes[0, 1].set_title('Bootstrap Confidence Intervals')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(methods, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Monte Carlo p-values
    mc_p_values = [validation_results['monte_carlo_tests'][method].p_value for method in methods]
    mc_decisions = [validation_results['monte_carlo_tests'][method].decision for method in methods]
    mc_colors = ['red' if decision == 'reject' else 'green' for decision in mc_decisions]
    
    bars3 = axes[0, 2].bar(methods, mc_p_values, color=mc_colors)
    axes[0, 2].axhline(0.05, color='black', linestyle='--', alpha=0.7)
    axes[0, 2].set_ylabel('p-value')
    axes[0, 2].set_title('Monte Carlo Test p-values')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Cross-validation stability scores
    cv_scores = [validation_results['cross_validation'][method].stability_score for method in methods]
    bars4 = axes[1, 0].bar(methods, cv_scores, color='skyblue')
    axes[1, 0].set_ylabel('Stability Score')
    axes[1, 0].set_title('Cross-Validation Stability')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # Plot 5: Bootstrap bias
    bootstrap_bias = [validation_results['bootstrap_analyses'][method].bias for method in methods]
    colors_bias = ['red' if abs(bias) > 0.05 else 'green' for bias in bootstrap_bias]
    bars5 = axes[1, 1].bar(methods, bootstrap_bias, color=colors_bias)
    axes[1, 1].axhline(0, color='black', linewidth=0.5)
    axes[1, 1].set_ylabel('Bias')
    axes[1, 1].set_title('Bootstrap Bias')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Effect sizes
    effect_sizes = [validation_results['hypothesis_tests'][method].effect_size for method in methods]
    colors_effect = ['green' if abs(es) > 0.5 else 'orange' for es in effect_sizes]
    bars6 = axes[1, 2].bar(methods, effect_sizes, color=colors_effect)
    axes[1, 2].axhline(0, color='black', linewidth=0.5)
    axes[1, 2].set_ylabel('Effect Size')
    axes[1, 2].set_title('Effect Sizes')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Plot 7: Method comparison heatmap
    comparison_data = []
    for method in methods:
        row = [
            validation_results['hypothesis_tests'][method].p_value,
            validation_results['bootstrap_analyses'][method].bias,
            validation_results['cross_validation'][method].stability_score,
            validation_results['monte_carlo_tests'][method].p_value
        ]
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data, 
                                index=methods,
                                columns=['Hypothesis\np-value', 'Bootstrap\nBias', 
                                       'CV\nStability', 'Monte Carlo\np-value'])
    
    sns.heatmap(comparison_df, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                ax=axes[2, 0], cbar_kws={'label': 'Value'})
    axes[2, 0].set_title('Method Comparison Heatmap')
    
    # Plot 8: Decision summary
    decisions_summary = {}
    for test_type in ['hypothesis_tests', 'monte_carlo_tests']:
        for method in methods:
            decision = validation_results[test_type][method].decision
            if decision not in decisions_summary:
                decisions_summary[decision] = 0
            decisions_summary[decision] += 1
    
    if decisions_summary:
        axes[2, 1].pie(decisions_summary.values(), labels=decisions_summary.keys(), 
                      autopct='%1.1f%%', startangle=90)
        axes[2, 1].set_title('Decision Summary')
    
    # Plot 9: Overall assessment
    assessment_text = f"""
    Methods Tested: {len(methods)}
    
    Hypothesis Tests:
    - Rejected: {sum(1 for d in decisions if d == 'reject')}
    - Failed to Reject: {sum(1 for d in decisions if d == 'fail_to_reject')}
    
    Monte Carlo Tests:
    - Rejected: {sum(1 for d in mc_decisions if d == 'reject')}
    - Failed to Reject: {sum(1 for d in mc_decisions if d == 'fail_to_reject')}
    
    Average Stability: {np.mean(cv_scores):.3f}
    Average Bias: {np.mean(bootstrap_bias):.4f}
    """
    
    axes[2, 2].text(0.1, 0.5, assessment_text, transform=axes[2, 2].transAxes, fontsize=10,
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
    axes[2, 2].set_xlim(0, 1)
    axes[2, 2].set_ylim(0, 1)
    axes[2, 2].axis('off')
    axes[2, 2].set_title('Overall Assessment')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_validation_report(validation_results: Dict[str, Any],
                           save_dir: str = "results/validation") -> str:
    """
    Create a comprehensive validation report with all plots.
    
    Parameters:
    -----------
    validation_results : Dict[str, Any]
        Comprehensive validation results
    save_dir : str
        Directory to save the report
        
    Returns:
    --------
    str
        Path to the saved report
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    methods = list(validation_results['hypothesis_tests'].keys())
    
    # Create individual plots for each method
    for method in methods:
        method_dir = save_path / method
        method_dir.mkdir(exist_ok=True)
        
        # Hypothesis test plot
        plot_hypothesis_test_result(
            validation_results['hypothesis_tests'][method],
            save_path=str(method_dir / f"{method}_hypothesis_test.png")
        )
        
        # Bootstrap plot
        plot_bootstrap_result(
            validation_results['bootstrap_analyses'][method],
            save_path=str(method_dir / f"{method}_bootstrap.png")
        )
        
        # Monte Carlo plot
        plot_monte_carlo_result(
            validation_results['monte_carlo_tests'][method],
            save_path=str(method_dir / f"{method}_monte_carlo.png")
        )
        
        # Cross-validation plot
        plot_cross_validation_result(
            validation_results['cross_validation'][method],
            save_path=str(method_dir / f"{method}_cross_validation.png")
        )
    
    # Create comprehensive summary plot
    plot_comprehensive_validation_summary(
        validation_results,
        save_path=str(save_path / "comprehensive_summary.png")
    )
    
    return str(save_path)


# Convenience functions
def plot_validation_result(result: Union[HypothesisTestResult, BootstrapResult, 
                                       MonteCarloResult, CrossValidationResult],
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Convenience function to plot any validation result.
    
    Parameters:
    -----------
    result : Union[HypothesisTestResult, BootstrapResult, MonteCarloResult, CrossValidationResult]
        Validation result to plot
    save_path : Optional[str]
        Path to save the plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        The created figure
    """
    if isinstance(result, HypothesisTestResult):
        return plot_hypothesis_test_result(result, save_path, figsize)
    elif isinstance(result, BootstrapResult):
        return plot_bootstrap_result(result, save_path, figsize)
    elif isinstance(result, MonteCarloResult):
        return plot_monte_carlo_result(result, save_path, figsize)
    elif isinstance(result, CrossValidationResult):
        return plot_cross_validation_result(result, save_path, figsize)
    else:
        raise ValueError(f"Unknown result type: {type(result)}")

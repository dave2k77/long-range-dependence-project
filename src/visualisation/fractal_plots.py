"""
Fractal Analysis Plotting Module for Long-Range Dependence Analysis

This module provides visualization tools for fractal analysis methods including
DFA (Detrended Fluctuation Analysis), R/S (Rescaled Range Analysis), and 
MFDFA (Multifractal Detrended Fluctuation Analysis).
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings

# Set up matplotlib for non-interactive backend
matplotlib.use('Agg')

class FractalPlotter:
    """A comprehensive fractal analysis plotting utility."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the fractal plotter.
        
        Parameters:
        -----------
        style : str
            Matplotlib style to use for plots
        figsize : tuple
            Default figure size (width, height)
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_dfa_analysis(self, dfa_results: Dict[str, Dict], 
                         series_names: List[str],
                         save_path: Optional[str] = None,
                         figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot DFA analysis results.
        
        Parameters:
        -----------
        dfa_results : Dict[str, Dict]
            Dictionary containing DFA results for each series
        series_names : List[str]
            List of series names to plot
        save_path : Optional[str]
            Path to save the plot
        figsize : Optional[Tuple[int, int]]
            Figure size
        """
        if figsize is None:
            figsize = self.figsize
        
        n_series = len(series_names)
        n_cols = min(3, n_series)
        n_rows = (n_series + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_series == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.ravel()
        else:
            axes = axes.ravel()
        
        for i, name in enumerate(series_names):
            if name in dfa_results and "error" not in dfa_results[name]:
                try:
                    dfa_results[name]["model"].plot_loglog(ax=axes[i], annotate=True)
                    axes[i].set_title(f'DFA: {name.replace("_", " ").title()}')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Plot Error: {str(e)[:50]}', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'DFA: {name.replace("_", " ").title()}')
            else:
                error_msg = dfa_results.get(name, {}).get("error", "No data available")
                axes[i].text(0.5, 0.5, f'Error: {error_msg[:50]}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'DFA: {name.replace("_", " ").title()}')
        
        # Hide unused subplots
        for i in range(n_series, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('DFA Analysis Results', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_rs_analysis(self, rs_results: Dict[str, Dict], 
                        series_names: List[str],
                        save_path: Optional[str] = None,
                        figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot R/S analysis results.
        
        Parameters:
        -----------
        rs_results : Dict[str, Dict]
            Dictionary containing R/S results for each series
        series_names : List[str]
            List of series names to plot
        save_path : Optional[str]
            Path to save the plot
        figsize : Optional[Tuple[int, int]]
            Figure size
        """
        if figsize is None:
            figsize = self.figsize
        
        n_series = len(series_names)
        n_cols = min(3, n_series)
        n_rows = (n_series + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_series == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.ravel()
        else:
            axes = axes.ravel()
        
        for i, name in enumerate(series_names):
            if name in rs_results and "error" not in rs_results[name]:
                try:
                    rs_results[name]["model"].plot_loglog(ax=axes[i], annotate=True)
                    axes[i].set_title(f'R/S: {name.replace("_", " ").title()}')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Plot Error: {str(e)[:50]}', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'R/S: {name.replace("_", " ").title()}')
            else:
                error_msg = rs_results.get(name, {}).get("error", "No data available")
                axes[i].text(0.5, 0.5, f'Error: {error_msg[:50]}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'R/S: {name.replace("_", " ").title()}')
        
        # Hide unused subplots
        for i in range(n_series, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('R/S Analysis Results', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_mfdfa_analysis(self, mfdfa_results: Dict[str, Dict], 
                           series_names: List[str],
                           save_path: Optional[str] = None,
                           figsize: Optional[Tuple[int, int]] = None,
                           q_indices: Optional[List[int]] = None) -> None:
        """
        Plot MFDFA analysis results.
        
        Parameters:
        -----------
        mfdfa_results : Dict[str, Dict]
            Dictionary containing MFDFA results for each series
        series_names : List[str]
            List of series names to plot
        save_path : Optional[str]
            Path to save the plot
        figsize : Optional[Tuple[int, int]]
            Figure size
        q_indices : Optional[List[int]]
            List of q indices to plot (default: [0, 3, -1])
        """
        if figsize is None:
            figsize = self.figsize
        
        if q_indices is None:
            q_indices = [0, 3, -1]
        
        n_series = len(series_names)
        n_cols = min(3, n_series)
        n_rows = (n_series + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_series == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.ravel()
        else:
            axes = axes.ravel()
        
        for i, name in enumerate(series_names):
            if name in mfdfa_results and "error" not in mfdfa_results[name]:
                try:
                    mfdfa_results[name]["model"].plot_loglog(ax=axes[i], q_indices=q_indices)
                    axes[i].set_title(f'MFDFA: {name.replace("_", " ").title()}')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Plot Error: {str(e)[:50]}', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'MFDFA: {name.replace("_", " ").title()}')
            else:
                error_msg = mfdfa_results.get(name, {}).get("error", "No data available")
                axes[i].text(0.5, 0.5, f'Error: {error_msg[:50]}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'MFDFA: {name.replace("_", " ").title()}')
        
        # Hide unused subplots
        for i in range(n_series, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('MFDFA Analysis Results', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_multifractal_spectrum(self, mfdfa_results: Dict[str, Dict], 
                                  series_names: List[str],
                                  save_path: Optional[str] = None,
                                  figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot multifractal spectrum (f(α) vs α).
        
        Parameters:
        -----------
        mfdfa_results : Dict[str, Dict]
            Dictionary containing MFDFA results for each series
        series_names : List[str]
            List of series names to plot
        save_path : Optional[str]
            Path to save the plot
        figsize : Optional[Tuple[int, int]]
            Figure size
        """
        if figsize is None:
            figsize = (12, 8)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Multifractal spectrum
        for name in series_names:
            if name in mfdfa_results and "error" not in mfdfa_results[name]:
                try:
                    model = mfdfa_results[name]["model"]
                    if hasattr(model, 'alpha') and hasattr(model, 'f_alpha'):
                        axes[0, 0].plot(model.alpha, model.f_alpha, 'o-', 
                                       label=name.replace("_", " ").title(), alpha=0.7)
                except Exception as e:
                    print(f"Error plotting multifractal spectrum for {name}: {e}")
        
        axes[0, 0].set_xlabel('α (singularity strength)')
        axes[0, 0].set_ylabel('f(α) (multifractal spectrum)')
        axes[0, 0].set_title('Multifractal Spectrum')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. h(q) vs q
        for name in series_names:
            if name in mfdfa_results and "error" not in mfdfa_results[name]:
                try:
                    model = mfdfa_results[name]["model"]
                    if hasattr(model, 'q') and hasattr(model, 'h_q'):
                        axes[0, 1].plot(model.q, model.h_q, 'o-', 
                                       label=name.replace("_", " ").title(), alpha=0.7)
                except Exception as e:
                    print(f"Error plotting h(q) for {name}: {e}")
        
        axes[0, 1].set_xlabel('q')
        axes[0, 1].set_ylabel('h(q)')
        axes[0, 1].set_title('Generalized Hurst Exponent')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. τ(q) vs q
        for name in series_names:
            if name in mfdfa_results and "error" not in mfdfa_results[name]:
                try:
                    model = mfdfa_results[name]["model"]
                    if hasattr(model, 'q') and hasattr(model, 'tau_q'):
                        axes[1, 0].plot(model.q, model.tau_q, 'o-', 
                                       label=name.replace("_", " ").title(), alpha=0.7)
                except Exception as e:
                    print(f"Error plotting τ(q) for {name}: {e}")
        
        axes[1, 0].set_xlabel('q')
        axes[1, 0].set_ylabel('τ(q)')
        axes[1, 0].set_title('Mass Exponent')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary statistics
        summary_data = []
        for name in series_names:
            if name in mfdfa_results and "error" not in mfdfa_results[name]:
                try:
                    hq_2 = mfdfa_results[name].get("hq_2", np.nan)
                    alpha_min = mfdfa_results[name].get("alpha_min", np.nan)
                    alpha_max = mfdfa_results[name].get("alpha_max", np.nan)
                    delta_alpha = mfdfa_results[name].get("delta_alpha", np.nan)
                    
                    summary_data.append([name.replace("_", " ").title(), 
                                       f"{hq_2:.3f}", f"{alpha_min:.3f}", 
                                       f"{alpha_max:.3f}", f"{delta_alpha:.3f}"])
                except Exception as e:
                    print(f"Error getting summary for {name}: {e}")
        
        if summary_data:
            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')
            table = axes[1, 1].table(cellText=summary_data,
                                    colLabels=['Series', 'h(q=2)', 'α_min', 'α_max', 'Δα'],
                                    cellLoc='center',
                                    loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            axes[1, 1].set_title('MFDFA Summary Statistics')
        else:
            axes[1, 1].text(0.5, 0.5, 'No valid data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('MFDFA Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_fractal_comparison(self, dfa_results: Dict[str, Dict],
                               rs_results: Dict[str, Dict],
                               mfdfa_results: Dict[str, Dict],
                               series_names: List[str],
                               save_path: Optional[str] = None,
                               figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Create comparison plots for different fractal analysis methods.
        
        Parameters:
        -----------
        dfa_results : Dict[str, Dict]
            DFA analysis results
        rs_results : Dict[str, Dict]
            R/S analysis results
        mfdfa_results : Dict[str, Dict]
            MFDFA analysis results
        series_names : List[str]
            List of series names
        save_path : Optional[str]
            Path to save the plot
        figsize : Optional[Tuple[int, int]]
            Figure size
        """
        if figsize is None:
            figsize = (15, 10)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. DFA alpha comparison
        alphas = []
        names_clean = []
        for name in series_names:
            if name in dfa_results and "error" not in dfa_results[name]:
                alphas.append(dfa_results[name]["alpha"])
                names_clean.append(name.replace("_", " ").title())
        
        if alphas:
            bars1 = axes[0, 0].bar(names_clean, alphas, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('DFA Scaling Exponent (α)')
            axes[0, 0].set_ylabel('α')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, alpha in zip(bars1, alphas):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{alpha:.3f}', ha='center', va='bottom')
        
        # 2. R/S Hurst exponent comparison
        hurst_rs = []
        names_clean2 = []
        for name in series_names:
            if name in rs_results and "error" not in rs_results[name]:
                hurst_rs.append(rs_results[name]["hurst"])
                names_clean2.append(name.replace("_", " ").title())
        
        if hurst_rs:
            bars2 = axes[0, 1].bar(names_clean2, hurst_rs, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('R/S Hurst Exponent (H)')
            axes[0, 1].set_ylabel('H')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, h_val in zip(bars2, hurst_rs):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{h_val:.3f}', ha='center', va='bottom')
        
        # 3. MFDFA h(q=2) comparison
        hq_2_values = []
        names_clean3 = []
        for name in series_names:
            if name in mfdfa_results and "error" not in mfdfa_results[name]:
                hq_2_values.append(mfdfa_results[name]["hq_2"])
                names_clean3.append(name.replace("_", " ").title())
        
        if hq_2_values:
            bars3 = axes[0, 2].bar(names_clean3, hq_2_values, alpha=0.7, color='purple')
            axes[0, 2].set_title('MFDFA h(q=2)')
            axes[0, 2].set_ylabel('h(q=2)')
            axes[0, 2].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, hq_val in zip(bars3, hq_2_values):
                axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{hq_val:.3f}', ha='center', va='bottom')
        
        # 4. Method comparison for each series
        comparison_data = []
        for name in series_names:
            dfa_alpha = dfa_results.get(name, {}).get("alpha", np.nan)
            rs_hurst = rs_results.get(name, {}).get("hurst", np.nan)
            mfdfa_hq2 = mfdfa_results.get(name, {}).get("hq_2", np.nan)
            
            comparison_data.append([name.replace("_", " ").title(), 
                                  f"{dfa_alpha:.3f}", f"{rs_hurst:.3f}", f"{mfdfa_hq2:.3f}"])
        
        if comparison_data:
            axes[1, 0].axis('tight')
            axes[1, 0].axis('off')
            table = axes[1, 0].table(cellText=comparison_data,
                                    colLabels=['Series', 'DFA α', 'R/S H', 'MFDFA h(q=2)'],
                                    cellLoc='center',
                                    loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            axes[1, 0].set_title('Method Comparison')
        
        # 5. Scatter plot: DFA vs R/S
        dfa_values = []
        rs_values = []
        valid_names = []
        for name in series_names:
            dfa_alpha = dfa_results.get(name, {}).get("alpha", np.nan)
            rs_hurst = rs_results.get(name, {}).get("hurst", np.nan)
            if not np.isnan(dfa_alpha) and not np.isnan(rs_hurst):
                dfa_values.append(dfa_alpha)
                rs_values.append(rs_hurst)
                valid_names.append(name.replace("_", " ").title())
        
        if dfa_values and rs_values:
            axes[1, 1].scatter(dfa_values, rs_values, alpha=0.7, s=100)
            for i, name in enumerate(valid_names):
                axes[1, 1].annotate(name, (dfa_values[i], rs_values[i]), 
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[1, 1].set_xlabel('DFA α')
            axes[1, 1].set_ylabel('R/S H')
            axes[1, 1].set_title('DFA vs R/S Comparison')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Scatter plot: DFA vs MFDFA
        dfa_values = []
        mfdfa_values = []
        valid_names = []
        for name in series_names:
            dfa_alpha = dfa_results.get(name, {}).get("alpha", np.nan)
            mfdfa_hq2 = mfdfa_results.get(name, {}).get("hq_2", np.nan)
            if not np.isnan(dfa_alpha) and not np.isnan(mfdfa_hq2):
                dfa_values.append(dfa_alpha)
                mfdfa_values.append(mfdfa_hq2)
                valid_names.append(name.replace("_", " ").title())
        
        if dfa_values and mfdfa_values:
            axes[1, 2].scatter(dfa_values, mfdfa_values, alpha=0.7, s=100, color='orange')
            for i, name in enumerate(valid_names):
                axes[1, 2].annotate(name, (dfa_values[i], mfdfa_values[i]), 
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[1, 2].set_xlabel('DFA α')
            axes[1, 2].set_ylabel('MFDFA h(q=2)')
            axes[1, 2].set_title('DFA vs MFDFA Comparison')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_dfa_analysis(dfa_results: Dict[str, Dict], 
                     series_names: List[str],
                     save_path: Optional[str] = None,
                     **kwargs) -> None:
    """
    Convenience function to plot DFA analysis results.
    
    Parameters:
    -----------
    dfa_results : Dict[str, Dict]
        DFA analysis results
    series_names : List[str]
        List of series names
    save_path : Optional[str]
        Path to save the plot
    **kwargs : dict
        Additional arguments passed to FractalPlotter.plot_dfa_analysis
    """
    plotter = FractalPlotter()
    plotter.plot_dfa_analysis(dfa_results, series_names, save_path, **kwargs)


def plot_rs_analysis(rs_results: Dict[str, Dict], 
                    series_names: List[str],
                    save_path: Optional[str] = None,
                    **kwargs) -> None:
    """
    Convenience function to plot R/S analysis results.
    
    Parameters:
    -----------
    rs_results : Dict[str, Dict]
        R/S analysis results
    series_names : List[str]
        List of series names
    save_path : Optional[str]
        Path to save the plot
    **kwargs : dict
        Additional arguments passed to FractalPlotter.plot_rs_analysis
    """
    plotter = FractalPlotter()
    plotter.plot_rs_analysis(rs_results, series_names, save_path, **kwargs)


def plot_mfdfa_analysis(mfdfa_results: Dict[str, Dict], 
                       series_names: List[str],
                       save_path: Optional[str] = None,
                       **kwargs) -> None:
    """
    Convenience function to plot MFDFA analysis results.
    
    Parameters:
    -----------
    mfdfa_results : Dict[str, Dict]
        MFDFA analysis results
    series_names : List[str]
        List of series names
    save_path : Optional[str]
        Path to save the plot
    **kwargs : dict
        Additional arguments passed to FractalPlotter.plot_mfdfa_analysis
    """
    plotter = FractalPlotter()
    plotter.plot_mfdfa_analysis(mfdfa_results, series_names, save_path, **kwargs)


def plot_multifractal_spectrum(mfdfa_results: Dict[str, Dict], 
                              series_names: List[str],
                              save_path: Optional[str] = None,
                              **kwargs) -> None:
    """
    Convenience function to plot multifractal spectrum.
    
    Parameters:
    -----------
    mfdfa_results : Dict[str, Dict]
        MFDFA analysis results
    series_names : List[str]
        List of series names
    save_path : Optional[str]
        Path to save the plot
    **kwargs : dict
        Additional arguments passed to FractalPlotter.plot_multifractal_spectrum
    """
    plotter = FractalPlotter()
    plotter.plot_multifractal_spectrum(mfdfa_results, series_names, save_path, **kwargs)


def plot_fractal_comparison(dfa_results: Dict[str, Dict],
                           rs_results: Dict[str, Dict],
                           mfdfa_results: Dict[str, Dict],
                           series_names: List[str],
                           save_path: Optional[str] = None,
                           **kwargs) -> None:
    """
    Convenience function to plot fractal analysis comparison.
    
    Parameters:
    -----------
    dfa_results : Dict[str, Dict]
        DFA analysis results
    rs_results : Dict[str, Dict]
        R/S analysis results
    mfdfa_results : Dict[str, Dict]
        MFDFA analysis results
    series_names : List[str]
        List of series names
    save_path : Optional[str]
        Path to save the plot
    **kwargs : dict
        Additional arguments passed to FractalPlotter.plot_fractal_comparison
    """
    plotter = FractalPlotter()
    plotter.plot_fractal_comparison(dfa_results, rs_results, mfdfa_results, 
                                   series_names, save_path, **kwargs)

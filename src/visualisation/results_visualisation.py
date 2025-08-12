"""
Comprehensive Results Visualization Module for Long-Range Dependence Analysis

This module provides comprehensive visualization tools for all analysis results,
including wavelet analysis, spectral analysis, ARFIMA analysis, and parameter comparisons.
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

class ResultsVisualizer:
    """A comprehensive results visualization utility."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the results visualizer.
        
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
    
    def plot_wavelet_analysis(self, wavelet_results: Dict[str, Dict], 
                             series_names: List[str],
                             save_path: Optional[str] = None,
                             figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot wavelet analysis results (Leaders and Whittle).
        
        Parameters:
        -----------
        wavelet_results : Dict[str, Dict]
            Dictionary containing wavelet results for each series
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
        
        # Wavelet Leaders plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_series == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.ravel()
        else:
            axes = axes.ravel()
        
        for i, name in enumerate(series_names):
            if name in wavelet_results and "error" not in wavelet_results[name]:
                try:
                    wavelet_results[name]["leaders_model"].plot_scalogram(ax=axes[i], plot_fitted=True)
                    axes[i].set_title(f'Wavelet Leaders: {name.replace("_", " ").title()}')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Plot Error: {str(e)[:50]}', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'Wavelet Leaders: {name.replace("_", " ").title()}')
            else:
                error_msg = wavelet_results.get(name, {}).get("error", "No data available")
                axes[i].text(0.5, 0.5, f'Error: {error_msg[:50]}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Wavelet Leaders: {name.replace("_", " ").title()}')
        
        # Hide unused subplots
        for i in range(n_series, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Wavelet Leaders Analysis Results', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            # Create filename for leaders
            path = Path(save_path)
            stem = path.stem
            suffix = path.suffix
            leaders_save_path = str(path.parent / f"{stem}_leaders{suffix}")
            plt.savefig(leaders_save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Wavelet Whittle plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_series == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.ravel()
        else:
            axes = axes.ravel()
        
        for i, name in enumerate(series_names):
            if name in wavelet_results and "error" not in wavelet_results[name]:
                try:
                    wavelet_results[name]["whittle_model"].plot_scalogram(ax=axes[i], plot_fitted=True)
                    axes[i].set_title(f'Wavelet Whittle: {name.replace("_", " ").title()}')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Plot Error: {str(e)[:50]}', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'Wavelet Whittle: {name.replace("_", " ").title()}')
            else:
                error_msg = wavelet_results.get(name, {}).get("error", "No data available")
                axes[i].text(0.5, 0.5, f'Error: {error_msg[:50]}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Wavelet Whittle: {name.replace("_", " ").title()}')
        
        # Hide unused subplots
        for i in range(n_series, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Wavelet Whittle Analysis Results', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            # Create filename for whittle
            path = Path(save_path)
            stem = path.stem
            suffix = path.suffix
            whittle_save_path = str(path.parent / f"{stem}_whittle{suffix}")
            plt.savefig(whittle_save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_spectral_analysis(self, spectral_results: Dict[str, Dict], 
                              series_names: List[str],
                              save_path: Optional[str] = None,
                              figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot spectral analysis results (Whittle MLE and Periodogram).
        
        Parameters:
        -----------
        spectral_results : Dict[str, Dict]
            Dictionary containing spectral results for each series
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
        
        # Spectral Whittle MLE plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_series == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.ravel()
        else:
            axes = axes.ravel()
        
        for i, name in enumerate(series_names):
            if name in spectral_results and "error" not in spectral_results[name]:
                try:
                    spectral_results[name]["whittle_model"].plot_spectrum(ax=axes[i], plot_fitted=True)
                    axes[i].set_title(f'Spectral Whittle MLE: {name.replace("_", " ").title()}')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Plot Error: {str(e)[:50]}', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'Spectral Whittle MLE: {name.replace("_", " ").title()}')
            else:
                error_msg = spectral_results.get(name, {}).get("error", "No data available")
                axes[i].text(0.5, 0.5, f'Error: {error_msg[:50]}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Spectral Whittle MLE: {name.replace("_", " ").title()}')
        
        # Hide unused subplots
        for i in range(n_series, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Spectral Whittle MLE Analysis Results', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            # Create filename for whittle
            path = Path(save_path)
            stem = path.stem
            suffix = path.suffix
            whittle_save_path = str(path.parent / f"{stem}_whittle{suffix}")
            plt.savefig(whittle_save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Spectral Periodogram plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_series == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.ravel()
        else:
            axes = axes.ravel()
        
        for i, name in enumerate(series_names):
            if name in spectral_results and "error" not in spectral_results[name]:
                try:
                    spectral_results[name]["periodogram_model"].plot_spectrum(ax=axes[i], plot_fitted=True)
                    axes[i].set_title(f'Spectral Periodogram: {name.replace("_", " ").title()}')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Plot Error: {str(e)[:50]}', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'Spectral Periodogram: {name.replace("_", " ").title()}')
            else:
                error_msg = spectral_results.get(name, {}).get("error", "No data available")
                axes[i].text(0.5, 0.5, f'Error: {error_msg[:50]}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Spectral Periodogram: {name.replace("_", " ").title()}')
        
        # Hide unused subplots
        for i in range(n_series, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Spectral Periodogram Analysis Results', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            # Create filename for periodogram
            path = Path(save_path)
            stem = path.stem
            suffix = path.suffix
            periodogram_save_path = str(path.parent / f"{stem}_periodogram{suffix}")
            plt.savefig(periodogram_save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_arfima_forecasts(self, arfima_results: Dict[str, Dict], 
                             data: Dict[str, np.ndarray],
                             series_names: List[str],
                             save_path: Optional[str] = None,
                             figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot ARFIMA forecasts.
        
        Parameters:
        -----------
        arfima_results : Dict[str, Dict]
            Dictionary containing ARFIMA results for each series
        data : Dict[str, np.ndarray]
            Original time series data
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
            if name in arfima_results and "error" not in arfima_results[name]:
                try:
                    y = data[name]
                    fc = arfima_results[name]["forecast"]
                    lo = arfima_results[name]["forecast_lo"]
                    hi = arfima_results[name]["forecast_hi"]
                    
                    # Plot historical data
                    axes[i].plot(y, label='Historical', linewidth=0.8)
                    
                    # Plot forecast
                    t_forecast = np.arange(len(y), len(y) + len(fc))
                    axes[i].plot(t_forecast, fc, 'r-', label='Forecast', linewidth=2)
                    axes[i].fill_between(t_forecast, lo, hi, alpha=0.3, 
                                       color='red', label='95% CI')
                    
                    axes[i].set_title(f'ARFIMA Forecast: {name.replace("_", " ").title()}')
                    axes[i].set_xlabel('Time')
                    axes[i].set_ylabel('Value')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Plot Error: {str(e)[:50]}', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'ARFIMA: {name.replace("_", " ").title()}')
            else:
                error_msg = arfima_results.get(name, {}).get("error", "No data available")
                axes[i].text(0.5, 0.5, f'Error: {error_msg[:50]}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'ARFIMA: {name.replace("_", " ").title()}')
        
        # Hide unused subplots
        for i in range(n_series, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('ARFIMA Forecasts', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_comprehensive_comparison(self, dfa_results: Dict[str, Dict],
                                     rs_results: Dict[str, Dict],
                                     mfdfa_results: Dict[str, Dict],
                                     wavelet_results: Dict[str, Dict],
                                     spectral_results: Dict[str, Dict],
                                     arfima_results: Dict[str, Dict],
                                     series_names: List[str],
                                     save_path: Optional[str] = None,
                                     figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Create comprehensive comparison plots for all analysis methods.
        
        Parameters:
        -----------
        dfa_results : Dict[str, Dict]
            DFA analysis results
        rs_results : Dict[str, Dict]
            R/S analysis results
        mfdfa_results : Dict[str, Dict]
            MFDFA analysis results
        wavelet_results : Dict[str, Dict]
            Wavelet analysis results
        spectral_results : Dict[str, Dict]
            Spectral analysis results
        arfima_results : Dict[str, Dict]
            ARFIMA analysis results
        series_names : List[str]
            List of series names
        save_path : Optional[str]
            Path to save the plot
        figsize : Optional[Tuple[int, int]]
            Figure size
        """
        if figsize is None:
            figsize = (20, 12)
        
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=figsize)
        
        # 1. DFA alpha comparison
        alphas = []
        names_clean = []
        for name in series_names:
            if name in dfa_results and "error" not in dfa_results[name]:
                alphas.append(dfa_results[name]["alpha"])
                names_clean.append(name.replace("_", " ").title())
        
        if alphas:
            bars1 = ax1.bar(names_clean, alphas, alpha=0.7, color='skyblue')
            ax1.set_title('DFA Scaling Exponent (α)')
            ax1.set_ylabel('α')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, alpha in zip(bars1, alphas):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{alpha:.3f}', ha='center', va='bottom')
        
        # 2. R/S Hurst exponent comparison
        hurst_rs = []
        names_clean2 = []
        for name in series_names:
            if name in rs_results and "error" not in rs_results[name]:
                hurst_rs.append(rs_results[name]["hurst"])
                names_clean2.append(name.replace("_", " ").title())
        
        if hurst_rs:
            bars2 = ax2.bar(names_clean2, hurst_rs, alpha=0.7, color='lightgreen')
            ax2.set_title('R/S Hurst Exponent (H)')
            ax2.set_ylabel('H')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, h_val in zip(bars2, hurst_rs):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{h_val:.3f}', ha='center', va='bottom')
        
        # 3. ARFIMA d parameter comparison
        d_params = []
        names_clean3 = []
        for name in series_names:
            if name in arfima_results and "error" not in arfima_results[name]:
                d_params.append(arfima_results[name]["model"].params.d)
                names_clean3.append(name.replace("_", " ").title())
        
        if d_params:
            bars3 = ax3.bar(names_clean3, d_params, alpha=0.7, color='orange')
            ax3.set_title('ARFIMA Fractional Parameter (d)')
            ax3.set_ylabel('d')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, d_val in zip(bars3, d_params):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{d_val:.3f}', ha='center', va='bottom')
        
        # 4. MFDFA h(q=2) comparison
        hq_2_values = []
        names_clean4 = []
        for name in series_names:
            if name in mfdfa_results and "error" not in mfdfa_results[name]:
                hq_2_values.append(mfdfa_results[name]["hq_2"])
                names_clean4.append(name.replace("_", " ").title())
        
        if hq_2_values:
            bars4 = ax4.bar(names_clean4, hq_2_values, alpha=0.7, color='purple')
            ax4.set_title('MFDFA h(q=2)')
            ax4.set_ylabel('h(q=2)')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, hq_val in zip(bars4, hq_2_values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{hq_val:.3f}', ha='center', va='bottom')
        
        # 5. Wavelet Leaders d parameter comparison
        d_wavelet_leaders = []
        names_clean5 = []
        for name in series_names:
            if name in wavelet_results and "error" not in wavelet_results[name]:
                d_wavelet_leaders.append(wavelet_results[name]["summary_leaders"].d)
                names_clean5.append(name.replace("_", " ").title())
        
        if d_wavelet_leaders:
            bars5 = ax5.bar(names_clean5, d_wavelet_leaders, alpha=0.7, color='red')
            ax5.set_title('Wavelet Leaders d Parameter')
            ax5.set_ylabel('d')
            ax5.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, d_val in zip(bars5, d_wavelet_leaders):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{d_val:.3f}', ha='center', va='bottom')
        
        # 6. Wavelet Whittle d parameter comparison
        d_wavelet_whittle = []
        names_clean6 = []
        for name in series_names:
            if name in wavelet_results and "error" not in wavelet_results[name]:
                d_wavelet_whittle.append(wavelet_results[name]["summary_whittle"].d)
                names_clean6.append(name.replace("_", " ").title())
        
        if d_wavelet_whittle:
            bars6 = ax6.bar(names_clean6, d_wavelet_whittle, alpha=0.7, color='brown')
            ax6.set_title('Wavelet Whittle d Parameter')
            ax6.set_ylabel('d')
            ax6.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, d_val in zip(bars6, d_wavelet_whittle):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{d_val:.3f}', ha='center', va='bottom')
        
        # 7. Spectral Whittle MLE d parameter comparison
        d_spectral_whittle = []
        names_clean7 = []
        for name in series_names:
            if name in spectral_results and "error" not in spectral_results[name]:
                d_spectral_whittle.append(spectral_results[name]["summary_whittle"].d)
                names_clean7.append(name.replace("_", " ").title())
        
        if d_spectral_whittle:
            bars7 = ax7.bar(names_clean7, d_spectral_whittle, alpha=0.7, color='darkgreen')
            ax7.set_title('Spectral Whittle MLE d Parameter')
            ax7.set_ylabel('d')
            ax7.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, d_val in zip(bars7, d_spectral_whittle):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{d_val:.3f}', ha='center', va='bottom')
        
        # 8. Spectral Periodogram d parameter comparison
        d_spectral_periodogram = []
        names_clean8 = []
        for name in series_names:
            if name in spectral_results and "error" not in spectral_results[name]:
                d_spectral_periodogram.append(spectral_results[name]["summary_periodogram"].d)
                names_clean8.append(name.replace("_", " ").title())
        
        if d_spectral_periodogram:
            bars8 = ax8.bar(names_clean8, d_spectral_periodogram, alpha=0.7, color='darkblue')
            ax8.set_title('Spectral Periodogram d Parameter')
            ax8.set_ylabel('d')
            ax8.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, d_val in zip(bars8, d_spectral_periodogram):
                ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{d_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_summary_table(self, dfa_results: Dict[str, Dict],
                            rs_results: Dict[str, Dict],
                            mfdfa_results: Dict[str, Dict],
                            wavelet_results: Dict[str, Dict],
                            spectral_results: Dict[str, Dict],
                            arfima_results: Dict[str, Dict],
                            series_names: List[str],
                            save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create a comprehensive summary table of all results.
        
        Parameters:
        -----------
        dfa_results : Dict[str, Dict]
            DFA analysis results
        rs_results : Dict[str, Dict]
            R/S analysis results
        mfdfa_results : Dict[str, Dict]
            MFDFA analysis results
        wavelet_results : Dict[str, Dict]
            Wavelet analysis results
        spectral_results : Dict[str, Dict]
            Spectral analysis results
        arfima_results : Dict[str, Dict]
            ARFIMA analysis results
        series_names : List[str]
            List of series names
        save_path : Optional[str]
            Path to save the table as CSV
        
        Returns:
        --------
        pd.DataFrame
            Summary table with all results
        """
        summary_data = []
        
        for name in series_names:
            row = {'Series': name.replace("_", " ").title()}
            
            # DFA results
            if name in dfa_results and "error" not in dfa_results[name]:
                summary = dfa_results[name]['summary']
                row.update({
                    'DFA_alpha': f"{dfa_results[name]['alpha']:.4f}",
                    'DFA_hurst': f"{dfa_results[name]['hurst']:.4f}",
                    'DFA_rvalue': f"{summary.rvalue:.4f}",
                    'DFA_pvalue': f"{summary.pvalue:.4f}"
                })
            else:
                row.update({
                    'DFA_alpha': 'N/A',
                    'DFA_hurst': 'N/A',
                    'DFA_rvalue': 'N/A',
                    'DFA_pvalue': 'N/A'
                })
            
            # R/S results
            if name in rs_results and "error" not in rs_results[name]:
                summary = rs_results[name]['summary']
                row.update({
                    'RS_hurst': f"{rs_results[name]['hurst']:.4f}",
                    'RS_alpha': f"{rs_results[name]['alpha_rs']:.4f}",
                    'RS_rvalue': f"{summary.rvalue:.4f}",
                    'RS_pvalue': f"{summary.pvalue:.4f}"
                })
            else:
                row.update({
                    'RS_hurst': 'N/A',
                    'RS_alpha': 'N/A',
                    'RS_rvalue': 'N/A',
                    'RS_pvalue': 'N/A'
                })
            
            # MFDFA results
            if name in mfdfa_results and "error" not in mfdfa_results[name]:
                row.update({
                    'MFDFA_hq2': f"{mfdfa_results[name]['hq_2']:.4f}",
                    'MFDFA_alpha_min': f"{mfdfa_results[name].get('alpha_min', np.nan):.4f}",
                    'MFDFA_alpha_max': f"{mfdfa_results[name].get('alpha_max', np.nan):.4f}",
                    'MFDFA_delta_alpha': f"{mfdfa_results[name].get('delta_alpha', np.nan):.4f}"
                })
            else:
                row.update({
                    'MFDFA_hq2': 'N/A',
                    'MFDFA_alpha_min': 'N/A',
                    'MFDFA_alpha_max': 'N/A',
                    'MFDFA_delta_alpha': 'N/A'
                })
            
            # Wavelet results
            if name in wavelet_results and "error" not in wavelet_results[name]:
                row.update({
                    'Wavelet_Leaders_d': f"{wavelet_results[name]['summary_leaders'].d:.4f}",
                    'Wavelet_Leaders_hurst': f"{wavelet_results[name]['summary_leaders'].hurst:.4f}",
                    'Wavelet_Leaders_alpha': f"{wavelet_results[name]['summary_leaders'].alpha:.4f}",
                    'Wavelet_Leaders_rvalue': f"{wavelet_results[name]['summary_leaders'].rvalue:.4f}",
                    'Wavelet_Leaders_pvalue': f"{wavelet_results[name]['summary_leaders'].pvalue:.4f}",
                    'Wavelet_Whittle_d': f"{wavelet_results[name]['summary_whittle'].d:.4f}",
                    'Wavelet_Whittle_hurst': f"{wavelet_results[name]['summary_whittle'].hurst:.4f}",
                    'Wavelet_Whittle_alpha': f"{wavelet_results[name]['summary_whittle'].alpha:.4f}",
                    'Wavelet_Whittle_rvalue': f"{wavelet_results[name]['summary_whittle'].rvalue:.4f}",
                    'Wavelet_Whittle_pvalue': f"{wavelet_results[name]['summary_whittle'].pvalue:.4f}"
                })
            else:
                row.update({
                    'Wavelet_Leaders_d': 'N/A', 'Wavelet_Leaders_hurst': 'N/A',
                    'Wavelet_Leaders_alpha': 'N/A', 'Wavelet_Leaders_rvalue': 'N/A',
                    'Wavelet_Leaders_pvalue': 'N/A',
                    'Wavelet_Whittle_d': 'N/A', 'Wavelet_Whittle_hurst': 'N/A',
                    'Wavelet_Whittle_alpha': 'N/A', 'Wavelet_Whittle_rvalue': 'N/A',
                    'Wavelet_Whittle_pvalue': 'N/A'
                })
            
            # Spectral results
            if name in spectral_results and "error" not in spectral_results[name]:
                row.update({
                    'Spectral_Whittle_d': f"{spectral_results[name]['summary_whittle'].d:.4f}",
                    'Spectral_Whittle_hurst': f"{spectral_results[name]['summary_whittle'].hurst:.4f}",
                    'Spectral_Whittle_alpha': f"{spectral_results[name]['summary_whittle'].alpha:.4f}",
                    'Spectral_Whittle_rvalue': f"{spectral_results[name]['summary_whittle'].rvalue:.4f}",
                    'Spectral_Whittle_pvalue': f"{spectral_results[name]['summary_whittle'].pvalue:.4f}",
                    'Spectral_Periodogram_d': f"{spectral_results[name]['summary_periodogram'].d:.4f}",
                    'Spectral_Periodogram_hurst': f"{spectral_results[name]['summary_periodogram'].hurst:.4f}",
                    'Spectral_Periodogram_alpha': f"{spectral_results[name]['summary_periodogram'].alpha:.4f}",
                    'Spectral_Periodogram_rvalue': f"{spectral_results[name]['summary_periodogram'].rvalue:.4f}",
                    'Spectral_Periodogram_pvalue': f"{spectral_results[name]['summary_periodogram'].pvalue:.4f}"
                })
            else:
                row.update({
                    'Spectral_Whittle_d': 'N/A', 'Spectral_Whittle_hurst': 'N/A',
                    'Spectral_Whittle_alpha': 'N/A', 'Spectral_Whittle_rvalue': 'N/A',
                    'Spectral_Whittle_pvalue': 'N/A',
                    'Spectral_Periodogram_d': 'N/A', 'Spectral_Periodogram_hurst': 'N/A',
                    'Spectral_Periodogram_alpha': 'N/A', 'Spectral_Periodogram_rvalue': 'N/A',
                    'Spectral_Periodogram_pvalue': 'N/A'
                })
            
            # ARFIMA results
            if name in arfima_results and "error" not in arfima_results[name]:
                row.update({
                    'ARFIMA_d': f"{arfima_results[name]['model'].params.d:.4f}",
                    'ARFIMA_p': f"{arfima_results[name]['model'].params.p:.4f}",
                    'ARFIMA_q': f"{arfima_results[name]['model'].params.q:.4f}",
                    'ARFIMA_aic': f"{arfima_results[name]['model'].aic:.4f}",
                    'ARFIMA_bic': f"{arfima_results[name]['model'].bic:.4f}"
                })
            else:
                row.update({
                    'ARFIMA_d': 'N/A', 'ARFIMA_p': 'N/A', 'ARFIMA_q': 'N/A',
                    'ARFIMA_aic': 'N/A', 'ARFIMA_bic': 'N/A'
                })
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        if save_path:
            summary_df.to_csv(save_path, index=False)
        
        return summary_df


def plot_wavelet_analysis(wavelet_results: Dict[str, Dict], 
                         series_names: List[str],
                         save_path: Optional[str] = None,
                         **kwargs) -> None:
    """
    Convenience function to plot wavelet analysis results.
    
    Parameters:
    -----------
    wavelet_results : Dict[str, Dict]
        Wavelet analysis results
    series_names : List[str]
        List of series names
    save_path : Optional[str]
        Path to save the plot
    **kwargs : dict
        Additional arguments passed to ResultsVisualizer.plot_wavelet_analysis
    """
    visualizer = ResultsVisualizer()
    visualizer.plot_wavelet_analysis(wavelet_results, series_names, save_path, **kwargs)


def plot_spectral_analysis(spectral_results: Dict[str, Dict], 
                          series_names: List[str],
                          save_path: Optional[str] = None,
                          **kwargs) -> None:
    """
    Convenience function to plot spectral analysis results.
    
    Parameters:
    -----------
    spectral_results : Dict[str, Dict]
        Spectral analysis results
    series_names : List[str]
        List of series names
    save_path : Optional[str]
        Path to save the plot
    **kwargs : dict
        Additional arguments passed to ResultsVisualizer.plot_spectral_analysis
    """
    visualizer = ResultsVisualizer()
    visualizer.plot_spectral_analysis(spectral_results, series_names, save_path, **kwargs)


def plot_arfima_forecasts(arfima_results: Dict[str, Dict], 
                         data: Dict[str, np.ndarray],
                         series_names: List[str],
                         save_path: Optional[str] = None,
                         **kwargs) -> None:
    """
    Convenience function to plot ARFIMA forecasts.
    
    Parameters:
    -----------
    arfima_results : Dict[str, Dict]
        ARFIMA analysis results
    data : Dict[str, np.ndarray]
        Original time series data
    series_names : List[str]
        List of series names
    save_path : Optional[str]
        Path to save the plot
    **kwargs : dict
        Additional arguments passed to ResultsVisualizer.plot_arfima_forecasts
    """
    visualizer = ResultsVisualizer()
    visualizer.plot_arfima_forecasts(arfima_results, data, series_names, save_path, **kwargs)


def plot_comprehensive_comparison(dfa_results: Dict[str, Dict],
                                 rs_results: Dict[str, Dict],
                                 mfdfa_results: Dict[str, Dict],
                                 wavelet_results: Dict[str, Dict],
                                 spectral_results: Dict[str, Dict],
                                 arfima_results: Dict[str, Dict],
                                 series_names: List[str],
                                 save_path: Optional[str] = None,
                                 **kwargs) -> None:
    """
    Convenience function to plot comprehensive comparison.
    
    Parameters:
    -----------
    dfa_results : Dict[str, Dict]
        DFA analysis results
    rs_results : Dict[str, Dict]
        R/S analysis results
    mfdfa_results : Dict[str, Dict]
        MFDFA analysis results
    wavelet_results : Dict[str, Dict]
        Wavelet analysis results
    spectral_results : Dict[str, Dict]
        Spectral analysis results
    arfima_results : Dict[str, Dict]
        ARFIMA analysis results
    series_names : List[str]
        List of series names
    save_path : Optional[str]
        Path to save the plot
    **kwargs : dict
        Additional arguments passed to ResultsVisualizer.plot_comprehensive_comparison
    """
    visualizer = ResultsVisualizer()
    visualizer.plot_comprehensive_comparison(dfa_results, rs_results, mfdfa_results,
                                           wavelet_results, spectral_results, arfima_results,
                                           series_names, save_path, **kwargs)


def create_summary_table(dfa_results: Dict[str, Dict],
                        rs_results: Dict[str, Dict],
                        mfdfa_results: Dict[str, Dict],
                        wavelet_results: Dict[str, Dict],
                        spectral_results: Dict[str, Dict],
                        arfima_results: Dict[str, Dict],
                        series_names: List[str],
                        save_path: Optional[str] = None,
                        **kwargs) -> pd.DataFrame:
    """
    Convenience function to create summary table.
    
    Parameters:
    -----------
    dfa_results : Dict[str, Dict]
        DFA analysis results
    rs_results : Dict[str, Dict]
        R/S analysis results
    mfdfa_results : Dict[str, Dict]
        MFDFA analysis results
    wavelet_results : Dict[str, Dict]
        Wavelet analysis results
    spectral_results : Dict[str, Dict]
        Spectral analysis results
    arfima_results : Dict[str, Dict]
        ARFIMA analysis results
    series_names : List[str]
        List of series names
    save_path : Optional[str]
        Path to save the table
    **kwargs : dict
        Additional arguments passed to ResultsVisualizer.create_summary_table
    
    Returns:
    --------
    pd.DataFrame
        Summary table with all results
    """
    visualizer = ResultsVisualizer()
    return visualizer.create_summary_table(dfa_results, rs_results, mfdfa_results,
                                         wavelet_results, spectral_results, arfima_results,
                                         series_names, save_path, **kwargs)

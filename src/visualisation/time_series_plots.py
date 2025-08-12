"""
Time Series Plotting Module for Long-Range Dependence Analysis

This module provides comprehensive visualization tools for time series data,
including basic plots, data quality assessment, and preprocessing visualization.
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

class TimeSeriesPlotter:
    """A comprehensive time series plotting utility."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the time series plotter.
        
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
    
    def plot_time_series(self, data: Dict[str, np.ndarray], 
                        save_path: Optional[str] = None,
                        figsize: Optional[Tuple[int, int]] = None,
                        title: str = "Time Series Data",
                        max_series_per_plot: int = 6) -> None:
        """
        Plot multiple time series in a grid layout.
        
        Parameters:
        -----------
        data : Dict[str, np.ndarray]
            Dictionary of time series data with series names as keys
        save_path : Optional[str]
            Path to save the plot (if None, plot is displayed)
        figsize : Optional[Tuple[int, int]]
            Figure size (width, height)
        title : str
            Main title for the plot
        max_series_per_plot : int
            Maximum number of series to plot in a single figure
        """
        if figsize is None:
            figsize = self.figsize
        
        series_names = list(data.keys())
        n_series = len(series_names)
        
        if n_series == 0:
            print("Warning: No data provided for plotting")
            return
        
        # Split into multiple plots if too many series
        if n_series > max_series_per_plot:
            self._plot_multiple_figures(data, save_path, figsize, title, max_series_per_plot)
            return
        
        # Create single plot
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
        
        # Plot each time series
        for i, name in enumerate(series_names):
            y = data[name]
            axes[i].plot(y, linewidth=0.8, alpha=0.8)
            axes[i].set_title(f'{name.replace("_", " ").title()}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_series, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _plot_multiple_figures(self, data: Dict[str, np.ndarray], 
                              save_path: Optional[str], 
                              figsize: Tuple[int, int],
                              title: str,
                              max_series_per_plot: int) -> None:
        """Plot multiple figures when there are too many series."""
        series_names = list(data.keys())
        n_series = len(series_names)
        n_plots = (n_series + max_series_per_plot - 1) // max_series_per_plot
        
        for plot_idx in range(n_plots):
            start_idx = plot_idx * max_series_per_plot
            end_idx = min(start_idx + max_series_per_plot, n_series)
            
            plot_data = {name: data[name] for name in series_names[start_idx:end_idx]}
            plot_title = f"{title} (Part {plot_idx + 1}/{n_plots})"
            
            if save_path:
                # Create filename with part number
                path = Path(save_path)
                stem = path.stem
                suffix = path.suffix
                plot_save_path = str(path.parent / f"{stem}_part{plot_idx + 1}{suffix}")
            else:
                plot_save_path = None
            
            self.plot_time_series(plot_data, plot_save_path, figsize, plot_title)
    
    def plot_data_quality(self, df: pd.DataFrame, 
                         save_path: Optional[str] = None,
                         figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Create comprehensive data quality visualization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing time series data
        save_path : Optional[str]
            Path to save the plot
        figsize : Optional[Tuple[int, int]]
            Figure size
        """
        if figsize is None:
            figsize = (15, 10)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. Missing values heatmap
        missing_data = df.isnull()
        sns.heatmap(missing_data, cbar=True, ax=axes[0, 0], cmap='viridis')
        axes[0, 0].set_title('Missing Values Pattern')
        
        # 2. Missing values percentage
        missing_pct = df.isnull().sum() / len(df) * 100
        axes[0, 1].bar(range(len(missing_pct)), missing_pct)
        axes[0, 1].set_title('Missing Values Percentage')
        axes[0, 1].set_xlabel('Columns')
        axes[0, 1].set_ylabel('Percentage (%)')
        axes[0, 1].set_xticks(range(len(missing_pct)))
        axes[0, 1].set_xticklabels(df.columns, rotation=45)
        
        # 3. Data distribution (first few columns)
        n_cols_to_plot = min(3, len(df.columns))
        for i in range(n_cols_to_plot):
            col = df.columns[i]
            if df[col].notna().sum() > 0:
                axes[0, 2].hist(df[col].dropna(), alpha=0.7, label=col, bins=30)
        axes[0, 2].set_title('Data Distribution')
        axes[0, 2].legend()
        
        # 4. Time series overview (first few columns)
        for i in range(n_cols_to_plot):
            col = df.columns[i]
            if df[col].notna().sum() > 0:
                axes[1, 0].plot(df[col], alpha=0.7, label=col, linewidth=0.8)
        axes[1, 0].set_title('Time Series Overview')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Correlation matrix
        if len(df.columns) > 1:
            corr_matrix = df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 1], fmt='.2f')
            axes[1, 1].set_title('Correlation Matrix')
        else:
            axes[1, 1].text(0.5, 0.5, 'Single column\nNo correlation', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Correlation Matrix')
        
        # 6. Summary statistics
        summary_stats = df.describe()
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        table = axes[1, 2].table(cellText=summary_stats.values,
                                rowLabels=summary_stats.index,
                                colLabels=summary_stats.columns,
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        axes[1, 2].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_preprocessing_comparison(self, original: pd.DataFrame, 
                                    processed: pd.DataFrame,
                                    save_path: Optional[str] = None,
                                    figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Compare original and processed data.
        
        Parameters:
        -----------
        original : pd.DataFrame
            Original data
        processed : pd.DataFrame
            Processed data
        save_path : Optional[str]
            Path to save the plot
        figsize : Optional[Tuple[int, int]]
            Figure size
        """
        if figsize is None:
            figsize = (15, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot first column comparison
        if len(original.columns) > 0 and len(processed.columns) > 0:
            col_name = original.columns[0]
            
            # Original vs processed
            axes[0, 0].plot(original[col_name], label='Original', alpha=0.7)
            axes[0, 0].plot(processed[col_name], label='Processed', alpha=0.7)
            axes[0, 0].set_title(f'Original vs Processed: {col_name}')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Distribution comparison
            axes[0, 1].hist(original[col_name].dropna(), alpha=0.7, label='Original', bins=30)
            axes[0, 1].hist(processed[col_name].dropna(), alpha=0.7, label='Processed', bins=30)
            axes[0, 1].set_title(f'Distribution Comparison: {col_name}')
            axes[0, 1].set_xlabel('Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            
            # Missing values comparison
            missing_orig = original.isnull().sum() / len(original) * 100
            missing_proc = processed.isnull().sum() / len(processed) * 100
            
            x = np.arange(len(missing_orig))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, missing_orig, width, label='Original', alpha=0.7)
            axes[1, 0].bar(x + width/2, missing_proc, width, label='Processed', alpha=0.7)
            axes[1, 0].set_title('Missing Values Comparison')
            axes[1, 0].set_xlabel('Columns')
            axes[1, 0].set_ylabel('Missing Percentage (%)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(original.columns, rotation=45)
            axes[1, 0].legend()
            
            # Statistics comparison
            stats_orig = original.describe()
            stats_proc = processed.describe()
            
            if len(original.columns) > 0:
                col = original.columns[0]
                stats_comparison = pd.DataFrame({
                    'Original': stats_orig[col],
                    'Processed': stats_proc[col]
                })
                
                axes[1, 1].axis('tight')
                axes[1, 1].axis('off')
                table = axes[1, 1].table(cellText=stats_comparison.values,
                                        rowLabels=stats_comparison.index,
                                        colLabels=stats_comparison.columns,
                                        cellLoc='center',
                                        loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)
                axes[1, 1].set_title('Statistics Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_time_series(data: Union[Dict[str, np.ndarray], pd.DataFrame, np.ndarray],
                    save_path: Optional[str] = None,
                    **kwargs) -> None:
    """
    Convenience function to plot time series data.
    
    Parameters:
    -----------
    data : Union[Dict[str, np.ndarray], pd.DataFrame, np.ndarray]
        Time series data in various formats
    save_path : Optional[str]
        Path to save the plot
    **kwargs : dict
        Additional arguments passed to TimeSeriesPlotter.plot_time_series
    """
    plotter = TimeSeriesPlotter()
    
    # Convert data to standard format
    if isinstance(data, np.ndarray):
        data_dict = {"series": data}
    elif isinstance(data, pd.DataFrame):
        data_dict = {col: data[col].values for col in data.columns}
    elif isinstance(data, dict):
        data_dict = data
    else:
        raise ValueError("Data must be numpy array, pandas DataFrame, or dictionary")
    
    plotter.plot_time_series(data_dict, save_path, **kwargs)


def plot_data_quality(df: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> None:
    """
    Convenience function to plot data quality assessment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    save_path : Optional[str]
        Path to save the plot
    **kwargs : dict
        Additional arguments passed to TimeSeriesPlotter.plot_data_quality
    """
    plotter = TimeSeriesPlotter()
    plotter.plot_data_quality(df, save_path, **kwargs)


def plot_preprocessing_comparison(original: pd.DataFrame, 
                                processed: pd.DataFrame,
                                save_path: Optional[str] = None,
                                **kwargs) -> None:
    """
    Convenience function to compare original and processed data.
    
    Parameters:
    -----------
    original : pd.DataFrame
        Original data
    processed : pd.DataFrame
        Processed data
    save_path : Optional[str]
        Path to save the plot
    **kwargs : dict
        Additional arguments passed to TimeSeriesPlotter.plot_preprocessing_comparison
    """
    plotter = TimeSeriesPlotter()
    plotter.plot_preprocessing_comparison(original, processed, save_path, **kwargs)

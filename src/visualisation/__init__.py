"""
Visualization Module for Long-Range Dependence Analysis

This package provides comprehensive visualization tools for time series analysis,
fractal analysis, results comparison, and statistical validation.
"""

from .time_series_plots import (
    TimeSeriesPlotter, plot_time_series, plot_data_quality, plot_preprocessing_comparison
)
from .fractal_plots import (
    FractalPlotter, plot_dfa_analysis, plot_rs_analysis, plot_mfdfa_analysis,
    plot_multifractal_spectrum, plot_fractal_comparison
)
from .higuchi_plots import (
    plot_higuchi_analysis, plot_higuchi_comparison, plot_higuchi_quality_assessment,
    create_higuchi_report, plot_higuchi_result
)
from .results_visualisation import (
    ResultsVisualizer, plot_wavelet_analysis, plot_spectral_analysis,
    plot_arfima_forecasts, plot_comprehensive_comparison, create_summary_table
)
from .validation_plots import (
    plot_hypothesis_test_result, plot_bootstrap_result, plot_monte_carlo_result,
    plot_cross_validation_result, plot_comprehensive_validation_summary,
    create_validation_report, plot_validation_result
)

__all__ = [
    # Time series plotting
    'TimeSeriesPlotter', 'plot_time_series', 'plot_data_quality', 'plot_preprocessing_comparison',
    
    # Fractal analysis plotting
    'FractalPlotter', 'plot_dfa_analysis', 'plot_rs_analysis', 'plot_mfdfa_analysis',
    'plot_multifractal_spectrum', 'plot_fractal_comparison',
    
    # Higuchi analysis plotting
    'plot_higuchi_analysis', 'plot_higuchi_comparison', 'plot_higuchi_quality_assessment',
    'create_higuchi_report', 'plot_higuchi_result',
    
    # Results visualization
    'ResultsVisualizer', 'plot_wavelet_analysis', 'plot_spectral_analysis',
    'plot_arfima_forecasts', 'plot_comprehensive_comparison', 'create_summary_table',
    
    # Statistical validation plotting
    'plot_hypothesis_test_result', 'plot_bootstrap_result', 'plot_monte_carlo_result',
    'plot_cross_validation_result', 'plot_comprehensive_validation_summary',
    'create_validation_report', 'plot_validation_result'
]

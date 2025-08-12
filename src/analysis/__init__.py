"""
Analysis Module for Long-Range Dependence Project

This module provides comprehensive analysis tools for long-range dependence:
- ARFIMA modelling and simulation
- Detrended Fluctuation Analysis (DFA)
- Rescaled Range Analysis (R/S)
- Multifractal Detrended Fluctuation Analysis (MFDFA)
- Wavelet Analysis (Wavelet Leaders, Wavelet Whittle)
- Spectral Analysis (Whittle MLE, Periodogram)
- Statistical Validation (Hypothesis testing, Cross-validation, Bootstrap, Monte Carlo)
"""

# ARFIMA Analysis
from .arfima_modelling import (
    ARFIMAModel,
    arfima_simulation,
    estimate_arfima_order
)

# DFA Analysis
from .dfa_analysis import (
    DFAModel,
    dfa,
    hurst_from_dfa_alpha,
    d_from_hurst
)

# R/S Analysis
from .rs_analysis import (
    RSModel,
    rs_analysis,
    d_from_hurst_rs,
    alpha_from_hurst_rs
)

# MFDFA Analysis
from .mfdfa_analysis import (
    MFDFAModel,
    mfdfa,
    hurst_from_mfdfa,
    alpha_from_mfdfa
)

# Wavelet Analysis
from .wavelet_analysis import (
    WaveletModel,
    wavelet_leaders_estimation,
    wavelet_whittle_estimation
)

# Spectral Analysis
from .spectral_analysis import (
    SpectralModel,
    whittle_mle,
    periodogram_estimation
)

# Higuchi Analysis
from .higuchi_analysis import (
    higuchi_fractal_dimension,
    estimate_higuchi_dimension,
    higuchi_analysis_batch,
    validate_higuchi_results,
    HiguchiSummary
)

# Statistical Validation
from .statistical_validation import (
    StatisticalValidator,
    HypothesisTestResult,
    CrossValidationResult,
    BootstrapResult,
    MonteCarloResult,
    test_lrd_hypothesis,
    bootstrap_confidence_interval,
    monte_carlo_test,
    cross_validate_lrd,
    comprehensive_validation
)

__all__ = [
    # ARFIMA
    'ARFIMAModel',
    'arfima_simulation',
    'estimate_arfima_order',
    
    # DFA
    'DFAModel',
    'dfa',
    'hurst_from_dfa_alpha',
    'd_from_hurst',
    
    # R/S
    'RSModel',
    'rs_analysis',
    'd_from_hurst_rs',
    'alpha_from_hurst_rs',
    
    # MFDFA
    'MFDFAModel',
    'mfdfa',
    'hurst_from_mfdfa',
    'alpha_from_mfdfa',
    
    # Wavelet
    'WaveletModel',
    'wavelet_leaders_estimation',
    'wavelet_whittle_estimation',
    
    # Spectral
    'SpectralModel',
    'whittle_mle',
    'periodogram_estimation',
    
    # Higuchi
    'higuchi_fractal_dimension',
    'estimate_higuchi_dimension',
    'higuchi_analysis_batch',
    'validate_higuchi_results',
    'HiguchiSummary',
    
    # Statistical Validation
    'StatisticalValidator',
    'HypothesisTestResult',
    'CrossValidationResult',
    'BootstrapResult',
    'MonteCarloResult',
    'test_lrd_hypothesis',
    'bootstrap_confidence_interval',
    'monte_carlo_test',
    'cross_validate_lrd',
    'comprehensive_validation'
]

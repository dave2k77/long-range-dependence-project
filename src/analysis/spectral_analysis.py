"""
Spectral Analysis Methods for Long-Range Dependence

This module implements spectral-based methods for estimating long-range dependence:
- Whittle Maximum Likelihood Estimation (MLE)
- Periodogram-based estimation
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.signal import periodogram
from scipy.special import gamma
import warnings
from typing import Tuple, Optional, Dict, List, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class SpectralSummary:
    """Container for spectral analysis results."""
    method: str
    hurst: float
    d: float
    alpha: float
    rvalue: float
    pvalue: float
    stderr: float
    frequencies: np.ndarray
    power_spectrum: np.ndarray
    fitted_spectrum: Optional[np.ndarray] = None


def _validate_signal(y: np.ndarray) -> np.ndarray:
    """Validate and clean input signal."""
    if y is None or len(y) == 0:
        raise ValueError("Signal cannot be empty")
    
    y = np.asarray(y, dtype=float)
    
    # Remove NaN values
    if np.any(np.isnan(y)):
        warnings.warn("Signal contains NaN values, removing them")
        y = y[~np.isnan(y)]
    
    if len(y) < 50:
        warnings.warn("Signal length is very short for spectral analysis")
    
    return y


def _generate_frequencies(n: int, sampling_rate: float = 1.0) -> np.ndarray:
    """Generate frequency array for spectral analysis."""
    # Use only positive frequencies up to Nyquist
    freqs = np.fft.fftfreq(n, d=1.0/sampling_rate)
    positive_freqs = freqs[freqs > 0]
    return positive_freqs


def _theoretical_spectrum(frequencies: np.ndarray, d: float, sigma2: float = 1.0) -> np.ndarray:
    """
    Compute theoretical power spectrum for fractional noise.
    
    Parameters:
    -----------
    frequencies : np.ndarray
        Frequency array
    d : float
        Fractional differencing parameter
    sigma2 : float
        Innovation variance
        
    Returns:
    --------
    np.ndarray
        Theoretical power spectrum
    """
    # Power spectrum of fractional noise: S(f) = sigma2 / |2*sin(pi*f)|^(2*d)
    # For small frequencies: S(f) ≈ sigma2 / (2*pi*f)^(2*d)
    
    # Avoid division by zero
    eps = 1e-10
    f_safe = np.maximum(frequencies, eps)
    
    # Theoretical spectrum
    spectrum = sigma2 / (2 * np.pi * f_safe) ** (2 * d)
    
    return spectrum


def _whittle_log_likelihood(params: np.ndarray, frequencies: np.ndarray, 
                           periodogram_values: np.ndarray) -> float:
    """
    Compute Whittle log-likelihood for spectral estimation.
    
    Parameters:
    -----------
    params : np.ndarray
        [d, log_sigma2] parameters
    frequencies : np.ndarray
        Frequency array
    periodogram_values : np.ndarray
        Periodogram values
        
    Returns:
    --------
    float
        Negative log-likelihood (to be minimized)
    """
    d, log_sigma2 = params
    sigma2 = np.exp(log_sigma2)
    
    # Theoretical spectrum
    theoretical = _theoretical_spectrum(frequencies, d, sigma2)
    
    # Whittle log-likelihood
    # L = -sum(log(S(f)) + I(f)/S(f))
    log_likelihood = -np.sum(np.log(theoretical) + periodogram_values / theoretical)
    
    return log_likelihood


def whittle_mle(y: np.ndarray, sampling_rate: float = 1.0, 
                freq_range: Optional[Tuple[float, float]] = None,
                initial_d: float = 0.3) -> Tuple[np.ndarray, np.ndarray, SpectralSummary]:
    """
    Estimate long-range dependence using Whittle Maximum Likelihood Estimation.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    sampling_rate : float
        Sampling rate of the signal
    freq_range : Optional[Tuple[float, float]]
        Frequency range to use for estimation (min_freq, max_freq)
    initial_d : float
        Initial guess for d parameter
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, SpectralSummary]
        Frequencies, periodogram values, and summary statistics
    """
    y = _validate_signal(y)
    n = len(y)
    
    # Compute periodogram
    freqs, periodogram_vals = periodogram(y, fs=sampling_rate, scaling='density')
    
    # Use only positive frequencies
    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    periodogram_vals = periodogram_vals[positive_mask]
    
    # Filter frequency range if specified
    if freq_range is not None:
        min_freq, max_freq = freq_range
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        freqs = freqs[freq_mask]
        periodogram_vals = periodogram_vals[freq_mask]
    
    if len(freqs) < 10:
        raise ValueError("Insufficient frequency points for estimation")
    
    # Initial parameters
    initial_params = np.array([initial_d, np.log(np.var(y))])
    
    # Bounds for optimization
    bounds = [(0.01, 0.49), (-20.0, 20.0)]  # d bounds, log_sigma2 bounds
    
    # Optimize using L-BFGS-B
    try:
        result = optimize.minimize(
            fun=lambda params: _whittle_log_likelihood(params, freqs, periodogram_vals),
            x0=initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'disp': False}
        )
        
        if not result.success:
            warnings.warn(f"Whittle optimization may not have converged: {result.message}")
        
        d_est, log_sigma2_est = result.x
        sigma2_est = np.exp(log_sigma2_est)
        
    except Exception as e:
        warnings.warn(f"Whittle optimization failed: {e}, using initial values")
        d_est, sigma2_est = initial_d, np.var(y)
    
    # Compute fitted spectrum
    fitted_spectrum = _theoretical_spectrum(freqs, d_est, sigma2_est)
    
    # Compute correlation coefficient between observed and fitted
    valid_mask = (periodogram_vals > 0) & (fitted_spectrum > 0)
    if np.sum(valid_mask) > 10:
        correlation = np.corrcoef(
            np.log(periodogram_vals[valid_mask]), 
            np.log(fitted_spectrum[valid_mask])
        )[0, 1]
        # Approximate p-value (assuming normal distribution)
        n_valid = np.sum(valid_mask)
        t_stat = correlation * np.sqrt((n_valid - 2) / (1 - correlation**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_valid - 2))
    else:
        correlation = 0.0
        p_value = 1.0
    
    # Convert to other parameters
    hurst = d_est + 0.5
    alpha = 2 * hurst
    
    # Approximate standard error (simplified)
    stderr = 1.0 / np.sqrt(len(freqs))
    
    summary = SpectralSummary(
        method="Whittle MLE",
        hurst=hurst,
        d=d_est,
        alpha=alpha,
        rvalue=correlation,
        pvalue=p_value,
        stderr=stderr,
        frequencies=freqs,
        power_spectrum=periodogram_vals,
        fitted_spectrum=fitted_spectrum
    )
    
    return freqs, periodogram_vals, summary


def periodogram_estimation(y: np.ndarray, sampling_rate: float = 1.0,
                          freq_range: Optional[Tuple[float, float]] = None,
                          min_freq_points: int = 20) -> Tuple[np.ndarray, np.ndarray, SpectralSummary]:
    """
    Estimate long-range dependence using periodogram regression.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    sampling_rate : float
        Sampling rate of the signal
    freq_range : Optional[Tuple[float, float]]
        Frequency range to use for estimation (min_freq, max_freq)
    min_freq_points : int
        Minimum number of frequency points to use
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, SpectralSummary]
        Frequencies, periodogram values, and summary statistics
    """
    y = _validate_signal(y)
    n = len(y)
    
    # Compute periodogram
    freqs, periodogram_vals = periodogram(y, fs=sampling_rate, scaling='density')
    
    # Use only positive frequencies
    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    periodogram_vals = periodogram_vals[positive_mask]
    
    # Filter frequency range if specified
    if freq_range is not None:
        min_freq, max_freq = freq_range
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        freqs = freqs[freq_mask]
        periodogram_vals = periodogram_vals[freq_mask]
    
    if len(freqs) < min_freq_points:
        raise ValueError(f"Insufficient frequency points ({len(freqs)}) for estimation")
    
    # Use only low frequencies for long-range dependence estimation
    # Take the lowest frequencies that give us enough points
    n_use = min(len(freqs), max(min_freq_points, len(freqs) // 4))
    freqs_use = freqs[:n_use]
    periodogram_use = periodogram_vals[:n_use]
    
    # Log-log regression: log(I(f)) = log(C) - 2*d*log(f)
    log_freqs = np.log(freqs_use)
    log_periodogram = np.log(periodogram_use)
    
    # Remove any infinite or NaN values
    valid_mask = np.isfinite(log_freqs) & np.isfinite(log_periodogram)
    if np.sum(valid_mask) < 5:
        raise ValueError("Insufficient valid points for regression")
    
    log_freqs_valid = log_freqs[valid_mask]
    log_periodogram_valid = log_periodogram[valid_mask]
    
    # Linear regression
    slope, intercept, r_value, p_value, stderr = stats.linregress(
        log_freqs_valid, log_periodogram_valid
    )
    
    # Extract parameters
    d_est = -slope / 2  # From log(I(f)) = log(C) - 2*d*log(f)
    hurst = d_est + 0.5
    alpha = 2 * hurst
    
    # Ensure d is in reasonable range
    d_est = np.clip(d_est, 0.01, 0.49)
    
    # Compute fitted values
    fitted_log_periodogram = intercept + slope * log_freqs_valid
    fitted_spectrum = np.exp(fitted_log_periodogram)
    
    summary = SpectralSummary(
        method="Periodogram Regression",
        hurst=hurst,
        d=d_est,
        alpha=alpha,
        rvalue=r_value,
        pvalue=p_value,
        stderr=stderr,
        frequencies=freqs_use,
        power_spectrum=periodogram_use,
        fitted_spectrum=fitted_spectrum
    )
    
    return freqs_use, periodogram_use, summary


class SpectralModel:
    """
    Spectral analysis model for long-range dependence estimation.
    
    Supports both Whittle MLE and Periodogram regression methods.
    """
    
    def __init__(self, method: str = 'whittle', sampling_rate: float = 1.0):
        """
        Initialize spectral model.
        
        Parameters:
        -----------
        method : str
            'whittle' or 'periodogram'
        sampling_rate : float
            Sampling rate of the signal
        """
        if method not in ['whittle', 'periodogram']:
            raise ValueError("Method must be 'whittle' or 'periodogram'")
        
        self.method = method
        self.sampling_rate = sampling_rate
        self.frequencies = None
        self.power_spectrum = None
        self.summary = None
        self.is_fitted = False
    
    def fit(self, y: np.ndarray, freq_range: Optional[Tuple[float, float]] = None) -> 'SpectralModel':
        """
        Fit spectral model to time series.
        
        Parameters:
        -----------
        y : np.ndarray
            Time series data
        freq_range : Optional[Tuple[float, float]]
            Frequency range for estimation
            
        Returns:
        --------
        SpectralModel
            Fitted model instance
        """
        if self.method == 'whittle':
            self.frequencies, self.power_spectrum, self.summary = whittle_mle(
                y, self.sampling_rate, freq_range
            )
        else:  # periodogram
            self.frequencies, self.power_spectrum, self.summary = periodogram_estimation(
                y, self.sampling_rate, freq_range
            )
        
        self.is_fitted = True
        return self
    
    def get_hurst(self) -> float:
        """Get Hurst exponent estimate."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.summary.hurst
    
    def get_d(self) -> float:
        """Get fractional differencing parameter estimate."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.summary.d
    
    def get_alpha(self) -> float:
        """Get scaling exponent estimate."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.summary.alpha
    
    def get_summary(self) -> SpectralSummary:
        """Get full summary statistics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.summary
    
    def plot_spectrum(self, ax: Optional[plt.Axes] = None, 
                     plot_fitted: bool = True) -> plt.Axes:
        """
        Plot power spectrum and fitted curve.
        
        Parameters:
        -----------
        ax : Optional[plt.Axes]
            Matplotlib axes to plot on
        plot_fitted : bool
            Whether to plot fitted spectrum
            
        Returns:
        --------
        plt.Axes
            Matplotlib axes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot periodogram
        ax.loglog(self.frequencies, self.power_spectrum, 'o', 
                 alpha=0.6, label='Periodogram', markersize=4)
        
        # Plot fitted spectrum if available
        if plot_fitted and self.summary.fitted_spectrum is not None:
            ax.loglog(self.frequencies, self.summary.fitted_spectrum, 'r-', 
                     linewidth=2, label=f'Fitted (d={self.summary.d:.3f})')
        
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title(f'{self.summary.method} Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add parameter annotation
        ax.text(0.05, 0.95, 
                f'H={self.summary.hurst:.3f}\nd={self.summary.d:.3f}\nr={self.summary.rvalue:.3f}',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return ax


# Utility functions for parameter conversion
def hurst_from_spectral_d(d: float) -> float:
    """Convert spectral d parameter to Hurst exponent."""
    return d + 0.5


def d_from_spectral_hurst(hurst: float) -> float:
    """Convert Hurst exponent to spectral d parameter."""
    return hurst - 0.5


def alpha_from_spectral_d(d: float) -> float:
    """Convert spectral d parameter to scaling exponent."""
    return 2 * (d + 0.5)

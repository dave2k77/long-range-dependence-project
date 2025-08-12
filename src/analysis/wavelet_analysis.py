"""
Wavelet Analysis Methods for Long-Range Dependence

This module implements wavelet-based methods for estimating long-range dependence:
- Wavelet Leaders
- Wavelet Whittle estimation
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
# Simple Morlet wavelet implementation
def _morlet_wavelet(n, scale, width=6.0):
    """Generate Morlet wavelet."""
    t = np.arange(-n//2, n//2)
    t = t / scale
    w = np.exp(1j * width * t) * np.exp(-0.5 * t**2)
    return w
import warnings
from typing import Tuple, Optional, Dict, List, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class WaveletSummary:
    """Container for wavelet analysis results."""
    method: str
    hurst: float
    d: float
    alpha: float
    rvalue: float
    pvalue: float
    stderr: float
    scales: np.ndarray
    coefficients: np.ndarray
    fitted_values: Optional[np.ndarray] = None


def _validate_signal(y: np.ndarray) -> np.ndarray:
    """Validate and clean input signal."""
    if y is None or len(y) == 0:
        raise ValueError("Signal cannot be empty")
    
    y = np.asarray(y, dtype=float)
    
    # Remove NaN values
    if np.any(np.isnan(y)):
        warnings.warn("Signal contains NaN values, removing them")
        y = y[~np.isnan(y)]
    
    if len(y) < 100:
        warnings.warn("Signal length is very short for wavelet analysis")
    
    return y


def _generate_scales(n: int, min_scale: int = 4, max_scale: Optional[int] = None) -> np.ndarray:
    """Generate scale array for wavelet analysis."""
    if max_scale is None:
        max_scale = n // 8
    
    # Use dyadic scales: 2^j where j goes from log2(min_scale) to log2(max_scale)
    min_j = int(np.log2(min_scale))
    max_j = int(np.log2(max_scale))
    
    # Ensure we have at least 5 scales by extending the range if needed
    if max_j - min_j < 4:
        max_j = min_j + 4
    
    # Generate scales
    scales = 2.0 ** np.arange(min_j, max_j + 1)
    
    # Filter scales to respect max_scale if specified
    if max_scale is not None:
        scales = scales[scales <= max_scale]
    
    # If we still don't have enough scales, try to generate more while respecting max_scale
    if len(scales) < 5:
        # Only generate more scales if we can do so without violating max_scale
        # Calculate the maximum possible scales within max_scale
        max_possible_j = int(np.log2(max_scale)) if max_scale is not None else max_j
        if max_possible_j - min_j >= 4:  # Can get 5 scales
            scales = 2.0 ** np.arange(min_j, min_j + 5)
            if max_scale is not None:
                scales = scales[scales <= max_scale]
        # Otherwise, keep the current scales (which is less than 5)
    
    return scales


def _compute_wavelet_coefficients(y: np.ndarray, scales: np.ndarray, 
                                wavelet: str = 'morlet') -> np.ndarray:
    """
    Compute wavelet coefficients for given scales.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    scales : np.ndarray
        Scale array
    wavelet : str
        Wavelet type ('morlet' or 'haar')
        
    Returns:
    --------
    np.ndarray
        Wavelet coefficients [scales, time]
    """
    n = len(y)
    
    if wavelet.lower() == 'morlet':
        # Use Morlet wavelet
        coefficients = np.zeros((len(scales), n))
        
        for i, scale in enumerate(scales):
            # Generate Morlet wavelet
            w = _morlet_wavelet(n, scale)
            
            # Compute wavelet transform using convolution
            coeffs = np.convolve(y, w.real, mode='same')
            coefficients[i, :] = coeffs
            
    elif wavelet.lower() == 'haar':
        # Use Haar wavelet (simplified implementation)
        coefficients = np.zeros((len(scales), n))
        
        for i, scale in enumerate(scales):
            scale_int = int(scale)
            if scale_int < 2:
                continue
                
            # Haar wavelet coefficients
            coeffs = np.zeros(n)
            for t in range(scale_int, n):
                # Haar wavelet at scale s
                if t + scale_int <= n:
                    coeffs[t] = np.sum(y[t:t+scale_int]) - np.sum(y[t-scale_int:t])
            
            coefficients[i, :] = coeffs
            
    else:
        raise ValueError("Wavelet must be 'morlet' or 'haar'")
    
    return coefficients


def _compute_wavelet_leaders(y: np.ndarray, scales: np.ndarray, 
                           wavelet: str = 'morlet') -> np.ndarray:
    """
    Compute wavelet leaders (local maxima of wavelet coefficients).
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    scales : np.ndarray
        Scale array
    wavelet : str
        Wavelet type
        
    Returns:
    --------
    np.ndarray
        Wavelet leaders [scales]
    """
    coefficients = _compute_wavelet_coefficients(y, scales, wavelet)
    
    # Compute wavelet leaders as the maximum absolute values at each scale
    leaders = np.max(np.abs(coefficients), axis=1)
    
    return leaders


def _theoretical_wavelet_spectrum(scales: np.ndarray, d: float, sigma2: float = 1.0) -> np.ndarray:
    """
    Compute theoretical wavelet spectrum for fractional noise.
    
    Parameters:
    -----------
    scales : np.ndarray
        Scale array
    d : float
        Fractional differencing parameter
    sigma2 : float
        Innovation variance
        
    Returns:
    --------
    np.ndarray
        Theoretical wavelet spectrum
    """
    # For fractional noise: E[|W(s)|^2] = sigma2 * s^(2*d + 1)
    spectrum = sigma2 * scales ** (2 * d + 1)
    return spectrum


def _wavelet_whittle_log_likelihood(params: np.ndarray, scales: np.ndarray, 
                                   wavelet_coefficients: np.ndarray) -> float:
    """
    Compute wavelet Whittle log-likelihood.
    
    Parameters:
    -----------
    params : np.ndarray
        [d, log_sigma2] parameters
    scales : np.ndarray
        Scale array
    wavelet_coefficients : np.ndarray
        Wavelet coefficient variances
        
    Returns:
    --------
    float
        Negative log-likelihood (to be minimized)
    """
    d, log_sigma2 = params
    sigma2 = np.exp(log_sigma2)
    
    # Theoretical spectrum
    theoretical = _theoretical_wavelet_spectrum(scales, d, sigma2)
    
    # Wavelet Whittle log-likelihood
    # L = -sum(log(S(s)) + |W(s)|^2/S(s))
    log_likelihood = -np.sum(np.log(theoretical) + wavelet_coefficients / theoretical)
    
    return log_likelihood


def wavelet_leaders_estimation(y: np.ndarray, min_scale: int = 4, 
                              max_scale: Optional[int] = None,
                              wavelet: str = 'morlet') -> Tuple[np.ndarray, np.ndarray, WaveletSummary]:
    """
    Estimate long-range dependence using Wavelet Leaders method.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    min_scale : int
        Minimum scale to use
    max_scale : Optional[int]
        Maximum scale to use
    wavelet : str
        Wavelet type ('morlet' or 'haar')
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, WaveletSummary]
        Scales, wavelet leaders, and summary statistics
    """
    y = _validate_signal(y)
    n = len(y)
    
    # Generate scales
    scales = _generate_scales(n, min_scale, max_scale)
    
    if len(scales) < 3:
        raise ValueError("Insufficient scales for wavelet analysis")
    
    # Compute wavelet leaders
    leaders = _compute_wavelet_leaders(y, scales, wavelet)
    
    # Log-log regression: log(L(s)) = log(C) + (2*d + 1)*log(s)
    log_scales = np.log(scales)
    log_leaders = np.log(leaders)
    
    # Remove any infinite or NaN values
    valid_mask = np.isfinite(log_scales) & np.isfinite(log_leaders)
    if np.sum(valid_mask) < 3:
        raise ValueError("Insufficient valid points for regression")
    
    log_scales_valid = log_scales[valid_mask]
    log_leaders_valid = log_leaders[valid_mask]
    
    # Linear regression
    slope, intercept, r_value, p_value, stderr = stats.linregress(
        log_scales_valid, log_leaders_valid
    )
    
    # Extract parameters: log(L(s)) = log(C) + (2*d + 1)*log(s)
    d_est = (slope - 1) / 2  # From slope = 2*d + 1
    hurst = d_est + 0.5
    alpha = 2 * hurst
    
    # Ensure d is in reasonable range
    d_est = np.clip(d_est, 0.01, 0.49)
    
    # Compute fitted values
    fitted_log_leaders = intercept + slope * log_scales_valid
    fitted_leaders = np.exp(fitted_log_leaders)
    
    summary = WaveletSummary(
        method="Wavelet Leaders",
        hurst=hurst,
        d=d_est,
        alpha=alpha,
        rvalue=r_value,
        pvalue=p_value,
        stderr=stderr,
        scales=scales,
        coefficients=leaders,
        fitted_values=fitted_leaders
    )
    
    return scales, leaders, summary


def wavelet_whittle_estimation(y: np.ndarray, min_scale: int = 4,
                              max_scale: Optional[int] = None,
                              wavelet: str = 'morlet',
                              initial_d: float = 0.3) -> Tuple[np.ndarray, np.ndarray, WaveletSummary]:
    """
    Estimate long-range dependence using Wavelet Whittle method.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    min_scale : int
        Minimum scale to use
    max_scale : Optional[int]
        Maximum scale to use
    wavelet : str
        Wavelet type ('morlet' or 'haar')
    initial_d : float
        Initial guess for d parameter
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, WaveletSummary]
        Scales, wavelet coefficient variances, and summary statistics
    """
    y = _validate_signal(y)
    n = len(y)
    
    # Generate scales
    scales = _generate_scales(n, min_scale, max_scale)
    
    if len(scales) < 3:
        raise ValueError("Insufficient scales for wavelet analysis")
    
    # Compute wavelet coefficients
    coefficients = _compute_wavelet_coefficients(y, scales, wavelet)
    
    # Compute variance of wavelet coefficients at each scale
    coeff_variances = np.var(coefficients, axis=1)
    
    # Initial parameters
    initial_params = np.array([initial_d, np.log(np.var(y))])
    
    # Bounds for optimization
    bounds = [(0.01, 0.49), (-20.0, 20.0)]  # d bounds, log_sigma2 bounds
    
    # Optimize using L-BFGS-B
    try:
        result = optimize.minimize(
            fun=lambda params: _wavelet_whittle_log_likelihood(params, scales, coeff_variances),
            x0=initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'disp': False}
        )
        
        if not result.success:
            warnings.warn(f"Wavelet Whittle optimization may not have converged: {result.message}")
        
        d_est, log_sigma2_est = result.x
        sigma2_est = np.exp(log_sigma2_est)
        
    except Exception as e:
        warnings.warn(f"Wavelet Whittle optimization failed: {e}, using initial values")
        d_est, sigma2_est = initial_d, np.var(y)
    
    # Compute fitted spectrum
    fitted_spectrum = _theoretical_wavelet_spectrum(scales, d_est, sigma2_est)
    
    # Compute correlation coefficient between observed and fitted
    valid_mask = (coeff_variances > 0) & (fitted_spectrum > 0)
    if np.sum(valid_mask) > 3:
        correlation = np.corrcoef(
            np.log(coeff_variances[valid_mask]), 
            np.log(fitted_spectrum[valid_mask])
        )[0, 1]
        # Approximate p-value
        n_valid = np.sum(valid_mask)
        t_stat = correlation * np.sqrt((n_valid - 2) / (1 - correlation**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_valid - 2))
    else:
        correlation = 0.0
        p_value = 1.0
    
    # Convert to other parameters
    hurst = d_est + 0.5
    alpha = 2 * hurst
    
    # Approximate standard error
    stderr = 1.0 / np.sqrt(len(scales))
    
    summary = WaveletSummary(
        method="Wavelet Whittle",
        hurst=hurst,
        d=d_est,
        alpha=alpha,
        rvalue=correlation,
        pvalue=p_value,
        stderr=stderr,
        scales=scales,
        coefficients=coeff_variances,
        fitted_values=fitted_spectrum
    )
    
    return scales, coeff_variances, summary


class WaveletModel:
    """
    Wavelet analysis model for long-range dependence estimation.
    
    Supports both Wavelet Leaders and Wavelet Whittle methods.
    """
    
    def __init__(self, method: str = 'leaders', wavelet: str = 'morlet'):
        """
        Initialize wavelet model.
        
        Parameters:
        -----------
        method : str
            'leaders' or 'whittle'
        wavelet : str
            'morlet' or 'haar'
        """
        if method not in ['leaders', 'whittle']:
            raise ValueError("Method must be 'leaders' or 'whittle'")
        if wavelet not in ['morlet', 'haar']:
            raise ValueError("Wavelet must be 'morlet' or 'haar'")
        
        self.method = method
        self.wavelet = wavelet
        self.scales = None
        self.coefficients = None
        self.summary = None
        self.is_fitted = False
    
    def fit(self, y: np.ndarray, min_scale: int = 4, 
            max_scale: Optional[int] = None) -> 'WaveletModel':
        """
        Fit wavelet model to time series.
        
        Parameters:
        -----------
        y : np.ndarray
            Time series data
        min_scale : int
            Minimum scale to use
        max_scale : Optional[int]
            Maximum scale to use
            
        Returns:
        --------
        WaveletModel
            Fitted model instance
        """
        if self.method == 'leaders':
            self.scales, self.coefficients, self.summary = wavelet_leaders_estimation(
                y, min_scale, max_scale, self.wavelet
            )
        else:  # whittle
            self.scales, self.coefficients, self.summary = wavelet_whittle_estimation(
                y, min_scale, max_scale, self.wavelet
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
    
    def get_summary(self) -> WaveletSummary:
        """Get full summary statistics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.summary
    
    def plot_scalogram(self, ax: Optional[plt.Axes] = None, 
                      plot_fitted: bool = True) -> plt.Axes:
        """
        Plot wavelet scalogram and fitted curve.
        
        Parameters:
        -----------
        ax : Optional[plt.Axes]
            Matplotlib axes to plot on
        plot_fitted : bool
            Whether to plot fitted curve
            
        Returns:
        --------
        plt.Axes
            Matplotlib axes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot wavelet coefficients
        ax.loglog(self.scales, self.coefficients, 'o', 
                 alpha=0.6, label=f'{self.summary.method}', markersize=6)
        
        # Plot fitted curve if available
        if plot_fitted and self.summary.fitted_values is not None:
            ax.loglog(self.scales, self.summary.fitted_values, 'r-', 
                     linewidth=2, label=f'Fitted (d={self.summary.d:.3f})')
        
        ax.set_xlabel('Scale')
        if self.method == 'leaders':
            ax.set_ylabel('Wavelet Leaders')
        else:
            ax.set_ylabel('Wavelet Coefficient Variance')
        ax.set_title(f'{self.summary.method} Analysis ({self.wavelet.capitalize()} Wavelet)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add parameter annotation
        ax.text(0.05, 0.95, 
                f'H={self.summary.hurst:.3f}\nd={self.summary.d:.3f}\nr={self.summary.rvalue:.3f}',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        return ax


# Utility functions for parameter conversion
def hurst_from_wavelet_d(d: float) -> float:
    """Convert wavelet d parameter to Hurst exponent."""
    return d + 0.5


def d_from_wavelet_hurst(hurst: float) -> float:
    """Convert Hurst exponent to wavelet d parameter."""
    return hurst - 0.5


def alpha_from_wavelet_d(d: float) -> float:
    """Convert wavelet d parameter to scaling exponent."""
    return 2 * (d + 0.5)

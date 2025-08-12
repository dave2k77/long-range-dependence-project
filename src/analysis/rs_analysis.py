"""
Rescaled Range (R/S) Analysis

This module implements R/S analysis for long-range dependence detection.
It provides both functional and object-oriented APIs.

References:
- Hurst (1951) for the original R/S methodology
- Mandelbrot and Wallis (1969) for theoretical foundations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
import matplotlib.pyplot as plt


@dataclass
class RSSummary:
    """Summary of an R/S analysis."""
    hurst: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float
    scales: np.ndarray
    rs_values: np.ndarray

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hurst": float(self.hurst),
            "intercept": float(self.intercept),
            "rvalue": float(self.rvalue),
            "pvalue": float(self.pvalue),
            "stderr": float(self.stderr),
            "scales": self.scales.copy(),
            "rs_values": self.rs_values.copy(),
        }


def _validate_signal(y: ArrayLike) -> np.ndarray:
    """Validate and prepare input signal."""
    y_np = np.asarray(y, dtype=float).ravel()
    if y_np.size < 16:
        raise ValueError("Input series is too short for R/S analysis (need at least 16 points)")
    if not np.all(np.isfinite(y_np)):
        finite_mask = np.isfinite(y_np)
        if finite_mask.sum() < 8:
            raise ValueError("Too many non-finite values in input series for R/S analysis")
        idx = np.arange(len(y_np))
        y_np[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], y_np[finite_mask])
    return y_np


def _generate_log_scales(n: int,
                        min_scale: int = 4,
                        max_scale: Optional[int] = None,
                        num_scales: int = 20,
                        base: float = 2.0) -> np.ndarray:
    """Generate log-spaced scales for R/S analysis."""
    if max_scale is None:
        max_scale = max(min(int(n // 4), 1024), min_scale + 1)
    if min_scale < 2:
        min_scale = 2
    if max_scale <= min_scale:
        max_scale = min_scale + 1
    
    logs = np.linspace(np.log(min_scale) / np.log(base), np.log(max_scale) / np.log(base), num_scales)
    scales = np.unique(np.clip(np.round(base ** logs).astype(int), min_scale, max_scale))
    
    if scales.size < 6:
        scales = np.unique(np.clip(np.round(np.linspace(min_scale, max_scale, 6)).astype(int), min_scale, max_scale))
    return scales


def _rs_for_scale(y: np.ndarray, s: int) -> float:
    """
    Calculate R/S statistic for a given scale s.
    
    Parameters:
    - y: input series
    - s: scale (window size)
    
    Returns:
    - R/S value for this scale
    """
    n = len(y)
    if s < 2 or s > n:
        return np.nan
    
    # Calculate number of segments
    num_segments = n // s
    if num_segments < 1:
        return np.nan
    
    rs_values = []
    
    for i in range(num_segments):
        start_idx = i * s
        end_idx = start_idx + s
        segment = y[start_idx:end_idx]
        
        # Calculate mean
        mean_seg = np.mean(segment)
        
        # Calculate cumulative deviation
        dev = segment - mean_seg
        cum_dev = np.cumsum(dev)
        
        # Calculate range R
        R = np.max(cum_dev) - np.min(cum_dev)
        
        # Calculate standard deviation S
        S = np.std(segment, ddof=1)
        
        # Avoid division by zero
        if S > 0:
            rs_values.append(R / S)
    
    if not rs_values:
        return np.nan
    
    return np.mean(rs_values)


def rs_analysis(y: ArrayLike,
                scales: Optional[Sequence[int]] = None,
                min_scale: int = 4,
                max_scale: Optional[int] = None,
                num_scales: int = 20,
                base: float = 2.0) -> Tuple[np.ndarray, np.ndarray, RSSummary]:
    """
    Perform Rescaled Range (R/S) analysis on a 1D series.
    
    Parameters:
    - y: input series (1D)
    - scales: window sizes to evaluate; if None, automatically generated
    - min_scale, max_scale, num_scales, base: parameters for automatic scale generation
    
    Returns:
    - scales (np.ndarray)
    - rs_values (np.ndarray): R/S values
    - summary (RSSummary) with Hurst exponent and regression stats
    """
    y_np = _validate_signal(y)
    
    if scales is None:
        scales = _generate_log_scales(len(y_np), min_scale=min_scale, max_scale=max_scale,
                                      num_scales=num_scales, base=base)
    else:
        scales = np.unique(np.asarray(scales, dtype=int))
        scales = scales[scales >= 2]
        if scales.size < 2:
            raise ValueError("Need at least two valid scales (>= 2)")
    
    rs_values = np.zeros_like(scales, dtype=float)
    for i, s in enumerate(scales):
        rs_values[i] = _rs_for_scale(y_np, int(s))
    
    # Remove any non-positive R/S values for log fit
    valid = np.isfinite(rs_values) & (rs_values > 0)
    if valid.sum() < 2:
        raise ValueError("Not enough valid R/S values to estimate Hurst exponent")
    
    x = np.log(scales[valid])
    ylog = np.log(rs_values[valid])
    reg = stats.linregress(x, ylog)
    
    hurst = reg.slope
    intercept = reg.intercept
    
    summary = RSSummary(
        hurst=hurst,
        intercept=intercept,
        rvalue=reg.rvalue,
        pvalue=reg.pvalue,
        stderr=reg.stderr,
        scales=scales.astype(float),
        rs_values=rs_values.astype(float),
    )
    return scales.astype(float), rs_values.astype(float), summary


class RSModel:
    """Object-oriented interface for R/S analysis."""
    
    def __init__(self):
        self.scales: Optional[np.ndarray] = None
        self.rs_values: Optional[np.ndarray] = None
        self.summary: Optional[RSSummary] = None
        self.is_fitted: bool = False
    
    def fit(self,
            y: ArrayLike,
            scales: Optional[Sequence[int]] = None,
            min_scale: int = 4,
            max_scale: Optional[int] = None,
            num_scales: int = 20,
            base: float = 2.0) -> "RSModel":
        scales_out, rs_values_out, summary = rs_analysis(
            y,
            scales=scales,
            min_scale=min_scale,
            max_scale=max_scale,
            num_scales=num_scales,
            base=base,
        )
        self.scales = scales_out
        self.rs_values = rs_values_out
        self.summary = summary
        self.is_fitted = True
        return self
    
    def get_hurst(self) -> float:
        if not self.is_fitted or self.summary is None:
            raise ValueError("Model not fitted")
        return float(self.summary.hurst)
    
    def get_summary(self) -> RSSummary:
        if not self.is_fitted or self.summary is None:
            raise ValueError("Model not fitted")
        return self.summary
    
    def plot_loglog(self, ax: Optional[plt.Axes] = None, annotate: bool = True) -> plt.Axes:
        if not self.is_fitted or self.scales is None or self.rs_values is None or self.summary is None:
            raise ValueError("Model not fitted")
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.loglog(self.scales, self.rs_values, 'o', label='R/S')
        
        # regression line over range
        x = np.log(self.scales)
        yhat = self.summary.intercept + self.summary.hurst * x
        ax.plot(self.scales, np.exp(yhat), '-', label=f"H={self.summary.hurst:.3f}")
        
        ax.set_xlabel('Scale s')
        ax.set_ylabel('R/S')
        ax.grid(True, which='both', ls='--', alpha=0.4)
        ax.legend()
        
        if annotate:
            ax.annotate(f"H={self.summary.hurst:.3f}\nr={self.summary.rvalue:.3f}",
                        xy=(0.05, 0.95), xycoords='axes fraction', va='top')
        return ax


def d_from_hurst_rs(H: float) -> float:
    """Convert Hurst exponent from R/S analysis to fractional differencing parameter d."""
    return float(H) - 0.5


def alpha_from_hurst_rs(H: float) -> float:
    """Convert Hurst exponent from R/S analysis to DFA scaling exponent α."""
    # For stationary processes, α ≈ H
    return float(H)

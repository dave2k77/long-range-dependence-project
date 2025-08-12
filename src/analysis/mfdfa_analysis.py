"""
Multifractal Detrended Fluctuation Analysis (MFDFA)

This module implements MFDFA for multifractal analysis, extending DFA
to detect multifractal scaling behavior through q-order moments.

References:
- Kantelhardt et al. (2002) for the original MFDFA methodology
- Ihlen (2012) for theoretical foundations and applications
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
import matplotlib.pyplot as plt


@dataclass
class MFDFASummary:
    """Summary of an MFDFA analysis."""
    hq: np.ndarray  # Generalized Hurst exponents
    tau: np.ndarray  # Mass exponents
    alpha: np.ndarray  # Singularity strengths
    f_alpha: np.ndarray  # Multifractal spectrum
    q_values: np.ndarray  # q values used
    scales: np.ndarray  # Scales used
    fq: np.ndarray  # Fluctuation functions F(q,s)
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            "hq": self.hq.copy(),
            "tau": self.tau.copy(),
            "alpha": self.alpha.copy(),
            "f_alpha": self.f_alpha.copy(),
            "q_values": self.q_values.copy(),
            "scales": self.scales.copy(),
            "fq": self.fq.copy(),
        }


def _validate_signal(y: ArrayLike) -> np.ndarray:
    """Validate and prepare input signal."""
    y_np = np.asarray(y, dtype=float).ravel()
    if y_np.size < 64:
        raise ValueError("Input series is too short for MFDFA analysis (need at least 64 points)")
    if not np.all(np.isfinite(y_np)):
        finite_mask = np.isfinite(y_np)
        if finite_mask.sum() < 32:
            raise ValueError("Too many non-finite values in input series for MFDFA analysis")
        idx = np.arange(len(y_np))
        y_np[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], y_np[finite_mask])
    return y_np


def _generate_log_scales(n: int,
                        min_scale: int = 8,
                        max_scale: Optional[int] = None,
                        num_scales: int = 25,
                        base: float = 2.0) -> np.ndarray:
    """Generate log-spaced scales for MFDFA analysis."""
    if max_scale is None:
        max_scale = max(min(int(n // 8), 512), min_scale + 1)
    if min_scale < 4:
        min_scale = 4
    if max_scale <= min_scale:
        max_scale = min_scale + 1
    
    logs = np.linspace(np.log(min_scale) / np.log(base), np.log(max_scale) / np.log(base), num_scales)
    scales = np.unique(np.clip(np.round(base ** logs).astype(int), min_scale, max_scale))
    
    if scales.size < 8:
        scales = np.unique(np.clip(np.round(np.linspace(min_scale, max_scale, 8)).astype(int), min_scale, max_scale))
    return scales


def _profile(y: np.ndarray) -> np.ndarray:
    """Calculate the profile (cumulative sum minus mean)."""
    return np.cumsum(y - np.mean(y))


def _poly_detrend(x: np.ndarray, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Detrend using polynomial fitting."""
    if order == 0:
        trend = np.mean(x)
        resid = x - trend
    else:
        coeffs = np.polyfit(np.arange(len(x)), x, order)
        trend = np.polyval(coeffs, np.arange(len(x)))
        resid = x - trend
    return trend, resid


def _fluctuation_for_scale_q(profile: np.ndarray, s: int, q: float, order: int = 1, 
                           use_reverse: bool = True, overlap: bool = False) -> float:
    """
    Calculate q-th order fluctuation function for a given scale.
    
    Parameters:
    - profile: integrated profile
    - s: scale (window size)
    - q: moment order
    - order: polynomial order for detrending
    - use_reverse: whether to use reverse segmentation
    - overlap: whether to use overlapping windows
    
    Returns:
    - F_q(s) value for this scale and q
    """
    n = len(profile)
    if s < 4 or s > n:
        return np.nan
    
    # Calculate number of segments
    if overlap:
        num_segments = n - s + 1
    else:
        num_segments = n // s
    
    if num_segments < 1:
        return np.nan
    
    f2_values = []
    
    for i in range(num_segments):
        if overlap:
            start_idx = i
            end_idx = i + s
        else:
            start_idx = i * s
            end_idx = start_idx + s
        
        if end_idx > n:
            break
            
        segment = profile[start_idx:end_idx]
        
        # Detrend
        trend, resid = _poly_detrend(segment, order)
        
        # Calculate variance
        f2 = np.mean(resid**2)
        
        if f2 > 0:
            f2_values.append(f2)
    
    if not f2_values:
        return np.nan
    
    # Calculate q-th order moment
    if q == 0:
        # Special case for q=0
        fq = np.exp(0.5 * np.mean(np.log(f2_values)))
    else:
        fq = np.mean(np.array(f2_values) ** (q/2)) ** (1/q)
    
    return fq


def mfdfa(y: ArrayLike,
          q_values: Optional[Sequence[float]] = None,
          scales: Optional[Sequence[int]] = None,
          order: int = 1,
          min_scale: int = 8,
          max_scale: Optional[int] = None,
          num_scales: int = 25,
          base: float = 2.0,
          use_reverse: bool = True,
          overlap: bool = False) -> Tuple[np.ndarray, np.ndarray, MFDFASummary]:
    """
    Perform Multifractal Detrended Fluctuation Analysis (MFDFA) on a 1D series.
    
    Parameters:
    - y: input series (1D)
    - q_values: moment orders to evaluate; if None, uses [-5, -3, -1, 0, 1, 3, 5]
    - scales: window sizes to evaluate; if None, automatically generated
    - order: polynomial order for detrending
    - min_scale, max_scale, num_scales, base: parameters for automatic scale generation
    - use_reverse: whether to use reverse segmentation
    - overlap: whether to use overlapping windows
    
    Returns:
    - scales (np.ndarray)
    - fq (np.ndarray): fluctuation functions F(q,s)
    - summary (MFDFASummary) with multifractal parameters
    """
    y_np = _validate_signal(y)
    
    # Set default q values if not provided
    if q_values is None:
        q_values = np.array([-5, -3, -1, 0, 1, 3, 5])
    else:
        q_values = np.asarray(q_values, dtype=float)
    
    # Generate scales if not provided
    if scales is None:
        scales = _generate_log_scales(len(y_np), min_scale=min_scale, max_scale=max_scale,
                                     num_scales=num_scales, base=base)
    else:
        scales = np.unique(np.asarray(scales, dtype=int))
        scales = scales[scales >= 4]
        if scales.size < 2:
            raise ValueError("Need at least two valid scales (>= 4)")
    
    # Calculate profile
    profile = _profile(y_np)
    
    # Calculate fluctuation functions for all q and s
    fq = np.zeros((len(q_values), len(scales)))
    
    for i, q in enumerate(q_values):
        for j, s in enumerate(scales):
            fq[i, j] = _fluctuation_for_scale_q(profile, int(s), q, order, use_reverse, overlap)
    
    # Remove any non-positive values for log fit
    valid = np.isfinite(fq) & (fq > 0)
    if valid.sum() < len(q_values) * 2:
        raise ValueError("Not enough valid fluctuation function values for MFDFA analysis")
    
    # Calculate generalized Hurst exponents h(q)
    hq = np.zeros(len(q_values))
    for i, q in enumerate(q_values):
        valid_q = valid[i, :]
        if valid_q.sum() >= 2:
            x = np.log(scales[valid_q])
            y = np.log(fq[i, valid_q])
            reg = stats.linregress(x, y)
            hq[i] = reg.slope
        else:
            hq[i] = np.nan
    
    # Calculate mass exponents tau(q)
    tau = q_values * hq - 1
    
    # Calculate singularity spectrum
    # alpha = dtau/dq
    alpha = np.gradient(tau, q_values)
    
    # f(alpha) = q * alpha - tau
    f_alpha = q_values * alpha - tau
    
    summary = MFDFASummary(
        hq=hq,
        tau=tau,
        alpha=alpha,
        f_alpha=f_alpha,
        q_values=q_values,
        scales=scales.astype(float),
        fq=fq.astype(float),
    )
    
    return scales.astype(float), fq.astype(float), summary


class MFDFAModel:
    """Object-oriented interface for MFDFA analysis."""
    
    def __init__(self, order: int = 1, use_reverse: bool = True, overlap: bool = False):
        self.order = order
        self.use_reverse = use_reverse
        self.overlap = overlap
        self.scales: Optional[np.ndarray] = None
        self.fq: Optional[np.ndarray] = None
        self.summary: Optional[MFDFASummary] = None
        self.is_fitted: bool = False
    
    def fit(self,
            y: ArrayLike,
            q_values: Optional[Sequence[float]] = None,
            scales: Optional[Sequence[int]] = None,
            min_scale: int = 8,
            max_scale: Optional[int] = None,
            num_scales: int = 25,
            base: float = 2.0) -> "MFDFAModel":
        scales_out, fq_out, summary = mfdfa(
            y,
            q_values=q_values,
            scales=scales,
            order=self.order,
            min_scale=min_scale,
            max_scale=max_scale,
            num_scales=num_scales,
            base=base,
            use_reverse=self.use_reverse,
            overlap=self.overlap,
        )
        self.scales = scales_out
        self.fq = fq_out
        self.summary = summary
        self.is_fitted = True
        return self
    
    def get_hq(self) -> np.ndarray:
        if not self.is_fitted or self.summary is None:
            raise ValueError("Model not fitted")
        return self.summary.hq.copy()
    
    def get_tau(self) -> np.ndarray:
        if not self.is_fitted or self.summary is None:
            raise ValueError("Model not fitted")
        return self.summary.tau.copy()
    
    def get_alpha(self) -> np.ndarray:
        if not self.is_fitted or self.summary is None:
            raise ValueError("Model not fitted")
        return self.summary.alpha.copy()
    
    def get_f_alpha(self) -> np.ndarray:
        if not self.is_fitted or self.summary is None:
            raise ValueError("Model not fitted")
        return self.summary.f_alpha.copy()
    
    def get_summary(self) -> MFDFASummary:
        if not self.is_fitted or self.summary is None:
            raise ValueError("Model not fitted")
        return self.summary
    
    def plot_loglog(self, ax: Optional[plt.Axes] = None, q_indices: Optional[Sequence[int]] = None) -> plt.Axes:
        """Plot log-log plots for selected q values."""
        if not self.is_fitted or self.scales is None or self.fq is None or self.summary is None:
            raise ValueError("Model not fitted")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        if q_indices is None:
            q_indices = [0, len(self.summary.q_values)//2, -1]  # First, middle, last q
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(q_indices)))
        
        for i, q_idx in enumerate(q_indices):
            if 0 <= q_idx < len(self.summary.q_values):
                q = self.summary.q_values[q_idx]
                hq = self.summary.hq[q_idx]
                
                # Plot data points
                valid = np.isfinite(self.fq[q_idx, :]) & (self.fq[q_idx, :] > 0)
                if valid.sum() > 0:
                    ax.loglog(self.scales[valid], self.fq[q_idx, valid], 'o', 
                             color=colors[i], alpha=0.7, label=f'q={q:.1f}, h(q)={hq:.3f}')
                    
                    # Plot regression line
                    x = np.log(self.scales[valid])
                    y = np.log(self.fq[q_idx, valid])
                    reg = stats.linregress(x, y)
                    yhat = reg.intercept + reg.slope * x
                    ax.plot(self.scales[valid], np.exp(yhat), '-', color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Scale s')
        ax.set_ylabel('F(q,s)')
        ax.set_title('MFDFA: Fluctuation Functions')
        ax.grid(True, which='both', ls='--', alpha=0.4)
        ax.legend()
        return ax
    
    def plot_multifractal_spectrum(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot the multifractal spectrum f(α) vs α."""
        if not self.is_fitted or self.summary is None:
            raise ValueError("Model not fitted")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Sort by alpha for smooth curve
        valid = np.isfinite(self.summary.alpha) & np.isfinite(self.summary.f_alpha)
        if valid.sum() > 0:
            alpha_sorted = self.summary.alpha[valid]
            f_alpha_sorted = self.summary.f_alpha[valid]
            
            # Sort by alpha
            sort_idx = np.argsort(alpha_sorted)
            alpha_plot = alpha_sorted[sort_idx]
            f_alpha_plot = f_alpha_sorted[sort_idx]
            
            ax.plot(alpha_plot, f_alpha_plot, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('α (Singularity Strength)')
            ax.set_ylabel('f(α) (Multifractal Spectrum)')
            ax.set_title('Multifractal Spectrum')
            ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_hq(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot generalized Hurst exponents h(q) vs q."""
        if not self.is_fitted or self.summary is None:
            raise ValueError("Model not fitted")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        valid = np.isfinite(self.summary.hq)
        if valid.sum() > 0:
            ax.plot(self.summary.q_values[valid], self.summary.hq[valid], 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('q')
            ax.set_ylabel('h(q) (Generalized Hurst Exponent)')
            ax.set_title('Generalized Hurst Exponents')
            ax.grid(True, alpha=0.3)
        
        return ax


def hurst_from_mfdfa(hq: np.ndarray, q_values: np.ndarray, q_target: float = 2.0) -> float:
    """Extract Hurst exponent from MFDFA results at a specific q value."""
    if q_target in q_values:
        idx = np.where(q_values == q_target)[0][0]
        return float(hq[idx])
    else:
        # Interpolate to get h(q=2)
        valid = np.isfinite(hq)
        if valid.sum() >= 2:
            return float(np.interp(q_target, q_values[valid], hq[valid]))
        else:
            return np.nan


def alpha_from_mfdfa(alpha: np.ndarray, q_values: np.ndarray, q_target: float = 2.0) -> float:
    """Extract scaling exponent from MFDFA results at a specific q value."""
    if q_target in q_values:
        idx = np.where(q_values == q_target)[0][0]
        return float(alpha[idx])
    else:
        # Interpolate to get α(q=2)
        valid = np.isfinite(alpha)
        if valid.sum() >= 2:
            return float(np.interp(q_target, q_values[valid], alpha[valid]))
        else:
            return np.nan

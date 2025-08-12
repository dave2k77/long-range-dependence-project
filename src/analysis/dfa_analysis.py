"""
Detrended Fluctuation Analysis (DFA)

This module implements DFA for long-range dependence analysis.
It provides a functional API and an object-oriented API, along with plotting utilities.

References:
- Peng et al. (1994, 1995) for DFA methodology
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
import matplotlib.pyplot as plt


Number = Union[int, float]


@dataclass
class DFASummary:
    """Summary of a DFA fit."""
    alpha: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float
    scales: np.ndarray
    flucts: np.ndarray

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "intercept": float(self.intercept),
            "rvalue": float(self.rvalue),
            "pvalue": float(self.pvalue),
            "stderr": float(self.stderr),
            "scales": self.scales.copy(),
            "flucts": self.flucts.copy(),
        }


def _validate_signal(y: ArrayLike) -> np.ndarray:
    y_np = np.asarray(y, dtype=float).ravel()
    if y_np.size < 16:
        raise ValueError("Input series is too short for DFA (need at least 16 points)")
    if not np.all(np.isfinite(y_np)):
        # Replace non-finites by interpolation where possible
        finite_mask = np.isfinite(y_np)
        if finite_mask.sum() < 8:
            raise ValueError("Too many non-finite values in input series for DFA")
        idx = np.arange(len(y_np))
        y_np[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], y_np[finite_mask])
    return y_np


def _generate_log_scales(n: int,
                         min_scale: int = 4,
                         max_scale: Optional[int] = None,
                         num_scales: int = 20,
                         base: float = 2.0) -> np.ndarray:
    if max_scale is None:
        max_scale = max(min(int(n // 4), 1024), min_scale + 1)
    if min_scale < 2:
        min_scale = 2
    if max_scale <= min_scale:
        max_scale = min_scale + 1
    # Log-spaced on [min_scale, max_scale]
    logs = np.linspace(np.log(min_scale) / np.log(base), np.log(max_scale) / np.log(base), num_scales)
    scales = np.unique(np.clip(np.round(base ** logs).astype(int), min_scale, max_scale))
    # Ensure at least 6 distinct scales
    if scales.size < 6:
        scales = np.unique(np.clip(np.round(np.linspace(min_scale, max_scale, 6)).astype(int), min_scale, max_scale))
    return scales


def _profile(y: np.ndarray) -> np.ndarray:
    y_centered = y - np.mean(y)
    return np.cumsum(y_centered)


def _poly_detrend(x: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit polynomial of given order and return (trend, residuals).
    """
    t = np.arange(len(x), dtype=float)
    # Use polyfit; for small windows, fall back to lower order if needed
    o = min(order, len(x) - 1)
    if o <= 0:
        trend = np.full_like(x, fill_value=x.mean())
        return trend, x - trend
    coeffs = np.polyfit(t, x, o)
    trend = np.polyval(coeffs, t)
    return trend, x - trend


def _fluctuation_for_scale(profile: np.ndarray, s: int, order: int, use_reverse: bool, overlap: bool) -> float:
    n = len(profile)
    if s < 2:
        return np.nan

    seg_starts: List[int] = []
    if overlap:
        step = max(1, s // 2)
        seg_starts = list(range(0, n - s + 1, step))
    else:
        seg_starts = list(range(0, (n // s) * s, s))

    segments: List[np.ndarray] = []
    for start in seg_starts:
        segments.append(profile[start:start + s])

    if use_reverse:
        # Start from the end to utilize remainder
        rev_profile = profile[::-1]
        if overlap:
            rev_starts = list(range(0, n - s + 1, max(1, s // 2)))
        else:
            rev_starts = list(range(0, (n // s) * s, s))
        for start in rev_starts:
            segments.append(rev_profile[start:start + s])

    if len(segments) == 0:
        return np.nan

    variances = []
    for seg in segments:
        trend, resid = _poly_detrend(seg, order)
        v = np.mean(resid ** 2)
        variances.append(v)

    F_s = np.sqrt(np.mean(variances))
    return float(F_s)


def dfa(y: ArrayLike,
        scales: Optional[Sequence[int]] = None,
        order: int = 1,
        overlap: bool = False,
        min_scale: int = 4,
        max_scale: Optional[int] = None,
        num_scales: int = 20,
        base: float = 2.0) -> Tuple[np.ndarray, np.ndarray, DFASummary]:
    """
    Perform Detrended Fluctuation Analysis (DFA) on a 1D series.

    Parameters:
    - y: input series (1D)
    - scales: window sizes to evaluate; if None, automatically generated
    - order: polynomial order for detrending (1 = linear DFA1)
    - overlap: if True, use overlapping windows (step = s/2)
    - min_scale, max_scale, num_scales, base: parameters for automatic scale generation

    Returns:
    - scales (np.ndarray)
    - flucts (np.ndarray): F(s) values
    - summary (DFASummary) with slope alpha and regression stats
    """
    y_np = _validate_signal(y)
    prof = _profile(y_np)

    if scales is None:
        scales = _generate_log_scales(len(y_np), min_scale=min_scale, max_scale=max_scale,
                                      num_scales=num_scales, base=base)
    else:
        scales = np.unique(np.asarray(scales, dtype=int))
        scales = scales[scales >= 2]
        if scales.size < 2:
            raise ValueError("Need at least two valid scales (>= 2)")

    flucts = np.zeros_like(scales, dtype=float)
    for i, s in enumerate(scales):
        flucts[i] = _fluctuation_for_scale(prof, int(s), order=order, use_reverse=True, overlap=overlap)

    # Remove any non-positive F(s) values for log fit
    valid = np.isfinite(flucts) & (flucts > 0)
    if valid.sum() < 2:
        raise ValueError("Not enough valid fluctuation values to estimate scaling exponent")

    x = np.log(scales[valid])
    ylog = np.log(flucts[valid])
    reg = stats.linregress(x, ylog)

    alpha = reg.slope
    intercept = reg.intercept

    summary = DFASummary(
        alpha=alpha,
        intercept=intercept,
        rvalue=reg.rvalue,
        pvalue=reg.pvalue,
        stderr=reg.stderr,
        scales=scales.astype(float),
        flucts=flucts.astype(float),
    )
    return scales.astype(float), flucts.astype(float), summary


class DFAModel:
    """Object-oriented interface for DFA."""

    def __init__(self, order: int = 1, overlap: bool = False):
        self.order = order
        self.overlap = overlap
        self.scales: Optional[np.ndarray] = None
        self.flucts: Optional[np.ndarray] = None
        self.summary: Optional[DFASummary] = None
        self.is_fitted: bool = False

    def fit(self,
            y: ArrayLike,
            scales: Optional[Sequence[int]] = None,
            min_scale: int = 4,
            max_scale: Optional[int] = None,
            num_scales: int = 20,
            base: float = 2.0) -> "DFAModel":
        scales_out, flucts_out, summary = dfa(
            y,
            scales=scales,
            order=self.order,
            overlap=self.overlap,
            min_scale=min_scale,
            max_scale=max_scale,
            num_scales=num_scales,
            base=base,
        )
        self.scales = scales_out
        self.flucts = flucts_out
        self.summary = summary
        self.is_fitted = True
        return self

    def get_alpha(self) -> float:
        if not self.is_fitted or self.summary is None:
            raise ValueError("Model not fitted")
        return float(self.summary.alpha)

    def get_summary(self) -> DFASummary:
        if not self.is_fitted or self.summary is None:
            raise ValueError("Model not fitted")
        return self.summary

    def plot_loglog(self, ax: Optional[plt.Axes] = None, annotate: bool = True) -> plt.Axes:
        if not self.is_fitted or self.scales is None or self.flucts is None or self.summary is None:
            raise ValueError("Model not fitted")
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(self.scales, self.flucts, 'o', label='F(s)')
        # regression line over range
        x = np.log(self.scales)
        yhat = self.summary.intercept + self.summary.alpha * x
        ax.plot(self.scales, np.exp(yhat), '-', label=f"alpha={self.summary.alpha:.3f}")
        ax.set_xlabel('Scale s')
        ax.set_ylabel('F(s)')
        ax.grid(True, which='both', ls='--', alpha=0.4)
        ax.legend()
        if annotate:
            ax.annotate(f"alpha={self.summary.alpha:.3f}\nr={self.summary.rvalue:.3f}",
                        xy=(0.05, 0.95), xycoords='axes fraction', va='top')
        return ax


def hurst_from_dfa_alpha(alpha: float) -> float:
    """
    Convert DFA scaling exponent alpha to Hurst exponent H.
    For stationary fGn-like processes, H ~= alpha.
    For integrated processes (fBm), alpha ~= H + 1.
    Here we return alpha as a proxy for H in typical DFA1 usage on stationary series.
    """
    return float(alpha)


def d_from_hurst(H: float) -> float:
    """Convert Hurst exponent to fractional differencing parameter d using H = d + 0.5."""
    return float(H) - 0.5

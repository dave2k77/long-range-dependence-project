"""
Higuchi's Fractal Dimension Analysis Module

This module implements Higuchi's method for estimating the fractal dimension of time series data.
Higuchi's method is particularly useful for analyzing the complexity and irregularity of time series
and can provide complementary information to other fractal analysis methods.

References:
- Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory.
  Physica D: Nonlinear Phenomena, 31(2), 277-283.
- Esteller, R., Vachtsevanos, G., Echauz, J., & Litt, B. (2001). A comparison of fractal
  dimension algorithms using synthetic and experimental data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings


@dataclass
class HiguchiSummary:
    """Container for Higuchi fractal dimension analysis results."""
    fractal_dimension: float
    std_error: float
    r_squared: float
    p_value: float
    k_values: np.ndarray
    l_values: np.ndarray
    log_k: np.ndarray
    log_l: np.ndarray
    slope: float
    intercept: float
    residuals: np.ndarray
    k_range: Tuple[int, int]
    n_points: int
    method: str = 'higuchi'
    additional_info: Optional[Dict[str, Any]] = None


def higuchi_fractal_dimension(y: np.ndarray,
                             k_max: Optional[int] = None,
                             k_min: int = 2,
                             optimize_k: bool = True,
                             method: str = 'linear') -> Tuple[np.ndarray, np.ndarray, HiguchiSummary]:
    """
    Calculate Higuchi's fractal dimension of a time series.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data (1D array)
    k_max : Optional[int]
        Maximum value of k (time interval). If None, will be set to len(y)//4
    k_min : int
        Minimum value of k (time interval). Must be >= 2
    optimize_k : bool
        Whether to optimize the k range for better linearity
    method : str
        Method for calculating the fractal dimension: 'linear' or 'robust'
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, HiguchiSummary]
        k_values: Array of k values used
        l_values: Array of corresponding L(k) values
        summary: HiguchiSummary object with results
        
    Raises:
    -------
    ValueError
        If input parameters are invalid
    """
    if len(y) < 10:
        raise ValueError("Time series must have at least 10 points")
    
    if k_min < 2:
        raise ValueError("k_min must be >= 2")
    
    if k_max is None:
        k_max = len(y) // 4
    
    if k_max <= k_min:
        raise ValueError("k_max must be > k_min")
    
    if k_max > len(y) // 2:
        warnings.warn("k_max is large relative to series length, results may be unreliable")
    
    # Generate k values
    k_values = np.arange(k_min, k_max + 1)
    
    # Calculate L(k) for each k
    l_values = _calculate_l_values(y, k_values)
    
    # Optimize k range if requested
    if optimize_k and len(k_values) > 5:
        k_values, l_values = _optimize_k_range(k_values, l_values)
    
    # Calculate fractal dimension
    log_k = np.log(k_values)
    log_l = np.log(l_values)
    
    if method == 'robust':
        slope, intercept, r_squared, p_value, std_error = _robust_regression(log_k, log_l)
    else:
        slope, intercept, r_squared, p_value, std_error = _linear_regression(log_k, log_l)
    
    # Fractal dimension is the negative of the slope
    fractal_dimension = -slope
    
    # Calculate residuals
    predicted = slope * log_k + intercept
    residuals = log_l - predicted
    
    # Create summary
    summary = HiguchiSummary(
        fractal_dimension=fractal_dimension,
        std_error=std_error,
        r_squared=r_squared,
        p_value=p_value,
        k_values=k_values,
        l_values=l_values,
        log_k=log_k,
        log_l=log_l,
        slope=slope,
        intercept=intercept,
        residuals=residuals,
        k_range=(k_values[0], k_values[-1]),
        n_points=len(y),
        method='higuchi',
        additional_info={
            'optimization_method': method,
            'k_optimized': optimize_k,
            'n_k_values': len(k_values)
        }
    )
    
    return k_values, l_values, summary


def _calculate_l_values(y: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    """
    Calculate L(k) values for Higuchi's method.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    k_values : np.ndarray
        Array of k values
        
    Returns:
    --------
    np.ndarray
        Array of L(k) values
    """
    n = len(y)
    l_values = np.zeros(len(k_values))
    
    for i, k in enumerate(k_values):
        l_sum = 0
        
        for m in range(k):
            # Calculate L(m,k) for this m
            l_mk = 0
            n_m = (n - m - 1) // k
            
            if n_m > 0:
                for j in range(1, n_m + 1):
                    l_mk += abs(y[m + j * k] - y[m + (j - 1) * k])
                
                # Normalize by k^2 and (n-1)/k^2
                l_mk = l_mk * (n - 1) / (k * k * n_m)
            
            l_sum += l_mk
        
        l_values[i] = l_sum / k
    
    return l_values


def _optimize_k_range(k_values: np.ndarray, l_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimize the k range to improve linearity of log(L) vs log(k).
    
    Parameters:
    -----------
    k_values : np.ndarray
        Original k values
    l_values : np.ndarray
        Corresponding L(k) values
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Optimized k values and L(k) values
    """
    if len(k_values) < 6:
        return k_values, l_values
    
    log_k = np.log(k_values)
    log_l = np.log(l_values)
    
    best_r2 = 0
    best_start = 0
    best_end = len(k_values)
    
    # Try different ranges of k values
    for start in range(len(k_values) - 4):
        for end in range(start + 5, len(k_values) + 1):
            k_subset = k_values[start:end]
            l_subset = l_values[start:end]
            
            if len(k_subset) < 5:
                continue
            
            try:
                _, _, r2, _, _ = _linear_regression(np.log(k_subset), np.log(l_subset))
                if r2 > best_r2:
                    best_r2 = r2
                    best_start = start
                    best_end = end
            except:
                continue
    
    # Use the range with best R² if it's significantly better
    if best_r2 > 0.95:  # Only use if very good fit
        return k_values[best_start:best_end], l_values[best_start:best_end]
    
    return k_values, l_values


def _linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Perform linear regression and return statistics.
    
    Parameters:
    -----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
        
    Returns:
    --------
    Tuple[float, float, float, float, float]
        slope, intercept, r_squared, p_value, std_error
    """
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Invalid input for regression")
    
    # Linear regression
    slope, intercept, r_value, p_value, std_error = stats.linregress(x, y)
    r_squared = r_value ** 2
    
    return slope, intercept, r_squared, p_value, std_error


def _robust_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Perform robust linear regression using Huber's method.
    
    Parameters:
    -----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
        
    Returns:
    --------
    Tuple[float, float, float, float, float]
        slope, intercept, r_squared, p_value, std_error
    """
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Invalid input for regression")
    
    # Add constant term for intercept
    X = np.column_stack([x, np.ones_like(x)])
    
    # Robust regression using Huber's method
    try:
        from sklearn.linear_model import HuberRegressor
        model = HuberRegressor(epsilon=1.35, max_iter=100)
        model.fit(X, y)
        slope, intercept = model.coef_[0], model.intercept_
    except ImportError:
        # Fallback to regular linear regression
        slope, intercept, _, _, _ = stats.linregress(x, y)
    
    # Calculate statistics
    predicted = slope * x + intercept
    residuals = y - predicted
    
    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Standard error
    n = len(x)
    if n > 2:
        std_error = np.sqrt(ss_res / (n - 2))
    else:
        std_error = np.inf
    
    # P-value (approximate)
    if n > 2 and std_error > 0:
        t_stat = slope / (std_error / np.sqrt(np.sum((x - np.mean(x)) ** 2)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    else:
        p_value = 1.0
    
    return slope, intercept, r_squared, p_value, std_error


def estimate_higuchi_dimension(y: np.ndarray,
                              k_max: Optional[int] = None,
                              k_min: int = 2,
                              optimize_k: bool = True) -> float:
    """
    Convenience function to get just the fractal dimension estimate.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    k_max : Optional[int]
        Maximum k value
    k_min : int
        Minimum k value
    optimize_k : bool
        Whether to optimize k range
        
    Returns:
    --------
    float
        Estimated fractal dimension
    """
    _, _, summary = higuchi_fractal_dimension(y, k_max, k_min, optimize_k)
    return summary.fractal_dimension


def higuchi_analysis_batch(time_series_list: List[np.ndarray],
                          names: Optional[List[str]] = None,
                          k_max: Optional[int] = None,
                          k_min: int = 2,
                          optimize_k: bool = True) -> Dict[str, HiguchiSummary]:
    """
    Perform Higuchi analysis on multiple time series.
    
    Parameters:
    -----------
    time_series_list : List[np.ndarray]
        List of time series arrays
    names : Optional[List[str]]
        Names for the time series
    k_max : Optional[int]
        Maximum k value
    k_min : int
        Minimum k value
    optimize_k : bool
        Whether to optimize k range
        
    Returns:
    --------
    Dict[str, HiguchiSummary]
        Dictionary mapping names to HiguchiSummary objects
    """
    if names is None:
        names = [f"series_{i}" for i in range(len(time_series_list))]
    
    if len(names) != len(time_series_list):
        raise ValueError("Names list must have same length as time series list")
    
    results = {}
    
    for name, series in zip(names, time_series_list):
        try:
            _, _, summary = higuchi_fractal_dimension(series, k_max, k_min, optimize_k)
            results[name] = summary
        except Exception as e:
            warnings.warn(f"Failed to analyze {name}: {e}")
            continue
    
    return results


def validate_higuchi_results(summary: HiguchiSummary,
                           min_r2: float = 0.8,
                           max_std_error: float = 0.1) -> Dict[str, Any]:
    """
    Validate Higuchi analysis results.
    
    Parameters:
    -----------
    summary : HiguchiSummary
        Higuchi analysis results
    min_r2 : float
        Minimum acceptable R² value
    max_std_error : float
        Maximum acceptable standard error
        
    Returns:
    --------
    Dict[str, Any]
        Validation results
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'quality_score': 1.0,
        'issues': []
    }
    
    # Check R²
    if summary.r_squared < min_r2:
        validation['is_valid'] = False
        validation['issues'].append(f"Low R² ({summary.r_squared:.3f} < {min_r2})")
        validation['quality_score'] *= 0.5
    
    # Check standard error
    if summary.std_error > max_std_error:
        validation['warnings'].append(f"High standard error ({summary.std_error:.3f} > {max_std_error})")
        validation['quality_score'] *= 0.8
    
    # Check number of k values
    if len(summary.k_values) < 5:
        validation['warnings'].append(f"Few k values ({len(summary.k_values)})")
        validation['quality_score'] *= 0.9
    
    # Check k range
    k_range = summary.k_range[1] - summary.k_range[0]
    if k_range < 10:
        validation['warnings'].append(f"Small k range ({k_range})")
        validation['quality_score'] *= 0.9
    
    # Check residuals
    if np.any(np.abs(summary.residuals) > 2 * np.std(summary.residuals)):
        validation['warnings'].append("Large residuals detected")
        validation['quality_score'] *= 0.95
    
    validation['quality_score'] = max(0.0, min(1.0, validation['quality_score']))
    
    return validation

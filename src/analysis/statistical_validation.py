"""
Statistical Validation Module for Long-Range Dependence Analysis

This module provides comprehensive statistical validation tools for long-range dependence analysis:
- Hypothesis testing for long-range dependence vs. no long-range dependence
- Cross-validation mechanisms (split-sample, bootstrap, Monte Carlo)
- Bootstrap sampling for confidence intervals
- Monte Carlo significance tests
- Robustness testing and sensitivity analysis

References:
- Beran (1994) for theoretical foundations of LRD testing
- Taqqu et al. (1995) for practical testing approaches
- Giraitis et al. (2003) for modern testing methods
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

from .dfa_analysis import dfa, DFASummary
from .rs_analysis import rs_analysis, RSSummary
from .wavelet_analysis import wavelet_leaders_estimation, WaveletSummary
from .spectral_analysis import whittle_mle, SpectralSummary


@dataclass
class HypothesisTestResult:
    """Container for hypothesis test results."""
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    critical_value: float
    significance_level: float
    decision: str  # 'reject' or 'fail_to_reject'
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class CrossValidationResult:
    """Container for cross-validation results."""
    method: str
    n_folds: int
    hurst_estimates: List[float]
    confidence_intervals: List[Tuple[float, float]]
    mean_hurst: float
    std_hurst: float
    cv_score: float  # Coefficient of variation
    stability_score: float  # Measure of estimate stability
    additional_metrics: Optional[Dict[str, Any]] = None


@dataclass
class BootstrapResult:
    """Container for bootstrap analysis results."""
    original_estimate: float
    bootstrap_estimates: np.ndarray
    mean_estimate: float
    std_estimate: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    bias: float
    standard_error: float
    additional_stats: Optional[Dict[str, Any]] = None


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo significance test results."""
    test_statistic: float
    null_distribution: np.ndarray
    p_value: float
    significance_level: float
    n_simulations: int
    decision: str
    effect_size: float
    power: Optional[float] = None
    additional_stats: Optional[Dict[str, Any]] = None


class StatisticalValidator:
    """
    Comprehensive statistical validation for long-range dependence analysis.
    
    This class provides methods for:
    - Hypothesis testing (LRD vs. no LRD)
    - Cross-validation (split-sample, bootstrap, Monte Carlo)
    - Bootstrap confidence intervals
    - Monte Carlo significance tests
    - Robustness testing
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the statistical validator.
        
        Parameters:
        -----------
        random_state : Optional[int]
            Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def test_lrd_hypothesis(self, 
                           y: np.ndarray,
                           method: str = 'dfa',
                           alpha: float = 0.05,
                           h0_hurst: float = 0.5,
                           alternative: str = 'greater') -> HypothesisTestResult:
        """
        Test the hypothesis of long-range dependence vs. no long-range dependence.
        
        Parameters:
        -----------
        y : np.ndarray
            Time series data
        method : str
            Analysis method ('dfa', 'rs', 'wavelet', 'spectral')
        alpha : float
            Significance level
        h0_hurst : float
            Null hypothesis Hurst exponent (default: 0.5 for no LRD)
        alternative : str
            Alternative hypothesis ('greater', 'less', 'two-sided')
            
        Returns:
        --------
        HypothesisTestResult
            Test results and decision
        """
        # Estimate Hurst exponent
        hurst_est, std_error = self._estimate_hurst_with_error(y, method)
        
        # Perform t-test
        t_stat = (hurst_est - h0_hurst) / std_error
        df = len(y) - 1  # Degrees of freedom
        
        if alternative == 'greater':
            p_value = 1 - stats.t.cdf(t_stat, df)
            critical_value = stats.t.ppf(1 - alpha, df)
        elif alternative == 'less':
            p_value = stats.t.cdf(t_stat, df)
            critical_value = stats.t.ppf(alpha, df)
        else:  # two-sided
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            critical_value = stats.t.ppf(1 - alpha/2, df)
        
        # Decision
        if alternative == 'greater':
            decision = 'reject' if t_stat > critical_value else 'fail_to_reject'
        elif alternative == 'less':
            decision = 'reject' if t_stat < critical_value else 'fail_to_reject'
        else:
            decision = 'reject' if abs(t_stat) > critical_value else 'fail_to_reject'
        
        # Effect size (Cohen's d)
        effect_size = (hurst_est - h0_hurst) / std_error
        
        # Confidence interval
        ci_lower = hurst_est - stats.t.ppf(1 - alpha/2, df) * std_error
        ci_upper = hurst_est + stats.t.ppf(1 - alpha/2, df) * std_error
        
        return HypothesisTestResult(
            test_name=f"LRD Hypothesis Test ({method.upper()})",
            null_hypothesis=f"H = {h0_hurst} (no long-range dependence)",
            alternative_hypothesis=f"H ≠ {h0_hurst} (long-range dependence present)",
            test_statistic=t_stat,
            p_value=p_value,
            critical_value=critical_value,
            significance_level=alpha,
            decision=decision,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            additional_info={
                'hurst_estimate': hurst_est,
                'std_error': std_error,
                'method': method,
                'sample_size': len(y)
            }
        )
    
    def split_sample_validation(self,
                               y: np.ndarray,
                               method: str = 'dfa',
                               n_folds: int = 5,
                               test_size: float = 0.2) -> CrossValidationResult:
        """
        Perform split-sample cross-validation for Hurst exponent estimation.
        
        Parameters:
        -----------
        y : np.ndarray
            Time series data
        method : str
            Analysis method ('dfa', 'rs', 'wavelet', 'spectral')
        n_folds : int
            Number of cross-validation folds
        test_size : float
            Proportion of data for testing
            
        Returns:
        --------
        CrossValidationResult
            Cross-validation results
        """
        n = len(y)
        test_length = int(n * test_size)
        hurst_estimates = []
        confidence_intervals = []
        
        for i in range(n_folds):
            # Create different train/test splits
            start_idx = (i * test_length) % n
            end_idx = start_idx + test_length
            
            if end_idx > n:
                # Handle wrap-around
                test_indices = list(range(start_idx, n)) + list(range(0, end_idx - n))
            else:
                test_indices = list(range(start_idx, end_idx))
            
            train_indices = [j for j in range(n) if j not in test_indices]
            
            # Estimate on training data
            train_data = y[train_indices]
            hurst_est, std_error = self._estimate_hurst_with_error(train_data, method)
            
            hurst_estimates.append(hurst_est)
            
            # Approximate confidence interval
            ci_lower = hurst_est - 1.96 * std_error
            ci_upper = hurst_est + 1.96 * std_error
            confidence_intervals.append((ci_lower, ci_upper))
        
        # Calculate summary statistics
        hurst_estimates = np.array(hurst_estimates)
        mean_hurst = np.mean(hurst_estimates)
        std_hurst = np.std(hurst_estimates)
        cv_score = std_hurst / mean_hurst if mean_hurst != 0 else np.inf
        
        # Stability score (inverse of coefficient of variation)
        stability_score = 1 / (1 + cv_score) if cv_score != np.inf else 0
        
        return CrossValidationResult(
            method=f"Split-Sample CV ({method.upper()})",
            n_folds=n_folds,
            hurst_estimates=hurst_estimates.tolist(),
            confidence_intervals=confidence_intervals,
            mean_hurst=mean_hurst,
            std_hurst=std_hurst,
            cv_score=cv_score,
            stability_score=stability_score,
            additional_metrics={
                'test_size': test_size,
                'min_estimate': np.min(hurst_estimates),
                'max_estimate': np.max(hurst_estimates),
                'range': np.max(hurst_estimates) - np.min(hurst_estimates)
            }
        )
    
    def bootstrap_validation(self,
                            y: np.ndarray,
                            method: str = 'dfa',
                            n_bootstrap: int = 1000,
                            confidence_level: float = 0.95,
                            block_size: Optional[int] = None) -> BootstrapResult:
        """
        Perform bootstrap validation for Hurst exponent estimation.
        
        Parameters:
        -----------
        y : np.ndarray
            Time series data
        method : str
            Analysis method ('dfa', 'rs', 'wavelet', 'spectral')
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level for intervals
        block_size : Optional[int]
            Block size for block bootstrap (None for iid bootstrap)
            
        Returns:
        --------
        BootstrapResult
            Bootstrap analysis results
        """
        # Original estimate
        original_estimate, _ = self._estimate_hurst_with_error(y, method)
        
        # Generate bootstrap samples
        bootstrap_estimates = []
        
        for _ in range(n_bootstrap):
            if block_size is None:
                # IID bootstrap (may not preserve time series structure)
                bootstrap_sample = np.random.choice(y, size=len(y), replace=True)
            else:
                # Block bootstrap (preserves local structure)
                bootstrap_sample = self._block_bootstrap(y, block_size)
            
            try:
                hurst_est, _ = self._estimate_hurst_with_error(bootstrap_sample, method)
                bootstrap_estimates.append(hurst_est)
            except (ValueError, RuntimeWarning):
                # Skip failed estimates
                continue
        
        if len(bootstrap_estimates) < n_bootstrap // 2:
            raise ValueError("Too many bootstrap estimates failed")
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        # Calculate bootstrap statistics
        mean_estimate = np.mean(bootstrap_estimates)
        std_estimate = np.std(bootstrap_estimates)
        bias = mean_estimate - original_estimate
        standard_error = std_estimate
        
        # Confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
        
        return BootstrapResult(
            original_estimate=original_estimate,
            bootstrap_estimates=bootstrap_estimates,
            mean_estimate=mean_estimate,
            std_estimate=std_estimate,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=confidence_level,
            bias=bias,
            standard_error=standard_error,
            additional_stats={
                'n_successful_bootstrap': len(bootstrap_estimates),
                'bootstrap_method': 'block' if block_size else 'iid',
                'block_size': block_size,
                'skewness': stats.skew(bootstrap_estimates),
                'kurtosis': stats.kurtosis(bootstrap_estimates)
            }
        )
    
    def monte_carlo_significance_test(self,
                                     y: np.ndarray,
                                     method: str = 'dfa',
                                     n_simulations: int = 1000,
                                     alpha: float = 0.05,
                                     null_model: str = 'iid') -> MonteCarloResult:
        """
        Perform Monte Carlo significance test for long-range dependence.
        
        Parameters:
        -----------
        y : np.ndarray
            Time series data
        method : str
            Analysis method ('dfa', 'rs', 'wavelet', 'spectral')
        n_simulations : int
            Number of Monte Carlo simulations
        alpha : float
            Significance level
        null_model : str
            Null model type ('iid', 'ar1', 'ma1')
            
        Returns:
        --------
        MonteCarloResult
            Monte Carlo test results
        """
        # Observed test statistic
        observed_hurst, _ = self._estimate_hurst_with_error(y, method)
        
        # Generate null distribution
        null_distribution = []
        
        for _ in range(n_simulations):
            # Generate null model data
            if null_model == 'iid':
                null_data = np.random.normal(0, 1, len(y))
            elif null_model == 'ar1':
                phi = 0.5  # AR(1) parameter
                null_data = self._generate_ar1_process(len(y), phi)
            elif null_model == 'ma1':
                theta = 0.5  # MA(1) parameter
                null_data = self._generate_ma1_process(len(y), theta)
            else:
                raise ValueError(f"Unknown null model: {null_model}")
            
            try:
                hurst_est, _ = self._estimate_hurst_with_error(null_data, method)
                null_distribution.append(hurst_est)
            except (ValueError, RuntimeWarning):
                continue
        
        if len(null_distribution) < n_simulations // 2:
            raise ValueError("Too many null model estimates failed")
        
        null_distribution = np.array(null_distribution)
        
        # Calculate p-value
        if observed_hurst > np.median(null_distribution):
            # One-sided test for positive LRD
            p_value = np.mean(null_distribution >= observed_hurst)
        else:
            # One-sided test for negative LRD
            p_value = np.mean(null_distribution <= observed_hurst)
        
        # Decision
        decision = 'reject' if p_value < alpha else 'fail_to_reject'
        
        # Effect size (how many standard deviations from null mean)
        null_mean = np.mean(null_distribution)
        null_std = np.std(null_distribution)
        effect_size = (observed_hurst - null_mean) / null_std if null_std > 0 else 0
        
        return MonteCarloResult(
            test_statistic=observed_hurst,
            null_distribution=null_distribution,
            p_value=p_value,
            significance_level=alpha,
            n_simulations=len(null_distribution),
            decision=decision,
            effect_size=effect_size,
            additional_stats={
                'null_model': null_model,
                'null_mean': null_mean,
                'null_std': null_std,
                'null_median': np.median(null_distribution)
            }
        )
    
    def robustness_test(self,
                       y: np.ndarray,
                       methods: List[str] = None,
                       parameter_ranges: Dict[str, List] = None) -> Dict[str, Any]:
        """
        Perform robustness testing across different methods and parameters.
        
        Parameters:
        -----------
        y : np.ndarray
            Time series data
        methods : List[str]
            List of analysis methods to test
        parameter_ranges : Dict[str, List]
            Parameter ranges to test for each method
            
        Returns:
        --------
        Dict[str, Any]
            Robustness test results
        """
        if methods is None:
            methods = ['dfa', 'rs', 'wavelet', 'spectral']
        
        if parameter_ranges is None:
            parameter_ranges = {
                'dfa': {'order': [1, 2, 3]},
                'rs': {'min_scale': [4, 8, 16]},
                'wavelet': {'wavelet': ['morlet', 'haar']},
                'spectral': {'method': ['whittle', 'periodogram']}
            }
        
        results = {}
        
        for method in methods:
            method_results = {}
            
            if method in parameter_ranges:
                for param_name, param_values in parameter_ranges[method].items():
                    param_results = []
                    
                    for param_value in param_values:
                        try:
                            hurst_est, std_error = self._estimate_hurst_with_error(
                                y, method, **{param_name: param_value}
                            )
                            param_results.append({
                                'parameter_value': param_value,
                                'hurst_estimate': hurst_est,
                                'std_error': std_error
                            })
                        except Exception as e:
                            param_results.append({
                                'parameter_value': param_value,
                                'error': str(e)
                            })
                    
                    method_results[param_name] = param_results
            
            # Also test with default parameters
            try:
                hurst_est, std_error = self._estimate_hurst_with_error(y, method)
                method_results['default'] = {
                    'hurst_estimate': hurst_est,
                    'std_error': std_error
                }
            except Exception as e:
                method_results['default'] = {'error': str(e)}
            
            results[method] = method_results
        
        return results
    
    def _estimate_hurst_with_error(self,
                                  y: np.ndarray,
                                  method: str,
                                  **kwargs) -> Tuple[float, float]:
        """
        Estimate Hurst exponent with standard error for a given method.
        
        Parameters:
        -----------
        y : np.ndarray
            Time series data
        method : str
            Analysis method
        **kwargs
            Method-specific parameters
            
        Returns:
        --------
        Tuple[float, float]
            Hurst exponent estimate and standard error
        """
        if method == 'dfa':
            _, _, summary = dfa(y, **kwargs)
            return summary.alpha / 2, summary.stderr / 2
        
        elif method == 'rs':
            _, _, summary = rs_analysis(y, **kwargs)
            return summary.hurst, summary.stderr
        
        elif method == 'wavelet':
            _, _, summary = wavelet_leaders_estimation(y, **kwargs)
            return summary.hurst, summary.stderr
        
        elif method == 'spectral':
            if 'method' in kwargs and kwargs['method'] == 'periodogram':
                _, _, summary = whittle_mle(y, **{k: v for k, v in kwargs.items() if k != 'method'})
            else:
                _, _, summary = whittle_mle(y, **kwargs)
            return summary.hurst, summary.stderr
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _block_bootstrap(self, y: np.ndarray, block_size: int) -> np.ndarray:
        """
        Generate block bootstrap sample.
        
        Parameters:
        -----------
        y : np.ndarray
            Original time series
        block_size : int
            Size of blocks
            
        Returns:
        --------
        np.ndarray
            Bootstrap sample
        """
        n = len(y)
        n_blocks = n // block_size
        bootstrap_sample = []
        
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n - block_size + 1)
            block = y[start_idx:start_idx + block_size]
            bootstrap_sample.extend(block)
        
        # Add remaining points if needed
        remaining = n - len(bootstrap_sample)
        if remaining > 0:
            start_idx = np.random.randint(0, n - remaining + 1)
            bootstrap_sample.extend(y[start_idx:start_idx + remaining])
        
        return np.array(bootstrap_sample)
    
    def _generate_ar1_process(self, n: int, phi: float) -> np.ndarray:
        """Generate AR(1) process."""
        y = np.zeros(n)
        y[0] = np.random.normal(0, 1)
        
        for i in range(1, n):
            y[i] = phi * y[i-1] + np.random.normal(0, 1)
        
        return y
    
    def _generate_ma1_process(self, n: int, theta: float) -> np.ndarray:
        """Generate MA(1) process."""
        epsilon = np.random.normal(0, 1, n)
        y = np.zeros(n)
        
        for i in range(n):
            if i == 0:
                y[i] = epsilon[i]
            else:
                y[i] = epsilon[i] + theta * epsilon[i-1]
        
        return y


# Convenience functions for easy access
def test_lrd_hypothesis(y: np.ndarray,
                       method: str = 'dfa',
                       alpha: float = 0.05,
                       h0_hurst: float = 0.5,
                       alternative: str = 'greater',
                       random_state: Optional[int] = None) -> HypothesisTestResult:
    """
    Convenience function for testing LRD hypothesis.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    method : str
        Analysis method ('dfa', 'rs', 'wavelet', 'spectral')
    alpha : float
        Significance level
    h0_hurst : float
        Null hypothesis Hurst exponent
    alternative : str
        Alternative hypothesis ('greater', 'less', 'two-sided')
    random_state : Optional[int]
        Random seed
        
    Returns:
    --------
    HypothesisTestResult
        Test results
    """
    validator = StatisticalValidator(random_state=random_state)
    return validator.test_lrd_hypothesis(y, method, alpha, h0_hurst, alternative)


def bootstrap_confidence_interval(y: np.ndarray,
                                method: str = 'dfa',
                                n_bootstrap: int = 1000,
                                confidence_level: float = 0.95,
                                block_size: Optional[int] = None,
                                random_state: Optional[int] = None) -> BootstrapResult:
    """
    Convenience function for bootstrap confidence intervals.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    method : str
        Analysis method
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level
    block_size : Optional[int]
        Block size for block bootstrap
    random_state : Optional[int]
        Random seed
        
    Returns:
    --------
    BootstrapResult
        Bootstrap analysis results
    """
    validator = StatisticalValidator(random_state=random_state)
    return validator.bootstrap_validation(y, method, n_bootstrap, confidence_level, block_size)


def monte_carlo_test(y: np.ndarray,
                    method: str = 'dfa',
                    n_simulations: int = 1000,
                    alpha: float = 0.05,
                    null_model: str = 'iid',
                    random_state: Optional[int] = None) -> MonteCarloResult:
    """
    Convenience function for Monte Carlo significance test.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    method : str
        Analysis method
    n_simulations : int
        Number of simulations
    alpha : float
        Significance level
    null_model : str
        Null model type
    random_state : Optional[int]
        Random seed
        
    Returns:
    --------
    MonteCarloResult
        Monte Carlo test results
    """
    validator = StatisticalValidator(random_state=random_state)
    return validator.monte_carlo_significance_test(y, method, n_simulations, alpha, null_model)


def cross_validate_lrd(y: np.ndarray,
                      method: str = 'dfa',
                      n_folds: int = 5,
                      test_size: float = 0.2,
                      random_state: Optional[int] = None) -> CrossValidationResult:
    """
    Convenience function for cross-validation.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    method : str
        Analysis method
    n_folds : int
        Number of folds
    test_size : float
        Test set size
    random_state : Optional[int]
        Random seed
        
    Returns:
    --------
    CrossValidationResult
        Cross-validation results
    """
    validator = StatisticalValidator(random_state=random_state)
    return validator.split_sample_validation(y, method, n_folds, test_size)


def comprehensive_validation(y: np.ndarray,
                           methods: List[str] = None,
                           alpha: float = 0.05,
                           n_bootstrap: int = 1000,
                           n_simulations: int = 1000,
                           random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Perform comprehensive statistical validation.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    methods : List[str]
        Analysis methods to test
    alpha : float
        Significance level
    n_bootstrap : int
        Number of bootstrap samples
    n_simulations : int
        Number of Monte Carlo simulations
    random_state : Optional[int]
        Random seed
        
    Returns:
    --------
    Dict[str, Any]
        Comprehensive validation results
    """
    if methods is None:
        methods = ['dfa', 'rs', 'wavelet', 'spectral']
    
    validator = StatisticalValidator(random_state=random_state)
    
    results = {
        'hypothesis_tests': {},
        'bootstrap_analyses': {},
        'monte_carlo_tests': {},
        'cross_validation': {},
        'robustness_tests': {}
    }
    
    for method in methods:
        try:
            # Hypothesis test
            results['hypothesis_tests'][method] = validator.test_lrd_hypothesis(
                y, method, alpha
            )
            
            # Bootstrap analysis
            results['bootstrap_analyses'][method] = validator.bootstrap_validation(
                y, method, n_bootstrap
            )
            
            # Monte Carlo test
            results['monte_carlo_tests'][method] = validator.monte_carlo_significance_test(
                y, method, n_simulations, alpha
            )
            
            # Cross-validation
            results['cross_validation'][method] = validator.split_sample_validation(
                y, method
            )
            
        except Exception as e:
            print(f"Warning: Method {method} failed: {e}")
            continue
    
    # Robustness test
    results['robustness_tests'] = validator.robustness_test(y, methods)
    
    return results

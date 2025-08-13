"""
ARFIMA (Autoregressive Fractionally Integrated Moving Average) Model Implementation.

This module provides a robust implementation of ARFIMA models for time series analysis,
including parameter estimation, forecasting, and diagnostics.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gamma
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from typing import Tuple, Optional, Dict, List, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class ARFIMAParams:
    """Container for ARFIMA model parameters."""
    d: float  # Fractional differencing parameter
    p: int    # AR order
    q: int    # MA order
    ar_params: Optional[np.ndarray] = None
    ma_params: Optional[np.ndarray] = None
    intercept: float = 0.0
    sigma2: float = 1.0  # Innovation variance


class ARFIMAModel:
    """
    ARFIMA (Autoregressive Fractionally Integrated Moving Average) Model.
    
    This class implements a custom ARFIMA model with the following features:
    - Fractional differencing with parameter d
    - Autoregressive component of order p
    - Moving average component of order q
    - Maximum likelihood estimation
    - Forecasting capabilities
    - Model diagnostics
    """
    
    def __init__(self, p: int = 1, d: float = 0.5, q: int = 1, fast_mode: bool = True):
        """
        Initialize ARFIMA model.
        
        Parameters:
        -----------
        p : int
            Order of autoregressive component
        d : float
            Fractional differencing parameter (0 < d < 0.5 for stationarity)
        q : int
            Order of moving average component
        fast_mode : bool
            Use fast estimation mode (approximate but much faster)
        """
        self.p = p
        self.d = d
        self.q = q
        self.fast_mode = fast_mode
        self.params = None
        self.fitted_values = None
        self.residuals = None
        self.is_fitted = False
        self.n_obs = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        
    def _fractional_difference_fast(self, x: np.ndarray, d: float) -> np.ndarray:
        """
        Fast fractional differencing using FFT-based convolution.
        
        Parameters:
        -----------
        x : np.ndarray
            Input time series
        d : float
            Fractional differencing parameter
            
        Returns:
        --------
        np.ndarray
            Fractionally differenced series
        """
        n = len(x)
        if d == 0:
            return x.copy()
        
        # Ensure d is in valid range
        d = np.clip(d, 0.01, 0.49)
        
        # Use FFT for efficient convolution
        # Compute weights using vectorized operations with bounds checking
        k = np.arange(n)
        
        # Check for valid gamma function arguments
        valid_k = k + 1 - d > 0
        weights = np.zeros(n)
        
        # Only compute weights for valid k values
        weights[valid_k] = np.exp(
            np.log(gamma(k[valid_k] + 1)) - 
            np.log(gamma(k[valid_k] + 1 - d)) - 
            np.log(gamma(1 - d))
        )
        
        # Set first weight to 1
        weights[0] = 1.0
        
        # Apply convolution using FFT
        x_fft = np.fft.fft(x)
        weights_fft = np.fft.fft(weights)
        result_fft = x_fft * weights_fft
        result = np.real(np.fft.ifft(result_fft))
        
        return result
    
    def _fractional_difference(self, x: np.ndarray, d: float) -> np.ndarray:
        """
        Apply fractional differencing to time series.
        
        Parameters:
        -----------
        x : np.ndarray
            Input time series
        d : float
            Fractional differencing parameter
            
        Returns:
        --------
        np.ndarray
            Fractionally differenced series
        """
        if self.fast_mode:
            return self._fractional_difference_fast(x, d)
        
        # Original implementation for comparison
        n = len(x)
        if d == 0:
            return x.copy()
        
        # Compute fractional differencing weights w_k recursively:
        # w_0 = 1, w_k = w_{k-1} * (k-1 - d) / k
        weights = np.zeros(n)
        weights[0] = 1.0
        for k in range(1, n):
            weights[k] = weights[k - 1] * (k - 1 - d) / k
        
        # Convolution-style application: y_t = sum_{k=0}^t w_k x_{t-k}
        y = np.zeros(n)
        for t in range(n):
            k_max = t + 1
            y[t] = np.dot(weights[:k_max], x[t::-1])
        return y
    
    def _fractional_integrate_fast(self, x: np.ndarray, d: float) -> np.ndarray:
        """
        Fast fractional integration using FFT-based convolution.
        
        Parameters:
        -----------
        x : np.ndarray
            Input time series
        d : float
            Fractional integration parameter
            
        Returns:
        --------
        np.ndarray
            Fractionally integrated series
        """
        n = len(x)
        if d == 0:
            return x.copy()
        
        # Use FFT for efficient convolution
        # Compute weights using vectorized operations
        k = np.arange(n)
        weights = np.exp(np.log(gamma(k + d)) - np.log(gamma(k + 1)) - np.log(gamma(d)))
        weights[0] = 1.0
        
        # Apply convolution using FFT
        x_fft = np.fft.fft(x)
        weights_fft = np.fft.fft(weights)
        result_fft = x_fft * weights_fft
        result = np.real(np.fft.ifft(result_fft))
        
        return result
    
    def _fractional_integrate(self, x: np.ndarray, d: float) -> np.ndarray:
        """
        Apply fractional integration (inverse of fractional differencing).
        
        Parameters:
        -----------
        x : np.ndarray
            Input time series
        d : float
            Fractional integration parameter
            
        Returns:
        --------
        np.ndarray
            Fractionally integrated series
        """
        if self.fast_mode:
            return self._fractional_integrate_fast(x, d)
        
        # Original implementation for comparison
        n = len(x)
        if d == 0:
            return x.copy()
        
        # Compute fractional integration weights c_k recursively for (1 - B)^{-d}:
        # c_0 = 1, c_k = c_{k-1} * (d + k - 1) / k
        weights = np.zeros(n)
        weights[0] = 1.0
        for k in range(1, n):
            weights[k] = weights[k - 1] * (d + k - 1) / k
        
        y = np.zeros(n)
        for t in range(n):
            k_max = t + 1
            y[t] = np.dot(weights[:k_max], x[t::-1])
        return y
    
    def _compute_arfima_residuals_fast(self, y: np.ndarray, params: ARFIMAParams) -> np.ndarray:
        """
        Fast computation of residuals for ARFIMA model using vectorized operations.
        
        Parameters:
        -----------
        y : np.ndarray
            Original time series
        params : ARFIMAParams
            Model parameters
            
        Returns:
        --------
        np.ndarray
            Model residuals
        """
        n = len(y)
        
        # Apply fractional differencing (with fallback for stability)
        try:
            y_diff = self._fractional_difference(y, params.d)
            # Check if we got NaN values
            if np.any(np.isnan(y_diff)):
                # Fallback to simple differencing
                y_diff = np.diff(y, prepend=y[0])
        except:
            # Fallback to simple differencing
            y_diff = np.diff(y, prepend=y[0])
        
        # Initialize residuals
        residuals = np.zeros(n)
        
        # Use vectorized operations for AR and MA components
        max_order = max(self.p, self.q)
        
        for t in range(max_order, n):
            # AR component (vectorized)
            ar_term = 0
            if params.ar_params is not None and self.p > 0:
                ar_term = np.dot(params.ar_params, y_diff[t-self.p:t][::-1])
            
            # MA component (vectorized)
            ma_term = 0
            if params.ma_params is not None and self.q > 0:
                ma_term = np.dot(params.ma_params, residuals[t-self.q:t][::-1])
            
            # Compute residual
            residuals[t] = y_diff[t] - params.intercept - ar_term + ma_term
        
        return residuals
    
    def _compute_arfima_residuals(self, y: np.ndarray, params: ARFIMAParams) -> np.ndarray:
        """
        Compute residuals for ARFIMA model.
        
        Parameters:
        -----------
        y : np.ndarray
            Original time series
        params : ARFIMAParams
            Model parameters
            
        Returns:
        --------
        np.ndarray
            Model residuals
        """
        if self.fast_mode:
            return self._compute_arfima_residuals_fast(y, params)
        
        # Original implementation
        n = len(y)
        residuals = np.zeros(n)
        
        # Apply fractional differencing
        y_diff = self._fractional_difference(y, params.d)
        
        # Compute ARFIMA residuals
        for t in range(max(self.p, self.q), n):
            # AR component
            ar_term = 0
            if params.ar_params is not None:
                for i in range(self.p):
                    ar_term += params.ar_params[i] * y_diff[t - i - 1]
            
            # MA component
            ma_term = 0
            if params.ma_params is not None:
                for i in range(self.q):
                    ma_term += params.ma_params[i] * residuals[t - i - 1]
            
            # Compute residual
            residuals[t] = y_diff[t] - params.intercept - ar_term + ma_term
        
        return residuals
    
    def _log_likelihood_fast(self, params: np.ndarray, y: np.ndarray) -> float:
        """
        Fast log-likelihood computation using approximate methods.
        
        Parameters:
        -----------
        params : np.ndarray
            Parameter vector [d, ar_params, ma_params, intercept, log_sigma2]
        y : np.ndarray
            Time series data
            
        Returns:
        --------
        float
            Log-likelihood value
        """
        # Extract parameters
        d = params[0]
        ar_start = 1
        ar_end = ar_start + self.p
        ma_start = ar_end
        ma_end = ma_start + self.q
        
        ar_params = params[ar_start:ar_end] if self.p > 0 else None
        ma_params = params[ma_start:ma_end] if self.q > 0 else None
        intercept = params[ma_end] if self.q > 0 else params[ar_end]
        log_sigma2 = params[-1]
        sigma2 = np.exp(log_sigma2)
        
        # Create parameter object
        model_params = ARFIMAParams(
            d=d, p=self.p, q=self.q,
            ar_params=ar_params, ma_params=ma_params,
            intercept=intercept, sigma2=sigma2
        )
        
        # Compute residuals using fast method
        residuals = self._compute_arfima_residuals_fast(y, model_params)
        
        # Use only the last portion of residuals for stability
        start_idx = max(self.p, self.q, len(y) // 4)
        residuals_use = residuals[start_idx:]
        
        # Compute log-likelihood
        n = len(residuals_use)
        ll = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(residuals_use**2) / sigma2
        
        return ll
    
    def _log_likelihood(self, params: np.ndarray, y: np.ndarray) -> float:
        """
        Compute log-likelihood for ARFIMA model.
        
        Parameters:
        -----------
        params : np.ndarray
            Parameter vector [d, ar_params, ma_params, intercept, log_sigma2]
        y : np.ndarray
            Time series data
            
        Returns:
        --------
        float
            Log-likelihood value
        """
        if self.fast_mode:
            return self._log_likelihood_fast(params, y)
        
        # Original implementation
        # Extract parameters
        d = params[0]
        ar_start = 1
        ar_end = ar_start + self.p
        ma_start = ar_end
        ma_end = ma_start + self.q
        
        ar_params = params[ar_start:ar_end] if self.p > 0 else None
        ma_params = params[ma_start:ma_end] if self.q > 0 else None
        intercept = params[ma_end] if self.q > 0 else params[ar_end]
        log_sigma2 = params[-1]
        sigma2 = np.exp(log_sigma2)
        
        # Create parameter object
        model_params = ARFIMAParams(
            d=d, p=self.p, q=self.q,
            ar_params=ar_params, ma_params=ma_params,
            intercept=intercept, sigma2=sigma2
        )
        
        # Compute residuals
        residuals = self._compute_arfima_residuals(y, model_params)
        
        # Compute log-likelihood
        n = len(residuals)
        ll = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(residuals**2) / sigma2
        
        return ll
    
    def _constraint_function(self, params: np.ndarray) -> float:
        """
        Constraint function for optimization (ensures stationarity and invertibility).
        
        Parameters:
        -----------
        params : np.ndarray
            Parameter vector
            
        Returns:
        --------
        float
            Constraint value (should be > 0 for feasible parameters)
        """
        d = params[0]
        
        # Constraint on d: 0 < d < 0.5 for stationarity
        d_constraint = min(d, 0.5 - d)
        
        # AR constraints (roots outside unit circle)
        ar_constraint = 1.0
        if self.p > 0:
            ar_start = 1
            ar_end = ar_start + self.p
            ar_params = params[ar_start:ar_end]
            
            # Check if AR polynomial has roots outside unit circle
            ar_poly = np.concatenate([[1.0], -ar_params])
            roots = np.roots(ar_poly)
            ar_constraint = np.min(np.abs(roots)) - 1.0
        
        # MA constraints (roots outside unit circle)
        ma_constraint = 1.0
        if self.q > 0:
            ma_start = 1 + self.p
            ma_end = ma_start + self.q
            ma_params = params[ma_start:ma_end]
            
            # Check if MA polynomial has roots outside unit circle
            ma_poly = np.concatenate([[1], ma_params])
            roots = np.roots(ma_poly)
            ma_constraint = np.min(np.abs(roots)) - 1.0
        
        # Return the minimum constraint value, ensuring it's positive for valid parameters
        constraint_value = min(d_constraint, ar_constraint, ma_constraint)
        return constraint_value
    
    def fit_fast(self, y: Union[np.ndarray, pd.Series]) -> 'ARFIMAModel':
        """
        Fast fit ARFIMA model using approximate methods.
        
        Parameters:
        -----------
        y : Union[np.ndarray, pd.Series]
            Time series data
            
        Returns:
        --------
        ARFIMAModel
            Fitted model instance
        """
        # Convert to numpy array
        if isinstance(y, pd.Series):
            y = y.values
        
        self.n_obs = len(y)
        
        # Use simple estimation methods for speed
        # Estimate d using Geweke-Porter-Hudak method (approximate)
        d_est = self._estimate_d_gph(y)
        
        # Simple AR/MA parameter estimation
        ar_params = np.zeros(self.p) if self.p > 0 else None
        ma_params = np.zeros(self.q) if self.q > 0 else None
        
        # Estimate intercept and variance
        intercept = np.mean(y)
        sigma2 = np.var(y)
        
        # Create parameter object
        self.params = ARFIMAParams(
            d=d_est, p=self.p, q=self.q,
            ar_params=ar_params, ma_params=ma_params,
            intercept=intercept, sigma2=sigma2
        )
        
        # Compute residuals and fitted values
        self.residuals = self._compute_arfima_residuals_fast(y, self.params)
        self.fitted_values = y - self.residuals
        
        self.is_fitted = True
        
        # Compute model statistics
        self._compute_model_statistics()
        
        return self
    
    def _compute_model_statistics(self):
        """
        Compute model statistics including log-likelihood, AIC, and BIC.
        """
        if not self.is_fitted or self.residuals is None:
            return
        
        # Get number of observations
        self.n_obs = len(self.residuals)
        
        # Check for NaN values in residuals or parameters
        if np.any(np.isnan(self.residuals)) or np.isnan(self.params.sigma2):
            # Set default values if we have NaN issues
            self.log_likelihood = np.nan
            self.aic = np.nan
            self.bic = np.nan
            return
        
        # Compute log-likelihood
        sigma2 = self.params.sigma2
        n = self.n_obs
        
        # Use residuals to compute log-likelihood
        self.log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(self.residuals**2) / sigma2
        
        # Count number of parameters (d, ar_params, ma_params, intercept, sigma2)
        n_params = 1 + self.p + self.q + 1 + 1  # d + AR + MA + intercept + sigma2
        
        # Compute AIC and BIC
        self.aic = -2 * self.log_likelihood + 2 * n_params
        self.bic = -2 * self.log_likelihood + n_params * np.log(n)
    
    def _estimate_d_gph(self, y: np.ndarray) -> float:
        """
        Estimate d using Geweke-Porter-Hudak method (fast approximation).
        
        Parameters:
        -----------
        y : np.ndarray
            Time series data
            
        Returns:
        --------
        float
            Estimated d parameter
        """
        n = len(y)
        
        # Use only a subset of the data for speed
        if n > 1000:
            y_subset = y[:1000]
        else:
            y_subset = y
        
        # Compute periodogram
        freqs = np.fft.fftfreq(len(y_subset))
        periodogram = np.abs(np.fft.fft(y_subset))**2
        
        # Use only positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        periodogram_pos = periodogram[pos_mask]
        
        # Use only low frequencies for estimation
        low_freq_mask = freqs_pos < 0.1
        if np.sum(low_freq_mask) < 5:
            # Fallback to simple estimate
            return 0.3
        
        freqs_low = freqs_pos[low_freq_mask]
        periodogram_low = periodogram_pos[low_freq_mask]
        
        # Log-log regression: log(I(f)) = log(C) - 2*d*log(f)
        log_freqs = np.log(freqs_low)
        log_periodogram = np.log(periodogram_low)
        
        # Remove any infinite or NaN values
        valid_mask = np.isfinite(log_freqs) & np.isfinite(log_periodogram)
        if np.sum(valid_mask) < 3:
            return 0.3
        
        log_freqs_valid = log_freqs[valid_mask]
        log_periodogram_valid = log_periodogram[valid_mask]
        
        # Linear regression
        slope, _, _, _, _ = stats.linregress(log_freqs_valid, log_periodogram_valid)
        
        # Extract d: d = -slope / 2
        d_est = -slope / 2
        
        # Ensure d is in reasonable range
        d_est = np.clip(d_est, 0.01, 0.49)
        
        return d_est
    
    def fit(self, y: Union[np.ndarray, pd.Series], 
            method: str = 'mle', 
            initial_params: Optional[np.ndarray] = None,
            max_iter: int = 500,
            tol: float = 1e-6) -> 'ARFIMAModel':
        """
        Fit ARFIMA model to time series data with robust optimization.
        
        Parameters:
        -----------
        y : Union[np.ndarray, pd.Series]
            Time series data
        method : str
            Estimation method ('mle' for maximum likelihood, 'fast' for fast estimation)
        initial_params : Optional[np.ndarray]
            Initial parameter values
        max_iter : int
            Maximum number of optimization iterations
        tol : float
            Optimization tolerance

            
        Returns:
        --------
        ARFIMAModel
            Fitted model instance
        """
        # Use fast method if requested or if fast_mode is enabled
        if method == 'fast' or (self.fast_mode and len(y) > 500):
            return self.fit_fast(y)
        
        # Convert to numpy array
        if isinstance(y, pd.Series):
            y = y.values
        
        self.n_obs = len(y)
        
        # Set initial parameters if not provided
        if initial_params is None:
            initial_params = np.zeros(1 + self.p + self.q + 2)  # d, ar, ma, intercept, log_sigma2
            initial_params[0] = 0.3  # Initial d
            initial_params[-2] = np.mean(y)  # Initial intercept
            initial_params[-1] = np.log(np.var(y))  # Initial log variance
        
        # Define bounds for parameters
        bounds = []
        bounds.append((0.01, 0.49))  # d bounds
        for _ in range(self.p):  # AR parameters
            bounds.append((-0.99, 0.99))
        for _ in range(self.q):  # MA parameters
            bounds.append((-0.99, 0.99))
        bounds.append((None, None))  # intercept
        bounds.append((-20.0, 20.0))  # log_sigma2 to avoid numeric overflow
        
        # Optimize parameters with multiple fallback methods
        if method == 'mle':
            result = None
            optimization_success = False
            
            # Try different optimization methods
            methods_to_try = [
                ('L-BFGS-B', {}),  # No constraints, but faster
                ('SLSQP', {'constraints': ({'type': 'ineq', 'fun': self._constraint_function},)}),
                ('TNC', {}),  # Another constrained method
            ]
            
            for opt_method, extra_options in methods_to_try:
                try:
                    # Set up optimization options
                    options = {
                        'maxiter': max_iter, 
                        'ftol': tol, 
                        'disp': False
                    }
                    
                    # Add method-specific options
                    if opt_method == 'SLSQP':
                        options.update(extra_options)
                    
                    # Try optimization with reduced iterations for stability
                    try:
                        if opt_method == 'SLSQP':
                            result = optimize.minimize(
                                fun=lambda params: -self._log_likelihood(params, y),
                                x0=initial_params,
                                method=opt_method,
                                bounds=bounds,
                                constraints=extra_options['constraints'],
                                options=options
                            )
                        else:
                            result = optimize.minimize(
                                fun=lambda params: -self._log_likelihood(params, y),
                                x0=initial_params,
                                method=opt_method,
                                bounds=bounds,
                                options=options
                            )
                        
                        # Check if optimization was successful
                        if result.success or result.fun < float('inf'):
                            optimization_success = True
                            break
                            
                    except (ValueError, RuntimeError) as e:
                        warnings.warn(f"Optimization method {opt_method} failed: {e}")
                        continue
                        
                except Exception as e:
                    warnings.warn(f"Optimization method {opt_method} failed: {e}")
                    continue
            
            # If all methods failed, use initial parameters as fallback
            if not optimization_success or result is None:
                warnings.warn("All optimization methods failed, using initial parameters")
                fitted_params = initial_params
            else:
                fitted_params = result.x
                
                if not result.success:
                    warnings.warn(f"Optimization completed but may not be optimal: {result.message}")
        
        # Extract fitted parameters
        d = fitted_params[0]
        ar_start = 1
        ar_end = ar_start + self.p
        ma_start = ar_end
        ma_end = ma_start + self.q
        
        ar_params = fitted_params[ar_start:ar_end] if self.p > 0 else None
        ma_params = fitted_params[ma_start:ma_end] if self.q > 0 else None
        intercept = fitted_params[ma_end] if self.q > 0 else fitted_params[ar_end]
        sigma2 = np.exp(fitted_params[-1])
        
        # Store fitted parameters
        self.params = ARFIMAParams(
            d=d, p=self.p, q=self.q,
            ar_params=ar_params, ma_params=ma_params,
            intercept=intercept, sigma2=sigma2
        )
        
        # Mark as fitted before computing fitted values (uses self.params)
        self.is_fitted = True
        
        try:
            # Compute fitted values and residuals
            self.fitted_values, self.residuals = self._compute_fitted_values(y)
            
            # Compute information criteria
            self.log_likelihood = self._log_likelihood(fitted_params, y)
            n_params = 1 + self.p + self.q + 2  # d, ar, ma, intercept, sigma2
            self.aic = 2 * n_params - 2 * self.log_likelihood
            self.bic = np.log(self.n_obs) * n_params - 2 * self.log_likelihood
        except Exception as e:
            warnings.warn(f"Error computing fitted values: {e}")
            # Set default values
            self.fitted_values = y.copy()
            self.residuals = np.zeros_like(y)
            self.log_likelihood = float('-inf')
            self.aic = float('inf')
            self.bic = float('inf')
        
        return self
    
    def _compute_fitted_values(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute fitted values for the ARFIMA model.
        
        Parameters:
        -----------
        y : np.ndarray
            Original time series
            
        Returns:
        --------
        np.ndarray
            Fitted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing fitted values")
        
        n = len(y)
        fitted = np.zeros(n)
        residuals = np.zeros(n)
        
        # Apply fractional differencing
        y_diff = self._fractional_difference(y, self.params.d)
        
        # Compute fitted values
        start_idx = max(self.p, self.q)
        # For initial observations, set fitted equal to observed
        if start_idx > 0:
            fitted[:start_idx] = y[:start_idx]
        for t in range(start_idx, n):
            # AR component
            ar_term = 0
            if self.params.ar_params is not None:
                for i in range(self.p):
                    ar_term += self.params.ar_params[i] * y_diff[t - i - 1]
            
            # MA component
            ma_term = 0
            if self.params.ma_params is not None:
                for i in range(self.q):
                    ma_term += self.params.ma_params[i] * residuals[t - i - 1]
            
            # Compute fitted value
            fitted_diff = self.params.intercept + ar_term - ma_term
            fitted[t] = y[t] - (y_diff[t] - fitted_diff)
            # Update residuals in differenced space
            residuals[t] = y_diff[t] - fitted_diff
        
        return fitted, residuals
    
    def _forecast_path(self, steps: int, last_values: Optional[np.ndarray], future_eps: Optional[np.ndarray]) -> np.ndarray:
        """
        Internal: generate a single forecast path in the differenced space with optional future innovations.
        If future_eps is None, assumes zero-mean future residuals (point forecast).
        Returns the forecasts in the original space after fractional integration.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        if last_values is None:
            last_values = self.fitted_values[-max(self.p, self.q):]

        # Differenced tail of series
        y_diff_hist = self._fractional_difference(last_values, self.params.d).tolist()

        # Residual history for MA terms
        res_hist = []
        if self.q > 0 and self.residuals is not None and len(self.residuals) > 0:
            res_hist = self.residuals[-self.q:].tolist()

        forecasts_diff = []
        for t in range(steps):
            # AR term
            ar_term = 0.0
            if self.params.ar_params is not None:
                for i in range(self.p):
                    idx = - (i + 1)
                    if len(y_diff_hist) >= self.p:
                        ar_term += self.params.ar_params[i] * y_diff_hist[idx]

            # MA term (using residual history)
            ma_term = 0.0
            if self.params.ma_params is not None and len(res_hist) > 0:
                for i in range(min(self.q, len(res_hist))):
                    ma_term += self.params.ma_params[i] * res_hist[-(i + 1)]

            eps_t = 0.0 if future_eps is None else future_eps[t]
            # y_diff_t = intercept + ar_term - ma_term + eps_t
            y_diff_t = self.params.intercept + ar_term - ma_term + eps_t

            forecasts_diff.append(y_diff_t)
            y_diff_hist.append(y_diff_t)
            # Update residual history: future residual equals eps_t
            if self.q > 0:
                res_hist.append(eps_t)
                if len(res_hist) > self.q:
                    res_hist = res_hist[-self.q:]

        # Transform differenced forecasts to original space via fractional integration
        forecasts = self._fractional_integrate(np.array(forecasts_diff), self.params.d)
        return forecasts

    def forecast(self, steps: int, last_values: Optional[np.ndarray] = None,
                 alpha: Optional[float] = None,
                 interval_method: str = 'bootstrap',
                 B: int = 500,
                 seed: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate forecasts for future time steps.
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
        last_values : Optional[np.ndarray]
            Last values of the series (if None, uses fitted data)
        alpha : Optional[float]
            If provided (e.g., 0.05), also return (lower, upper) confidence bands at 1-alpha level.
        interval_method : str
            'bootstrap' (recommended) or 'analytic'.
        B : int
            Number of bootstrap paths for interval estimation (if method='bootstrap').
        seed : Optional[int]
            Random seed for reproducibility in bootstrap.
            
        Returns:
        --------
        np.ndarray or Tuple[np.ndarray, np.ndarray, np.ndarray]
            Forecasted values; if alpha is provided, also returns (lower, upper) arrays.
        """
        # Point forecasts (future_eps = 0)
        point_fc = self._forecast_path(steps, last_values, future_eps=None)

        if alpha is None:
            return point_fc

        # Compute intervals
        if interval_method == 'analytic':
            # Naive SE approximation: constant sigma for all horizons
            sigma = float(np.sqrt(self.params.sigma2))
            z = stats.norm.ppf(1 - alpha / 2)
            se = np.full(steps, sigma)
            lower = point_fc - z * se
            upper = point_fc + z * se
            return point_fc, lower, upper
        elif interval_method == 'bootstrap':
            rng = np.random.default_rng(seed)
            paths = np.zeros((B, steps))
            for b in range(B):
                future_eps = rng.normal(0.0, np.sqrt(self.params.sigma2), size=steps)
                paths[b, :] = self._forecast_path(steps, last_values, future_eps)
            lower = np.quantile(paths, alpha / 2, axis=0)
            upper = np.quantile(paths, 1 - alpha / 2, axis=0)
            return point_fc, lower, upper
        else:
            raise ValueError("interval_method must be 'bootstrap' or 'analytic'")
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate model summary statistics.
        
        Returns:
        --------
        Dict[str, Any]
            Model summary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating summary")
        
        summary = {
            'model': f'ARFIMA({self.p},{self.d},{self.q})',
            'parameters': {
                'd': self.params.d,
                'ar_params': self.params.ar_params.tolist() if self.params.ar_params is not None else None,
                'ma_params': self.params.ma_params.tolist() if self.params.ma_params is not None else None,
                'intercept': self.params.intercept,
                'sigma2': self.params.sigma2
            },
            'fit_metrics': {
                'log_likelihood': self.log_likelihood,
                'aic': self.aic,
                'bic': self.bic,
                'n_observations': self.n_obs
            },
            'residuals': {
                'mean': np.mean(self.residuals),
                'std': np.std(self.residuals),
                'skewness': stats.skew(self.residuals),
                'kurtosis': stats.kurtosis(self.residuals)
            }
        }
        
        return summary
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot model diagnostics.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting diagnostics")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Residuals plot
        axes[0, 0].plot(self.residuals)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True)
        
        # Residuals histogram
        axes[0, 1].hist(self.residuals, bins=30, density=True, alpha=0.7)
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Density')
        
        # Q-Q plot
        stats.probplot(self.residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True)
        
        # ACF of residuals
        from statsmodels.tsa.stattools import acf
        acf_residuals = acf(self.residuals, nlags=20)
        axes[1, 1].bar(range(len(acf_residuals)), acf_residuals)
        axes[1, 1].set_title('ACF of Residuals')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('ACF')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, y: np.ndarray) -> np.ndarray:
        """
        Generate in-sample predictions.
        
        Parameters:
        -----------
        y : np.ndarray
            Time series data
            
        Returns:
        --------
        np.ndarray
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        fitted, _ = self._compute_fitted_values(y)
        return fitted
    
    def estimate_hurst(self) -> float:
        """
        Estimate Hurst exponent from the fitted ARFIMA model.
        
        Returns:
        --------
        float
            Estimated Hurst exponent
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before estimating Hurst exponent")
        
        # For ARFIMA models, H = d + 0.5
        return self.params.d + 0.5
    
    def estimate_alpha(self) -> float:
        """
        Estimate alpha parameter from the fitted ARFIMA model.
        
        Returns:
        --------
        float
            Estimated alpha parameter
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before estimating alpha")
        
        # For ARFIMA models, alpha = 2 * d
        return 2 * self.params.d
    
    def get_confidence_intervals(self) -> Dict[str, float]:
        """
        Get confidence intervals for model parameters.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary with confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing confidence intervals")
        
        # Simple confidence intervals based on parameter estimates
        # In a full implementation, these would be computed from the Hessian matrix
        return {
            'd_ci': (max(0.01, self.params.d - 0.1), min(0.49, self.params.d + 0.1)),
            'hurst_ci': (max(0.51, self.estimate_hurst() - 0.1), min(0.99, self.estimate_hurst() + 0.1)),
            'alpha_ci': (max(0.02, self.estimate_alpha() - 0.2), min(0.98, self.estimate_alpha() + 0.2))
        }


def estimate_arfima_order(y: np.ndarray, max_p: int = 3, max_q: int = 3, 
                         d_values: List[float] = None) -> Tuple[int, float, int]:
    """
    Estimate optimal ARFIMA model order using information criteria.
    
    Parameters:
    -----------
    y : np.ndarray
        Time series data
    max_p : int
        Maximum AR order to consider
    max_q : int
        Maximum MA order to consider
    d_values : List[float]
        List of d values to consider

        
    Returns:
    --------
    Tuple[int, float, int]
        Optimal (p, d, q) values
    """
    if d_values is None:
        d_values = [0.1, 0.2, 0.3, 0.4]
    
    best_aic = np.inf
    best_params = (0, 0.0, 0)
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for d in d_values:
                try:
                    model = ARFIMAModel(p=p, d=d, q=q)
                    model.fit(y, max_iter=100)  # Reduced iterations for stability
                    
                    if model.aic < best_aic and model.is_fitted:
                        best_aic = model.aic
                        best_params = (p, d, q)
                        
                except Exception as e:
                    continue
    
    return best_params


def arfima_simulation(n: int, d: float, ar_params: np.ndarray = None, 
                     ma_params: np.ndarray = None, sigma: float = 1.0,
                     seed: int = None) -> np.ndarray:
    """
    Simulate ARFIMA time series.
    
    Parameters:
    -----------
    n : int
        Length of time series
    d : float
        Fractional differencing parameter
    ar_params : np.ndarray
        AR parameters
    ma_params : np.ndarray
        MA parameters
    sigma : float
        Innovation standard deviation
    seed : int
        Random seed
        
    Returns:
    --------
    np.ndarray
        Simulated ARFIMA time series
    """
    # Parameter validation
    if n is None or n <= 0:
        raise ValueError("n must be a positive integer")
    if d is None or d < 0:
        raise ValueError("d must be non-negative for simulation")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if seed is not None:
        np.random.seed(seed)
    
    # Generate white noise
    innovations = np.random.normal(0, sigma, n)
    
    # Apply fractional integration
    y = np.zeros(n)
    for t in range(n):
        y[t] = innovations[t]
        
        # Add AR component
        if ar_params is not None:
            for i in range(min(t, len(ar_params))):
                y[t] += ar_params[i] * y[t - i - 1]
        
        # Add MA component
        if ma_params is not None:
            for i in range(min(t, len(ma_params))):
                y[t] += ma_params[i] * innovations[t - i - 1]
    
    # Apply fractional integration
    model = ARFIMAModel()
    y_integrated = model._fractional_integrate(y, d)
    
    return y_integrated

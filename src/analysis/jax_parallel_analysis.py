"""
JAX-based Parallel Computation for Long-Range Dependence Analysis

This module provides JAX-accelerated implementations of various fractal analysis methods
with support for parallel computation across multiple datasets and parameters.

Key features:
- GPU/TPU acceleration via JAX
- Vectorized operations for batch processing
- Parallel computation across multiple datasets
- Automatic differentiation for optimization
- Memory-efficient computation with JIT compilation

References:
- JAX documentation: https://jax.readthedocs.io/
- Parallel computation patterns for scientific computing
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, grad, hessian
from jax.scipy import stats as jax_stats
from jax.scipy.optimize import minimize
import jax.random as random
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import numpy as np
from functools import partial
import warnings

# Configure JAX for optimal performance
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')  # Change to 'gpu' or 'tpu' if available


@dataclass
class JAXAnalysisConfig:
    """Configuration for JAX-based analysis"""
    use_gpu: bool = False
    use_tpu: bool = False
    batch_size: int = 32
    num_parallel: int = 4
    precision: str = 'float64'  # 'float32' or 'float64'
    enable_jit: bool = True
    enable_vmap: bool = True
    enable_pmap: bool = False  # Only for multi-device setups
    memory_efficient: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_parallel <= 0:
            raise ValueError("num_parallel must be positive")
        if self.precision not in ['float32', 'float64']:
            raise ValueError("precision must be 'float32' or 'float64'")


class JAXDeviceManager:
    """Manages JAX device configuration and parallel computation"""
    
    def __init__(self, config: JAXAnalysisConfig):
        self.config = config
        self.devices = self._setup_devices()
        self.key = random.PRNGKey(42)
    
    def _setup_devices(self) -> List[jax.Device]:
        """Setup available devices for computation"""
        if self.config.use_tpu and jax.device_count('tpu') > 0:
            return jax.devices('tpu')
        elif self.config.use_gpu and jax.device_count('gpu') > 0:
            return jax.devices('gpu')
        else:
            return jax.devices('cpu')
    
    def get_parallel_devices(self) -> List[jax.Device]:
        """Get devices for parallel computation"""
        num_devices = min(self.config.num_parallel, len(self.devices))
        return self.devices[:num_devices]
    
    def split_key(self) -> jax.Array:
        """Split the random key for parallel operations"""
        self.key, subkey = random.split(self.key)
        return subkey


class JAXDFAAnalysis:
    """JAX-accelerated Detrended Fluctuation Analysis"""
    
    def __init__(self, config: JAXAnalysisConfig):
        self.config = config
        self.device_manager = JAXDeviceManager(config)
        
        # JIT-compiled functions
        if config.enable_jit:
            self._profile_jitted = jit(self._profile)
            self._poly_detrend_jitted = jit(self._poly_detrend)
            self._calculate_fluctuation_jitted = jit(self._calculate_fluctuation)
            self._fit_scaling_jitted = jit(self._fit_scaling)
        else:
            # Fallback to non-jitted versions
            self._profile_jitted = self._profile
            self._poly_detrend_jitted = self._poly_detrend
            self._calculate_fluctuation_jitted = self._calculate_fluctuation
            self._fit_scaling_jitted = self._fit_scaling
    
    def _profile(self, y: jax.Array) -> jax.Array:
        """Calculate the profile (cumulative sum of centered data)"""
        y_centered = y - jnp.mean(y)
        return jnp.cumsum(y_centered)
    
    def _poly_detrend(self, x: jax.Array, order: int) -> Tuple[jax.Array, jax.Array]:
        """Polynomial detrending using JAX"""
        t = jnp.arange(len(x), dtype=jnp.float64)
        # Use lower order if needed
        order = min(order, len(x) - 1)
        
        if order <= 0:
            trend = jnp.full_like(x, jnp.mean(x))
            return trend, x - trend
        
        # Manual polynomial fitting for JAX compatibility
        A = jnp.vander(t, order + 1)
        coeffs = jnp.linalg.lstsq(A, x, rcond=None)[0]
        trend = jnp.polyval(coeffs, t)
        return trend, x - trend
    
    def _calculate_fluctuation(self, profile: jax.Array, scale: int) -> jax.Array:
        """Calculate fluctuation for a given scale"""
        n = len(profile)
        n_segments = n // scale
        
        def segment_fluctuation(segment_idx):
            start_idx = segment_idx * scale
            end_idx = start_idx + scale
            segment = profile[start_idx:end_idx]
            
            # Detrend segment
            trend, detrended = self._poly_detrend_jitted(segment, 1)
            
            # Calculate RMS
            return jnp.sqrt(jnp.mean(detrended ** 2))
        
        if self.config.enable_vmap:
            fluctuations = vmap(segment_fluctuation)(jnp.arange(n_segments))
        else:
            fluctuations = jnp.array([segment_fluctuation(i) for i in range(n_segments)])
        
        return jnp.mean(fluctuations)
    
    def _fit_scaling(self, scales: jax.Array, flucts: jax.Array) -> Dict[str, jax.Array]:
        """Fit scaling relationship using JAX"""
        log_scales = jnp.log(scales)
        log_flucts = jnp.log(flucts)
        
        # Linear regression using JAX
        n = len(log_scales)
        A = jnp.column_stack([log_scales, jnp.ones(n)])
        coeffs = jnp.linalg.lstsq(A, log_flucts, rcond=None)[0]
        
        alpha = coeffs[0]
        intercept = coeffs[1]
        
        # Calculate R-squared
        y_pred = A @ coeffs
        ss_res = jnp.sum((log_flucts - y_pred) ** 2)
        ss_tot = jnp.sum((log_flucts - jnp.mean(log_flucts)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        
        return {
            'alpha': alpha,
            'intercept': intercept,
            'r_squared': r_squared,
            'scales': scales,
            'fluctuations': flucts
        }
    
    def analyze_single(self, y: jax.Array, 
                      min_scale: int = 4,
                      max_scale: Optional[int] = None,
                      num_scales: int = 20) -> Dict[str, jax.Array]:
        """Analyze a single time series"""
        y = jnp.asarray(y, dtype=jnp.float64)
        
        if max_scale is None:
            max_scale = max(min(len(y) // 4, 1024), min_scale + 1)
        
        # Generate scales
        scales = jnp.unique(jnp.logspace(jnp.log2(min_scale), jnp.log2(max_scale), num_scales, base=2).astype(int))
        
        # Calculate profile
        profile = self._profile_jitted(y)
        
        # Calculate fluctuations for each scale
        if self.config.enable_vmap:
            flucts = vmap(lambda s: self._calculate_fluctuation_jitted(profile, s))(scales)
        else:
            flucts = jnp.array([self._calculate_fluctuation_jitted(profile, s) for s in scales])
        
        # Fit scaling relationship
        results = self._fit_scaling_jitted(scales, flucts)
        
        return results
    
    def analyze_batch(self, data_batch: jax.Array, **kwargs) -> List[Dict[str, jax.Array]]:
        """Analyze a batch of time series in parallel"""
        if self.config.enable_vmap:
            # For vmap, we need to handle the dictionary structure differently
            # Since JAX can't handle dictionaries in arrays, we'll use a different approach
            results = vmap(lambda y: self.analyze_single(y, **kwargs))(data_batch)
            # Convert the structured array back to a list of dictionaries
            return [dict(zip(results.keys(), [results[key][i] for key in results.keys()])) 
                   for i in range(len(data_batch))]
        else:
            return [self.analyze_single(y, **kwargs) for y in data_batch]


class JAXHiguchiAnalysis:
    """JAX-accelerated Higuchi Fractal Dimension Analysis"""
    
    def __init__(self, config: JAXAnalysisConfig):
        self.config = config
        self.device_manager = JAXDeviceManager(config)
        
        if config.enable_jit:
            self._calculate_l_values_jitted = jit(self._calculate_l_values)
            self._fit_fractal_dimension_jitted = jit(self._fit_fractal_dimension)
        else:
            # Fallback to non-jitted versions
            self._calculate_l_values_jitted = self._calculate_l_values
            self._fit_fractal_dimension_jitted = self._fit_fractal_dimension
    
    def _calculate_l_values(self, y: jax.Array, k_values: jax.Array) -> jax.Array:
        """Calculate L(k) values for Higuchi method"""
        n = len(y)
        
        def l_value_for_k(k):
            # Calculate L(k) for a specific k
            l_sum = 0.0
            
            for m in range(k):
                # Calculate L_m(k)
                l_m = 0.0
                for i in range(1, (n - m) // k):
                    l_m += jnp.abs(y[m + i * k] - y[m + (i - 1) * k])
                
                if (n - m) // k > 0:
                    l_m = l_m * (n - 1) / (k ** 2 * ((n - m) // k))
                    l_sum += l_m
            
            return l_sum / k
        
        if self.config.enable_vmap:
            return vmap(l_value_for_k)(k_values)
        else:
            return jnp.array([l_value_for_k(k) for k in k_values])
    
    def _fit_fractal_dimension(self, k_values: jax.Array, l_values: jax.Array) -> Dict[str, jax.Array]:
        """Fit fractal dimension using linear regression"""
        log_k = jnp.log(k_values)
        log_l = jnp.log(l_values)
        
        # Linear regression
        n = len(log_k)
        A = jnp.column_stack([log_k, jnp.ones(n)])
        coeffs = jnp.linalg.lstsq(A, log_l, rcond=None)[0]
        
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Calculate R-squared
        y_pred = A @ coeffs
        ss_res = jnp.sum((log_l - y_pred) ** 2)
        ss_tot = jnp.sum((log_l - jnp.mean(log_l)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        
        # Fractal dimension is related to the slope
        fractal_dimension = 5 - slope  # Higuchi's relationship
        
        return {
            'fractal_dimension': fractal_dimension,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'k_values': k_values,
            'l_values': l_values
        }
    
    def analyze_single(self, y: jax.Array,
                      k_max: Optional[int] = None,
                      k_min: int = 2) -> Dict[str, jax.Array]:
        """Analyze a single time series using Higuchi method"""
        y = jnp.asarray(y, dtype=jnp.float64)
        
        if k_max is None:
            k_max = len(y) // 4
        
        k_values = jnp.arange(k_min, k_max + 1)
        l_values = self._calculate_l_values_jitted(y, k_values)
        
        results = self._fit_fractal_dimension_jitted(k_values, l_values)
        return results
    
    def analyze_batch(self, data_batch: jax.Array, **kwargs) -> List[Dict[str, jax.Array]]:
        """Analyze a batch of time series in parallel"""
        if self.config.enable_vmap:
            # For vmap, we need to handle the dictionary structure differently
            # Since JAX can't handle dictionaries in arrays, we'll use a different approach
            results = vmap(lambda y: self.analyze_single(y, **kwargs))(data_batch)
            # Convert the structured array back to a list of dictionaries
            return [dict(zip(results.keys(), [results[key][i] for key in results.keys()])) 
                   for i in range(len(data_batch))]
        else:
            return [self.analyze_single(y, **kwargs) for y in data_batch]


class JAXParallelProcessor:
    """Main class for parallel processing of multiple datasets"""
    
    def __init__(self, config: JAXAnalysisConfig):
        self.config = config
        self.device_manager = JAXDeviceManager(config)
        self.dfa_analyzer = JAXDFAAnalysis(config)
        self.higuchi_analyzer = JAXHiguchiAnalysis(config)
    
    def process_datasets(self, 
                        datasets: Dict[str, jax.Array],
                        methods: List[str] = ['dfa', 'higuchi'],
                        **kwargs) -> Dict[str, Dict[str, Any]]:
        """Process multiple datasets with multiple methods in parallel"""
        results = {}
        
        # Convert datasets to JAX arrays
        jax_datasets = {name: jnp.asarray(data, dtype=jnp.float64) 
                       for name, data in datasets.items()}
        
        # Process each method
        for method in methods:
            if method == 'dfa':
                method_results = self._process_dfa(jax_datasets, **kwargs)
            elif method == 'higuchi':
                method_results = self._process_higuchi(jax_datasets, **kwargs)
            else:
                warnings.warn(f"Unknown method: {method}")
                continue
            
            results[method] = method_results
        
        return results
    
    def _process_dfa(self, datasets: Dict[str, jax.Array], **kwargs) -> Dict[str, Any]:
        """Process datasets using DFA"""
        # Stack datasets for batch processing
        dataset_names = list(datasets.keys())
        data_batch = jnp.stack([datasets[name] for name in dataset_names])
        
        # Analyze batch
        batch_results = self.dfa_analyzer.analyze_batch(data_batch, **kwargs)
        
        # Convert back to dictionary format
        results = {}
        for i, name in enumerate(dataset_names):
            results[name] = {
                'alpha': float(batch_results[i]['alpha']),
                'intercept': float(batch_results[i]['intercept']),
                'r_squared': float(batch_results[i]['r_squared']),
                'scales': np.array(batch_results[i]['scales']),
                'fluctuations': np.array(batch_results[i]['fluctuations'])
            }
        
        return results
    
    def _process_higuchi(self, datasets: Dict[str, jax.Array], **kwargs) -> Dict[str, Any]:
        """Process datasets using Higuchi method"""
        # Stack datasets for batch processing
        dataset_names = list(datasets.keys())
        data_batch = jnp.stack([datasets[name] for name in dataset_names])
        
        # Analyze batch
        batch_results = self.higuchi_analyzer.analyze_batch(data_batch, **kwargs)
        
        # Convert back to dictionary format
        results = {}
        for i, name in enumerate(dataset_names):
            results[name] = {
                'fractal_dimension': float(batch_results[i]['fractal_dimension']),
                'slope': float(batch_results[i]['slope']),
                'r_squared': float(batch_results[i]['r_squared']),
                'k_values': np.array(batch_results[i]['k_values']),
                'l_values': np.array(batch_results[i]['l_values'])
            }
        
        return results
    
    def monte_carlo_analysis(self, 
                           data: jax.Array,
                           n_simulations: int = 1000,
                           methods: List[str] = ['dfa', 'higuchi'],
                           **kwargs) -> Dict[str, Dict[str, jax.Array]]:
        """Perform Monte Carlo analysis with JAX acceleration"""
        key = self.device_manager.split_key()
        
        # Generate bootstrap samples
        n = len(data)
        indices = random.randint(key, (n_simulations, n), 0, n)
        bootstrap_samples = data[indices]
        
        # Process bootstrap samples
        results = {}
        for method in methods:
            if method == 'dfa':
                method_results = self.dfa_analyzer.analyze_batch(bootstrap_samples, **kwargs)
                results[method] = {
                    'alphas': jnp.array([result['alpha'] for result in method_results]),
                    'r_squared': jnp.array([result['r_squared'] for result in method_results])
                }
            elif method == 'higuchi':
                method_results = self.higuchi_analyzer.analyze_batch(bootstrap_samples, **kwargs)
                results[method] = {
                    'fractal_dimensions': jnp.array([result['fractal_dimension'] for result in method_results]),
                    'r_squared': jnp.array([result['r_squared'] for result in method_results])
                }
        
        return results


# Utility functions for JAX-based analysis
def create_jax_config(use_gpu: bool = False, 
                     batch_size: int = 32,
                     num_parallel: int = 4) -> JAXAnalysisConfig:
    """Create a JAX analysis configuration"""
    return JAXAnalysisConfig(
        use_gpu=use_gpu,
        batch_size=batch_size,
        num_parallel=num_parallel,
        enable_jit=True,
        enable_vmap=True
    )


def jax_parallel_analysis(datasets: Dict[str, np.ndarray],
                         methods: List[str] = ['dfa', 'higuchi'],
                         config: Optional[JAXAnalysisConfig] = None,
                         **kwargs) -> Dict[str, Dict[str, Any]]:
    """Convenience function for parallel JAX analysis"""
    if config is None:
        config = create_jax_config()
    
    processor = JAXParallelProcessor(config)
    return processor.process_datasets(datasets, methods, **kwargs)


def jax_monte_carlo_analysis(data: np.ndarray,
                            n_simulations: int = 1000,
                            methods: List[str] = ['dfa', 'higuchi'],
                            config: Optional[JAXAnalysisConfig] = None,
                            **kwargs) -> Dict[str, Dict[str, np.ndarray]]:
    """Convenience function for JAX Monte Carlo analysis"""
    if config is None:
        config = create_jax_config()
    
    processor = JAXParallelProcessor(config)
    jax_data = jnp.asarray(data, dtype=jnp.float64)
    
    results = processor.monte_carlo_analysis(jax_data, n_simulations, methods, **kwargs)
    
    # Convert back to numpy arrays
    numpy_results = {}
    for method, method_results in results.items():
        numpy_results[method] = {
            key: np.array(value) for key, value in method_results.items()
        }
    
    return numpy_results

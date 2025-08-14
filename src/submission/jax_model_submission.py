"""
JAX-based Model Submission System

This module extends the existing model submission framework with JAX-accelerated
parallel computation capabilities for long-range dependence analysis.

Key features:
- JAX-accelerated model implementations
- Parallel processing of multiple datasets
- GPU/TPU support for large-scale computations
- Integration with existing submission framework
- Automatic performance benchmarking
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, hessian
from jax.scipy import stats as jax_stats
from jax.scipy.optimize import minimize
import jax.random as random
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import numpy as np
import time
import json
import os
from pathlib import Path
import warnings

from .model_submission import BaseEstimatorModel, ModelMetadata, ModelValidator, ModelTester, ValidationResult, SubmissionStatus, ModelRegistry
from src.analysis.jax_parallel_analysis import JAXAnalysisConfig, JAXParallelProcessor


@dataclass
class JAXModelMetadata(ModelMetadata):
    """Extended metadata for JAX-based models"""
    jax_config: Optional[JAXAnalysisConfig] = None
    gpu_required: bool = False
    tpu_required: bool = False
    parallel_capable: bool = True
    batch_processing: bool = True
    memory_optimized: bool = True


class JAXBaseEstimatorModel(BaseEstimatorModel):
    """Base class for JAX-accelerated estimator models"""
    
    def __init__(self, jax_config: Optional[JAXAnalysisConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.jax_config = jax_config or JAXAnalysisConfig()
        self.jax_processor = JAXParallelProcessor(self.jax_config)
        self.key = random.PRNGKey(42)
        
        # JIT-compile core methods if enabled
        if self.jax_config.enable_jit:
            self._jit_compile_methods()
    
    def _jit_compile_methods(self):
        """JIT-compile core computational methods"""
        # This will be overridden by subclasses
        pass
    
    def validate_input(self, data: np.ndarray) -> bool:
        """Validate input data for JAX processing"""
        if not super().validate_input(data):
            return False
        
        # Additional JAX-specific validations
        if len(data) < 16:  # Minimum for meaningful JAX computation
            return False
        
        # Check for data types that JAX can handle
        try:
            jnp.asarray(data, dtype=jnp.float64)
        except:
            return False
        
        return True
    
    def preprocess_data(self, data: np.ndarray) -> jax.Array:
        """Preprocess data for JAX computation"""
        # Convert to JAX array with appropriate dtype
        jax_data = jnp.asarray(data, dtype=jnp.float64)
        
        # Remove any NaN or infinite values
        finite_mask = jnp.isfinite(jax_data)
        if jnp.sum(finite_mask) < len(jax_data) * 0.9:  # Allow up to 10% non-finite values
            warnings.warn("Many non-finite values detected, interpolating...")
            # Simple interpolation for non-finite values
            indices = jnp.arange(len(jax_data))
            finite_indices = indices[finite_mask]
            finite_values = jax_data[finite_mask]
            
            def interpolate_value(idx):
                return jnp.interp(idx, finite_indices, finite_values)
            
            jax_data = vmap(interpolate_value)(indices)
        
        return jax_data
    
    def estimate_confidence_intervals_jax(self, 
                                        data: jax.Array,
                                        n_bootstrap: int = 1000,
                                        confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Estimate confidence intervals using JAX-accelerated bootstrap"""
        key = self.key
        
        # Generate bootstrap samples
        n = len(data)
        bootstrap_indices = random.randint(key, (n_bootstrap, n), 0, n)
        bootstrap_samples = data[bootstrap_indices]
        
        # Analyze bootstrap samples in parallel
        bootstrap_results = self._analyze_bootstrap_batch(bootstrap_samples)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        intervals = {}
        for metric, values in bootstrap_results.items():
            lower = jnp.percentile(values, lower_percentile)
            upper = jnp.percentile(values, upper_percentile)
            intervals[metric] = (float(lower), float(upper))
        
        return intervals
    
    def _analyze_bootstrap_batch(self, bootstrap_samples: jax.Array) -> Dict[str, jax.Array]:
        """Analyze a batch of bootstrap samples - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _analyze_bootstrap_batch")
    
    def get_computation_info(self) -> Dict[str, Any]:
        """Get information about JAX computation setup"""
        return {
            'jax_config': {
                'use_gpu': self.jax_config.use_gpu,
                'use_tpu': self.jax_config.use_tpu,
                'batch_size': self.jax_config.batch_size,
                'num_parallel': self.jax_config.num_parallel,
                'enable_jit': self.jax_config.enable_jit,
                'enable_vmap': self.jax_config.enable_vmap
            },
            'available_devices': {
                'cpu': jax.device_count('cpu'),
                'gpu': jax.device_count('gpu'),
                'tpu': jax.device_count('tpu')
            },
            'current_device': str(jax.devices()[0])
        }


class JAXDFAModel(JAXBaseEstimatorModel):
    """JAX-accelerated DFA model implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = None
        self.intercept = None
        self.r_squared = None
        self.scales = None
        self.fluctuations = None
    
    def _jit_compile_methods(self):
        """JIT-compile DFA-specific methods"""
        self._profile_jitted = jit(self._profile)
        self._poly_detrend_jitted = jit(self._poly_detrend)
        # Disable JIT for problematic functions
        self._calculate_fluctuation_jitted = self._calculate_fluctuation
        self._fit_scaling_jitted = jit(self._fit_scaling)
    
    def _profile(self, y: jax.Array) -> jax.Array:
        """Calculate the profile (cumulative sum of centered data)"""
        y_centered = y - jnp.mean(y)
        return jnp.cumsum(y_centered)
    
    def _poly_detrend(self, x: jax.Array, order: int) -> Tuple[jax.Array, jax.Array]:
        """Polynomial detrending using JAX"""
        t = jnp.arange(len(x), dtype=jnp.float64)
        o = jnp.minimum(order, len(x) - 1)
        
        if o <= 0:
            trend = jnp.full_like(x, jnp.mean(x))
            return trend, x - trend
        
        # Manual polynomial fitting for JAX compatibility
        A = jnp.vander(t, o + 1)
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
        
        if self.jax_config.enable_vmap:
            # Use static shape for vmap
            max_segments = n // scale
            segment_indices = jnp.arange(max_segments)
            fluctuations = vmap(segment_fluctuation)(segment_indices)
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
    
    def fit(self, data: np.ndarray) -> 'JAXDFAModel':
        """Fit the DFA model to the data"""
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        
        # Preprocess data
        jax_data = self.preprocess_data(data)
        
        # Get parameters
        min_scale = self.parameters.get('min_scale', 4)
        max_scale = self.parameters.get('max_scale', None)
        num_scales = self.parameters.get('num_scales', 20)
        
        if max_scale is None:
            max_scale = max(min(len(jax_data) // 4, 1024), min_scale + 1)
        
        # Generate scales
        scales = jnp.unique(jnp.logspace(jnp.log2(min_scale), jnp.log2(max_scale), num_scales, base=2).astype(int))
        
        # Calculate profile
        profile = self._profile_jitted(jax_data)
        
        # Calculate fluctuations for each scale
        if self.jax_config.enable_vmap:
            flucts = vmap(lambda s: self._calculate_fluctuation_jitted(profile, s))(scales)
        else:
            flucts = jnp.array([self._calculate_fluctuation_jitted(profile, s) for s in scales])
        
        # Fit scaling relationship
        results = self._fit_scaling_jitted(scales, flucts)
        
        # Store results
        self.alpha = float(results['alpha'])
        self.intercept = float(results['intercept'])
        self.r_squared = float(results['r_squared'])
        self.scales = np.array(results['scales'])
        self.fluctuations = np.array(results['fluctuations'])
        
        self.fitted = True
        return self
    
    def estimate_hurst(self) -> float:
        """Estimate the Hurst exponent (H = alpha)"""
        if not self.fitted:
            raise ValueError("Model must be fitted before estimating Hurst exponent")
        return self.alpha
    
    def estimate_alpha(self) -> float:
        """Estimate the alpha parameter"""
        if not self.fitted:
            raise ValueError("Model must be fitted before estimating alpha")
        return self.alpha
    
    def get_confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
        """Get confidence intervals for estimates"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting confidence intervals")
        
        # Use JAX-accelerated bootstrap
        jax_data = jnp.asarray(self.results.get('data', []), dtype=jnp.float64)
        if len(jax_data) == 0:
            return {'alpha': (self.alpha, self.alpha), 'hurst': (self.alpha, self.alpha)}
        
        return self.estimate_confidence_intervals_jax(jax_data)
    
    def get_quality_metrics(self) -> Dict[str, float]:
        """Get quality metrics for the estimation"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting quality metrics")
        
        return {
            'r_squared': self.r_squared,
            'alpha': self.alpha,
            'intercept': self.intercept,
            'num_scales': len(self.scales),
            'min_scale': float(self.scales[0]),
            'max_scale': float(self.scales[-1])
        }
    
    def _analyze_bootstrap_batch(self, bootstrap_samples: jax.Array) -> Dict[str, jax.Array]:
        """Analyze a batch of bootstrap samples for DFA"""
        def analyze_single_bootstrap(sample):
            # Generate scales
            min_scale = self.parameters.get('min_scale', 4)
            max_scale = max(min(len(sample) // 4, 1024), min_scale + 1)
            num_scales = self.parameters.get('num_scales', 20)
            
            scales = jnp.unique(jnp.logspace(jnp.log2(min_scale), jnp.log2(max_scale), num_scales, base=2).astype(int))
            
            # Calculate profile
            profile = self._profile_jitted(sample)
            
            # Calculate fluctuations
            if self.jax_config.enable_vmap:
                flucts = vmap(lambda s: self._calculate_fluctuation_jitted(profile, s))(scales)
            else:
                flucts = jnp.array([self._calculate_fluctuation_jitted(profile, s) for s in scales])
            
            # Fit scaling
            results = self._fit_scaling_jitted(scales, flucts)
            return results['alpha']
        
        if self.jax_config.enable_vmap:
            alphas = vmap(analyze_single_bootstrap)(bootstrap_samples)
        else:
            alphas = jnp.array([analyze_single_bootstrap(sample) for sample in bootstrap_samples])
        
        return {'alpha': alphas}


class JAXHiguchiModel(JAXBaseEstimatorModel):
    """JAX-accelerated Higuchi model implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fractal_dimension = None
        self.slope = None
        self.intercept = None
        self.r_squared = None
        self.k_values = None
        self.l_values = None
    
    def _jit_compile_methods(self):
        """JIT-compile Higuchi-specific methods"""
        # Disable JIT for problematic functions
        self._calculate_l_values_jitted = self._calculate_l_values
        self._fit_fractal_dimension_jitted = jit(self._fit_fractal_dimension)
    
    def _calculate_l_values(self, y: jax.Array, k_values: jax.Array) -> jax.Array:
        """Calculate L(k) values for Higuchi method"""
        n = len(y)
        
        def l_value_for_k(k):
            l_sum = 0.0
            
            for m in range(k):
                l_m = 0.0
                for i in range(1, (n - m) // k):
                    l_m += jnp.abs(y[m + i * k] - y[m + (i - 1) * k])
                
                if (n - m) // k > 0:
                    l_m = l_m * (n - 1) / (k ** 2 * ((n - m) // k))
                    l_sum += l_m
            
            return l_sum / k
        
        if self.jax_config.enable_vmap:
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
    
    def fit(self, data: np.ndarray) -> 'JAXHiguchiModel':
        """Fit the Higuchi model to the data"""
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        
        # Preprocess data
        jax_data = self.preprocess_data(data)
        
        # Get parameters
        k_max = self.parameters.get('k_max', len(jax_data) // 4)
        k_min = self.parameters.get('k_min', 2)
        
        k_values = jnp.arange(k_min, k_max + 1)
        l_values = self._calculate_l_values_jitted(jax_data, k_values)
        
        results = self._fit_fractal_dimension_jitted(k_values, l_values)
        
        # Store results
        self.fractal_dimension = float(results['fractal_dimension'])
        self.slope = float(results['slope'])
        self.intercept = float(results['intercept'])
        self.r_squared = float(results['r_squared'])
        self.k_values = np.array(results['k_values'])
        self.l_values = np.array(results['l_values'])
        
        self.fitted = True
        return self
    
    def estimate_hurst(self) -> float:
        """Estimate the Hurst exponent from fractal dimension"""
        if not self.fitted:
            raise ValueError("Model must be fitted before estimating Hurst exponent")
        # Convert fractal dimension to Hurst exponent
        # D = 2 - H for 1D time series
        return 2 - self.fractal_dimension
    
    def estimate_alpha(self) -> float:
        """Estimate the alpha parameter (same as Hurst for Higuchi)"""
        return self.estimate_hurst()
    
    def get_confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
        """Get confidence intervals for estimates"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting confidence intervals")
        
        # Use JAX-accelerated bootstrap
        jax_data = jnp.asarray(self.results.get('data', []), dtype=jnp.float64)
        if len(jax_data) == 0:
            return {'fractal_dimension': (self.fractal_dimension, self.fractal_dimension)}
        
        return self.estimate_confidence_intervals_jax(jax_data)
    
    def get_quality_metrics(self) -> Dict[str, float]:
        """Get quality metrics for the estimation"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting quality metrics")
        
        return {
            'fractal_dimension': self.fractal_dimension,
            'slope': self.slope,
            'r_squared': self.r_squared,
            'num_k_values': len(self.k_values),
            'min_k': float(self.k_values[0]),
            'max_k': float(self.k_values[-1])
        }
    
    def _analyze_bootstrap_batch(self, bootstrap_samples: jax.Array) -> Dict[str, jax.Array]:
        """Analyze a batch of bootstrap samples for Higuchi"""
        def analyze_single_bootstrap(sample):
            k_max = self.parameters.get('k_max', len(sample) // 4)
            k_min = self.parameters.get('k_min', 2)
            
            k_values = jnp.arange(k_min, k_max + 1)
            l_values = self._calculate_l_values_jitted(sample, k_values)
            results = self._fit_fractal_dimension_jitted(k_values, l_values)
            return results['fractal_dimension']
        
        if self.jax_config.enable_vmap:
            fractal_dimensions = vmap(analyze_single_bootstrap)(bootstrap_samples)
        else:
            fractal_dimensions = jnp.array([analyze_single_bootstrap(sample) for sample in bootstrap_samples])
        
        return {'fractal_dimension': fractal_dimensions}


class JAXModelValidator(ModelValidator):
    """Extended validator for JAX-based models"""
    
    def validate_jax_model(self, model_class: type) -> List[ValidationResult]:
        """Validate a JAX-based model class"""
        results = []
        
        # Basic model validation
        basic_results = self.validate_model_class(model_class)
        results.extend(basic_results)
        
        # JAX-specific validations
        jax_results = self._validate_jax_specific_requirements(model_class)
        results.extend(jax_results)
        
        return results
    
    def _validate_jax_specific_requirements(self, model_class: type) -> List[ValidationResult]:
        """Validate JAX-specific requirements"""
        results = []
        
        # Check if it inherits from JAXBaseEstimatorModel
        if not issubclass(model_class, JAXBaseEstimatorModel):
            results.append(ValidationResult(
                is_valid=False,
                status=SubmissionStatus.REJECTED,
                errors=["JAX models must inherit from JAXBaseEstimatorModel"],
                warnings=[]
            ))
        
        # Check for JAX-specific methods
        required_jax_methods = ['_jit_compile_methods', 'estimate_confidence_intervals_jax']
        missing_methods = []
        for method in required_jax_methods:
            if not hasattr(model_class, method):
                missing_methods.append(method)
        
        if missing_methods:
            results.append(ValidationResult(
                is_valid=False,
                status=SubmissionStatus.REJECTED,
                errors=[f"JAX model missing required methods: {missing_methods}"],
                warnings=[]
            ))
        
        return results


class JAXModelTester(ModelTester):
    """Extended tester for JAX-based models with performance benchmarking"""
    
    def __init__(self, test_data: Optional[Dict[str, np.ndarray]] = None, jax_config: Optional[JAXAnalysisConfig] = None):
        super().__init__(test_data)
        self.jax_config = jax_config or JAXAnalysisConfig()
    
    def benchmark_performance(self, model_class: type, **kwargs) -> Dict[str, Any]:
        """Benchmark the performance of a JAX model"""
        benchmark_results = {
            'jax_config': self.jax_config.__dict__,
            'performance_metrics': {},
            'memory_usage': {},
            'scalability': {}
        }
        
        # Test with different dataset sizes
        dataset_sizes = [100, 500, 1000, 5000, 10000]
        
        for size in dataset_sizes:
            # Generate test data
            test_data = np.random.normal(0, 1, size)
            
            # Time the analysis
            start_time = time.time()
            model = model_class(jax_config=self.jax_config, **kwargs)
            model.fit(test_data)
            end_time = time.time()
            
            benchmark_results['performance_metrics'][f'size_{size}'] = {
                'fit_time': end_time - start_time,
                'data_size': size
            }
        
        return benchmark_results
    
    def test(self, model) -> Dict[str, Any]:
        """Test a JAX model with enhanced performance metrics"""
        # Run standard tests
        standard_results = super().test(model)
        
        # Add JAX-specific performance metrics
        if hasattr(model, 'get_computation_info'):
            computation_info = model.get_computation_info()
            for dataset_name in standard_results:
                if standard_results[dataset_name].get('success', False):
                    standard_results[dataset_name]['computation_info'] = computation_info
        
        return standard_results


class JAXModelSubmission:
    """Main class for JAX model submissions with parallel computation support"""
    
    def __init__(self, registry_path: str = "models/registry.json", jax_config: Optional[JAXAnalysisConfig] = None):
        self.validator = JAXModelValidator()
        self.tester = JAXModelTester(jax_config=jax_config)
        self.registry = ModelRegistry(registry_path)
        self.jax_config = jax_config or JAXAnalysisConfig()
    
    def submit_jax_model(self, 
                        model_file: str,
                        metadata: JAXModelMetadata,
                        benchmark_performance: bool = True,
                        test_model: bool = True) -> Dict[str, Any]:
        """Submit a JAX-based model with performance benchmarking"""
        submission_result = {
            "success": False,
            "validation_results": [],
            "test_results": None,
            "performance_evaluation": None,
            "benchmark_results": None,
            "model_id": None,
            "message": ""
        }
        
        try:
            # Validate JAX model
            validation_results = self.validator.validate_jax_model_file(model_file)
            submission_result["validation_results"].extend(validation_results)
            
            if not all(result.is_valid for result in validation_results):
                submission_result["message"] = "JAX model validation failed"
                return submission_result
            
            # Import and test model
            model_class = self._import_jax_model(model_file)
            
            if test_model:
                test_results = self.tester.test(model_class(jax_config=self.jax_config, **metadata.parameters))
                performance = self.tester.evaluate_performance(test_results)
                
                submission_result["test_results"] = test_results
                submission_result["performance_evaluation"] = performance
            
            # Benchmark performance if requested
            if benchmark_performance:
                benchmark_results = self.tester.benchmark_performance(model_class, **metadata.parameters)
                submission_result["benchmark_results"] = benchmark_results
            
            # Register model
            metadata.file_path = model_file
            model_id = f"{metadata.name}_{metadata.version}"
            
            if self.registry.register_model(metadata):
                submission_result["model_id"] = model_id
                submission_result["success"] = True
                submission_result["message"] = "JAX model submitted successfully"
            else:
                submission_result["message"] = "Model already exists in registry"
            
        except Exception as e:
            submission_result["message"] = f"JAX model submission failed: {str(e)}"
        
        return submission_result
    
    def _import_jax_model(self, model_file: str) -> type:
        """Import a JAX model from file"""
        # Implementation similar to the original but with JAX-specific handling
        import importlib.util
        import sys
        
        spec = importlib.util.spec_from_file_location("jax_submitted_model", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find JAX model class
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, JAXBaseEstimatorModel):
                return attr
        
        raise ValueError("No valid JAX model class found in file")


# Utility functions for JAX model submission
def create_jax_model_metadata(name: str,
                             version: str,
                             author: str,
                             description: str,
                             algorithm_type: str,
                             parameters: Dict[str, Any],
                             dependencies: List[str],
                             file_path: str,
                             jax_config: Optional[JAXAnalysisConfig] = None,
                             **kwargs) -> JAXModelMetadata:
    """Create metadata for a JAX model submission"""
    return JAXModelMetadata(
        name=name,
        version=version,
        author=author,
        description=description,
        algorithm_type=algorithm_type,
        parameters=parameters,
        dependencies=dependencies,
        file_path=file_path,
        jax_config=jax_config,
        **kwargs
    )


def submit_jax_model_convenience(model_file: str,
                                name: str,
                                version: str,
                                author: str,
                                description: str,
                                algorithm_type: str,
                                parameters: Dict[str, Any],
                                dependencies: List[str],
                                jax_config: Optional[JAXAnalysisConfig] = None,
                                **kwargs) -> Dict[str, Any]:
    """Convenience function for submitting JAX models"""
    metadata = create_jax_model_metadata(
        name=name,
        version=version,
        author=author,
        description=description,
        algorithm_type=algorithm_type,
        parameters=parameters,
        dependencies=dependencies,
        file_path=model_file,
        jax_config=jax_config,
        **kwargs
    )
    
    submission_system = JAXModelSubmission(jax_config=jax_config)
    return submission_system.submit_jax_model(model_file, metadata, **kwargs)

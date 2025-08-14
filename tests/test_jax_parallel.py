"""
Tests for JAX Parallel Computation Implementation

This module contains comprehensive tests for the JAX-accelerated parallel computation
capabilities for long-range dependence analysis.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.jax_parallel_analysis import (
    JAXAnalysisConfig,
    JAXDeviceManager,
    JAXDFAAnalysis,
    JAXHiguchiAnalysis,
    JAXParallelProcessor,
    create_jax_config,
    jax_parallel_analysis,
    jax_monte_carlo_analysis
)
from submission.jax_model_submission import (
    JAXModelMetadata,
    JAXBaseEstimatorModel,
    JAXDFAModel,
    JAXHiguchiModel,
    JAXModelValidator,
    JAXModelTester,
    JAXModelSubmission,
    create_jax_model_metadata
)


class TestJAXAnalysisConfig:
    """Test JAX analysis configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = JAXAnalysisConfig()
        
        assert config.use_gpu is False
        assert config.use_tpu is False
        assert config.batch_size == 32
        assert config.num_parallel == 4
        assert config.precision == 'float64'
        assert config.enable_jit is True
        assert config.enable_vmap is True
        assert config.enable_pmap is False
        assert config.memory_efficient is True
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = JAXAnalysisConfig(
            use_gpu=True,
            batch_size=64,
            num_parallel=8,
            precision='float32',
            enable_jit=False
        )
        
        assert config.use_gpu is True
        assert config.batch_size == 64
        assert config.num_parallel == 8
        assert config.precision == 'float32'
        assert config.enable_jit is False
    
    def test_create_jax_config(self):
        """Test convenience function for creating config"""
        config = create_jax_config(use_gpu=True, batch_size=128, num_parallel=16)
        
        assert config.use_gpu is True
        assert config.batch_size == 128
        assert config.num_parallel == 16
        assert config.enable_jit is True
        assert config.enable_vmap is True


class TestJAXDeviceManager:
    """Test JAX device manager"""
    
    def test_device_setup(self):
        """Test device setup"""
        config = JAXAnalysisConfig()
        manager = JAXDeviceManager(config)
        
        assert len(manager.devices) > 0
        assert manager.key is not None
    
    def test_get_parallel_devices(self):
        """Test getting parallel devices"""
        config = JAXAnalysisConfig(num_parallel=2)
        manager = JAXDeviceManager(config)
        
        parallel_devices = manager.get_parallel_devices()
        assert len(parallel_devices) <= 2
    
    def test_split_key(self):
        """Test random key splitting"""
        config = JAXAnalysisConfig()
        manager = JAXDeviceManager(config)
        
        original_key = manager.key
        subkey = manager.split_key()
        
        assert subkey is not None
        assert manager.key is not original_key


class TestJAXDFAAnalysis:
    """Test JAX DFA analysis"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = JAXAnalysisConfig(enable_jit=False, enable_vmap=False)
        self.analyzer = JAXDFAAnalysis(self.config)
    
    def test_profile_calculation(self):
        """Test profile calculation"""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        profile = self.analyzer._profile(data)
        
        expected = jnp.cumsum(data - jnp.mean(data))
        np.testing.assert_array_almost_equal(profile, expected)
    
    def test_poly_detrend(self):
        """Test polynomial detrending"""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trend, detrended = self.analyzer._poly_detrend(x, 1)
        
        assert trend.shape == x.shape
        assert detrended.shape == x.shape
        assert jnp.allclose(jnp.mean(detrended), 0.0, atol=1e-10)
    
    def test_calculate_fluctuation(self):
        """Test fluctuation calculation"""
        profile = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        scale = 2
        
        fluct = self.analyzer._calculate_fluctuation(profile, scale)
        assert fluct > 0
        assert jnp.isfinite(fluct)
    
    def test_fit_scaling(self):
        """Test scaling relationship fitting"""
        scales = jnp.array([4, 8, 16, 32])
        flucts = jnp.array([1.0, 2.0, 4.0, 8.0])  # Perfect scaling
        
        results = self.analyzer._fit_scaling(scales, flucts)
        
        assert 'alpha' in results
        assert 'intercept' in results
        assert 'r_squared' in results
        assert results['r_squared'] > 0.9  # Should be very high for perfect scaling
    
    def test_analyze_single(self):
        """Test single dataset analysis"""
        # Generate test data with known properties
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        results = self.analyzer.analyze_single(data)
        
        assert 'alpha' in results
        assert 'intercept' in results
        assert 'r_squared' in results
        assert 'scales' in results
        assert 'fluctuations' in results
        assert jnp.isfinite(results['alpha'])
        assert results['r_squared'] > 0
    
    def test_analyze_batch(self):
        """Test batch analysis"""
        # Generate test data
        np.random.seed(42)
        data_batch = jnp.array([
            np.random.normal(0, 1, 100),
            np.random.normal(0, 1, 100),
            np.random.normal(0, 1, 100)
        ])
        
        results = self.analyzer.analyze_batch(data_batch)
        
        # Now results is a list of dictionaries
        assert len(results) == 3
        assert 'alpha' in results[0]
        assert 'intercept' in results[0]
        assert 'r_squared' in results[0]
        assert all('alpha' in result for result in results)


class TestJAXHiguchiAnalysis:
    """Test JAX Higuchi analysis"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = JAXAnalysisConfig(enable_jit=False, enable_vmap=False)
        self.analyzer = JAXHiguchiAnalysis(self.config)
    
    def test_calculate_l_values(self):
        """Test L(k) values calculation"""
        y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        k_values = jnp.array([2, 3, 4])
        
        l_values = self.analyzer._calculate_l_values(y, k_values)
        
        assert len(l_values) == len(k_values)
        assert jnp.all(l_values > 0)
        assert jnp.all(jnp.isfinite(l_values))
    
    def test_fit_fractal_dimension(self):
        """Test fractal dimension fitting"""
        k_values = jnp.array([2, 4, 8, 16])
        l_values = jnp.array([1.0, 2.0, 4.0, 8.0])  # Perfect scaling
        
        results = self.analyzer._fit_fractal_dimension(k_values, l_values)
        
        assert 'fractal_dimension' in results
        assert 'slope' in results
        assert 'intercept' in results
        assert 'r_squared' in results
        assert results['r_squared'] > 0.9
    
    def test_analyze_single(self):
        """Test single dataset analysis"""
        # Generate test data
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        results = self.analyzer.analyze_single(data)
        
        assert 'fractal_dimension' in results
        assert 'slope' in results
        assert 'r_squared' in results
        assert 'k_values' in results
        assert 'l_values' in results
        assert jnp.isfinite(results['fractal_dimension'])
        assert results['r_squared'] > 0
    
    def test_analyze_batch(self):
        """Test batch analysis"""
        # Generate test data
        np.random.seed(42)
        data_batch = jnp.array([
            np.random.normal(0, 1, 100),
            np.random.normal(0, 1, 100),
            np.random.normal(0, 1, 100)
        ])
        
        results = self.analyzer.analyze_batch(data_batch)
        
        # Now results is a list of dictionaries
        assert len(results) == 3
        assert 'fractal_dimension' in results[0]
        assert 'slope' in results[0]
        assert 'r_squared' in results[0]
        assert all('fractal_dimension' in result for result in results)


class TestJAXParallelProcessor:
    """Test JAX parallel processor"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = JAXAnalysisConfig(enable_jit=False, enable_vmap=False)
        self.processor = JAXParallelProcessor(self.config)
    
    def test_process_datasets(self):
        """Test processing multiple datasets"""
        # Generate test datasets
        np.random.seed(42)
        datasets = {
            'dataset1': np.random.normal(0, 1, 100),
            'dataset2': np.random.normal(0, 1, 100),
            'dataset3': np.random.normal(0, 1, 100)
        }
        
        results = self.processor.process_datasets(
            datasets=datasets,
            methods=['dfa', 'higuchi']
        )
        
        assert 'dfa' in results
        assert 'higuchi' in results
        assert len(results['dfa']) == 3
        assert len(results['higuchi']) == 3
    
    def test_monte_carlo_analysis(self):
        """Test Monte Carlo analysis"""
        # Generate test data
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        results = self.processor.monte_carlo_analysis(
            data=jnp.asarray(data),
            n_simulations=10,  # Small number for testing
            methods=['dfa', 'higuchi']
        )
        
        assert 'dfa' in results
        assert 'higuchi' in results
        assert len(results['dfa']['alphas']) == 10
        assert len(results['higuchi']['fractal_dimensions']) == 10


class TestJAXModelClasses:
    """Test JAX model classes"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = JAXAnalysisConfig(enable_jit=False, enable_vmap=False)
    
    def test_jax_base_estimator_model(self):
        """Test base JAX estimator model"""
        # Test that we can't instantiate the abstract base class
        with pytest.raises(TypeError):
            model = JAXBaseEstimatorModel(jax_config=self.config)
        
        # Test that concrete implementations work
        model = JAXDFAModel(jax_config=self.config)
        assert model.jax_config == self.config
        assert model.key is not None
        assert hasattr(model, 'validate_input')
        assert hasattr(model, 'preprocess_data')
    
    def test_jax_dfa_model(self):
        """Test JAX DFA model"""
        model = JAXDFAModel(jax_config=self.config)
        
        # Test fitting
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        model.fit(data)
        
        assert model.fitted is True
        assert model.alpha is not None
        assert model.r_squared is not None
        
        # Test estimation
        alpha = model.estimate_alpha()
        hurst = model.estimate_hurst()
        
        assert jnp.isfinite(alpha)
        assert jnp.isfinite(hurst)
        assert alpha == hurst  # For DFA, alpha = H
        
        # Test quality metrics
        metrics = model.get_quality_metrics()
        assert 'r_squared' in metrics
        assert 'alpha' in metrics
        assert 'intercept' in metrics
    
    def test_jax_higuchi_model(self):
        """Test JAX Higuchi model"""
        model = JAXHiguchiModel(jax_config=self.config)
        
        # Test fitting
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        model.fit(data)
        
        assert model.fitted is True
        assert model.fractal_dimension is not None
        assert model.r_squared is not None
        
        # Test estimation
        hurst = model.estimate_hurst()
        alpha = model.estimate_alpha()
        
        assert jnp.isfinite(hurst)
        assert jnp.isfinite(alpha)
        assert hurst == alpha  # For Higuchi, H = alpha
        
        # Test quality metrics
        metrics = model.get_quality_metrics()
        assert 'fractal_dimension' in metrics
        assert 'slope' in metrics
        assert 'r_squared' in metrics


class TestJAXModelSubmission:
    """Test JAX model submission system"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = JAXAnalysisConfig(enable_jit=False, enable_vmap=False)
    
    def test_jax_model_metadata(self):
        """Test JAX model metadata"""
        metadata = JAXModelMetadata(
            name="TestModel",
            version="1.0.0",
            author="Test Author",
            description="Test description",
            algorithm_type="dfa",
            parameters={'param1': 1},
            dependencies=['jax'],
            file_path="test.py",
            jax_config=self.config,
            gpu_required=False,
            parallel_capable=True
        )
        
        assert metadata.name == "TestModel"
        assert metadata.jax_config == self.config
        assert metadata.gpu_required is False
        assert metadata.parallel_capable is True
    
    def test_create_jax_model_metadata(self):
        """Test convenience function for creating metadata"""
        metadata = create_jax_model_metadata(
            name="TestModel",
            version="1.0.0",
            author="Test Author",
            description="Test description",
            algorithm_type="dfa",
            parameters={'param1': 1},
            dependencies=['jax'],
            file_path="test.py",
            jax_config=self.config
        )
        
        assert metadata.name == "TestModel"
        assert metadata.jax_config == self.config
    
    def test_jax_model_validator(self):
        """Test JAX model validator"""
        validator = JAXModelValidator()
        
        # Test validation of JAX model class
        results = validator.validate_jax_model(JAXDFAModel)
        
        # Should have some validation results
        assert len(results) > 0
    
    def test_jax_model_tester(self):
        """Test JAX model tester"""
        tester = JAXModelTester(jax_config=self.config)
        
        # Test performance benchmarking
        benchmark_results = tester.benchmark_performance(JAXDFAModel)
        
        assert 'jax_config' in benchmark_results
        assert 'performance_metrics' in benchmark_results
    
    def test_jax_model_submission(self):
        """Test JAX model submission"""
        submission_system = JAXModelSubmission(jax_config=self.config)
        
        assert submission_system.jax_config == self.config
        assert hasattr(submission_system, 'validator')
        assert hasattr(submission_system, 'tester')
        assert hasattr(submission_system, 'registry')


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_jax_parallel_analysis(self):
        """Test convenience function for parallel analysis"""
        # Generate test datasets
        np.random.seed(42)
        datasets = {
            'dataset1': np.random.normal(0, 1, 100),
            'dataset2': np.random.normal(0, 1, 100)
        }
        
        results = jax_parallel_analysis(
            datasets=datasets,
            methods=['dfa'],
            config=JAXAnalysisConfig(enable_jit=False, enable_vmap=False)
        )
        
        assert 'dfa' in results
        assert len(results['dfa']) == 2
    
    def test_jax_monte_carlo_analysis(self):
        """Test convenience function for Monte Carlo analysis"""
        # Generate test data
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        results = jax_monte_carlo_analysis(
            data=data,
            n_simulations=10,  # Small number for testing
            methods=['dfa'],
            config=JAXAnalysisConfig(enable_jit=False, enable_vmap=False)
        )
        
        assert 'dfa' in results
        assert len(results['dfa']['alphas']) == 10


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_analysis(self):
        """Test end-to-end analysis workflow"""
        # Generate test data
        np.random.seed(42)
        datasets = {
            'white_noise': np.random.normal(0, 1, 200),
            'fbm_low': np.cumsum(np.random.normal(0, 1, 200)) * 0.1,
            'fbm_high': np.cumsum(np.random.normal(0, 1, 200)) * 0.9
        }
        
        # Configure JAX
        config = JAXAnalysisConfig(enable_jit=False, enable_vmap=False)
        
        # Perform analysis
        results = jax_parallel_analysis(
            datasets=datasets,
            methods=['dfa', 'higuchi'],
            config=config
        )
        
        # Check results
        assert 'dfa' in results
        assert 'higuchi' in results
        
        for method in ['dfa', 'higuchi']:
            for dataset_name, result in results[method].items():
                if method == 'dfa':
                    assert 'alpha' in result
                    assert 'r_squared' in result
                    assert jnp.isfinite(result['alpha'])
                elif method == 'higuchi':
                    assert 'fractal_dimension' in result
                    assert 'r_squared' in result
                    assert jnp.isfinite(result['fractal_dimension'])
    
    def test_model_workflow(self):
        """Test complete model workflow"""
        # Create model
        config = JAXAnalysisConfig(enable_jit=False, enable_vmap=False)
        model = JAXDFAModel(jax_config=config)
        
        # Generate test data
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        
        # Fit model
        model.fit(data)
        
        # Get estimates
        alpha = model.estimate_alpha()
        hurst = model.estimate_hurst()
        metrics = model.get_quality_metrics()
        
        # Validate results
        assert jnp.isfinite(alpha)
        assert jnp.isfinite(hurst)
        assert alpha == hurst
        assert metrics['r_squared'] > 0
        assert metrics['r_squared'] <= 1


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_input_data(self):
        """Test handling of invalid input data"""
        config = JAXAnalysisConfig(enable_jit=False, enable_vmap=False)
        model = JAXDFAModel(jax_config=config)
        
        # Test with too short data
        with pytest.raises(ValueError):
            model.fit(np.array([1, 2, 3]))  # Too short
        
        # Test with invalid data
        with pytest.raises(ValueError):
            model.fit(np.array([np.nan, 1, 2, 3, 4, 5]))  # Contains NaN
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration"""
        # Test with invalid batch size
        with pytest.raises(ValueError):
            JAXAnalysisConfig(batch_size=0)
        
        # Test with invalid parallel count
        with pytest.raises(ValueError):
            JAXAnalysisConfig(num_parallel=0)


if __name__ == "__main__":
    pytest.main([__file__])

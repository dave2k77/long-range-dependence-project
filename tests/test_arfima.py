"""
Tests for ARFIMA model implementation.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.arfima_modelling import (
    ARFIMAModel, 
    ARFIMAParams, 
    estimate_arfima_order, 
    arfima_simulation
)


class TestARFIMAParams:
    """Test ARFIMAParams dataclass."""
    
    def test_arfima_params_creation(self):
        """Test creating ARFIMAParams instance."""
        params = ARFIMAParams(
            d=0.3,
            p=1,
            q=1,
            ar_params=np.array([0.5]),
            ma_params=np.array([0.3]),
            intercept=1.0,
            sigma2=2.0
        )
        
        assert params.d == 0.3
        assert params.p == 1
        assert params.q == 1
        assert np.array_equal(params.ar_params, np.array([0.5]))
        assert np.array_equal(params.ma_params, np.array([0.3]))
        assert params.intercept == 1.0
        assert params.sigma2 == 2.0


class TestARFIMAModel:
    """Test ARFIMA model functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n = 1000
        self.test_data = np.random.normal(0, 1, self.n)
        
    def test_model_initialization(self):
        """Test ARFIMA model initialization."""
        model = ARFIMAModel(p=1, d=0.3, q=1)
        
        assert model.p == 1
        assert model.d == 0.3
        assert model.q == 1
        assert model.params is None
        assert not model.is_fitted
        
    def test_fractional_difference(self):
        """Test fractional differencing."""
        model = ARFIMAModel()
        x = np.array([1, 2, 3, 4, 5])
        
        # Test with d=0 (should return original series)
        result = model._fractional_difference(x, 0)
        np.testing.assert_array_almost_equal(result, x)
        
        # Test with d=0.5
        result = model._fractional_difference(x, 0.5)
        assert len(result) == len(x)
        assert not np.array_equal(result, x)  # Should be different
        
    def test_fractional_integrate(self):
        """Test fractional integration."""
        model = ARFIMAModel()
        x = np.array([1, 2, 3, 4, 5])
        d = 0.3
        
        # Test that integration is inverse of differencing
        diffed = model._fractional_difference(x, d)
        integrated = model._fractional_integrate(diffed, d)
        
        # Should be approximately equal (within numerical precision)
        np.testing.assert_array_almost_equal(x, integrated, decimal=5)
        
    def test_model_fitting(self):
        """Test model fitting."""
        model = ARFIMAModel(p=1, d=0.3, q=1)
        
        # Create synthetic ARFIMA data
        test_data = arfima_simulation(
            n=500, 
            d=0.3, 
            ar_params=np.array([0.5]), 
            ma_params=np.array([0.3]), 
            sigma=1.0,
            seed=42
        )
        
        # Fit model
        fitted_model = model.fit(test_data)
        
        assert fitted_model.is_fitted
        assert fitted_model.params is not None
        assert fitted_model.fitted_values is not None
        assert fitted_model.residuals is not None
        assert fitted_model.log_likelihood is not None
        assert fitted_model.aic is not None
        assert fitted_model.bic is not None
        
    def test_model_fitting_with_pandas(self):
        """Test model fitting with pandas Series."""
        model = ARFIMAModel(p=1, d=0.3, q=1)
        test_series = pd.Series(self.test_data)
        
        fitted_model = model.fit(test_series)
        
        assert fitted_model.is_fitted
        assert isinstance(fitted_model.fitted_values, np.ndarray)
        
    def test_forecasting(self):
        """Test forecasting functionality."""
        model = ARFIMAModel(p=1, d=0.3, q=1)
        
        # Create synthetic data
        test_data = arfima_simulation(
            n=500, 
            d=0.3, 
            ar_params=np.array([0.5]), 
            ma_params=np.array([0.3]), 
            sigma=1.0,
            seed=42
        )
        
        # Fit model
        model.fit(test_data)
        
        # Generate forecasts
        forecasts = model.forecast(steps=10)
        
        assert len(forecasts) == 10
        assert isinstance(forecasts, np.ndarray)
        assert not np.any(np.isnan(forecasts))
        
    def test_model_summary(self):
        """Test model summary generation."""
        model = ARFIMAModel(p=1, d=0.3, q=1)
        
        # Create synthetic data
        test_data = arfima_simulation(
            n=500, 
            d=0.3, 
            ar_params=np.array([0.5]), 
            ma_params=np.array([0.3]), 
            sigma=1.0,
            seed=42
        )
        
        # Fit model
        model.fit(test_data)
        
        # Generate summary
        summary = model.summary()
        
        assert isinstance(summary, dict)
        assert 'model' in summary
        assert 'parameters' in summary
        assert 'fit_metrics' in summary
        assert 'residuals' in summary
        
        # Check parameter values
        assert 'd' in summary['parameters']
        assert 'ar_params' in summary['parameters']
        assert 'ma_params' in summary['parameters']
        assert 'intercept' in summary['parameters']
        assert 'sigma2' in summary['parameters']
        
    def test_model_diagnostics_plot(self):
        """Test diagnostic plotting."""
        model = ARFIMAModel(p=1, d=0.3, q=1)
        
        # Create synthetic data
        test_data = arfima_simulation(
            n=500, 
            d=0.3, 
            ar_params=np.array([0.5]), 
            ma_params=np.array([0.3]), 
            sigma=1.0,
            seed=42
        )
        
        # Fit model
        model.fit(test_data)
        
        # Test that plotting doesn't raise errors
        try:
            model.plot_diagnostics()
        except Exception as e:
            pytest.fail(f"plot_diagnostics raised an exception: {e}")
            
    def test_predict_method(self):
        """Test predict method."""
        model = ARFIMAModel(p=1, d=0.3, q=1)
        
        # Create synthetic data
        test_data = arfima_simulation(
            n=500, 
            d=0.3, 
            ar_params=np.array([0.5]), 
            ma_params=np.array([0.3]), 
            sigma=1.0,
            seed=42
        )
        
        # Fit model
        model.fit(test_data)
        
        # Generate predictions
        predictions = model.predict(test_data)
        
        assert len(predictions) == len(test_data)
        assert isinstance(predictions, np.ndarray)
        assert not np.any(np.isnan(predictions))
        
    def test_constraint_function(self):
        """Test constraint function for optimization."""
        model = ARFIMAModel(p=1, d=0.3, q=1)
        
        # Test valid parameters (AR and MA parameters must have roots outside unit circle)
        # Use parameters that result in roots outside unit circle (stationary/invertible)
        valid_params = np.array([0.3, 1.5, 1.5, 0.0, 0.0])  # d, ar, ma, intercept, log_sigma2
        constraint_value = model._constraint_function(valid_params)
        assert constraint_value > 0
        
        # Test invalid d parameter
        invalid_d_params = np.array([0.6, 0.5, 0.3, 0.0, 0.0])  # d > 0.5
        constraint_value = model._constraint_function(invalid_d_params)
        assert constraint_value <= 0
        
    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        model = ARFIMAModel(p=1, d=0.3, q=1)
        
        # Create synthetic data
        test_data = arfima_simulation(
            n=100, 
            d=0.3, 
            ar_params=np.array([0.5]), 
            ma_params=np.array([0.3]), 
            sigma=1.0,
            seed=42
        )
        
        # Test parameters
        params = np.array([0.3, 0.5, 0.3, 0.0, 0.0])  # d, ar, ma, intercept, log_sigma2
        
        ll = model._log_likelihood(params, test_data)
        
        assert isinstance(ll, float)
        assert not np.isnan(ll)
        assert not np.isinf(ll)
        
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        model = ARFIMAModel(p=1, d=0.3, q=1)
        
        # Test forecasting without fitting
        with pytest.raises(ValueError, match="Model must be fitted before forecasting"):
            model.forecast(10)
            
        # Test summary without fitting
        with pytest.raises(ValueError, match="Model must be fitted before generating summary"):
            model.summary()
            
        # Test predict without fitting
        with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
            model.predict(np.array([1, 2, 3]))
            
        # Test diagnostics without fitting
        with pytest.raises(ValueError, match="Model must be fitted before plotting diagnostics"):
            model.plot_diagnostics()


class TestEstimateARFIMAOrder:
    """Test ARFIMA order estimation."""
    
    def test_order_estimation(self):
        """Test automatic order estimation."""
        # Create synthetic data
        test_data = arfima_simulation(
            n=500, 
            d=0.3, 
            ar_params=np.array([0.5]), 
            ma_params=np.array([0.3]), 
            sigma=1.0,
            seed=42
        )
        
        # Estimate order
        p, d, q = estimate_arfima_order(test_data, max_p=2, max_q=2)
        
        assert isinstance(p, int)
        assert isinstance(d, float)
        assert isinstance(q, int)
        assert p >= 0
        assert 0 < d < 0.5
        assert q >= 0
        
    def test_order_estimation_with_custom_d_values(self):
        """Test order estimation with custom d values."""
        test_data = arfima_simulation(
            n=300, 
            d=0.2, 
            ar_params=np.array([0.4]), 
            ma_params=np.array([0.2]), 
            sigma=1.0,
            seed=42
        )
        
        custom_d_values = [0.1, 0.2, 0.3]
        p, d, q = estimate_arfima_order(test_data, d_values=custom_d_values)
        
        assert d in custom_d_values


class TestARFIMASimulation:
    """Test ARFIMA simulation functionality."""
    
    def test_simulation_basic(self):
        """Test basic ARFIMA simulation."""
        n = 1000
        d = 0.3
        ar_params = np.array([0.5])
        ma_params = np.array([0.3])
        sigma = 1.0
        
        # Simulate data
        simulated_data = arfima_simulation(
            n=n, 
            d=d, 
            ar_params=ar_params, 
            ma_params=ma_params, 
            sigma=sigma,
            seed=42
        )
        
        assert len(simulated_data) == n
        assert isinstance(simulated_data, np.ndarray)
        assert not np.any(np.isnan(simulated_data))
        assert not np.any(np.isinf(simulated_data))
        
    def test_simulation_reproducibility(self):
        """Test that simulation is reproducible with same seed."""
        params = {
            'n': 500,
            'd': 0.3,
            'ar_params': np.array([0.5]),
            'ma_params': np.array([0.3]),
            'sigma': 1.0,
            'seed': 42
        }
        
        # Simulate twice with same parameters
        data1 = arfima_simulation(**params)
        data2 = arfima_simulation(**params)
        
        np.testing.assert_array_equal(data1, data2)
        
    def test_simulation_without_ar_ma(self):
        """Test simulation with only fractional integration."""
        n = 500
        d = 0.3
        
        # Simulate pure fractional noise
        simulated_data = arfima_simulation(
            n=n, 
            d=d, 
            ar_params=None, 
            ma_params=None, 
            sigma=1.0,
            seed=42
        )
        
        assert len(simulated_data) == n
        assert isinstance(simulated_data, np.ndarray)
        
    def test_simulation_parameter_validation(self):
        """Test simulation parameter validation."""
        # Test with invalid d
        with pytest.raises(ValueError):
            arfima_simulation(n=100, d=-0.1)
            
        # Test with invalid n
        with pytest.raises(ValueError):
            arfima_simulation(n=0, d=0.3)


class TestIntegration:
    """Integration tests for the complete ARFIMA workflow."""
    
    def test_complete_workflow(self):
        """Test complete ARFIMA analysis workflow."""
        # 1. Simulate data
        true_d = 0.3
        true_ar = np.array([0.5])
        true_ma = np.array([0.3])
        
        simulated_data = arfima_simulation(
            n=1000, 
            d=true_d, 
            ar_params=true_ar, 
            ma_params=true_ma, 
            sigma=1.0,
            seed=42
        )
        
        # 2. Estimate model order
        p, d, q = estimate_arfima_order(simulated_data, max_p=2, max_q=2)
        
        # 3. Fit model
        model = ARFIMAModel(p=p, d=d, q=q)
        fitted_model = model.fit(simulated_data)
        
        # 4. Generate forecasts
        forecasts = fitted_model.forecast(steps=20)
        
        # 5. Generate summary
        summary = fitted_model.summary()
        
        # 6. Validate results
        assert fitted_model.is_fitted
        assert len(forecasts) == 20
        assert 'model' in summary
        assert 'parameters' in summary
        
        # Check that estimated d is reasonable
        assert 0 < fitted_model.params.d < 0.5
        
    def test_model_comparison(self):
        """Test comparing different ARFIMA models."""
        # Simulate data
        test_data = arfima_simulation(
            n=800, 
            d=0.25, 
            ar_params=np.array([0.4]), 
            ma_params=np.array([0.2]), 
            sigma=1.0,
            seed=42
        )
        
        # Fit different models
        models = []
        for p in [0, 1, 2]:
            for q in [0, 1, 2]:
                try:
                    model = ARFIMAModel(p=p, d=0.25, q=q)
                    model.fit(test_data)
                    models.append((model, model.aic))
                except:
                    continue
        
        # Find best model by AIC
        best_model, best_aic = min(models, key=lambda x: x[1])
        
        assert best_model.is_fitted
        assert best_aic < float('inf')
        
        # Test that best model can forecast
        forecasts = best_model.forecast(steps=10)
        assert len(forecasts) == 10


if __name__ == "__main__":
    pytest.main([__file__])

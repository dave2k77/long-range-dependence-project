"""
Tests for wavelet analysis methods.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from src.analysis.wavelet_analysis import (
    WaveletSummary, _validate_signal, _generate_scales, _compute_wavelet_coefficients,
    _compute_wavelet_leaders, _theoretical_wavelet_spectrum, _wavelet_whittle_log_likelihood,
    wavelet_leaders_estimation, wavelet_whittle_estimation, WaveletModel,
    hurst_from_wavelet_d, d_from_wavelet_hurst, alpha_from_wavelet_d
)


class TestWaveletSummary:
    """Test WaveletSummary dataclass."""
    
    def test_wavelet_summary_creation(self):
        """Test creating WaveletSummary instance."""
        scales = np.array([4, 8, 16, 32])
        coeffs = np.array([1.0, 0.5, 0.25, 0.125])
        fitted = np.array([1.1, 0.6, 0.3, 0.15])
        
        summary = WaveletSummary(
            method="Test Method",
            hurst=0.7,
            d=0.2,
            alpha=1.4,
            rvalue=0.95,
            pvalue=0.001,
            stderr=0.05,
            scales=scales,
            coefficients=coeffs,
            fitted_values=fitted
        )
        
        assert summary.method == "Test Method"
        assert summary.hurst == 0.7
        assert summary.d == 0.2
        assert summary.alpha == 1.4
        assert summary.rvalue == 0.95
        assert summary.pvalue == 0.001
        assert summary.stderr == 0.05
        np.testing.assert_array_equal(summary.scales, scales)
        np.testing.assert_array_equal(summary.coefficients, coeffs)
        np.testing.assert_array_equal(summary.fitted_values, fitted)


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_validate_signal_valid(self):
        """Test signal validation with valid input."""
        y = np.random.randn(100)
        result = _validate_signal(y)
        np.testing.assert_array_equal(result, y)
    
    def test_validate_signal_with_nan(self):
        """Test signal validation with NaN values."""
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = _validate_signal(y)
        assert len(result) == 4
        assert not np.any(np.isnan(result))
    
    def test_validate_signal_empty(self):
        """Test signal validation with empty input."""
        with pytest.raises(ValueError, match="Signal cannot be empty"):
            _validate_signal([])
    
    def test_validate_signal_none(self):
        """Test signal validation with None input."""
        with pytest.raises(ValueError, match="Signal cannot be empty"):
            _validate_signal(None)
    
    def test_validate_signal_short(self):
        """Test signal validation with short signal."""
        y = np.random.randn(50)  # Short signal
        
        with pytest.warns(UserWarning, match="Signal length is very short"):
            result = _validate_signal(y)
        
        np.testing.assert_array_equal(result, y)
    
    def test_generate_scales(self):
        """Test scale generation."""
        n = 100
        scales = _generate_scales(n)
        
        assert len(scales) > 0
        assert np.all(scales > 0)
        assert np.all(scales <= n // 8)  # Max scale
        # Check that scales are powers of 2
        assert np.allclose(scales, 2.0 ** np.arange(2, int(np.log2(n // 8)) + 1))
    
    def test_generate_scales_custom_bounds(self):
        """Test scale generation with custom bounds."""
        n = 100
        min_scale = 8
        max_scale = 32
        scales = _generate_scales(n, min_scale, max_scale)
        
        assert len(scales) > 0
        assert np.all(scales >= min_scale)
        assert np.all(scales <= max_scale)
    
    def test_compute_wavelet_coefficients_morlet(self):
        """Test wavelet coefficient computation with Morlet wavelet."""
        y = np.random.randn(50)
        scales = np.array([4, 8, 16])
        
        coefficients = _compute_wavelet_coefficients(y, scales, 'morlet')
        
        assert coefficients.shape == (len(scales), len(y))
        assert not np.any(np.isnan(coefficients))
    
    def test_compute_wavelet_coefficients_haar(self):
        """Test wavelet coefficient computation with Haar wavelet."""
        y = np.random.randn(50)
        scales = np.array([4, 8, 16])
        
        coefficients = _compute_wavelet_coefficients(y, scales, 'haar')
        
        assert coefficients.shape == (len(scales), len(y))
        assert not np.any(np.isnan(coefficients))
    
    def test_compute_wavelet_coefficients_invalid_wavelet(self):
        """Test wavelet coefficient computation with invalid wavelet."""
        y = np.random.randn(50)
        scales = np.array([4, 8])
        
        with pytest.raises(ValueError, match="Wavelet must be 'morlet' or 'haar'"):
            _compute_wavelet_coefficients(y, scales, 'invalid')
    
    def test_compute_wavelet_leaders(self):
        """Test wavelet leaders computation."""
        y = np.random.randn(50)
        scales = np.array([4, 8, 16])
        
        leaders = _compute_wavelet_leaders(y, scales, 'morlet')
        
        assert len(leaders) == len(scales)
        assert np.all(leaders > 0)
        assert not np.any(np.isnan(leaders))
    
    def test_theoretical_wavelet_spectrum(self):
        """Test theoretical wavelet spectrum computation."""
        scales = np.array([4, 8, 16, 32])
        d = 0.3
        sigma2 = 1.0
        
        spectrum = _theoretical_wavelet_spectrum(scales, d, sigma2)
        
        assert len(spectrum) == len(scales)
        assert np.all(spectrum > 0)
        # Check that spectrum increases with scale
        assert spectrum[0] < spectrum[1] < spectrum[2] < spectrum[3]


class TestWaveletWhittleLogLikelihood:
    """Test wavelet Whittle log-likelihood function."""
    
    def test_wavelet_whittle_log_likelihood(self):
        """Test wavelet Whittle log-likelihood computation."""
        params = np.array([0.3, np.log(1.0)])  # d=0.3, sigma2=1.0
        scales = np.array([4, 8, 16])
        coeff_variances = np.array([1.0, 2.0, 4.0])
        
        ll = _wavelet_whittle_log_likelihood(params, scales, coeff_variances)
        
        assert isinstance(ll, float)
        assert not np.isnan(ll)
        assert not np.isinf(ll)
    
    def test_wavelet_whittle_log_likelihood_extreme_params(self):
        """Test wavelet Whittle log-likelihood with extreme parameters."""
        params = np.array([0.01, np.log(0.1)])  # Very small d and sigma2
        scales = np.array([4, 8])
        coeff_variances = np.array([1.0, 2.0])
        
        ll = _wavelet_whittle_log_likelihood(params, scales, coeff_variances)
        
        assert isinstance(ll, float)
        assert not np.isnan(ll)


class TestWaveletLeadersEstimation:
    """Test Wavelet Leaders estimation."""
    
    def test_wavelet_leaders_estimation_basic(self):
        """Test basic Wavelet Leaders estimation."""
        np.random.seed(42)
        y = np.random.randn(200)
        
        scales, leaders, summary = wavelet_leaders_estimation(y)
        
        assert isinstance(scales, np.ndarray)
        assert isinstance(leaders, np.ndarray)
        assert isinstance(summary, WaveletSummary)
        assert summary.method == "Wavelet Leaders"
        assert 0.01 <= summary.d <= 0.49
        # Allow for edge cases in Hurst estimation
        assert 0.1 <= summary.hurst <= 0.99
        assert summary.alpha == 2 * summary.hurst
        assert -1 <= summary.rvalue <= 1
        assert 0 <= summary.pvalue <= 1
        assert summary.stderr > 0
    
    def test_wavelet_leaders_estimation_with_custom_scales(self):
        """Test Wavelet Leaders estimation with custom scale bounds."""
        np.random.seed(42)
        y = np.random.randn(200)
        
        scales, leaders, summary = wavelet_leaders_estimation(
            y, min_scale=8, max_scale=64
        )
        
        assert len(scales) > 0
        assert np.all(scales >= 8)
        assert np.all(scales <= 64)
    
    def test_wavelet_leaders_estimation_haar_wavelet(self):
        """Test Wavelet Leaders estimation with Haar wavelet."""
        np.random.seed(42)
        y = np.random.randn(200)
        
        scales, leaders, summary = wavelet_leaders_estimation(
            y, wavelet='haar'
        )
        
        assert isinstance(summary, WaveletSummary)
        assert summary.method == "Wavelet Leaders"
    
    def test_wavelet_leaders_estimation_insufficient_scales(self):
        """Test Wavelet Leaders estimation with insufficient scales."""
        y = np.random.randn(20)  # Very short signal
        
        with pytest.raises(ValueError, match="Insufficient scales for wavelet analysis"):
            wavelet_leaders_estimation(y, min_scale=100, max_scale=200)
    
    def test_wavelet_leaders_estimation_insufficient_points(self):
        """Test Wavelet Leaders estimation with insufficient valid points."""
        y = np.array([0.0] * 50)  # Constant signal
        
        # For constant signal, we should get insufficient valid points error
        # but first we need to pass the scale check
        with pytest.raises(ValueError, match="Insufficient valid points for regression"):
            wavelet_leaders_estimation(y, min_scale=2, max_scale=16)


class TestWaveletWhittleEstimation:
    """Test Wavelet Whittle estimation."""
    
    def test_wavelet_whittle_estimation_basic(self):
        """Test basic Wavelet Whittle estimation."""
        np.random.seed(42)
        y = np.random.randn(200)
        
        scales, coeff_variances, summary = wavelet_whittle_estimation(y)
        
        assert isinstance(scales, np.ndarray)
        assert isinstance(coeff_variances, np.ndarray)
        assert isinstance(summary, WaveletSummary)
        assert summary.method == "Wavelet Whittle"
        assert 0.01 <= summary.d <= 0.49
        assert 0.51 <= summary.hurst <= 0.99
        assert summary.alpha == 2 * summary.hurst
        assert -1 <= summary.rvalue <= 1
        assert 0 <= summary.pvalue <= 1
        assert summary.stderr > 0
    
    def test_wavelet_whittle_estimation_with_custom_scales(self):
        """Test Wavelet Whittle estimation with custom scale bounds."""
        np.random.seed(42)
        y = np.random.randn(200)
        
        scales, coeff_variances, summary = wavelet_whittle_estimation(
            y, min_scale=8, max_scale=64
        )
        
        assert len(scales) > 0
        assert np.all(scales >= 8)
        assert np.all(scales <= 64)
    
    def test_wavelet_whittle_estimation_haar_wavelet(self):
        """Test Wavelet Whittle estimation with Haar wavelet."""
        np.random.seed(42)
        y = np.random.randn(200)
        
        scales, coeff_variances, summary = wavelet_whittle_estimation(
            y, wavelet='haar'
        )
        
        assert isinstance(summary, WaveletSummary)
        assert summary.method == "Wavelet Whittle"
    
    def test_wavelet_whittle_estimation_insufficient_scales(self):
        """Test Wavelet Whittle estimation with insufficient scales."""
        y = np.random.randn(20)  # Very short signal
        
        with pytest.raises(ValueError, match="Insufficient scales for wavelet analysis"):
            wavelet_whittle_estimation(y, min_scale=100, max_scale=200)


class TestWaveletModel:
    """Test WaveletModel class."""
    
    def test_wavelet_model_init_leaders(self):
        """Test WaveletModel initialization with Leaders method."""
        model = WaveletModel(method='leaders', wavelet='morlet')
        
        assert model.method == 'leaders'
        assert model.wavelet == 'morlet'
        assert not model.is_fitted
        assert model.scales is None
        assert model.coefficients is None
        assert model.summary is None
    
    def test_wavelet_model_init_whittle(self):
        """Test WaveletModel initialization with Whittle method."""
        model = WaveletModel(method='whittle', wavelet='haar')
        
        assert model.method == 'whittle'
        assert model.wavelet == 'haar'
    
    def test_wavelet_model_invalid_method(self):
        """Test WaveletModel initialization with invalid method."""
        with pytest.raises(ValueError, match="Method must be 'leaders' or 'whittle'"):
            WaveletModel(method='invalid')
    
    def test_wavelet_model_invalid_wavelet(self):
        """Test WaveletModel initialization with invalid wavelet."""
        with pytest.raises(ValueError, match="Wavelet must be 'morlet' or 'haar'"):
            WaveletModel(wavelet='invalid')
    
    def test_wavelet_model_fit_leaders(self):
        """Test WaveletModel fitting with Leaders method."""
        model = WaveletModel(method='leaders')
        y = np.random.randn(200)
        
        result = model.fit(y)
        
        assert result is model
        assert model.is_fitted
        assert model.scales is not None
        assert model.coefficients is not None
        assert model.summary is not None
        assert model.summary.method == "Wavelet Leaders"
    
    def test_wavelet_model_fit_whittle(self):
        """Test WaveletModel fitting with Whittle method."""
        model = WaveletModel(method='whittle')
        y = np.random.randn(200)
        
        result = model.fit(y)
        
        assert result is model
        assert model.is_fitted
        assert model.summary.method == "Wavelet Whittle"
    
    def test_wavelet_model_getters_not_fitted(self):
        """Test WaveletModel getters when not fitted."""
        model = WaveletModel()
        
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.get_hurst()
        
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.get_d()
        
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.get_alpha()
        
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.get_summary()
    
    def test_wavelet_model_getters_fitted(self):
        """Test WaveletModel getters when fitted."""
        model = WaveletModel(method='leaders')
        y = np.random.randn(200)
        model.fit(y)
        
        hurst = model.get_hurst()
        d = model.get_d()
        alpha = model.get_alpha()
        summary = model.get_summary()
        
        assert isinstance(hurst, float)
        assert isinstance(d, float)
        assert isinstance(alpha, float)
        assert isinstance(summary, WaveletSummary)
        assert hurst == summary.hurst
        assert d == summary.d
        assert alpha == summary.alpha
    
    @patch('matplotlib.pyplot.subplots')
    def test_wavelet_model_plot_scalogram(self, mock_subplots):
        """Test WaveletModel plotting."""
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        model = WaveletModel(method='leaders')
        y = np.random.randn(200)
        model.fit(y)
        
        ax = model.plot_scalogram()
        
        assert ax == mock_ax
        mock_ax.loglog.assert_called()
        mock_ax.set_xlabel.assert_called_with('Scale')
        mock_ax.set_ylabel.assert_called_with('Wavelet Leaders')
    
    def test_wavelet_model_plot_scalogram_not_fitted(self):
        """Test WaveletModel plotting when not fitted."""
        model = WaveletModel()
        
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.plot_scalogram()


class TestUtilityFunctions:
    """Test utility functions for parameter conversion."""
    
    def test_hurst_from_wavelet_d(self):
        """Test conversion from wavelet d to Hurst exponent."""
        d = 0.3
        hurst = hurst_from_wavelet_d(d)
        expected = d + 0.5
        assert hurst == expected
    
    def test_d_from_wavelet_hurst(self):
        """Test conversion from Hurst exponent to wavelet d."""
        hurst = 0.8
        d = d_from_wavelet_hurst(hurst)
        expected = hurst - 0.5
        assert d == expected
    
    def test_alpha_from_wavelet_d(self):
        """Test conversion from wavelet d to scaling exponent."""
        d = 0.3
        alpha = alpha_from_wavelet_d(d)
        expected = 2 * (d + 0.5)
        assert alpha == expected
    
    def test_parameter_conversion_consistency(self):
        """Test consistency of parameter conversions."""
        d_original = 0.3
        hurst = hurst_from_wavelet_d(d_original)
        d_converted = d_from_wavelet_hurst(hurst)
        alpha = alpha_from_wavelet_d(d_original)
        
        assert abs(d_original - d_converted) < 1e-10
        assert alpha == 2 * hurst


class TestIntegration:
    """Integration tests."""
    
    def test_wavelet_leaders_with_pandas_series(self):
        """Test Wavelet Leaders with pandas Series input."""
        np.random.seed(42)
        y_series = pd.Series(np.random.randn(200))
        
        scales, leaders, summary = wavelet_leaders_estimation(y_series)
        
        assert isinstance(summary, WaveletSummary)
        assert summary.method == "Wavelet Leaders"
    
    def test_wavelet_whittle_with_pandas_series(self):
        """Test Wavelet Whittle with pandas Series input."""
        np.random.seed(42)
        y_series = pd.Series(np.random.randn(200))
        
        scales, coeff_variances, summary = wavelet_whittle_estimation(y_series)
        
        assert isinstance(summary, WaveletSummary)
        assert summary.method == "Wavelet Whittle"
    
    def test_wavelet_model_with_pandas_series(self):
        """Test WaveletModel with pandas Series input."""
        model = WaveletModel(method='leaders')
        y_series = pd.Series(np.random.randn(200))
        
        model.fit(y_series)
        
        assert model.is_fitted
        assert isinstance(model.get_hurst(), float)
        assert isinstance(model.get_d(), float)
        assert isinstance(model.get_alpha(), float)
    
    def test_wavelet_methods_consistency(self):
        """Test consistency between different wavelet methods."""
        np.random.seed(42)
        y = np.random.randn(300)
        
        # Test both methods on same data
        scales1, leaders, summary1 = wavelet_leaders_estimation(y)
        scales2, coeff_variances, summary2 = wavelet_whittle_estimation(y)
        
        # Both should give reasonable estimates
        assert 0.01 <= summary1.d <= 0.49
        assert 0.01 <= summary2.d <= 0.49
        # Allow for edge cases in Hurst estimation
        assert 0.1 <= summary1.hurst <= 0.99
        assert 0.1 <= summary2.hurst <= 0.99
        
        # Both should have similar scale ranges
        assert len(scales1) > 0
        assert len(scales2) > 0

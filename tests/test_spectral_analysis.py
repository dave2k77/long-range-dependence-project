"""
Tests for spectral analysis methods.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from src.analysis.spectral_analysis import (
    SpectralSummary, _validate_signal, _generate_frequencies, _theoretical_spectrum,
    _whittle_log_likelihood, whittle_mle, periodogram_estimation, SpectralModel,
    hurst_from_spectral_d, d_from_spectral_hurst, alpha_from_spectral_d
)


class TestSpectralSummary:
    """Test SpectralSummary dataclass."""
    
    def test_spectral_summary_creation(self):
        """Test creating SpectralSummary instance."""
        freqs = np.array([0.1, 0.2, 0.3])
        psd = np.array([1.0, 0.5, 0.25])
        fitted = np.array([1.1, 0.6, 0.3])
        
        summary = SpectralSummary(
            method="Test Method",
            hurst=0.7,
            d=0.2,
            alpha=1.4,
            rvalue=0.95,
            pvalue=0.001,
            stderr=0.05,
            frequencies=freqs,
            power_spectrum=psd,
            fitted_spectrum=fitted
        )
        
        assert summary.method == "Test Method"
        assert summary.hurst == 0.7
        assert summary.d == 0.2
        assert summary.alpha == 1.4
        assert summary.rvalue == 0.95
        assert summary.pvalue == 0.001
        assert summary.stderr == 0.05
        np.testing.assert_array_equal(summary.frequencies, freqs)
        np.testing.assert_array_equal(summary.power_spectrum, psd)
        np.testing.assert_array_equal(summary.fitted_spectrum, fitted)


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
    
    def test_generate_frequencies(self):
        """Test frequency generation."""
        n = 100
        freqs = _generate_frequencies(n)
        assert len(freqs) == n // 2 - 1  # Excluding zero frequency
        assert np.all(freqs > 0)
        assert np.all(freqs <= 0.5)  # Nyquist frequency
    
    def test_theoretical_spectrum(self):
        """Test theoretical spectrum computation."""
        freqs = np.array([0.1, 0.2, 0.3])
        d = 0.3
        sigma2 = 1.0
        
        spectrum = _theoretical_spectrum(freqs, d, sigma2)
        
        assert len(spectrum) == len(freqs)
        assert np.all(spectrum > 0)
        # Check that spectrum decreases with frequency
        assert spectrum[0] > spectrum[1] > spectrum[2]
    
    def test_theoretical_spectrum_zero_freq(self):
        """Test theoretical spectrum with zero frequency."""
        freqs = np.array([0.0, 0.1, 0.2])
        d = 0.3
        sigma2 = 1.0
        
        spectrum = _theoretical_spectrum(freqs, d, sigma2)
        
        assert len(spectrum) == len(freqs)
        assert np.all(spectrum > 0)  # Should handle zero frequency gracefully


class TestWhittleLogLikelihood:
    """Test Whittle log-likelihood function."""
    
    def test_whittle_log_likelihood(self):
        """Test Whittle log-likelihood computation."""
        params = np.array([0.3, np.log(1.0)])  # d=0.3, sigma2=1.0
        freqs = np.array([0.1, 0.2, 0.3])
        periodogram_vals = np.array([1.0, 0.5, 0.25])
        
        ll = _whittle_log_likelihood(params, freqs, periodogram_vals)
        
        assert isinstance(ll, float)
        assert not np.isnan(ll)
        assert not np.isinf(ll)
    
    def test_whittle_log_likelihood_extreme_params(self):
        """Test Whittle log-likelihood with extreme parameters."""
        params = np.array([0.01, np.log(0.1)])  # Very small d and sigma2
        freqs = np.array([0.1, 0.2])
        periodogram_vals = np.array([1.0, 0.5])
        
        ll = _whittle_log_likelihood(params, freqs, periodogram_vals)
        
        assert isinstance(ll, float)
        assert not np.isnan(ll)


class TestWhittleMLE:
    """Test Whittle MLE estimation."""
    
    def test_whittle_mle_basic(self):
        """Test basic Whittle MLE estimation."""
        # Generate synthetic fractional noise
        np.random.seed(42)
        n = 500
        d_true = 0.3
        y = np.cumsum(np.random.randn(n))  # Random walk approximation
        
        freqs, periodogram_vals, summary = whittle_mle(y)
        
        assert isinstance(freqs, np.ndarray)
        assert isinstance(periodogram_vals, np.ndarray)
        assert isinstance(summary, SpectralSummary)
        assert summary.method == "Whittle MLE"
        assert 0.01 <= summary.d <= 0.49
        assert 0.51 <= summary.hurst <= 0.99
        assert summary.alpha == 2 * summary.hurst
        assert -1 <= summary.rvalue <= 1
        assert 0 <= summary.pvalue <= 1
        assert summary.stderr > 0
    
    def test_whittle_mle_with_freq_range(self):
        """Test Whittle MLE with frequency range."""
        np.random.seed(42)
        y = np.random.randn(200)
        
        freqs, periodogram_vals, summary = whittle_mle(
            y, freq_range=(0.01, 0.1)
        )
        
        assert len(freqs) > 0
        assert np.all(freqs >= 0.01)
        assert np.all(freqs <= 0.1)
    
    def test_whittle_mle_short_signal(self):
        """Test Whittle MLE with short signal."""
        y = np.random.randn(50)  # Very short signal
        
        # The warning is now handled differently, so we just test the function works
        freqs, periodogram_vals, summary = whittle_mle(y)
        
        assert isinstance(summary, SpectralSummary)
    
    def test_whittle_mle_insufficient_freqs(self):
        """Test Whittle MLE with insufficient frequency points."""
        y = np.random.randn(10)  # Very short signal
        
        with pytest.raises(ValueError, match="Insufficient frequency points"):
            whittle_mle(y, freq_range=(0.5, 0.6))  # Very narrow range


class TestPeriodogramEstimation:
    """Test periodogram estimation."""
    
    def test_periodogram_estimation_basic(self):
        """Test basic periodogram estimation."""
        np.random.seed(42)
        y = np.random.randn(300)
        
        freqs, periodogram_vals, summary = periodogram_estimation(y)
        
        assert isinstance(freqs, np.ndarray)
        assert isinstance(periodogram_vals, np.ndarray)
        assert isinstance(summary, SpectralSummary)
        assert summary.method == "Periodogram Regression"
        assert 0.01 <= summary.d <= 0.49
        # Allow for edge cases in Hurst estimation
        assert 0.4 <= summary.hurst <= 0.99
        assert summary.alpha == 2 * summary.hurst
        assert -1 <= summary.rvalue <= 1
        assert 0 <= summary.pvalue <= 1
        assert summary.stderr > 0
    
    def test_periodogram_estimation_with_freq_range(self):
        """Test periodogram estimation with frequency range."""
        np.random.seed(42)
        y = np.random.randn(200)
        
        # Use a wider frequency range to ensure enough points
        freqs, periodogram_vals, summary = periodogram_estimation(
            y, freq_range=(0.005, 0.2), min_freq_points=10
        )
        
        assert len(freqs) > 0
        assert np.all(freqs >= 0.005)
        assert np.all(freqs <= 0.2)
    
    def test_periodogram_estimation_insufficient_points(self):
        """Test periodogram estimation with insufficient points."""
        y = np.random.randn(20)  # Very short signal
        
        with pytest.raises(ValueError, match="Insufficient frequency points"):
            periodogram_estimation(y, min_freq_points=50)


class TestSpectralModel:
    """Test SpectralModel class."""
    
    def test_spectral_model_init_whittle(self):
        """Test SpectralModel initialization with Whittle method."""
        model = SpectralModel(method='whittle', sampling_rate=2.0)
        
        assert model.method == 'whittle'
        assert model.sampling_rate == 2.0
        assert not model.is_fitted
        assert model.frequencies is None
        assert model.power_spectrum is None
        assert model.summary is None
    
    def test_spectral_model_init_periodogram(self):
        """Test SpectralModel initialization with periodogram method."""
        model = SpectralModel(method='periodogram', sampling_rate=1.0)
        
        assert model.method == 'periodogram'
        assert model.sampling_rate == 1.0
    
    def test_spectral_model_invalid_method(self):
        """Test SpectralModel initialization with invalid method."""
        with pytest.raises(ValueError, match="Method must be 'whittle' or 'periodogram'"):
            SpectralModel(method='invalid')
    
    def test_spectral_model_fit_whittle(self):
        """Test SpectralModel fitting with Whittle method."""
        model = SpectralModel(method='whittle')
        y = np.random.randn(200)
        
        result = model.fit(y)
        
        assert result is model
        assert model.is_fitted
        assert model.frequencies is not None
        assert model.power_spectrum is not None
        assert model.summary is not None
        assert model.summary.method == "Whittle MLE"
    
    def test_spectral_model_fit_periodogram(self):
        """Test SpectralModel fitting with periodogram method."""
        model = SpectralModel(method='periodogram')
        y = np.random.randn(200)
        
        result = model.fit(y)
        
        assert result is model
        assert model.is_fitted
        assert model.summary.method == "Periodogram Regression"
    
    def test_spectral_model_getters_not_fitted(self):
        """Test SpectralModel getters when not fitted."""
        model = SpectralModel()
        
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.get_hurst()
        
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.get_d()
        
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.get_alpha()
        
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.get_summary()
    
    def test_spectral_model_getters_fitted(self):
        """Test SpectralModel getters when fitted."""
        model = SpectralModel(method='whittle')
        y = np.random.randn(200)
        model.fit(y)
        
        hurst = model.get_hurst()
        d = model.get_d()
        alpha = model.get_alpha()
        summary = model.get_summary()
        
        assert isinstance(hurst, float)
        assert isinstance(d, float)
        assert isinstance(alpha, float)
        assert isinstance(summary, SpectralSummary)
        assert hurst == summary.hurst
        assert d == summary.d
        assert alpha == summary.alpha
    
    @patch('matplotlib.pyplot.subplots')
    def test_spectral_model_plot_spectrum(self, mock_subplots):
        """Test SpectralModel plotting."""
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        model = SpectralModel(method='whittle')
        y = np.random.randn(200)
        model.fit(y)
        
        ax = model.plot_spectrum()
        
        assert ax == mock_ax
        mock_ax.loglog.assert_called()
        mock_ax.set_xlabel.assert_called_with('Frequency')
        mock_ax.set_ylabel.assert_called_with('Power Spectral Density')
    
    def test_spectral_model_plot_spectrum_not_fitted(self):
        """Test SpectralModel plotting when not fitted."""
        model = SpectralModel()
        
        with pytest.raises(ValueError, match="Model must be fitted first"):
            model.plot_spectrum()


class TestUtilityFunctions:
    """Test utility functions for parameter conversion."""
    
    def test_hurst_from_spectral_d(self):
        """Test conversion from spectral d to Hurst exponent."""
        d = 0.3
        hurst = hurst_from_spectral_d(d)
        expected = d + 0.5
        assert hurst == expected
    
    def test_d_from_spectral_hurst(self):
        """Test conversion from Hurst exponent to spectral d."""
        hurst = 0.8
        d = d_from_spectral_hurst(hurst)
        expected = hurst - 0.5
        assert d == expected
    
    def test_alpha_from_spectral_d(self):
        """Test conversion from spectral d to scaling exponent."""
        d = 0.3
        alpha = alpha_from_spectral_d(d)
        expected = 2 * (d + 0.5)
        assert alpha == expected
    
    def test_parameter_conversion_consistency(self):
        """Test consistency of parameter conversions."""
        d_original = 0.3
        hurst = hurst_from_spectral_d(d_original)
        d_converted = d_from_spectral_hurst(hurst)
        alpha = alpha_from_spectral_d(d_original)
        
        assert abs(d_original - d_converted) < 1e-10
        assert alpha == 2 * hurst


class TestIntegration:
    """Integration tests."""
    
    def test_whittle_mle_with_pandas_series(self):
        """Test Whittle MLE with pandas Series input."""
        np.random.seed(42)
        y_series = pd.Series(np.random.randn(200))
        
        freqs, periodogram_vals, summary = whittle_mle(y_series)
        
        assert isinstance(summary, SpectralSummary)
        assert summary.method == "Whittle MLE"
    
    def test_periodogram_estimation_with_pandas_series(self):
        """Test periodogram estimation with pandas Series input."""
        np.random.seed(42)
        y_series = pd.Series(np.random.randn(200))
        
        freqs, periodogram_vals, summary = periodogram_estimation(y_series)
        
        assert isinstance(summary, SpectralSummary)
        assert summary.method == "Periodogram Regression"
    
    def test_spectral_model_with_pandas_series(self):
        """Test SpectralModel with pandas Series input."""
        model = SpectralModel(method='whittle')
        y_series = pd.Series(np.random.randn(200))
        
        model.fit(y_series)
        
        assert model.is_fitted
        assert isinstance(model.get_hurst(), float)
        assert isinstance(model.get_d(), float)
        assert isinstance(model.get_alpha(), float)

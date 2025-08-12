"""
Tests for MFDFA (Multifractal Detrended Fluctuation Analysis) implementation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.analysis.mfdfa_analysis import (
    mfdfa, MFDFAModel, MFDFASummary, _validate_signal, _generate_log_scales,
    _profile, _poly_detrend, _fluctuation_for_scale_q,
    hurst_from_mfdfa, alpha_from_mfdfa
)


class TestMFDFASummary:
    """Test MFDFASummary dataclass."""
    
    def test_mfdfa_summary_creation(self):
        """Test MFDFASummary creation and basic properties."""
        q_values = np.array([-3, -1, 0, 1, 3])
        scales = np.array([8, 16, 32, 64])
        hq = np.array([0.8, 0.7, 0.6, 0.7, 0.8])
        tau = np.array([-1.4, -0.7, -0.2, 0.4, 1.4])
        alpha = np.array([0.6, 0.7, 0.8, 0.7, 0.6])
        f_alpha = np.array([0.8, 0.7, 0.6, 0.7, 0.8])
        fq = np.random.rand(5, 4)
        
        summary = MFDFASummary(
            hq=hq,
            tau=tau,
            alpha=alpha,
            f_alpha=f_alpha,
            q_values=q_values,
            scales=scales,
            fq=fq
        )
        
        np.testing.assert_array_equal(summary.hq, hq)
        np.testing.assert_array_equal(summary.tau, tau)
        np.testing.assert_array_equal(summary.alpha, alpha)
        np.testing.assert_array_equal(summary.f_alpha, f_alpha)
        np.testing.assert_array_equal(summary.q_values, q_values)
        np.testing.assert_array_equal(summary.scales, scales)
        np.testing.assert_array_equal(summary.fq, fq)
    
    def test_as_dict(self):
        """Test MFDFASummary.as_dict method."""
        q_values = np.array([-1, 0, 1])
        scales = np.array([8, 16])
        hq = np.array([0.7, 0.6, 0.7])
        tau = np.array([-0.7, -0.2, 0.4])
        alpha = np.array([0.7, 0.8, 0.7])
        f_alpha = np.array([0.7, 0.6, 0.7])
        fq = np.random.rand(3, 2)
        
        summary = MFDFASummary(
            hq=hq,
            tau=tau,
            alpha=alpha,
            f_alpha=f_alpha,
            q_values=q_values,
            scales=scales,
            fq=fq
        )
        
        result = summary.as_dict()
        
        assert isinstance(result, dict)
        np.testing.assert_array_equal(result['hq'], hq)
        np.testing.assert_array_equal(result['tau'], tau)
        np.testing.assert_array_equal(result['alpha'], alpha)
        np.testing.assert_array_equal(result['f_alpha'], f_alpha)
        np.testing.assert_array_equal(result['q_values'], q_values)
        np.testing.assert_array_equal(result['scales'], scales)
        np.testing.assert_array_equal(result['fq'], fq)


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_validate_signal_valid(self):
        """Test _validate_signal with valid input."""
        y = np.random.randn(100)
        result = _validate_signal(y)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
        assert result.dtype == float
    
    def test_validate_signal_too_short(self):
        """Test _validate_signal with too short input."""
        y = np.random.randn(50)
        with pytest.raises(ValueError, match="too short"):
            _validate_signal(y)
    
    def test_validate_signal_with_nans(self):
        """Test _validate_signal with NaN values."""
        y = np.random.randn(100)
        y[10:20] = np.nan
        result = _validate_signal(y)
        assert np.all(np.isfinite(result))
    
    def test_validate_signal_too_many_nans(self):
        """Test _validate_signal with too many NaN values."""
        y = np.full(100, np.nan)
        y[:30] = np.random.randn(30)
        with pytest.raises(ValueError, match="Too many non-finite"):
            _validate_signal(y)
    
    def test_generate_log_scales(self):
        """Test _generate_log_scales function."""
        scales = _generate_log_scales(1000)
        assert isinstance(scales, np.ndarray)
        assert scales.size >= 8
        assert np.all(scales >= 8)
        assert np.all(scales <= 125)  # max_scale = 1000 // 8 = 125
    
    def test_generate_log_scales_custom_params(self):
        """Test _generate_log_scales with custom parameters."""
        scales = _generate_log_scales(500, min_scale=16, max_scale=128, num_scales=10)
        assert scales.size >= 8
        assert np.all(scales >= 16)
        assert np.all(scales <= 128)
    
    def test_profile(self):
        """Test _profile function."""
        y = np.array([1, 2, 3, 4, 5])
        prof = _profile(y)
        expected = np.cumsum(y - np.mean(y))
        np.testing.assert_array_almost_equal(prof, expected)
    
    def test_poly_detrend_linear(self):
        """Test _poly_detrend with linear trend."""
        x = np.array([1, 2, 3, 4, 5])
        trend, resid = _poly_detrend(x, order=1)
        assert trend.shape == x.shape
        assert resid.shape == x.shape
        # Residuals should sum to approximately zero
        assert abs(np.sum(resid)) < 1e-10
    
    def test_poly_detrend_constant(self):
        """Test _poly_detrend with order 0 (constant)."""
        x = np.array([1, 2, 3, 4, 5])
        trend, resid = _poly_detrend(x, order=0)
        assert np.allclose(trend, np.mean(x))
        np.testing.assert_array_almost_equal(resid, x - np.mean(x))
    
    def test_fluctuation_for_scale_q(self):
        """Test _fluctuation_for_scale_q function."""
        profile = np.random.randn(100)
        s = 10
        q = 2.0
        result = _fluctuation_for_scale_q(profile, s, q, order=1, use_reverse=True, overlap=False)
        assert isinstance(result, float)
        assert result > 0
        assert np.isfinite(result)
    
    def test_fluctuation_for_scale_q_zero(self):
        """Test _fluctuation_for_scale_q with q=0."""
        profile = np.random.randn(100)
        s = 10
        q = 0.0
        result = _fluctuation_for_scale_q(profile, s, q, order=1, use_reverse=True, overlap=False)
        assert isinstance(result, float)
        assert result > 0
        assert np.isfinite(result)


class TestMFDFAFunctionalAPI:
    """Test the functional MFDFA API."""
    
    def test_mfdfa_basic(self):
        """Test basic MFDFA functionality."""
        # Generate synthetic data with known scaling
        np.random.seed(42)
        n = 1000
        # Create fractional Gaussian noise-like data
        y = np.cumsum(np.random.randn(n))  # Random walk
        y = np.diff(y)  # Make it stationary
        
        scales, fq, summary = mfdfa(y)
        
        assert isinstance(scales, np.ndarray)
        assert isinstance(fq, np.ndarray)
        assert isinstance(summary, MFDFASummary)
        assert fq.shape == (len(summary.q_values), len(scales))
        assert scales.size >= 8
        assert np.all(fq > 0)
        assert np.isfinite(summary.hq).any()
    
    def test_mfdfa_with_custom_q(self):
        """Test MFDFA with custom q values."""
        y = np.random.randn(500)
        custom_q = [-3, -1, 0, 1, 3]
        
        scales, fq, summary = mfdfa(y, q_values=custom_q)
        
        np.testing.assert_array_equal(summary.q_values, custom_q)
        assert fq.shape == (len(custom_q), len(scales))
    
    def test_mfdfa_with_custom_scales(self):
        """Test MFDFA with custom scales."""
        y = np.random.randn(500)
        custom_scales = [8, 16, 32, 64, 128]
        
        scales, fq, summary = mfdfa(y, scales=custom_scales)
        
        np.testing.assert_array_equal(scales, custom_scales)
        assert fq.shape[1] == len(custom_scales)
    
    def test_mfdfa_edge_cases(self):
        """Test MFDFA with edge cases."""
        # Very short series (should fail)
        with pytest.raises(ValueError):
            mfdfa(np.random.randn(50))
        
        # Invalid scales
        with pytest.raises(ValueError):
            mfdfa(np.random.randn(100), scales=[2])  # Scale < 4
        
        with pytest.raises(ValueError):
            mfdfa(np.random.randn(100), scales=[8])  # Only one scale


class TestMFDFAModel:
    """Test MFDFAModel class."""
    
    def test_mfdfa_model_initialization(self):
        """Test MFDFAModel initialization."""
        model = MFDFAModel(order=2, use_reverse=True, overlap=False)
        assert model.order == 2
        assert model.use_reverse is True
        assert model.overlap is False
        assert model.scales is None
        assert model.fq is None
        assert model.summary is None
        assert model.is_fitted is False
    
    def test_mfdfa_model_fit(self):
        """Test MFDFAModel.fit method."""
        model = MFDFAModel()
        y = np.random.randn(500)
        
        result = model.fit(y)
        
        assert result is model
        assert model.is_fitted is True
        assert model.scales is not None
        assert model.fq is not None
        assert model.summary is not None
        assert np.isfinite(model.summary.hq).any()
    
    def test_mfdfa_model_getters(self):
        """Test MFDFAModel getter methods."""
        model = MFDFAModel()
        y = np.random.randn(500)
        model.fit(y)
        
        hq = model.get_hq()
        tau = model.get_tau()
        alpha = model.get_alpha()
        f_alpha = model.get_f_alpha()
        summary = model.get_summary()
        
        assert isinstance(hq, np.ndarray)
        assert isinstance(tau, np.ndarray)
        assert isinstance(alpha, np.ndarray)
        assert isinstance(f_alpha, np.ndarray)
        assert isinstance(summary, MFDFASummary)
        assert np.isfinite(hq).any()
    
    def test_mfdfa_model_getters_not_fitted(self):
        """Test MFDFAModel getters when not fitted."""
        model = MFDFAModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_hq()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_tau()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_alpha()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_f_alpha()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_summary()
    
    @patch('matplotlib.pyplot.subplots')
    def test_mfdfa_model_plot_loglog(self, mock_subplots):
        """Test MFDFAModel.plot_loglog method."""
        # Mock matplotlib
        from unittest.mock import MagicMock
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        model = MFDFAModel()
        y = np.random.randn(500)
        model.fit(y)
        
        ax = model.plot_loglog()
        
        assert ax is mock_ax
        mock_subplots.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    def test_mfdfa_model_plot_multifractal_spectrum(self, mock_subplots):
        """Test MFDFAModel.plot_multifractal_spectrum method."""
        # Mock matplotlib
        from unittest.mock import MagicMock
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        model = MFDFAModel()
        y = np.random.randn(500)
        model.fit(y)
        
        ax = model.plot_multifractal_spectrum()
        
        assert ax is mock_ax
        mock_subplots.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    def test_mfdfa_model_plot_hq(self, mock_subplots):
        """Test MFDFAModel.plot_hq method."""
        # Mock matplotlib
        from unittest.mock import MagicMock
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        model = MFDFAModel()
        y = np.random.randn(500)
        model.fit(y)
        
        ax = model.plot_hq()
        
        assert ax is mock_ax
        mock_subplots.assert_called_once()
    
    def test_mfdfa_model_plot_not_fitted(self):
        """Test MFDFAModel plotting when not fitted."""
        model = MFDFAModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.plot_loglog()
        with pytest.raises(ValueError, match="not fitted"):
            model.plot_multifractal_spectrum()
        with pytest.raises(ValueError, match="not fitted"):
            model.plot_hq()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_hurst_from_mfdfa(self):
        """Test hurst_from_mfdfa function."""
        hq = np.array([0.7, 0.6, 0.7])
        q_values = np.array([1, 2, 3])
        
        # Test exact match
        H = hurst_from_mfdfa(hq, q_values, q_target=2.0)
        assert H == 0.6
        assert isinstance(H, float)
        
        # Test interpolation
        H_interp = hurst_from_mfdfa(hq, q_values, q_target=2.5)
        assert isinstance(H_interp, float)
        assert np.isfinite(H_interp)
    
    def test_alpha_from_mfdfa(self):
        """Test alpha_from_mfdfa function."""
        alpha = np.array([0.7, 0.6, 0.7])
        q_values = np.array([1, 2, 3])
        
        # Test exact match
        a = alpha_from_mfdfa(alpha, q_values, q_target=2.0)
        assert a == 0.6
        assert isinstance(a, float)
        
        # Test interpolation
        a_interp = alpha_from_mfdfa(alpha, q_values, q_target=2.5)
        assert isinstance(a_interp, float)
        assert np.isfinite(a_interp)


class TestIntegration:
    """Integration tests for MFDFA."""
    
    def test_complete_workflow(self):
        """Test complete MFDFA workflow."""
        # Generate synthetic data
        np.random.seed(42)
        n = 1000
        y = np.cumsum(np.random.randn(n))
        y = np.diff(y)  # Make stationary
        
        # Functional API
        scales, fq, summary = mfdfa(y)
        
        # Object-oriented API
        model = MFDFAModel()
        model.fit(y)
        
        # Compare results
        np.testing.assert_array_equal(scales, model.scales)
        np.testing.assert_array_equal(fq, model.fq)
        np.testing.assert_array_equal(summary.hq, model.summary.hq)
    
    def test_pandas_series_input(self):
        """Test MFDFA with pandas Series input."""
        y_series = pd.Series(np.random.randn(500))
        
        # Functional API
        scales, fq, summary = mfdfa(y_series)
        
        # Object-oriented API
        model = MFDFAModel()
        model.fit(y_series)
        
        assert np.isfinite(summary.hq).any()
        assert np.isfinite(model.get_hq()).any()
    
    def test_different_parameters(self):
        """Test MFDFA with different parameters."""
        y = np.random.randn(500)
        
        # Default parameters
        scales1, fq1, summary1 = mfdfa(y)
        
        # Custom parameters
        scales2, fq2, summary2 = mfdfa(y, q_values=[-2, 0, 2], min_scale=16, max_scale=256)
        
        # Results should be different but both valid
        assert scales1.size != scales2.size
        assert summary1.q_values.size != summary2.q_values.size
        assert np.isfinite(summary1.hq).any()
        assert np.isfinite(summary2.hq).any()


if __name__ == "__main__":
    pytest.main([__file__])

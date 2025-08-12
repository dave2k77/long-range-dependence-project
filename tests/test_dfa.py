"""
Tests for DFA (Detrended Fluctuation Analysis) implementation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.analysis.dfa_analysis import (
    dfa, DFAModel, DFASummary, _validate_signal, _generate_log_scales,
    _profile, _poly_detrend, _fluctuation_for_scale,
    hurst_from_dfa_alpha, d_from_hurst
)


class TestDFASummary:
    """Test DFASummary dataclass."""
    
    def test_dfa_summary_creation(self):
        """Test DFASummary creation and basic properties."""
        scales = np.array([4, 8, 16, 32])
        flucts = np.array([1.0, 2.0, 4.0, 8.0])
        
        summary = DFASummary(
            alpha=1.0,
            intercept=0.0,
            rvalue=0.99,
            pvalue=0.001,
            stderr=0.01,
            scales=scales,
            flucts=flucts
        )
        
        assert summary.alpha == 1.0
        assert summary.intercept == 0.0
        assert summary.rvalue == 0.99
        assert summary.pvalue == 0.001
        assert summary.stderr == 0.01
        np.testing.assert_array_equal(summary.scales, scales)
        np.testing.assert_array_equal(summary.flucts, flucts)
    
    def test_as_dict(self):
        """Test DFASummary.as_dict method."""
        scales = np.array([4, 8])
        flucts = np.array([1.0, 2.0])
        
        summary = DFASummary(
            alpha=0.8,
            intercept=-0.5,
            rvalue=0.95,
            pvalue=0.01,
            stderr=0.05,
            scales=scales,
            flucts=flucts
        )
        
        result = summary.as_dict()
        
        assert isinstance(result, dict)
        assert result['alpha'] == 0.8
        assert result['intercept'] == -0.5
        assert result['rvalue'] == 0.95
        assert result['pvalue'] == 0.01
        assert result['stderr'] == 0.05
        np.testing.assert_array_equal(result['scales'], scales)
        np.testing.assert_array_equal(result['flucts'], flucts)


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
        y = np.random.randn(10)
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
        y[:5] = np.random.randn(5)
        with pytest.raises(ValueError, match="Too many non-finite"):
            _validate_signal(y)
    
    def test_generate_log_scales(self):
        """Test _generate_log_scales function."""
        scales = _generate_log_scales(1000)
        assert isinstance(scales, np.ndarray)
        assert scales.size >= 6
        assert np.all(scales >= 4)
        assert np.all(scales <= 250)  # max_scale = 1000 // 4 = 250
    
    def test_generate_log_scales_custom_params(self):
        """Test _generate_log_scales with custom parameters."""
        scales = _generate_log_scales(500, min_scale=8, max_scale=64, num_scales=10)
        assert scales.size >= 6
        assert np.all(scales >= 8)
        assert np.all(scales <= 64)
    
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
    
    def test_fluctuation_for_scale(self):
        """Test _fluctuation_for_scale function."""
        profile = np.random.randn(100)
        s = 10
        result = _fluctuation_for_scale(profile, s, order=1, use_reverse=True, overlap=False)
        assert isinstance(result, float)
        assert result > 0
        assert np.isfinite(result)


class TestDFAFunctionalAPI:
    """Test the functional DFA API."""
    
    def test_dfa_basic(self):
        """Test basic DFA functionality."""
        # Generate synthetic data with known scaling
        np.random.seed(42)
        n = 1000
        # Create fractional Gaussian noise-like data
        y = np.cumsum(np.random.randn(n))  # Random walk
        y = np.diff(y)  # Make it stationary
        
        scales, flucts, summary = dfa(y)
        
        assert isinstance(scales, np.ndarray)
        assert isinstance(flucts, np.ndarray)
        assert isinstance(summary, DFASummary)
        assert scales.size == flucts.size
        assert scales.size >= 6
        assert np.all(flucts > 0)
        assert np.isfinite(summary.alpha)
        assert summary.rvalue > 0
    
    def test_dfa_with_custom_scales(self):
        """Test DFA with custom scales."""
        y = np.random.randn(500)
        custom_scales = [4, 8, 16, 32, 64]
        
        scales, flucts, summary = dfa(y, scales=custom_scales)
        
        np.testing.assert_array_equal(scales, custom_scales)
        assert flucts.size == len(custom_scales)
    
    def test_dfa_different_orders(self):
        """Test DFA with different polynomial orders."""
        y = np.random.randn(500)
        
        # Test DFA1 (linear)
        scales1, flucts1, summary1 = dfa(y, order=1)
        
        # Test DFA2 (quadratic)
        scales2, flucts2, summary2 = dfa(y, order=2)
        
        assert summary1.alpha != summary2.alpha  # Should be different
        assert np.isfinite(summary1.alpha)
        assert np.isfinite(summary2.alpha)
    
    def test_dfa_with_overlap(self):
        """Test DFA with overlapping windows."""
        y = np.random.randn(500)
        
        scales, flucts, summary = dfa(y, overlap=True)
        
        assert np.isfinite(summary.alpha)
        assert summary.rvalue > 0
    
    def test_dfa_edge_cases(self):
        """Test DFA with edge cases."""
        # Very short series (should fail)
        with pytest.raises(ValueError):
            dfa(np.random.randn(10))
        
        # Invalid scales
        with pytest.raises(ValueError):
            dfa(np.random.randn(100), scales=[1])  # Scale < 2
        
        with pytest.raises(ValueError):
            dfa(np.random.randn(100), scales=[4])  # Only one scale


class TestDFAModel:
    """Test DFAModel class."""
    
    def test_dfa_model_initialization(self):
        """Test DFAModel initialization."""
        model = DFAModel(order=2, overlap=True)
        assert model.order == 2
        assert model.overlap is True
        assert model.scales is None
        assert model.flucts is None
        assert model.summary is None
        assert model.is_fitted is False
    
    def test_dfa_model_fit(self):
        """Test DFAModel.fit method."""
        model = DFAModel()
        y = np.random.randn(500)
        
        result = model.fit(y)
        
        assert result is model
        assert model.is_fitted is True
        assert model.scales is not None
        assert model.flucts is not None
        assert model.summary is not None
        assert np.isfinite(model.summary.alpha)
    
    def test_dfa_model_get_alpha(self):
        """Test DFAModel.get_alpha method."""
        model = DFAModel()
        y = np.random.randn(500)
        model.fit(y)
        
        alpha = model.get_alpha()
        assert isinstance(alpha, float)
        assert np.isfinite(alpha)
        assert alpha == model.summary.alpha
    
    def test_dfa_model_get_alpha_not_fitted(self):
        """Test DFAModel.get_alpha when not fitted."""
        model = DFAModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_alpha()
    
    def test_dfa_model_get_summary(self):
        """Test DFAModel.get_summary method."""
        model = DFAModel()
        y = np.random.randn(500)
        model.fit(y)
        
        summary = model.get_summary()
        assert isinstance(summary, DFASummary)
        assert summary is model.summary
    
    def test_dfa_model_get_summary_not_fitted(self):
        """Test DFAModel.get_summary when not fitted."""
        model = DFAModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_summary()
    
    @patch('matplotlib.pyplot.subplots')
    def test_dfa_model_plot_loglog(self, mock_subplots):
        """Test DFAModel.plot_loglog method."""
        # Mock matplotlib
        from unittest.mock import MagicMock
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        model = DFAModel()
        y = np.random.randn(500)
        model.fit(y)
        
        ax = model.plot_loglog()
        
        assert ax is mock_ax
        mock_subplots.assert_called_once()
    
    def test_dfa_model_plot_loglog_not_fitted(self):
        """Test DFAModel.plot_loglog when not fitted."""
        model = DFAModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.plot_loglog()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_hurst_from_dfa_alpha(self):
        """Test hurst_from_dfa_alpha function."""
        alpha = 0.8
        H = hurst_from_dfa_alpha(alpha)
        assert H == 0.8
        assert isinstance(H, float)
    
    def test_d_from_hurst(self):
        """Test d_from_hurst function."""
        H = 0.8
        d = d_from_hurst(H)
        assert abs(d - 0.3) < 1e-10  # 0.8 - 0.5
        assert isinstance(d, float)


class TestIntegration:
    """Integration tests for DFA."""
    
    def test_complete_workflow(self):
        """Test complete DFA workflow."""
        # Generate synthetic data
        np.random.seed(42)
        n = 1000
        y = np.cumsum(np.random.randn(n))
        y = np.diff(y)  # Make stationary
        
        # Functional API
        scales, flucts, summary = dfa(y)
        
        # Object-oriented API
        model = DFAModel()
        model.fit(y)
        
        # Compare results
        assert abs(summary.alpha - model.get_alpha()) < 1e-10
        np.testing.assert_array_equal(scales, model.scales)
        np.testing.assert_array_equal(flucts, model.flucts)
    
    def test_model_comparison(self):
        """Test comparing different DFA models."""
        y = np.random.randn(500)
        
        # DFA1
        model1 = DFAModel(order=1)
        model1.fit(y)
        
        # DFA2
        model2 = DFAModel(order=2)
        model2.fit(y)
        
        # Results should be different but both valid
        assert abs(model1.get_alpha() - model2.get_alpha()) > 1e-6
        assert np.isfinite(model1.get_alpha())
        assert np.isfinite(model2.get_alpha())
    
    def test_pandas_series_input(self):
        """Test DFA with pandas Series input."""
        y_series = pd.Series(np.random.randn(500))
        
        # Functional API
        scales, flucts, summary = dfa(y_series)
        
        # Object-oriented API
        model = DFAModel()
        model.fit(y_series)
        
        assert np.isfinite(summary.alpha)
        assert np.isfinite(model.get_alpha())
    
    def test_different_scale_generations(self):
        """Test DFA with different scale generation parameters."""
        y = np.random.randn(500)
        
        # Default scales
        scales1, flucts1, summary1 = dfa(y)
        
        # Custom scales
        scales2, flucts2, summary2 = dfa(y, min_scale=8, max_scale=128, num_scales=15)
        
        # Results should be different but both valid
        assert scales1.size != scales2.size
        assert np.isfinite(summary1.alpha)
        assert np.isfinite(summary2.alpha)


if __name__ == "__main__":
    pytest.main([__file__])

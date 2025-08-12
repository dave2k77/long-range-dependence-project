"""
Tests for R/S (Rescaled Range) Analysis implementation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.analysis.rs_analysis import (
    rs_analysis, RSModel, RSSummary, _validate_signal, _generate_log_scales,
    _rs_for_scale, d_from_hurst_rs, alpha_from_hurst_rs
)


class TestRSSummary:
    """Test RSSummary dataclass."""
    
    def test_rs_summary_creation(self):
        """Test RSSummary creation and basic properties."""
        scales = np.array([4, 8, 16, 32])
        rs_values = np.array([2.0, 4.0, 8.0, 16.0])
        
        summary = RSSummary(
            hurst=0.8,
            intercept=0.5,
            rvalue=0.99,
            pvalue=0.001,
            stderr=0.01,
            scales=scales,
            rs_values=rs_values
        )
        
        assert summary.hurst == 0.8
        assert summary.intercept == 0.5
        assert summary.rvalue == 0.99
        assert summary.pvalue == 0.001
        assert summary.stderr == 0.01
        np.testing.assert_array_equal(summary.scales, scales)
        np.testing.assert_array_equal(summary.rs_values, rs_values)
    
    def test_as_dict(self):
        """Test RSSummary.as_dict method."""
        scales = np.array([4, 8])
        rs_values = np.array([2.0, 4.0])
        
        summary = RSSummary(
            hurst=0.7,
            intercept=0.3,
            rvalue=0.95,
            pvalue=0.01,
            stderr=0.05,
            scales=scales,
            rs_values=rs_values
        )
        
        result = summary.as_dict()
        
        assert isinstance(result, dict)
        assert result['hurst'] == 0.7
        assert result['intercept'] == 0.3
        assert result['rvalue'] == 0.95
        assert result['pvalue'] == 0.01
        assert result['stderr'] == 0.05
        np.testing.assert_array_equal(result['scales'], scales)
        np.testing.assert_array_equal(result['rs_values'], rs_values)


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
    
    def test_rs_for_scale(self):
        """Test _rs_for_scale function."""
        y = np.random.randn(100)
        s = 10
        result = _rs_for_scale(y, s)
        assert isinstance(result, float)
        assert result > 0
        assert np.isfinite(result)
    
    def test_rs_for_scale_invalid(self):
        """Test _rs_for_scale with invalid parameters."""
        y = np.random.randn(100)
        
        # Scale too small
        result = _rs_for_scale(y, 1)
        assert np.isnan(result)
        
        # Scale too large
        result = _rs_for_scale(y, 200)
        assert np.isnan(result)


class TestRSFunctionalAPI:
    """Test the functional R/S API."""
    
    def test_rs_analysis_basic(self):
        """Test basic R/S analysis functionality."""
        # Generate synthetic data with known scaling
        np.random.seed(42)
        n = 1000
        # Create fractional Gaussian noise-like data
        y = np.cumsum(np.random.randn(n))  # Random walk
        y = np.diff(y)  # Make it stationary
        
        scales, rs_values, summary = rs_analysis(y)
        
        assert isinstance(scales, np.ndarray)
        assert isinstance(rs_values, np.ndarray)
        assert isinstance(summary, RSSummary)
        assert scales.size == rs_values.size
        assert scales.size >= 6
        assert np.all(rs_values > 0)
        assert np.isfinite(summary.hurst)
        assert summary.rvalue > 0
    
    def test_rs_analysis_with_custom_scales(self):
        """Test R/S analysis with custom scales."""
        y = np.random.randn(500)
        custom_scales = [4, 8, 16, 32, 64]
        
        scales, rs_values, summary = rs_analysis(y, scales=custom_scales)
        
        np.testing.assert_array_equal(scales, custom_scales)
        assert rs_values.size == len(custom_scales)
    
    def test_rs_analysis_edge_cases(self):
        """Test R/S analysis with edge cases."""
        # Very short series (should fail)
        with pytest.raises(ValueError):
            rs_analysis(np.random.randn(10))
        
        # Invalid scales
        with pytest.raises(ValueError):
            rs_analysis(np.random.randn(100), scales=[1])  # Scale < 2
        
        with pytest.raises(ValueError):
            rs_analysis(np.random.randn(100), scales=[4])  # Only one scale


class TestRSModel:
    """Test RSModel class."""
    
    def test_rs_model_initialization(self):
        """Test RSModel initialization."""
        model = RSModel()
        assert model.scales is None
        assert model.rs_values is None
        assert model.summary is None
        assert model.is_fitted is False
    
    def test_rs_model_fit(self):
        """Test RSModel.fit method."""
        model = RSModel()
        y = np.random.randn(500)
        
        result = model.fit(y)
        
        assert result is model
        assert model.is_fitted is True
        assert model.scales is not None
        assert model.rs_values is not None
        assert model.summary is not None
        assert np.isfinite(model.summary.hurst)
    
    def test_rs_model_get_hurst(self):
        """Test RSModel.get_hurst method."""
        model = RSModel()
        y = np.random.randn(500)
        model.fit(y)
        
        hurst = model.get_hurst()
        assert isinstance(hurst, float)
        assert np.isfinite(hurst)
        assert hurst == model.summary.hurst
    
    def test_rs_model_get_hurst_not_fitted(self):
        """Test RSModel.get_hurst when not fitted."""
        model = RSModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_hurst()
    
    def test_rs_model_get_summary(self):
        """Test RSModel.get_summary method."""
        model = RSModel()
        y = np.random.randn(500)
        model.fit(y)
        
        summary = model.get_summary()
        assert isinstance(summary, RSSummary)
        assert summary is model.summary
    
    def test_rs_model_get_summary_not_fitted(self):
        """Test RSModel.get_summary when not fitted."""
        model = RSModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_summary()
    
    @patch('matplotlib.pyplot.subplots')
    def test_rs_model_plot_loglog(self, mock_subplots):
        """Test RSModel.plot_loglog method."""
        # Mock matplotlib
        from unittest.mock import MagicMock
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        model = RSModel()
        y = np.random.randn(500)
        model.fit(y)
        
        ax = model.plot_loglog()
        
        assert ax is mock_ax
        mock_subplots.assert_called_once()
    
    def test_rs_model_plot_loglog_not_fitted(self):
        """Test RSModel.plot_loglog when not fitted."""
        model = RSModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.plot_loglog()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_d_from_hurst_rs(self):
        """Test d_from_hurst_rs function."""
        H = 0.8
        d = d_from_hurst_rs(H)
        assert abs(d - 0.3) < 1e-10  # 0.8 - 0.5
        assert isinstance(d, float)
    
    def test_alpha_from_hurst_rs(self):
        """Test alpha_from_hurst_rs function."""
        H = 0.8
        alpha = alpha_from_hurst_rs(H)
        assert alpha == 0.8
        assert isinstance(alpha, float)


class TestIntegration:
    """Integration tests for R/S analysis."""
    
    def test_complete_workflow(self):
        """Test complete R/S analysis workflow."""
        # Generate synthetic data
        np.random.seed(42)
        n = 1000
        y = np.cumsum(np.random.randn(n))
        y = np.diff(y)  # Make stationary
        
        # Functional API
        scales, rs_values, summary = rs_analysis(y)
        
        # Object-oriented API
        model = RSModel()
        model.fit(y)
        
        # Compare results
        assert abs(summary.hurst - model.get_hurst()) < 1e-10
        np.testing.assert_array_equal(scales, model.scales)
        np.testing.assert_array_equal(rs_values, model.rs_values)
    
    def test_pandas_series_input(self):
        """Test R/S analysis with pandas Series input."""
        y_series = pd.Series(np.random.randn(500))
        
        # Functional API
        scales, rs_values, summary = rs_analysis(y_series)
        
        # Object-oriented API
        model = RSModel()
        model.fit(y_series)
        
        assert np.isfinite(summary.hurst)
        assert np.isfinite(model.get_hurst())
    
    def test_different_scale_generations(self):
        """Test R/S analysis with different scale generation parameters."""
        y = np.random.randn(500)
        
        # Default scales
        scales1, rs_values1, summary1 = rs_analysis(y)
        
        # Custom scales
        scales2, rs_values2, summary2 = rs_analysis(y, min_scale=8, max_scale=128, num_scales=15)
        
        # Results should be different but both valid
        assert scales1.size != scales2.size
        assert np.isfinite(summary1.hurst)
        assert np.isfinite(summary2.hurst)


if __name__ == "__main__":
    pytest.main([__file__])

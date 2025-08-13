
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from submission.model_submission import BaseEstimatorModel

class DemoEstimatorModel(BaseEstimatorModel):
    """Example estimator model for demonstration"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hurst_estimate = None
        self.alpha_estimate = None
        self.confidence_intervals = {}
        self.quality_metrics = {}
    
    def fit(self, data):
        """Fit the model to the data"""
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        
        data = self.preprocess_data(data)
        
        # Simple variance-time plot analysis
        n = len(data)
        scales = [2**i for i in range(2, int(np.log2(n/4)))]
        variances = []
        
        for scale in scales:
            if scale < n:
                chunks = [data[i:i+scale] for i in range(0, n-scale+1, scale)]
                chunk_vars = [chunk.var() for chunk in chunks if len(chunk) == scale]
                if chunk_vars:
                    variances.append(np.mean(chunk_vars))
                else:
                    variances.append(np.nan)
            else:
                variances.append(np.nan)
        
        # Remove NaN values
        valid_scales = [s for s, v in zip(scales, variances) if not np.isnan(v)]
        valid_variances = [v for v in variances if not np.isnan(v)]
        
        if len(valid_scales) < 3:
            raise ValueError("Insufficient valid scales for analysis")
        
        # Fit power law: variance ~ scale^beta
        log_scales = np.log(valid_scales)
        log_variances = np.log(valid_variances)
        
        # Simple linear regression
        coeffs = np.polyfit(log_scales, log_variances, 1)
        beta = coeffs[0]
        
        self.hurst_estimate = 1 - beta / 2
        self.alpha_estimate = 2 * self.hurst_estimate
        
        # Calculate confidence intervals (simplified)
        std_error = np.sqrt(np.mean((log_variances - np.polyval(coeffs, log_scales))**2))
        self.confidence_intervals = {
            "hurst": (max(0, self.hurst_estimate - 2*std_error), 
                     min(1, self.hurst_estimate + 2*std_error)),
            "alpha": (max(0, self.alpha_estimate - 2*std_error), 
                     min(2, self.alpha_estimate + 2*std_error))
        }
        
        # Calculate quality metrics
        r_squared = 1 - np.sum((log_variances - np.polyval(coeffs, log_scales))**2) / np.sum((log_variances - np.mean(log_variances))**2)
        self.quality_metrics = {
            "r_squared": r_squared,
            "std_error": std_error,
            "n_scales": len(valid_scales)
        }
        
        self.fitted = True
        return self
    
    def estimate_hurst(self):
        """Estimate the Hurst exponent"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.hurst_estimate
    
    def estimate_alpha(self):
        """Estimate the alpha parameter"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.alpha_estimate
    
    def get_confidence_intervals(self):
        """Get confidence intervals for estimates"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.confidence_intervals
    
    def get_quality_metrics(self):
        """Get quality metrics for the estimation"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.quality_metrics

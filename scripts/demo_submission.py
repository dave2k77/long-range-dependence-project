#!/usr/bin/env python3
"""
Demo script for the submission system

This script demonstrates how to submit new models and datasets to the benchmark.
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from submission import (
    SubmissionManager,
    ModelMetadata,
    DatasetMetadata,
    BaseEstimatorModel
)


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


def create_demo_model_file():
    """Create a demo model file"""
    model_code = '''
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
'''
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Write model file
    model_file = models_dir / "demo_estimator_model.py"
    with open(model_file, 'w') as f:
        f.write(model_code)
    
    return str(model_file)


def create_demo_dataset():
    """Create a demo dataset"""
    # Generate synthetic data
    np.random.seed(42)
    n_points = 1000
    
    # Create timestamp
    start_date = datetime(2020, 1, 1)
    timestamps = [start_date + pd.Timedelta(hours=i) for i in range(n_points)]
    
    # Generate fractional Brownian motion
    hurst = 0.7
    d = hurst - 0.5
    
    fgn = np.zeros(n_points)
    for i in range(1, n_points):
        fgn[i] = fgn[i-1] + np.random.normal(0, 1) * (i ** d)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, n_points)
    values = fgn + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    # Create datasets directory if it doesn't exist
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    # Save dataset
    dataset_file = datasets_dir / "demo_fbm_dataset.csv"
    data.to_csv(dataset_file, index=False)
    
    return str(dataset_file)


def demo_model_submission():
    """Demonstrate model submission"""
    print("=== Model Submission Demo ===")
    
    # Create demo model file
    model_file = create_demo_model_file()
    print(f"Created demo model file: {model_file}")
    
    # Create model metadata
    model_metadata = ModelMetadata(
        name="DemoVarianceTimeEstimator",
        version="1.0.0",
        author="Demo User",
        description="A simple variance-time plot based estimator for demonstration purposes",
        algorithm_type="custom",
        parameters={
            "min_scale": 4,
            "max_scale": 256,
            "scale_type": "logarithmic"
        },
        dependencies=["numpy", "pandas"],
        file_path=model_file
    )
    
    # Submit model
    manager = SubmissionManager()
    result = manager.submit_model(
        model_file=model_file,
        metadata=model_metadata,
        run_full_analysis=True
    )
    
    print(f"\nSubmission ID: {result.submission_id}")
    print(f"Status: {result.status.value}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    if result.processing_time is not None:
        print(f"Processing time: {result.processing_time:.2f} seconds")
    else:
        print("Processing time: N/A")
    
    if result.validation_results:
        print("\nValidation Results:")
        for i, validation in enumerate(result.validation_results):
            status = "PASSED" if validation.is_valid else "FAILED"
            if validation.errors:
                print(f"  {i+1}. FAILED - Errors: {validation.errors}")
            elif validation.warnings:
                print(f"  {i+1}. PASSED with warnings: {validation.warnings}")
            else:
                print(f"  {i+1}. {status}")
    
    if result.performance_metrics:
        print("\nPerformance Metrics:")
        for key, value in result.performance_metrics.items():
            print(f"  {key}: {value}")
    
    return result


def demo_dataset_submission():
    """Demonstrate dataset submission"""
    print("\n=== Dataset Submission Demo ===")
    
    # Create demo dataset
    dataset_file = create_demo_dataset()
    print(f"Created demo dataset: {dataset_file}")
    
    # Create dataset metadata
    dataset_metadata = DatasetMetadata(
        name="DemoFBMData",
        version="1.0.0",
        author="Demo User",
        description="Synthetic fractional Brownian motion data for demonstration",
        source="Generated",
        format="csv",
        size=1000,
        columns=["timestamp", "value"],
        file_path=dataset_file,
        sampling_frequency="1 hour",
        units="arbitrary",
        collection_date="2024-01-01"
    )
    
    # Submit dataset
    manager = SubmissionManager()
    result = manager.submit_dataset(
        file_path=dataset_file,
        metadata=dataset_metadata,
        run_full_analysis=True
    )
    
    print(f"\nSubmission ID: {result.submission_id}")
    print(f"Status: {result.status.value}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    if result.processing_time is not None:
        print(f"Processing time: {result.processing_time:.2f} seconds")
    else:
        print("Processing time: N/A")
    
    if result.validation_results:
        print("\nValidation Results:")
        for i, validation in enumerate(result.validation_results):
            status = "PASSED" if validation.is_valid else "FAILED"
            if validation.errors:
                print(f"  {i+1}. FAILED - Errors: {validation.errors}")
            elif validation.warnings:
                print(f"  {i+1}. PASSED with warnings: {validation.warnings}")
            else:
                print(f"  {i+1}. {status}")
    
    if result.performance_metrics:
        print("\nQuality Evaluation:")
        for key, value in result.performance_metrics.items():
            print(f"  {key}: {value}")
    
    return result


def demo_submission_management():
    """Demonstrate submission management features"""
    print("\n=== Submission Management Demo ===")
    
    manager = SubmissionManager()
    
    # List all submissions
    all_submissions = manager.list_submissions()
    print(f"Total submissions: {len(all_submissions)}")
    
    model_submissions = manager.list_submissions("model")
    print(f"Model submissions: {len(model_submissions)}")
    
    dataset_submissions = manager.list_submissions("dataset")
    print(f"Dataset submissions: {len(dataset_submissions)}")
    
    # Get details of a specific submission
    if all_submissions:
        submission_id = all_submissions[0]
        result = manager.get_submission_result(submission_id)
        if result:
            print(f"\nSubmission {submission_id}:")
            print(f"  Type: {result.submission_type}")
            print(f"  Status: {result.status.value}")
            print(f"  Success: {result.success}")
            print(f"  Date: {result.submission_date}")


def main():
    """Main demo function"""
    print("Long-Range Dependence Project - Submission System Demo")
    print("=" * 60)
    
    try:
        # Demo model submission
        model_result = demo_model_submission()
        
        # Demo dataset submission
        dataset_result = demo_dataset_submission()
        
        # Demo submission management
        demo_submission_management()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print(f"Model submission: {'SUCCESS' if model_result.success else 'FAILED'}")
        print(f"Dataset submission: {'SUCCESS' if dataset_result.success else 'FAILED'}")
        
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

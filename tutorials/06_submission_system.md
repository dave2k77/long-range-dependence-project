# Submission System Tutorial

This tutorial explains how to submit new estimator models and datasets to the Long-Range Dependence benchmark system.

## Overview

The submission system allows users to contribute new models and datasets to the benchmark, ensuring they meet quality standards and integrate properly with the existing analysis pipeline.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Model Submission](#model-submission)
3. [Dataset Submission](#dataset-submission)
4. [Standards and Requirements](#standards-and-requirements)
5. [Validation Process](#validation-process)
6. [Integration Testing](#integration-testing)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

Before submitting, ensure you have:

- Python 3.8+ installed
- Required dependencies: `numpy`, `pandas`, `scipy`
- Understanding of long-range dependence analysis
- Your model or dataset ready for submission

## Model Submission

### Step 1: Create Your Model

Your model must inherit from `BaseEstimatorModel` and implement the required methods:

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

from submission.model_submission import BaseEstimatorModel
import numpy as np

class MyEstimatorModel(BaseEstimatorModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hurst_estimate = None
        self.alpha_estimate = None
        self.confidence_intervals = {}
        self.quality_metrics = {}
    
    def fit(self, data):
        """Fit the model to the data"""
        # Your implementation here
        # Must return self
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
        return self.confidence_intervals
    
    def get_quality_metrics(self):
        """Get quality metrics for the estimation"""
        return self.quality_metrics
```

### Step 2: Prepare Model Metadata

Create metadata for your model:

```python
from submission.model_submission import ModelMetadata

metadata = ModelMetadata(
    name="MyVarianceTimeEstimator",
    version="1.0.0",
    author="Your Name",
    description="A variance-time plot based estimator for long-range dependence",
    algorithm_type="custom",  # or "dfa", "rs", "wavelet", "spectral"
    parameters={
        "min_scale": 4,
        "max_scale": 256,
        "scale_type": "logarithmic"
    },
    dependencies=["numpy", "pandas"],
    file_path="path/to/your/model.py"
)
```

**Important Note**: The field is now `algorithm_type` instead of `category`.

### Step 3: Submit Your Model

```python
from submission.model_submission import ModelSubmission

# Create submission
submission = ModelSubmission()

# Submit the model
result = submission.submit_model(
    model_file="path/to/your/model.py",
    metadata=metadata
)

if result.is_valid:
    print("✓ Model submitted successfully!")
    print(f"Model ID: {result.model_id}")
else:
    print("✗ Model submission failed:")
    for error in result.errors:
        print(f"  - {error}")
```

## Dataset Submission

### Step 1: Prepare Dataset Metadata

```python
from submission.dataset_submission import DatasetMetadata

dataset_metadata = DatasetMetadata(
    name="MyFinancialDataset",
    description="Daily stock returns for technology companies",
    data_type="financial",
    n_samples=1000,
    n_features=5,
    sampling_frequency="daily",
    units="returns",
    collection_date="2024-01-01",
    parameters={
        "start_date": "2020-01-01",
        "end_date": "2024-01-01",
        "symbols": ["AAPL", "GOOGL", "MSFT"]
    },
    file_path="path/to/your/dataset.csv"
)
```

### Step 2: Submit Dataset

```python
from submission.dataset_submission import DatasetSubmission

# Create submission
submission = DatasetSubmission()

# Submit the dataset
result = submission.submit_dataset(
    dataset_file="path/to/your/dataset.csv",
    metadata=dataset_metadata
)

if result.is_valid:
    print("✓ Dataset submitted successfully!")
    print(f"Dataset ID: {result.dataset_id}")
else:
    print("✗ Dataset submission failed:")
    for error in result.errors:
        print(f"  - {error}")
```

## Standards and Requirements

### Model Requirements

1. **Inheritance**: Must inherit from `BaseEstimatorModel`
2. **Required Methods**:
   - `fit(data)`: Fit the model to data
   - `estimate_hurst()`: Return Hurst exponent estimate
   - `estimate_alpha()`: Return alpha parameter estimate
   - `get_confidence_intervals()`: Return confidence intervals
   - `get_quality_metrics()`: Return quality metrics

3. **Return Values**:
   - `estimate_hurst()`: Must return a float
   - `estimate_alpha()`: Must return a float
   - `get_confidence_intervals()`: Must return a dict with 'hurst' and 'alpha' keys
   - `get_quality_metrics()`: Must return a dict

### Dataset Requirements

1. **Format**: CSV, JSON, or NumPy (.npy) files
2. **Data Quality**: No missing values, reasonable range
3. **Metadata**: Complete description and parameters
4. **Size**: Minimum 100 points, maximum 1,000,000 points

## Validation Process

### Model Validation

The system automatically validates your model:

1. **Syntax Check**: Ensures Python code is valid
2. **Import Check**: Verifies all dependencies can be imported
3. **Class Check**: Confirms inheritance from `BaseEstimatorModel`
4. **Method Check**: Validates required methods exist
5. **Type Check**: Ensures return types are correct

### Dataset Validation

1. **File Format**: Checks file can be loaded
2. **Data Structure**: Validates data dimensions and types
3. **Quality Check**: Identifies potential issues
4. **Metadata Validation**: Ensures completeness

## Integration Testing

### Test Your Model

```python
from submission.model_submission import ModelTester

# Create tester
tester = ModelTester()

# Test with synthetic data
test_result = tester.test_model(
    model_file="path/to/your/model.py",
    test_data=test_dataset
)

print(f"Test Results:")
print(f"  Hurst estimate: {test_result.hurst_estimate:.3f}")
print(f"  Alpha estimate: {test_result.alpha_estimate:.3f}")
print(f"  Processing time: {test_result.processing_time:.3f}s")
print(f"  Memory usage: {test_result.memory_usage:.1f}MB")
```

### Test Your Dataset

```python
from submission.dataset_submission import DatasetTester

# Create tester
tester = DatasetTester()

# Test dataset quality
test_result = tester.test_dataset(
    dataset_file="path/to/your/dataset.csv"
)

print(f"Dataset Test Results:")
print(f"  Data quality score: {test_result.quality_score:.2f}")
print(f"  Missing values: {test_result.missing_values}")
print(f"  Outliers detected: {test_result.outliers_detected}")
```

## Best Practices

### 1. Model Design

- **Efficiency**: Optimize for speed and memory usage
- **Robustness**: Handle edge cases gracefully
- **Documentation**: Include clear docstrings
- **Testing**: Test with various data types and sizes

### 2. Dataset Preparation

- **Clean Data**: Remove or handle missing values
- **Normalization**: Consider if data needs scaling
- **Documentation**: Document data source and processing
- **Validation**: Verify data quality before submission

### 3. Submission Process

- **Test Locally**: Ensure your model/dataset works before submitting
- **Check Dependencies**: List all required packages
- **Provide Examples**: Include usage examples in metadata
- **Version Control**: Use semantic versioning

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` during validation
**Solution**: Ensure all dependencies are listed in `requirements.txt`

**Issue**: Model fails validation tests
**Solution**: Check that all required methods return correct types

**Issue**: Dataset too large
**Solution**: Consider sampling or compression

**Issue**: Import errors in model file
**Solution**: Use relative imports or add `src` to Python path

### Getting Help

1. **Check Logs**: Review validation error messages
2. **Test Locally**: Run validation tests on your machine
3. **Review Examples**: Study existing successful submissions
4. **Create Issue**: Report bugs on GitHub

## Example: Complete Submission

Here's a complete example of submitting a custom estimator:

```python
#!/usr/bin/env python3
"""
Example: Custom Variance-Time Estimator
"""

import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

import numpy as np
from submission.model_submission import BaseEstimatorModel, ModelMetadata, ModelSubmission

class VarianceTimeEstimator(BaseEstimatorModel):
    """Custom estimator using variance-time plot analysis."""
    
    def __init__(self, min_scale=4, max_scale=256, **kwargs):
        super().__init__(**kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.hurst_estimate = None
        self.alpha_estimate = None
        self.confidence_intervals = {}
        self.quality_metrics = {}
    
    def fit(self, data):
        """Fit the model using variance-time analysis."""
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        
        data = self.preprocess_data(data)
        n = len(data)
        
        # Generate scales
        scales = [2**i for i in range(int(np.log2(self.min_scale)), 
                                    int(np.log2(min(self.max_scale, n//4)))]
        
        # Calculate variances for each scale
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
        
        # Linear regression
        coeffs = np.polyfit(log_scales, log_variances, 1)
        beta = coeffs[0]
        
        # Calculate estimates
        self.hurst_estimate = 1 - beta / 2
        self.alpha_estimate = 2 * self.hurst_estimate
        
        # Calculate confidence intervals
        residuals = log_variances - np.polyval(coeffs, log_scales)
        std_error = np.sqrt(np.mean(residuals**2))
        
        self.confidence_intervals = {
            "hurst": (max(0, self.hurst_estimate - 2*std_error), 
                     min(1, self.hurst_estimate + 2*std_error)),
            "alpha": (max(0, self.alpha_estimate - 2*std_error), 
                     min(2, self.alpha_estimate + 2*std_error))
        }
        
        # Calculate quality metrics
        r_squared = 1 - np.sum(residuals**2) / np.sum((log_variances - np.mean(log_variances))**2)
        self.quality_metrics = {
            "r_squared": r_squared,
            "std_error": std_error,
            "n_scales": len(valid_scales)
        }
        
        self.fitted = True
        return self
    
    def estimate_hurst(self):
        """Estimate the Hurst exponent."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.hurst_estimate
    
    def estimate_alpha(self):
        """Estimate the alpha parameter."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.alpha_estimate
    
    def get_confidence_intervals(self):
        """Get confidence intervals for estimates."""
        return self.confidence_intervals
    
    def get_quality_metrics(self):
        """Get quality metrics for the estimation."""
        return self.quality_metrics

# Submit the model
def submit_custom_estimator():
    """Submit the custom variance-time estimator."""
    
    # Create metadata
    metadata = ModelMetadata(
        name="VarianceTimeEstimator",
        version="1.0.0",
        author="Your Name",
        description="Custom estimator using variance-time plot analysis for long-range dependence",
        algorithm_type="custom",
        parameters={
            "min_scale": 4,
            "max_scale": 256,
            "method": "variance_time"
        },
        dependencies=["numpy"],
        file_path="variance_time_estimator.py"
    )
    
    # Create submission
    submission = ModelSubmission()
    
    # Submit
    result = submission.submit_model(
        model_file="variance_time_estimator.py",
        metadata=metadata
    )
    
    if result.is_valid:
        print("✓ Custom estimator submitted successfully!")
        print(f"Model ID: {result.model_id}")
        return result
    else:
        print("✗ Submission failed:")
        for error in result.errors:
            print(f"  - {error}")
        return None

if __name__ == "__main__":
    submit_custom_estimator()
```

## Next Steps

After successful submission:

1. **Monitor Performance**: Check how your model performs on benchmark datasets
2. **Iterate**: Improve based on feedback and results
3. **Collaborate**: Share insights with the community
4. **Contribute**: Help improve the benchmark system

---

**You're now ready to contribute to the Long-Range Dependence benchmark!** 🎉

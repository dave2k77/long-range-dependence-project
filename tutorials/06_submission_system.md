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
from src.submission.model_submission import BaseEstimatorModel
import numpy as np

class MyEstimatorModel(BaseEstimatorModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hurst_estimate = None
        self.alpha_estimate = None
    
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
        return {
            "hurst": (lower_bound, upper_bound),
            "alpha": (lower_bound, upper_bound)
        }
    
    def get_quality_metrics(self):
        """Get quality metrics for the estimation"""
        return {
            "r_squared": r_squared_value,
            "std_error": standard_error,
            "convergence": convergence_status
        }
```

### Step 2: Prepare Model Metadata

Create metadata for your model:

```python
from src.submission.model_submission import ModelMetadata

metadata = ModelMetadata(
    name="MyVarianceTimeEstimator",
    version="1.0.0",
    author="Your Name",
    description="A variance-time plot based estimator for long-range dependence",
    category="custom",  # or "dfa", "rs", "wavelet", "spectral"
    parameters={
        "min_scale": 4,
        "max_scale": 256,
        "scale_type": "logarithmic"
    },
    dependencies=["numpy", "pandas"],
    file_path="path/to/your/model.py"
)
```

### Step 3: Submit Your Model

```python
from src.submission import SubmissionManager

manager = SubmissionManager()
result = manager.submit_model(
    model_file="path/to/your/model.py",
    metadata=metadata,
    run_full_analysis=True
)

print(f"Submission ID: {result.submission_id}")
print(f"Status: {result.status.value}")
print(f"Success: {result.success}")
```

## Dataset Submission

### Step 1: Prepare Your Dataset

Your dataset must be in one of the supported formats:
- CSV (recommended)
- JSON
- Parquet
- HDF5

Required columns:
- `timestamp`: Time index
- `value`: The time series values

Example CSV format:
```csv
timestamp,value
2020-01-01 00:00:00,1.234
2020-01-01 01:00:00,1.456
2020-01-01 02:00:00,1.789
...
```

### Step 2: Prepare Dataset Metadata

```python
from src.submission.dataset_submission import DatasetMetadata

metadata = DatasetMetadata(
    name="MyTimeSeriesData",
    version="1.0.0",
    author="Your Name",
    description="Financial time series data for long-range dependence analysis",
    category="financial",  # or "physiological", "environmental", "synthetic"
    source="Yahoo Finance",
    sampling_frequency="1 hour",
    units="USD",
    collection_date="2024-01-01",
    file_path="path/to/your/dataset.csv"
)
```

### Step 3: Submit Your Dataset

```python
from src.submission import SubmissionManager

manager = SubmissionManager()
result = manager.submit_dataset(
    file_path="path/to/your/dataset.csv",
    metadata=metadata,
    run_full_analysis=True
)

print(f"Submission ID: {result.submission_id}")
print(f"Status: {result.status.value}")
print(f"Success: {result.success}")
```

## Standards and Requirements

### Model Standards

Your model must meet these requirements:

1. **Interface Compliance**:
   - Inherit from `BaseEstimatorModel`
   - Implement all required methods
   - Follow the expected parameter ranges

2. **Performance Standards**:
   - Minimum R² of 0.7
   - Maximum standard error of 0.2
   - Minimum convergence rate of 80%
   - Maximum computation time of 300 seconds

3. **Documentation**:
   - README.md with usage examples
   - API documentation
   - Parameter descriptions
   - Performance characteristics

### Dataset Standards

Your dataset must meet these requirements:

1. **Format Requirements**:
   - Supported file format
   - Required columns present
   - Proper data types

2. **Quality Requirements**:
   - Minimum length: 100 points
   - Maximum length: 100,000 points
   - Maximum missing ratio: 10%
   - Maximum outlier ratio: 5%
   - Minimum quality score: 0.7

3. **Metadata Requirements**:
   - All required metadata fields
   - Accurate descriptions
   - Proper categorization

## Validation Process

The submission system performs comprehensive validation:

### Model Validation

1. **File Validation**:
   - File exists and is readable
   - Correct file extension
   - File size within limits

2. **Code Validation**:
   - Syntax check
   - Import validation
   - Interface compliance

3. **Testing**:
   - Unit tests on synthetic data
   - Performance benchmarks
   - Integration tests

### Dataset Validation

1. **Format Validation**:
   - File format check
   - Column structure validation
   - Data type verification

2. **Content Validation**:
   - Data length requirements
   - Missing value analysis
   - Outlier detection
   - Quality scoring

3. **Testing**:
   - Basic statistical tests
   - Long-range dependence analysis
   - Robustness testing

## Integration Testing

After validation, your submission undergoes integration testing:

### Model Integration

1. **Full Analysis Pipeline**:
   - Test on multiple synthetic datasets
   - Compare with existing models
   - Performance benchmarking

2. **Comparison Analysis**:
   - Accuracy comparison
   - Speed comparison
   - Memory efficiency

### Dataset Integration

1. **Multi-Model Analysis**:
   - Test with all available models
   - Consistency checking
   - Result comparison

2. **Quality Assessment**:
   - Long-range dependence detection
   - Stability analysis
   - Recommendation generation

## Best Practices

### For Model Development

1. **Follow Standards**:
   - Use the provided base class
   - Implement all required methods
   - Follow naming conventions

2. **Optimize Performance**:
   - Efficient algorithms
   - Memory management
   - Error handling

3. **Documentation**:
   - Clear docstrings
   - Usage examples
   - Parameter explanations

### For Dataset Preparation

1. **Data Quality**:
   - Clean and preprocess data
   - Handle missing values
   - Remove outliers appropriately

2. **Metadata**:
   - Complete and accurate metadata
   - Clear descriptions
   - Proper categorization

3. **Format**:
   - Use standard formats
   - Consistent timestamps
   - Proper data types

## Troubleshooting

### Common Model Issues

1. **Import Errors**:
   ```
   Error: Failed to import model
   Solution: Check dependencies and import statements
   ```

2. **Interface Errors**:
   ```
   Error: Missing required methods
   Solution: Implement all required methods from BaseEstimatorModel
   ```

3. **Performance Issues**:
   ```
   Error: Model performance below threshold
   Solution: Optimize algorithm and check parameter ranges
   ```

### Common Dataset Issues

1. **Format Errors**:
   ```
   Error: Unsupported file format
   Solution: Use CSV, JSON, Parquet, or HDF5 format
   ```

2. **Column Errors**:
   ```
   Error: Missing required column 'value'
   Solution: Ensure 'timestamp' and 'value' columns are present
   ```

3. **Quality Issues**:
   ```
   Error: Dataset quality below threshold
   Solution: Clean data and reduce missing/outlier ratios
   ```

### Getting Help

If you encounter issues:

1. Check the validation results for specific error messages
2. Review the standards and requirements
3. Run the demo script to see working examples
4. Check the logs in `results/submissions/`

## Example: Complete Submission

Here's a complete example of submitting a model:

```python
#!/usr/bin/env python3
"""Example model submission"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from submission import SubmissionManager, ModelMetadata

# Create model metadata
metadata = ModelMetadata(
    name="ExampleEstimator",
    version="1.0.0",
    author="John Doe",
    description="Example estimator for tutorial",
    category="custom",
    parameters={"scale_factor": 1.0},
    dependencies=["numpy"],
    file_path="models/example_model.py"
)

# Submit model
manager = SubmissionManager()
result = manager.submit_model(
    model_file="models/example_model.py",
    metadata=metadata,
    run_full_analysis=True
)

# Check results
if result.success:
    print("✅ Model submitted successfully!")
    print(f"Submission ID: {result.submission_id}")
else:
    print("❌ Submission failed:")
    print(f"Error: {result.message}")
    for validation in result.validation_results:
        if not validation.passed:
            print(f"  - {validation.message}")
```

## Next Steps

After successful submission:

1. **Monitor Status**: Check submission status using the submission ID
2. **Review Results**: Examine validation and test results
3. **Integration**: Approved submissions are automatically integrated
4. **Updates**: You can update your submission with new versions

## Conclusion

The submission system provides a robust framework for contributing to the long-range dependence benchmark. By following the standards and best practices outlined in this tutorial, you can successfully submit high-quality models and datasets that enhance the benchmark's capabilities.

For more information, see:
- [API Documentation](api_documentation.md)
- [Standards Reference](standards_reference.md)
- [Demo Scripts](scripts/demo_submission.py)

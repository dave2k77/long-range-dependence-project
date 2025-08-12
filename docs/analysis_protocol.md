# Long-Range Dependence Analysis Protocol

This document provides a standardized protocol for conducting long-range dependence analysis using the methods implemented in this project.

## Table of Contents

1. [Pre-Analysis Preparation](#pre-analysis-preparation)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Analysis Workflow](#analysis-workflow)
5. [Quality Assessment](#quality-assessment)
6. [Results Interpretation](#results-interpretation)
7. [Reporting Standards](#reporting-standards)
8. [Troubleshooting](#troubleshooting)

## Pre-Analysis Preparation

### 1.1 Project Setup

#### Environment Configuration

```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import numpy, pandas, scipy, statsmodels, pywt; print('All packages installed')"
```

#### Directory Structure

Ensure the following directory structure exists:

```
project_root/
├── data/
│   ├── raw/           # Original data files
│   ├── processed/     # Cleaned and preprocessed data
│   └── metadata/      # Data documentation
├── results/
│   ├── figures/       # Generated plots
│   ├── tables/        # Results tables
│   └── reports/       # Analysis reports
├── config/            # Configuration files
└── src/               # Source code
```

#### Configuration Setup

```python
from src.config_loader import get_config_loader

# Load configuration
config = get_config_loader()
data_config = config.get_data_config()
analysis_config = config.get_analysis_config()
plot_config = config.get_plot_config()
```

### 1.2 Data Requirements

#### Minimum Data Requirements

- **Sample Size**: Minimum 100 observations (recommended: 1000+)
- **Data Type**: Numeric time series data
- **Format**: CSV, Excel, or NumPy array
- **Quality**: No more than 10% missing values

#### Data Characteristics

- **Stationarity**: Check if data is stationary
- **Trends**: Identify and document any trends
- **Seasonality**: Note any seasonal patterns
- **Outliers**: Document extreme values

### 1.3 Analysis Planning

#### Research Questions

Define clear research questions:

1. **Primary Question**: Does the time series exhibit long-range dependence?
2. **Secondary Questions**: 
   - What is the strength of the dependence?
   - Is the dependence persistent or anti-persistent?
   - Are there multiple scaling regimes?

#### Method Selection

Choose appropriate methods based on data characteristics:

- **DFA**: General purpose, robust to trends
- **R/S**: Traditional method, requires detrending
- **MFDFA**: For multifractal analysis
- **Wavelet**: For non-stationary processes
- **Spectral**: For frequency domain analysis
- **ARFIMA**: For modeling and forecasting

## Data Preprocessing

### 2.1 Data Loading

#### Basic Loading

```python
from src.data_processing.data_loader import DataLoader

# Load data
loader = DataLoader()
data = loader.load_from_csv('data/raw/time_series.csv')

# Check basic properties
print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Missing values: {data.isnull().sum()}")
```

#### Financial Data Loading

```python
# Load financial data
symbols = ['AAPL', 'GOOGL', 'MSFT']
financial_data = loader.load_financial_data(
    symbols=symbols,
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

### 2.2 Data Cleaning

#### Missing Value Handling

```python
from src.data_processing.preprocessing import TimeSeriesPreprocessor

# Create preprocessor
preprocessor = TimeSeriesPreprocessor(
    missing_values_method='interpolation',
    outlier_method='iqr',
    outlier_threshold=1.5
)

# Clean data
cleaned_data = preprocessor.clean_time_series(data)
```

#### Outlier Detection and Treatment

```python
# Detect outliers
outliers = preprocessor.detect_outliers(data)

# Handle outliers
if len(outliers) > 0:
    print(f"Found {len(outliers)} outliers")
    # Choose method: removal, winsorization, or capping
    cleaned_data = preprocessor.handle_outliers(data, method='winsorization')
```

### 2.3 Data Transformation

#### Stationarity Testing

```python
# Test stationarity
stationarity_results = preprocessor.test_stationarity(data)

print("Stationarity Test Results:")
for test, result in stationarity_results.items():
    print(f"{test}: {'Stationary' if result['stationary'] else 'Non-stationary'}")
    print(f"  p-value: {result['p_value']:.4f}")
```

#### Making Data Stationary

```python
# If non-stationary, make stationary
if not stationarity_results['adf']['stationary']:
    # Try differencing
    stationary_data = preprocessor.make_stationary(data, method='differencing')
    
    # Test again
    new_stationarity = preprocessor.test_stationarity(stationary_data)
```

### 2.4 Data Quality Assessment

```python
from src.data_processing.quality_check import DataQualityChecker

# Assess data quality
checker = DataQualityChecker()
quality_report = checker.assess_data_quality(cleaned_data)

print("Data Quality Score:", quality_report['overall_score'])
print("Quality Issues:", quality_report['issues'])
```

## Exploratory Data Analysis

### 3.1 Basic Statistics

```python
import numpy as np
import pandas as pd

# Calculate basic statistics
stats = {
    'mean': np.mean(cleaned_data),
    'std': np.std(cleaned_data),
    'min': np.min(cleaned_data),
    'max': np.max(cleaned_data),
    'skewness': scipy.stats.skew(cleaned_data),
    'kurtosis': scipy.stats.kurtosis(cleaned_data)
}

print("Basic Statistics:")
for stat, value in stats.items():
    print(f"  {stat}: {value:.4f}")
```

### 3.2 Time Series Visualization

```python
from src.visualisation.time_series_plots import plot_time_series

# Create basic time series plot
fig, ax = plot_time_series(
    cleaned_data,
    title="Time Series Data",
    xlabel="Time",
    ylabel="Value"
)

# Save plot
fig.savefig('results/figures/time_series.png', dpi=300, bbox_inches='tight')
plt.close()
```

### 3.3 Autocorrelation Analysis

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot autocorrelation functions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(cleaned_data, ax=ax1, lags=50, title="Autocorrelation Function")
plot_pacf(cleaned_data, ax=ax2, lags=50, title="Partial Autocorrelation Function")

plt.tight_layout()
fig.savefig('results/figures/autocorrelation.png', dpi=300, bbox_inches='tight')
plt.close()
```

## Analysis Workflow

### 4.1 DFA Analysis

#### Basic DFA

```python
from src.analysis.fractal_analysis import dfa

# Run DFA analysis
dfa_results = dfa(
    cleaned_data,
    min_scale=10,
    max_scale=None,  # Will be set to n//4
    n_scales=20,
    detrend_order=1
)

print("DFA Results:")
print(f"  Alpha: {dfa_results['alpha']:.4f}")
print(f"  Hurst Exponent: {dfa_results['hurst']:.4f}")
print(f"  R²: {dfa_results['r_squared']:.4f}")
```

#### DFA Visualization

```python
from src.visualisation.fractal_plots import plot_dfa_results

# Plot DFA results
fig = plot_dfa_results(dfa_results, title="DFA Analysis Results")
fig.savefig('results/figures/dfa_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
```

### 4.2 R/S Analysis

#### Basic R/S

```python
from src.analysis.fractal_analysis import rescaled_range

# Run R/S analysis
rs_results = rescaled_range(
    cleaned_data,
    min_scale=10,
    max_scale=None,
    n_scales=20
)

print("R/S Results:")
print(f"  Hurst Exponent: {rs_results['hurst']:.4f}")
print(f"  Alpha: {rs_results['alpha_rs']:.4f}")
print(f"  R²: {rs_results['r_squared']:.4f}")
```

#### R/S Visualization

```python
from src.visualisation.fractal_plots import plot_rs_results

# Plot R/S results
fig = plot_rs_results(rs_results, title="R/S Analysis Results")
fig.savefig('results/figures/rs_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
```

### 4.3 MFDFA Analysis

#### Basic MFDFA

```python
from src.analysis.fractal_analysis import mfdfa

# Run MFDFA analysis
mfdfa_results = mfdfa(
    cleaned_data,
    min_scale=10,
    max_scale=None,
    n_scales=20,
    q_min=-5,
    q_max=5,
    q_step=0.5
)

print("MFDFA Results:")
print(f"  Generalized Hurst Exponents: {mfdfa_results['h_q']}")
print(f"  Multifractal Spectrum: {mfdfa_results['f_alpha']}")
```

#### MFDFA Visualization

```python
from src.visualisation.fractal_plots import plot_mfdfa_results

# Plot MFDFA results
fig = plot_mfdfa_results(mfdfa_results, title="MFDFA Analysis Results")
fig.savefig('results/figures/mfdfa_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
```

### 4.4 Wavelet Analysis

#### Wavelet Leaders

```python
from src.analysis.wavelet_analysis import wavelet_leaders_estimation

# Run wavelet leaders analysis
wavelet_results = wavelet_leaders_estimation(
    cleaned_data,
    wavelet_type='db4',
    min_scale=2,
    max_scale=None
)

print("Wavelet Leaders Results:")
print(f"  Hurst Exponent: {wavelet_results['hurst']:.4f}")
print(f"  Alpha: {wavelet_results['alpha']:.4f}")
```

#### Wavelet Whittle

```python
from src.analysis.wavelet_analysis import wavelet_whittle_estimation

# Run wavelet whittle analysis
whittle_results = wavelet_whittle_estimation(
    cleaned_data,
    wavelet_type='db4',
    freq_min=0.01,
    freq_max=0.5
)

print("Wavelet Whittle Results:")
print(f"  Hurst Exponent: {whittle_results['hurst']:.4f}")
print(f"  Alpha: {whittle_results['alpha']:.4f}")
```

### 4.5 Spectral Analysis

#### Periodogram Estimation

```python
from src.analysis.spectral_analysis import periodogram_estimation

# Run periodogram analysis
periodogram_results = periodogram_estimation(
    cleaned_data,
    method='welch',
    window='hann'
)

print("Periodogram Results:")
print(f"  Frequencies: {len(periodogram_results['frequencies'])}")
print(f"  Power Spectrum: {len(periodogram_results['power_spectrum'])}")
```

#### Whittle MLE

```python
from src.analysis.spectral_analysis import whittle_mle

# Run Whittle MLE analysis
whittle_results = whittle_mle(
    cleaned_data,
    freq_min=0.01,
    freq_max=0.5
)

print("Whittle MLE Results:")
print(f"  Hurst Exponent: {whittle_results['hurst']:.4f}")
print(f"  Alpha: {whittle_results['alpha']:.4f}")
```

### 4.6 ARFIMA Modeling

#### Model Fitting

```python
from src.analysis.arfima_analysis import ARFIMAModel

# Create ARFIMA model
arfima_model = ARFIMAModel(
    max_p=3,
    max_d=2,
    max_q=3,
    method='css-mle'
)

# Fit model
arfima_results = arfima_model.fit(cleaned_data)

print("ARFIMA Results:")
print(f"  Best Model: ARFIMA{arfima_results['best_order']}")
print(f"  AIC: {arfima_results['aic']:.4f}")
print(f"  BIC: {arfima_results['bic']:.4f}")
```

#### Model Diagnostics

```python
# Run diagnostics
diagnostics = arfima_model.diagnose(arfima_results)

print("Model Diagnostics:")
for test, result in diagnostics.items():
    print(f"  {test}: {'Pass' if result['passed'] else 'Fail'}")
    print(f"    p-value: {result['p_value']:.4f}")
```

## Quality Assessment

### 5.1 Method Comparison

#### Cross-Method Validation

```python
# Compare results across methods
methods = {
    'DFA': dfa_results,
    'R/S': rs_results,
    'Wavelet Leaders': wavelet_results,
    'Wavelet Whittle': whittle_results,
    'Whittle MLE': whittle_results,
    'ARFIMA': arfima_results
}

# Create comparison table
comparison_data = []
for method, results in methods.items():
    if 'hurst' in results:
        comparison_data.append({
            'Method': method,
            'Hurst Exponent': results['hurst'],
            'Alpha': results.get('alpha', results.get('alpha_rs', 'N/A')),
            'R²': results.get('r_squared', 'N/A')
        })

comparison_df = pd.DataFrame(comparison_data)
print("Method Comparison:")
print(comparison_df)
```

#### Consistency Check

```python
# Check consistency of Hurst exponents
hurst_values = [results['hurst'] for results in methods.values() if 'hurst' in results]
hurst_std = np.std(hurst_values)

print(f"Hurst Exponent Consistency:")
print(f"  Mean: {np.mean(hurst_values):.4f}")
print(f"  Std: {hurst_std:.4f}")
print(f"  CV: {hurst_std/np.mean(hurst_values):.4f}")

# Flag potential issues
if hurst_std > 0.1:
    print("  WARNING: High variability in Hurst estimates")
```

### 5.2 Robustness Analysis

#### Parameter Sensitivity

```python
# Test parameter sensitivity
scales_range = [10, 15, 20, 25, 30]
dfa_sensitivity = []

for n_scales in scales_range:
    results = dfa(cleaned_data, n_scales=n_scales)
    dfa_sensitivity.append({
        'n_scales': n_scales,
        'alpha': results['alpha'],
        'hurst': results['hurst']
    })

sensitivity_df = pd.DataFrame(dfa_sensitivity)
print("Parameter Sensitivity Analysis:")
print(sensitivity_df)
```

#### Subset Analysis

```python
# Test robustness with different data subsets
n = len(cleaned_data)
subset_sizes = [n//2, 3*n//4, n]
subset_results = []

for size in subset_sizes:
    subset_data = cleaned_data[:size]
    results = dfa(subset_data)
    subset_results.append({
        'subset_size': size,
        'alpha': results['alpha'],
        'hurst': results['hurst']
    })

subset_df = pd.DataFrame(subset_results)
print("Subset Analysis:")
print(subset_df)
```

## Results Interpretation

### 6.1 Long-Range Dependence Assessment

#### Hurst Exponent Interpretation

```python
def interpret_hurst(hurst_value):
    """Interpret Hurst exponent values."""
    if hurst_value < 0.5:
        return "Anti-persistent (negative correlations)"
    elif hurst_value == 0.5:
        return "No long-range dependence (white noise)"
    elif hurst_value < 0.7:
        return "Weak long-range dependence"
    elif hurst_value < 0.9:
        return "Moderate long-range dependence"
    else:
        return "Strong long-range dependence"

# Interpret results
for method, results in methods.items():
    if 'hurst' in results:
        interpretation = interpret_hurst(results['hurst'])
        print(f"{method}: H = {results['hurst']:.4f} - {interpretation}")
```

#### Statistical Significance

```python
def assess_significance(results, threshold=0.05):
    """Assess statistical significance of results."""
    if 'p_value' in results:
        return results['p_value'] < threshold
    elif 'r_squared' in results:
        return results['r_squared'] > 0.8
    else:
        return "Unknown"

# Assess significance
for method, results in methods.items():
    significant = assess_significance(results)
    print(f"{method}: {'Significant' if significant else 'Not Significant'}")
```

### 6.2 Practical Implications

#### Forecasting Implications

```python
def forecasting_implications(hurst_value):
    """Assess implications for forecasting."""
    if hurst_value > 0.5:
        return "Long memory suggests improved long-term forecasting"
    else:
        return "Short memory limits long-term forecasting accuracy"

# Assess forecasting implications
for method, results in methods.items():
    if 'hurst' in results:
        implications = forecasting_implications(results['hurst'])
        print(f"{method}: {implications}")
```

#### Risk Management Implications

```python
def risk_implications(hurst_value):
    """Assess risk management implications."""
    if hurst_value > 0.7:
        return "High persistence suggests clustering of extreme events"
    elif hurst_value < 0.3:
        return "Anti-persistence suggests mean reversion"
    else:
        return "Moderate dependence, standard risk models may apply"

# Assess risk implications
for method, results in methods.items():
    if 'hurst' in results:
        risk_implications = risk_implications(results['hurst'])
        print(f"{method}: {risk_implications}")
```

## Reporting Standards

### 7.1 Report Structure

#### Executive Summary

- Research question and objectives
- Key findings and conclusions
- Practical implications
- Recommendations

#### Methodology

- Data description and preprocessing
- Analysis methods used
- Parameter settings
- Quality control measures

#### Results

- Summary statistics
- Method comparison table
- Statistical significance assessment
- Robustness analysis results

#### Discussion

- Interpretation of results
- Comparison with literature
- Limitations and assumptions
- Future research directions

### 7.2 Results Tables

#### Standard Results Table

```python
from src.visualisation.results_visualisation import create_summary_table

# Create summary table
summary_table = create_summary_table(
    dfa_results=dfa_results,
    rs_results=rs_results,
    wavelet_results=wavelet_results,
    spectral_results=whittle_results,
    arfima_results=arfima_results
)

# Save table
summary_table.to_csv('results/tables/analysis_summary.csv', index=False)
```

#### Method Comparison Table

```python
# Create method comparison table
comparison_table = pd.DataFrame({
    'Method': list(methods.keys()),
    'Hurst Exponent': [methods[m].get('hurst', 'N/A') for m in methods.keys()],
    'Alpha': [methods[m].get('alpha', methods[m].get('alpha_rs', 'N/A')) for m in methods.keys()],
    'R²': [methods[m].get('r_squared', 'N/A') for m in methods.keys()],
    'Significance': [assess_significance(methods[m]) for m in methods.keys()]
})

comparison_table.to_csv('results/tables/method_comparison.csv', index=False)
```

### 7.3 Visualization Standards

#### Plot Requirements

- **Resolution**: Minimum 300 DPI
- **Format**: PNG for web, PDF for publication
- **Size**: 12x8 inches for single plots, 15x10 for multi-panel
- **Labels**: Clear axis labels and titles
- **Legend**: Descriptive legend for multiple series
- **Grid**: Subtle grid for readability

#### Color Scheme

- **Primary**: Blue (#1f77b4)
- **Secondary**: Orange (#ff7f0e)
- **Tertiary**: Green (#2ca02c)
- **Quaternary**: Red (#d62728)
- **Accessibility**: Color-blind friendly palette

## Troubleshooting

### 8.1 Common Issues

#### Insufficient Data

**Problem**: Error "Insufficient data for analysis"
**Solution**: 
- Increase minimum data length to 100+ observations
- Use shorter scale ranges
- Consider data augmentation techniques

#### Convergence Issues

**Problem**: Optimization algorithms fail to converge
**Solution**:
- Check data quality and preprocessing
- Adjust optimization parameters
- Try different initial values
- Use robust optimization methods

#### Memory Issues

**Problem**: Out of memory errors
**Solution**:
- Reduce scale ranges
- Use chunked processing
- Optimize data types
- Increase system memory

### 8.2 Quality Control

#### Data Quality Checks

```python
def quality_checklist(data):
    """Run quality control checklist."""
    checks = {
        'Sample size': len(data) >= 100,
        'No missing values': not np.any(np.isnan(data)),
        'No infinite values': not np.any(np.isinf(data)),
        'Sufficient variation': np.std(data) > 0,
        'Reasonable range': np.max(data) - np.min(data) > 0
    }
    
    failed_checks = [check for check, passed in checks.items() if not passed]
    
    if failed_checks:
        print("Quality checks failed:")
        for check in failed_checks:
            print(f"  - {check}")
        return False
    else:
        print("All quality checks passed")
        return True

# Run quality checks
quality_checklist(cleaned_data)
```

#### Method Validation

```python
def validate_methods(methods_results):
    """Validate method results."""
    validations = {}
    
    for method, results in methods_results.items():
        validation = {
            'has_hurst': 'hurst' in results,
            'hurst_range': 0 <= results.get('hurst', 0) <= 1,
            'has_alpha': 'alpha' in results or 'alpha_rs' in results,
            'has_r_squared': 'r_squared' in results,
            'r_squared_range': 0 <= results.get('r_squared', 0) <= 1
        }
        
        validations[method] = validation
    
    return validations

# Validate results
validations = validate_methods(methods)
for method, validation in validations.items():
    print(f"{method}:")
    for check, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
```

### 8.3 Performance Optimization

#### Parallel Processing

```python
from src.config_loader import get_performance_config

# Get performance settings
perf_config = get_performance_config()
if perf_config.get('parallel', {}).get('enabled', False):
    max_workers = perf_config['parallel']['max_workers']
    print(f"Using parallel processing with {max_workers} workers")
```

#### Memory Management

```python
# Monitor memory usage
import psutil
import os

def monitor_memory():
    """Monitor memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

# Monitor before and after analysis
monitor_memory()
# ... run analysis ...
monitor_memory()
```

---

*This protocol provides a standardized approach to long-range dependence analysis. Follow these steps to ensure reproducible and reliable results. For specific implementation details, refer to the API documentation.*

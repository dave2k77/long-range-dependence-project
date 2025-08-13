# Synthetic Data Generation Tutorial

## 🎯 Overview

This tutorial covers the comprehensive synthetic data generation capabilities of our framework. You'll learn how to generate pure signals (ARFIMA, fBm, fGn) and apply various contaminations for robust method testing.

## 📊 Pure Signal Generators

### ARFIMA Processes

ARFIMA (Autoregressive Fractionally Integrated Moving Average) processes are characterized by the fractional differencing parameter `d`.

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

from data_processing.synthetic_generator import SyntheticDataGenerator

# Initialize generator
generator = SyntheticDataGenerator(random_state=42)

# Generate ARFIMA with different d values using the new convenience method
arfima_weak = generator.generate_arfima(n=1000, d=0.1)  # Weak LRD
arfima_medium = generator.generate_arfima(n=1000, d=0.3)  # Medium LRD
arfima_strong = generator.generate_arfima(n=1000, d=0.4)  # Strong LRD

print(f"Weak LRD (d=0.1): {len(arfima_weak)} points")
print(f"Medium LRD (d=0.3): {len(arfima_medium)} points")
print(f"Strong LRD (d=0.4): {len(arfima_strong)} points")
```

**Parameter Guide:**
- `n`: Number of data points
- `d`: Fractional differencing parameter (0 < d < 0.5)
- `ar_params`: Optional AR parameters (list of floats)
- `ma_params`: Optional MA parameters (list of floats)
- `sigma`: Noise standard deviation (default: 1.0)
- `random_state`: For reproducibility

### Fractional Brownian Motion (fBm)

fBm is a generalization of Brownian motion with Hurst exponent H.

```python
# Generate fBm with different Hurst exponents using the new convenience method
fbm_anti = generator.generate_fbm(n=1000, hurst=0.3)  # Anti-persistent
fbm_random = generator.generate_fbm(n=1000, hurst=0.5)  # Random walk
fbm_persistent = generator.generate_fbm(n=1000, hurst=0.7)  # Persistent

print(f"Anti-persistent fBm (H=0.3): {len(fbm_anti)} points")
print(f"Random walk fBm (H=0.5): {len(fbm_random)} points")
print(f"Persistent fBm (H=0.7): {len(fbm_persistent)} points")
```

**Parameter Guide:**
- `n`: Number of data points
- `hurst`: Hurst exponent (0 < H < 1)
- `random_state`: For reproducibility

### Fractional Gaussian Noise (fGn)

fGn represents the increments of fBm.

```python
# Generate fGn with different Hurst exponents using the new convenience method
fgn_anti = generator.generate_fgn(n=1000, hurst=0.3)
fgn_random = generator.generate_fgn(n=1000, hurst=0.5)
fgn_persistent = generator.generate_fgn(n=1000, hurst=0.7)

print(f"Anti-persistent fGn (H=0.3): {len(fgn_anti)} points")
print(f"Random fGn (H=0.5): {len(fgn_random)} points")
print(f"Persistent fGn (H=0.7): {len(fgn_persistent)} points")
```

## 🎭 Data Contaminators

### Polynomial Trends

Add polynomial trends to simulate real-world non-stationarities.

```python
from data_processing.synthetic_generator import DataContaminator

# Initialize contaminator
contaminator = DataContaminator(random_state=42)

# Add different polynomial trends
linear_trend = contaminator.add_polynomial_trend(arfima_medium, degree=1, amplitude=0.1)
quadratic_trend = contaminator.add_polynomial_trend(arfima_medium, degree=2, amplitude=0.05)
cubic_trend = contaminator.add_polynomial_trend(arfima_medium, degree=3, amplitude=0.02)

print(f"Original signal variance: {np.var(arfima_medium):.4f}")
print(f"Linear trend variance: {np.var(linear_trend):.4f}")
print(f"Quadratic trend variance: {np.var(quadratic_trend):.4f}")
```

**Parameter Guide:**
- `signal`: Input time series
- `degree`: Polynomial degree (1=linear, 2=quadratic, etc.)
- `amplitude`: Trend strength relative to signal

### Periodicity

Add periodic components to simulate seasonal patterns.

```python
# Add periodicity (note: frequency is a positional argument, not keyword)
periodic_signal = contaminator.add_periodicity(arfima_medium, 50, amplitude=0.2)
seasonal_signal = contaminator.add_periodicity(arfima_medium, 100, amplitude=0.15)

print(f"Periodic signal variance: {np.var(periodic_signal):.4f}")
print(f"Seasonal signal variance: {np.var(seasonal_signal):.4f}")
```

**Parameter Guide:**
- `signal`: Input time series
- `frequency`: Period length (number of points)
- `amplitude`: Periodic component strength

### Outliers

Add outliers to test robustness of analysis methods.

```python
# Add different types of outliers
outlier_signal = contaminator.add_outliers(arfima_medium, fraction=0.02, magnitude=4.0)
spike_signal = contaminator.add_outliers(arfima_medium, fraction=0.01, magnitude=6.0)

print(f"Outlier signal variance: {np.var(outlier_signal):.4f}")
print(f"Spike signal variance: {np.var(spike_signal):.4f}")
```

**Parameter Guide:**
- `signal`: Input time series
- `fraction`: Proportion of points to convert to outliers
- `magnitude`: Outlier strength in standard deviations

### Heavy Tails

Add heavy-tailed noise for non-Gaussian processes.

```python
# Add heavy-tailed noise
heavy_tail_signal = contaminator.add_heavy_tails(arfima_medium, df=2.0, fraction=0.15)
cauchy_signal = contaminator.add_heavy_tails(arfima_medium, df=1.0, fraction=0.1)

print(f"Heavy tail signal variance: {np.var(heavy_tail_signal):.4f}")
print(f"Cauchy signal variance: {np.var(cauchy_signal):.4f}")
```

**Parameter Guide:**
- `signal`: Input time series
- `df`: Degrees of freedom for t-distribution (lower = heavier tails)
- `fraction`: Proportion of points to replace with heavy-tailed noise

## 🔧 Advanced Generation

### Comprehensive Dataset Generation

Generate a complete set of synthetic datasets for comprehensive testing.

```python
# Generate comprehensive dataset
comprehensive_dataset = generator.generate_comprehensive_dataset(
    n=1000,
    save=True
)

print("Generated datasets:")
print(f"Clean signals: {len(comprehensive_dataset['clean_signals'])}")
print(f"Contaminated signals: {len(comprehensive_dataset['contaminated_signals'])}")
print(f"Irregular signals: {len(comprehensive_dataset['irregular_signals'])}")
```

**Note**: The `generate_comprehensive_dataset` method automatically saves data to the default data directory. If you need to specify a custom data root, you can initialize the `SyntheticDataGenerator` with a custom `data_root` parameter:

```python
# Initialize with custom data root
generator = SyntheticDataGenerator(data_root="custom_data", random_state=42)

# Generate comprehensive dataset
comprehensive_dataset = generator.generate_comprehensive_dataset(n=1000, save=True)
```

### Custom Signal Generation

For more control, use the underlying pure generator directly.

```python
# Access the pure generator for advanced usage
pure_generator = generator.pure_generator

# Generate ARFIMA with custom parameters
custom_arfima = pure_generator.generate_arfima(
    n=1000, 
    d=0.25, 
    ar_params=[0.3, -0.1], 
    ma_params=[0.2], 
    sigma=0.8
)

print(f"Custom ARFIMA: {len(custom_arfima)} points")
print(f"AR parameters: [0.3, -0.1]")
print(f"MA parameters: [0.2]")
```

## 📊 Data Quality and Validation

### Signal Properties

Check the statistical properties of generated signals.

```python
import numpy as np

def analyze_signal_properties(signal, name):
    """Analyze basic properties of a generated signal."""
    print(f"\n{name} Properties:")
    print(f"  Length: {len(signal)}")
    print(f"  Mean: {np.mean(signal):.4f}")
    print(f"  Std: {np.std(signal):.4f}")
    print(f"  Min: {np.min(signal):.4f}")
    print(f"  Max: {np.max(signal):.4f}")
    print(f"  Variance: {np.var(signal):.4f}")

# Analyze different signal types
signals = {
    "ARFIMA (d=0.3)": arfima_medium,
    "fBm (H=0.7)": fbm_persistent,
    "fGn (H=0.6)": fgn_persistent,
    "Contaminated": contaminated_signal
}

for name, signal in signals.items():
    analyze_signal_properties(signal, name)
```

### Long-Range Dependence Validation

Verify that generated signals exhibit the expected long-range dependence.

```python
from analysis.dfa_analysis import dfa
from analysis.rs_analysis import rs_analysis

def validate_lrd(signal, name):
    """Validate long-range dependence properties."""
    print(f"\n{name} LRD Validation:")
    
    try:
        # DFA analysis
        scales, flucts, dfa_summary = dfa(signal, order=1)
        print(f"  DFA Hurst: {dfa_summary.hurst:.3f}")
        
        # R/S analysis
        scales_rs, rs_values, rs_summary = rs_analysis(signal)
        print(f"  R/S Hurst: {rs_summary.hurst:.3f}")
        
        # Check consistency
        hurst_diff = abs(dfa_summary.hurst - rs_summary.hurst)
        if hurst_diff < 0.1:
            print(f"  ✓ Hurst estimates consistent (diff: {hurst_diff:.3f})")
        else:
            print(f"  ⚠ Hurst estimates differ (diff: {hurst_diff:.3f})")
            
    except Exception as e:
        print(f"  ✗ Analysis failed: {e}")

# Validate all signals
for name, signal in signals.items():
    validate_lrd(signal, name)
```

## 💾 Data Storage and Management

### Saving Generated Data

Save generated datasets for later use.

```python
# Save individual signals
np.save("data/raw/arfima_medium.npy", arfima_medium)
np.save("data/raw/fbm_persistent.npy", fbm_persistent)

# Save comprehensive dataset
import pickle
with open("data/raw/comprehensive_dataset.pkl", "wb") as f:
    pickle.dump(comprehensive_dataset, f)

print("Data saved successfully!")
```

### Loading Saved Data

```python
# Load individual signals
loaded_arfima = np.load("data/raw/arfima_medium.npy")
loaded_fbm = np.load("data/raw/fbm_persistent.npy")

# Load comprehensive dataset
with open("data/raw/comprehensive_dataset.pkl", "rb") as f:
    loaded_comprehensive = pickle.load(f)

print(f"Loaded ARFIMA: {len(loaded_arfima)} points")
print(f"Loaded comprehensive dataset: {len(loaded_comprehensive['clean_signals'])} clean signals")
```

## 🎯 Best Practices

### 1. Reproducibility
- Always set `random_state` for reproducible results
- Document all generation parameters
- Use version control for generation scripts

### 2. Data Quality
- Generate sufficient data points (recommend ≥500)
- Validate statistical properties
- Test with different contamination levels

### 3. Performance
- Use batch generation for large datasets
- Save intermediate results
- Monitor memory usage for very long series

### 4. Validation
- Always validate generated signals with analysis methods
- Compare with theoretical expectations
- Test robustness with contaminated data

## 🔍 Troubleshooting

### Common Issues

**Issue**: Generated signals don't show expected LRD
**Solution**: Check parameter ranges and ensure sufficient data length

**Issue**: Memory errors with large datasets
**Solution**: Generate data in smaller batches or use streaming approaches

**Issue**: Inconsistent results between runs
**Solution**: Ensure random_state is set and check for global state changes

**Issue**: Contamination not visible
**Solution**: Increase amplitude parameters and check signal-to-noise ratios

**Issue**: `TypeError: ArmaProcess.generate_sample() got an unexpected keyword argument 'random_state'`
**Solution**: This issue has been fixed in the latest version. The method now properly handles reproducibility by setting the numpy random seed before calling `generate_sample()`. If you encounter this error, please update to the latest version.

**Issue**: `TypeError: generate_comprehensive_dataset() got an unexpected keyword argument 'data_root'`
**Solution**: The `generate_comprehensive_dataset()` method doesn't accept a `data_root` parameter. Use the constructor to set the data root: `SyntheticDataGenerator(data_root="custom_path", random_state=42)`.

### Recent Fixes Applied

The following issues have been resolved in recent updates:

1. **ArmaProcess Parameter Error**: Fixed `random_state` parameter issue in ARFIMA generation
2. **Method Parameter Validation**: Corrected parameter lists for all generation methods
3. **Import Path Updates**: Updated all import statements to match current codebase structure
4. **Tutorial Accuracy**: All code examples now work with the current implementation

### Getting Help

If you encounter issues not covered here:

1. **Check the project documentation**
2. **Review the API reference**
3. **Run the demo scripts**: `python scripts/demo_synthetic_data.py`
4. **Create an issue on GitHub** with:
   - Error message and traceback
   - Code that caused the error
   - Your system information (Python version, OS)
   - Expected vs. actual behavior

## 📚 Next Steps

- **Tutorial 3**: Learn advanced analysis methods
- **Tutorial 4**: Understand statistical validation techniques
- **Tutorial 5**: Create comprehensive visualizations
- **Tutorial 6**: Submit your own models and datasets

## 🆘 Getting Help

For additional support:
1. Check the project documentation
2. Review the API reference
3. Run the demo scripts
4. Create an issue on GitHub

---

**Congratulations!** You've mastered synthetic data generation for long-range dependence analysis. 🎉

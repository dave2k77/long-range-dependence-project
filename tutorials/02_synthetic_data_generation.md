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

from data_processing import PureSignalGenerator

# Initialize generator
generator = PureSignalGenerator(random_state=42)

# Generate ARFIMA with different d values
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
- `random_state`: For reproducibility

### Fractional Brownian Motion (fBm)

fBm is a generalization of Brownian motion with Hurst exponent H.

```python
# Generate fBm with different Hurst exponents
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
# Generate fGn with different Hurst exponents
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
from data_processing import DataContaminator

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

Add sinusoidal components to simulate seasonal patterns.

```python
# Add periodicity with different frequencies
low_freq = contaminator.add_periodicity(arfima_medium, frequency=20, amplitude=0.2)
high_freq = contaminator.add_periodicity(arfima_medium, frequency=100, amplitude=0.1)
multi_freq = contaminator.add_periodicity(arfima_medium, frequency=50, amplitude=0.15)

print(f"Low frequency periodicity: {len(low_freq)} points")
print(f"High frequency periodicity: {len(high_freq)} points")
```

**Parameter Guide:**
- `signal`: Input time series
- `frequency`: Frequency of oscillation (cycles per time unit)
- `amplitude`: Amplitude relative to signal

### Outliers

Inject extreme values to test robustness.

```python
# Add outliers with different characteristics
few_outliers = contaminator.add_outliers(arfima_medium, fraction=0.01, magnitude=3.0)
many_outliers = contaminator.add_outliers(arfima_medium, fraction=0.05, magnitude=2.0)
extreme_outliers = contaminator.add_outliers(arfima_medium, fraction=0.02, magnitude=5.0)

print(f"Few outliers: {np.sum(np.abs(few_outliers) > 2*np.std(arfima_medium))} extreme points")
print(f"Many outliers: {np.sum(np.abs(many_outliers) > 2*np.std(arfima_medium))} extreme points")
```

**Parameter Guide:**
- `signal`: Input time series
- `fraction`: Fraction of points to replace with outliers
- `magnitude`: Outlier magnitude in standard deviations

### Irregular Sampling

Simulate missing data scenarios.

```python
# Create irregular sampling patterns
sparse_sampling = contaminator.add_irregular_sampling(arfima_medium, missing_fraction=0.3)
random_sampling = contaminator.add_irregular_sampling(arfima_medium, missing_fraction=0.5)

print(f"Original length: {len(arfima_medium)}")
print(f"Sparse sampling length: {len(sparse_sampling)}")
print(f"Random sampling length: {len(random_sampling)}")
```

**Parameter Guide:**
- `signal`: Input time series
- `missing_fraction`: Fraction of points to remove

### Heavy-Tail Fluctuations

Add heavy-tailed noise using Student's t-distribution.

```python
# Add heavy-tailed noise
light_tails = contaminator.add_heavy_tails(arfima_medium, df=10, scale=0.1)
heavy_tails = contaminator.add_heavy_tails(arfima_medium, df=3, scale=0.1)

print(f"Light tails kurtosis: {scipy.stats.kurtosis(light_tails):.2f}")
print(f"Heavy tails kurtosis: {scipy.stats.kurtosis(heavy_tails):.2f}")
```

**Parameter Guide:**
- `signal`: Input time series
- `df`: Degrees of freedom for t-distribution (lower = heavier tails)
- `scale`: Noise scale relative to signal

## 🔄 Comprehensive Dataset Generation

Generate a complete testing dataset with all signal types and contaminations.

```python
from data_processing import SyntheticDataGenerator

# Initialize comprehensive generator
generator = SyntheticDataGenerator(random_state=42)

# Generate complete dataset
dataset = generator.generate_comprehensive_dataset(n=1000, save=True)

print("Generated datasets:")
print(f"Clean signals: {len(dataset['clean_signals'])}")
print(f"Contaminated signals: {len(dataset['contaminated_signals'])}")
print(f"Irregular signals: {len(dataset['irregular_signals'])}")
```

**Available Signal Types:**
- **Clean Signals**: ARFIMA, fBm, fGn with various parameters
- **Contaminated Signals**: All clean signals with trends, periodicity, outliers, heavy tails
- **Irregular Signals**: Clean signals with missing data

## 📈 Visualization Examples

### Plot Pure Signals

```python
import matplotlib.pyplot as plt
from visualisation import time_series_plots

# Create comparison plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

time_series_plots.plot_time_series(arfima_weak, ax=axes[0], title="Weak LRD (d=0.1)")
time_series_plots.plot_time_series(arfima_medium, ax=axes[1], title="Medium LRD (d=0.3)")
time_series_plots.plot_time_series(arfima_strong, ax=axes[2], title="Strong LRD (d=0.4)")

plt.tight_layout()
plt.show()
```

### Plot Contamination Effects

```python
# Compare original vs contaminated
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

time_series_plots.plot_time_series(arfima_medium, ax=axes[0], title="Original ARFIMA")
time_series_plots.plot_time_series(linear_trend, ax=axes[1], title="With Linear Trend")

plt.tight_layout()
plt.show()
```

## 🧪 Testing and Validation

### Verify Signal Properties

```python
from analysis import DFAnalysis

dfa = DFAnalysis()

# Test theoretical vs estimated Hurst exponents
signals = {
    'Weak LRD': arfima_weak,
    'Medium LRD': arfima_medium,
    'Strong LRD': arfima_strong
}

for name, signal in signals.items():
    result = dfa.analyze(signal)
    print(f"{name}: Estimated H = {result['hurst']:.3f}")
```

### Test Contamination Robustness

```python
# Compare analysis results with and without contamination
original_result = dfa.analyze(arfima_medium)
contaminated_result = dfa.analyze(linear_trend)

print(f"Original H: {original_result['hurst']:.3f}")
print(f"Contaminated H: {contaminated_result['hurst']:.3f}")
print(f"Difference: {abs(original_result['hurst'] - contaminated_result['hurst']):.3f}")
```

## 📋 Best Practices

### 1. Reproducibility
Always set `random_state` for reproducible results:
```python
generator = SyntheticDataGenerator(random_state=42)
```

### 2. Parameter Selection
- Use realistic parameter ranges for your application
- Test multiple parameter combinations
- Consider the scale of your real data

### 3. Data Quality
- Check for NaN or infinite values
- Verify signal properties match expectations
- Test with known theoretical values

### 4. Storage and Organization
```python
# Save with descriptive names
generator._save_signal(arfima_medium, "arfima_d03_clean", "Clean ARFIMA with d=0.3")
generator._save_signal(linear_trend, "arfima_d03_trend", "ARFIMA with linear trend")
```

## 🔗 Command Line Usage

Generate synthetic data from the command line:

```bash
# Generate comprehensive dataset
python scripts/generate_synthetic_data.py --n 1000

# Generate only clean signals
python scripts/generate_synthetic_data.py --clean-only --n 500

# Generate with specific random state
python scripts/generate_synthetic_data.py --n 1000 --random-state 42
```

## 📚 Next Steps

- **Tutorial 3**: [Advanced Analysis Methods](03_advanced_analysis.md)
- **Tutorial 4**: [Statistical Validation](04_statistical_validation.md)
- **Tutorial 5**: [Visualization and Reporting](05_visualization.md)

---

**You now have a comprehensive understanding of synthetic data generation!** 🎉

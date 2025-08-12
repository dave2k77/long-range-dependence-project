# Advanced Analysis Methods Tutorial

## 🎯 Overview

This tutorial covers the advanced analysis methods available in our framework: DFA, R/S Analysis, Higuchi Method, MFDFA, Spectral Analysis, and Wavelet Analysis. You'll learn how to use each method and interpret their results.

## 📊 Detrended Fluctuation Analysis (DFA)

DFA is a robust method for estimating the Hurst exponent that is resistant to non-stationarities.

### Basic DFA Analysis

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

from analysis import DFAnalysis
from data_processing import PureSignalGenerator

# Generate test data
generator = PureSignalGenerator(random_state=42)
signal = generator.generate_arfima(n=1000, d=0.3)

# Initialize DFA analyzer
dfa = DFAnalysis()

# Run analysis
result = dfa.analyze(signal)

print(f"Hurst exponent: {result['hurst']:.3f}")
print(f"R-squared: {result['r_squared']:.3f}")
print(f"Scales used: {len(result['scales'])}")
```

### DFA with Custom Parameters

```python
# Customize DFA parameters
dfa_custom = DFAnalysis(
    min_scale=10,
    max_scale=100,
    num_scales=20,
    polynomial_order=2
)

result = dfa_custom.analyze(signal)
print(f"Custom DFA Hurst: {result['hurst']:.3f}")
```

**Parameter Guide:**
- `min_scale`: Minimum scale for analysis
- `max_scale`: Maximum scale for analysis
- `num_scales`: Number of scales to use
- `polynomial_order`: Order of polynomial for detrending

### DFA Visualization

```python
from visualisation import fractal_plots

# Plot DFA results
fractal_plots.plot_dfa_results(result, title="DFA Analysis Results")

# Plot fluctuation function
fractal_plots.plot_fluctuation_function(result['scales'], result['fluctuations'], 
                                       result['hurst'], title="DFA Fluctuation Function")
```

## 📈 R/S Analysis (Rescaled Range)

R/S analysis is the classical method for estimating the Hurst exponent.

### Basic R/S Analysis

```python
from analysis import RSAnalysis

# Initialize R/S analyzer
rs = RSAnalysis()

# Run analysis
result = rs.analyze(signal)

print(f"Hurst exponent: {result['hurst']:.3f}")
print(f"R-squared: {result['r_squared']:.3f}")
print(f"Scales used: {len(result['scales'])}")
```

### R/S with Custom Parameters

```python
# Customize R/S parameters
rs_custom = RSAnalysis(
    min_scale=10,
    max_scale=200,
    num_scales=25
)

result = rs_custom.analyze(signal)
print(f"Custom R/S Hurst: {result['hurst']:.3f}")
```

### R/S Visualization

```python
# Plot R/S results
fractal_plots.plot_rs_results(result, title="R/S Analysis Results")

# Plot rescaled range
fractal_plots.plot_rescaled_range(result['scales'], result['rs_values'], 
                                 result['hurst'], title="R/S Rescaled Range")
```

## 🔍 Higuchi Method

The Higuchi method estimates the fractal dimension of time series.

### Basic Higuchi Analysis

```python
from analysis import HiguchiAnalysis

# Initialize Higuchi analyzer
higuchi = HiguchiAnalysis()

# Run analysis
result = higuchi.analyze(signal)

print(f"Fractal dimension: {result['fractal_dimension']:.3f}")
print(f"R-squared: {result['r_squared']:.3f}")
print(f"K values used: {len(result['k_values'])}")
```

### Higuchi with Custom Parameters

```python
# Customize Higuchi parameters
higuchi_custom = HiguchiAnalysis(
    k_max=20,
    k_min=2
)

result = higuchi_custom.analyze(signal)
print(f"Custom Higuchi fractal dimension: {result['fractal_dimension']:.3f}")
```

### Higuchi Visualization

```python
from visualisation import higuchi_plots

# Plot Higuchi results
higuchi_plots.plot_higuchi_results(result, title="Higuchi Analysis Results")
```

## 🌊 Multifractal Detrended Fluctuation Analysis (MFDFA)

MFDFA extends DFA to analyze multifractal properties.

### Basic MFDFA Analysis

```python
from analysis import MFDFAAnalysis

# Initialize MFDFA analyzer
mfdfa = MFDFAAnalysis()

# Run analysis
result = mfdfa.analyze(signal)

print(f"Generalized Hurst exponent (q=2): {result['h_q'][10]:.3f}")
print(f"Multifractal spectrum width: {result['spectrum_width']:.3f}")
print(f"Singularity strength range: {result['alpha_range']:.3f}")
```

### MFDFA with Custom Parameters

```python
# Customize MFDFA parameters
mfdfa_custom = MFDFAAnalysis(
    q_values=np.linspace(-5, 5, 21),
    min_scale=10,
    max_scale=100,
    num_scales=20
)

result = mfdfa_custom.analyze(signal)
print(f"Custom MFDFA spectrum width: {result['spectrum_width']:.3f}")
```

### MFDFA Visualization

```python
from visualisation import fractal_plots

# Plot MFDFA results
fractal_plots.plot_mfdfa_results(result, title="MFDFA Analysis Results")

# Plot multifractal spectrum
fractal_plots.plot_multifractal_spectrum(result['alpha'], result['f_alpha'], 
                                        title="Multifractal Spectrum")
```

## 📡 Spectral Analysis

Spectral analysis methods for long-range dependence detection.

### Periodogram Method

```python
from analysis import SpectralAnalysis

# Initialize spectral analyzer
spectral = SpectralAnalysis()

# Run periodogram analysis
result = spectral.analyze_periodogram(signal)

print(f"Spectral exponent: {result['spectral_exponent']:.3f}")
print(f"Hurst exponent: {result['hurst']:.3f}")
print(f"R-squared: {result['r_squared']:.3f}")
```

### Whittle Estimation

```python
# Run Whittle estimation
result = spectral.analyze_whittle(signal)

print(f"Whittle Hurst exponent: {result['hurst']:.3f}")
print(f"Confidence interval: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
```

### Spectral Visualization

```python
from visualisation import fractal_plots

# Plot periodogram results
fractal_plots.plot_periodogram_results(result, title="Periodogram Analysis")

# Plot power spectrum
fractal_plots.plot_power_spectrum(result['frequencies'], result['power_spectrum'], 
                                 result['fitted_spectrum'], title="Power Spectrum")
```

## 🌊 Wavelet Analysis

Wavelet-based methods for long-range dependence analysis.

### Wavelet Leaders Method

```python
from analysis import WaveletAnalysis

# Initialize wavelet analyzer
wavelet = WaveletAnalysis()

# Run wavelet leaders analysis
result = wavelet.analyze_leaders(signal)

print(f"Wavelet Hurst exponent: {result['hurst']:.3f}")
print(f"R-squared: {result['r_squared']:.3f}")
print(f"Scales used: {len(result['scales'])}")
```

### Wavelet Whittle Method

```python
# Run wavelet Whittle analysis
result = wavelet.analyze_whittle(signal)

print(f"Wavelet Whittle Hurst: {result['hurst']:.3f}")
print(f"Confidence interval: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
```

### Wavelet Visualization

```python
from visualisation import fractal_plots

# Plot wavelet results
fractal_plots.plot_wavelet_results(result, title="Wavelet Analysis Results")

# Plot wavelet coefficients
fractal_plots.plot_wavelet_coefficients(result['scales'], result['coefficients'], 
                                      title="Wavelet Coefficients")
```

## 🔄 Method Comparison

### Compare Multiple Methods

```python
# Generate test signal
signal = generator.generate_arfima(n=1000, d=0.3)

# Run all methods
methods = {
    'DFA': DFAnalysis(),
    'R/S': RSAnalysis(),
    'Higuchi': HiguchiAnalysis(),
    'Spectral': SpectralAnalysis(),
    'Wavelet': WaveletAnalysis()
}

results = {}
for name, method in methods.items():
    if name == 'Higuchi':
        results[name] = method.analyze(signal)['fractal_dimension']
    else:
        results[name] = method.analyze(signal)['hurst']

# Display results
for name, result in results.items():
    print(f"{name}: {result:.3f}")
```

### Method Robustness Analysis

```python
# Test with contaminated data
from data_processing import DataContaminator

contaminator = DataContaminator(random_state=42)
contaminated_signal = contaminator.add_polynomial_trend(signal, degree=1, amplitude=0.1)

# Compare results
clean_results = {}
contaminated_results = {}

for name, method in methods.items():
    if name == 'Higuchi':
        clean_results[name] = method.analyze(signal)['fractal_dimension']
        contaminated_results[name] = method.analyze(contaminated_signal)['fractal_dimension']
    else:
        clean_results[name] = method.analyze(signal)['hurst']
        contaminated_results[name] = method.analyze(contaminated_signal)['hurst']

# Calculate robustness
for name in methods.keys():
    difference = abs(clean_results[name] - contaminated_results[name])
    print(f"{name} robustness: {difference:.3f}")
```

## 📊 Advanced Visualization

### Method Comparison Plot

```python
import matplotlib.pyplot as plt
import numpy as np

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# DFA plot
fractal_plots.plot_dfa_results(dfa.analyze(signal), ax=axes[0,0], title="DFA")

# R/S plot
fractal_plots.plot_rs_results(rs.analyze(signal), ax=axes[0,1], title="R/S")

# Spectral plot
fractal_plots.plot_periodogram_results(spectral.analyze_periodogram(signal), 
                                      ax=axes[1,0], title="Spectral")

# Wavelet plot
fractal_plots.plot_wavelet_results(wavelet.analyze_leaders(signal), 
                                  ax=axes[1,1], title="Wavelet")

plt.tight_layout()
plt.show()
```

### Parameter Sensitivity Analysis

```python
# Test parameter sensitivity
scales_range = [10, 20, 50, 100, 200]
hurst_values = []

for scale in scales_range:
    dfa_temp = DFAnalysis(max_scale=scale)
    result = dfa_temp.analyze(signal)
    hurst_values.append(result['hurst'])

plt.figure(figsize=(10, 6))
plt.plot(scales_range, hurst_values, 'o-')
plt.xlabel('Maximum Scale')
plt.ylabel('Hurst Exponent')
plt.title('DFA Parameter Sensitivity')
plt.grid(True)
plt.show()
```

## 🧪 Best Practices

### 1. Method Selection
- **DFA**: Best for non-stationary data with trends
- **R/S**: Good baseline, sensitive to non-stationarities
- **Higuchi**: Useful for fractal dimension estimation
- **MFDFA**: For multifractal analysis
- **Spectral**: Good for stationary data
- **Wavelet**: Robust to various data characteristics

### 2. Parameter Tuning
- Test multiple parameter ranges
- Use domain knowledge to guide selection
- Consider data length and characteristics

### 3. Quality Assessment
- Check R-squared values for goodness of fit
- Verify convergence of results
- Compare with theoretical expectations

### 4. Robustness Testing
- Test with contaminated data
- Use bootstrap methods for confidence intervals
- Cross-validate results

## 📚 Next Steps

- **Tutorial 4**: [Statistical Validation](04_statistical_validation.md)
- **Tutorial 5**: [Visualization and Reporting](05_visualization.md)

---

**You now have comprehensive knowledge of advanced analysis methods!** 🎉

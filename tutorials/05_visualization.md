# Visualization and Reporting Tutorial

## 🎯 Overview

This tutorial covers the comprehensive visualization capabilities of our framework. You'll learn how to create publication-ready figures, customize plots, and generate comprehensive reports for your long-range dependence analysis.

## 📊 Time Series Visualization

### Basic Time Series Plots

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

import matplotlib.pyplot as plt
import numpy as np
from visualisation import time_series_plots
from data_processing import PureSignalGenerator

# Generate test data
generator = PureSignalGenerator(random_state=42)
signal = generator.generate_arfima(n=1000, d=0.3)

# Basic time series plot
time_series_plots.plot_time_series(signal, title="ARFIMA Process (d=0.3)")
plt.show()
```

### Customized Time Series Plots

```python
# Create customized plot
fig, ax = plt.subplots(figsize=(12, 6))
time_series_plots.plot_time_series(
    signal,
    ax=ax,
    title="Customized Time Series Plot",
    xlabel="Time",
    ylabel="Value",
    color='blue',
    linewidth=1.5,
    alpha=0.8
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Multiple Time Series Comparison

```python
# Generate multiple signals
signals = {
    'Weak LRD (d=0.1)': generator.generate_arfima(n=1000, d=0.1),
    'Medium LRD (d=0.3)': generator.generate_arfima(n=1000, d=0.3),
    'Strong LRD (d=0.4)': generator.generate_arfima(n=1000, d=0.4)
}

# Create comparison plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
for i, (name, data) in enumerate(signals.items()):
    time_series_plots.plot_time_series(
        data,
        ax=axes[i],
        title=name,
        xlabel="Time" if i == 2 else "",
        ylabel="Value"
    )
plt.tight_layout()
plt.show()
```

## 🔍 Fractal Analysis Visualization

### DFA Results Visualization

```python
from analysis import DFAnalysis
from visualisation import fractal_plots

# Run DFA analysis
dfa = DFAnalysis()
result = dfa.analyze(signal)

# Plot DFA results
fractal_plots.plot_dfa_results(result, title="DFA Analysis Results")
plt.show()

# Plot fluctuation function
fractal_plots.plot_fluctuation_function(
    result['scales'],
    result['fluctuations'],
    result['hurst'],
    title="DFA Fluctuation Function"
)
plt.show()
```

### R/S Analysis Visualization

```python
from analysis import RSAnalysis

# Run R/S analysis
rs = RSAnalysis()
result = rs.analyze(signal)

# Plot R/S results
fractal_plots.plot_rs_results(result, title="R/S Analysis Results")
plt.show()

# Plot rescaled range
fractal_plots.plot_rescaled_range(
    result['scales'],
    result['rs_values'],
    result['hurst'],
    title="R/S Rescaled Range"
)
plt.show()
```

### Higuchi Method Visualization

```python
from analysis import HiguchiAnalysis
from visualisation import higuchi_plots

# Run Higuchi analysis
higuchi = HiguchiAnalysis()
result = higuchi.analyze(signal)

# Plot Higuchi results
higuchi_plots.plot_higuchi_results(result, title="Higuchi Analysis Results")
plt.show()
```

### MFDFA Visualization

```python
from analysis import MFDFAAnalysis

# Run MFDFA analysis
mfdfa = MFDFAAnalysis()
result = mfdfa.analyze(signal)

# Plot MFDFA results
fractal_plots.plot_mfdfa_results(result, title="MFDFA Analysis Results")
plt.show()

# Plot multifractal spectrum
fractal_plots.plot_multifractal_spectrum(
    result['alpha'],
    result['f_alpha'],
    title="Multifractal Spectrum"
)
plt.show()
```

## 📡 Spectral Analysis Visualization

### Periodogram Visualization

```python
from analysis import SpectralAnalysis

# Run spectral analysis
spectral = SpectralAnalysis()
result = spectral.analyze_periodogram(signal)

# Plot periodogram results
fractal_plots.plot_periodogram_results(result, title="Periodogram Analysis")
plt.show()

# Plot power spectrum
fractal_plots.plot_power_spectrum(
    result['frequencies'],
    result['power_spectrum'],
    result['fitted_spectrum'],
    title="Power Spectrum"
)
plt.show()
```

### Whittle Estimation Visualization

```python
# Run Whittle estimation
result = spectral.analyze_whittle(signal)

# Plot Whittle results
fractal_plots.plot_whittle_results(result, title="Whittle Estimation Results")
plt.show()
```

## 🌊 Wavelet Analysis Visualization

### Wavelet Leaders Visualization

```python
from analysis import WaveletAnalysis

# Run wavelet analysis
wavelet = WaveletAnalysis()
result = wavelet.analyze_leaders(signal)

# Plot wavelet results
fractal_plots.plot_wavelet_results(result, title="Wavelet Leaders Analysis")
plt.show()

# Plot wavelet coefficients
fractal_plots.plot_wavelet_coefficients(
    result['scales'],
    result['coefficients'],
    title="Wavelet Coefficients"
)
plt.show()
```

## 📊 Validation Visualization

### Bootstrap Analysis Visualization

```python
from statistical_validation import BootstrapValidation
from visualisation import validation_plots

# Run bootstrap analysis
bootstrap = BootstrapValidation(DFAnalysis())
result = bootstrap.analyze(signal, n_bootstrap=1000)

# Plot bootstrap results
validation_plots.plot_bootstrap_results(result, title="Bootstrap Analysis")
plt.show()

# Plot bootstrap distribution
validation_plots.plot_bootstrap_distribution(
    result['bootstrap_samples'],
    result['original_estimate'],
    title="Bootstrap Distribution"
)
plt.show()
```

### Cross-Validation Visualization

```python
from statistical_validation import CrossValidation

# Run cross-validation
cv = CrossValidation(DFAnalysis())
result = cv.analyze(signal, n_folds=5)

# Plot cross-validation results
validation_plots.plot_cross_validation_results(result, title="Cross-Validation Results")
plt.show()

# Plot fold estimates
validation_plots.plot_fold_estimates(
    result['fold_estimates'],
    result['mean_estimate'],
    title="Cross-Validation Fold Estimates"
)
plt.show()
```

### Monte Carlo Visualization

```python
from statistical_validation import MonteCarloValidation

# Run Monte Carlo analysis
mc = MonteCarloValidation(DFAnalysis())
result = mc.analyze(signal_length=1000, true_hurst=0.3, n_simulations=100)

# Plot Monte Carlo results
validation_plots.plot_monte_carlo_results(result, title="Monte Carlo Analysis")
plt.show()
```

### Hypothesis Testing Visualization

```python
from statistical_validation import HypothesisTesting

# Run hypothesis testing
ht = HypothesisTesting(DFAnalysis())
result = ht.test_lrd(signal, alpha=0.05)

# Plot hypothesis test results
validation_plots.plot_hypothesis_test_results(result, title="Hypothesis Test Results")
plt.show()

# Plot test statistic distribution
validation_plots.plot_test_statistic_distribution(
    result['null_distribution'],
    result['test_statistic'],
    title="Test Statistic Distribution"
)
plt.show()
```

## 🔄 Method Comparison Visualization

### Comprehensive Method Comparison

```python
# Run multiple methods
methods = {
    'DFA': DFAnalysis(),
    'R/S': RSAnalysis(),
    'Spectral': SpectralAnalysis(),
    'Wavelet': WaveletAnalysis()
}

results = {}
for name, method in methods.items():
    if name == 'Higuchi':
        results[name] = method.analyze(signal)['fractal_dimension']
    else:
        results[name] = method.analyze(signal)['hurst']

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# DFA plot
fractal_plots.plot_dfa_results(methods['DFA'].analyze(signal), 
                              ax=axes[0,0], title="DFA")

# R/S plot
fractal_plots.plot_rs_results(methods['R/S'].analyze(signal), 
                             ax=axes[0,1], title="R/S")

# Spectral plot
fractal_plots.plot_periodogram_results(methods['Spectral'].analyze_periodogram(signal), 
                                      ax=axes[1,0], title="Spectral")

# Wavelet plot
fractal_plots.plot_wavelet_results(methods['Wavelet'].analyze_leaders(signal), 
                                  ax=axes[1,1], title="Wavelet")

plt.tight_layout()
plt.show()
```

### Method Performance Dashboard

```python
# Create performance dashboard
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Hurst exponent comparison
method_names = list(results.keys())
hurst_values = list(results.values())

axes[0,0].bar(method_names, hurst_values, color=['blue', 'green', 'red', 'orange'])
axes[0,0].set_title('Hurst Exponent Comparison')
axes[0,0].set_ylabel('Hurst Exponent')
axes[0,0].tick_params(axis='x', rotation=45)

# Method robustness (example with contaminated data)
from data_processing import DataContaminator
contaminator = DataContaminator(random_state=42)
contaminated_signal = contaminator.add_polynomial_trend(signal, degree=1, amplitude=0.1)

contaminated_results = {}
for name, method in methods.items():
    if name == 'Higuchi':
        contaminated_results[name] = method.analyze(contaminated_signal)['fractal_dimension']
    else:
        contaminated_results[name] = method.analyze(contaminated_signal)['hurst']

robustness = [abs(results[name] - contaminated_results[name]) for name in method_names]
axes[0,1].bar(method_names, robustness, color=['blue', 'green', 'red', 'orange'])
axes[0,1].set_title('Method Robustness')
axes[0,1].set_ylabel('Absolute Difference')
axes[0,1].tick_params(axis='x', rotation=45)

# Computational time comparison (example)
import time
times = {}
for name, method in methods.items():
    start_time = time.time()
    method.analyze(signal)
    times[name] = time.time() - start_time

time_values = list(times.values())
axes[1,0].bar(method_names, time_values, color=['blue', 'green', 'red', 'orange'])
axes[1,0].set_title('Computational Time')
axes[1,0].set_ylabel('Time (seconds)')
axes[1,0].tick_params(axis='x', rotation=45)

# R-squared comparison (example)
r_squared_values = [0.95, 0.92, 0.88, 0.94]  # Example values
axes[1,1].bar(method_names, r_squared_values, color=['blue', 'green', 'red', 'orange'])
axes[1,1].set_title('Goodness of Fit (R²)')
axes[1,1].set_ylabel('R-squared')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

## 📈 Publication-Ready Figures

### Setting Up Publication Style

```python
# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})
```

### Creating Multi-Panel Figures

```python
# Create publication-ready multi-panel figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Panel A: Time series
time_series_plots.plot_time_series(signal, ax=axes[0,0], title="(A) Time Series")

# Panel B: DFA
fractal_plots.plot_dfa_results(dfa.analyze(signal), ax=axes[0,1], title="(B) DFA")

# Panel C: R/S
fractal_plots.plot_rs_results(rs.analyze(signal), ax=axes[0,2], title="(C) R/S")

# Panel D: Spectral
fractal_plots.plot_periodogram_results(spectral.analyze_periodogram(signal), 
                                      ax=axes[1,0], title="(D) Spectral")

# Panel E: Wavelet
fractal_plots.plot_wavelet_results(wavelet.analyze_leaders(signal), 
                                  ax=axes[1,1], title="(E) Wavelet")

# Panel F: Bootstrap
validation_plots.plot_bootstrap_results(bootstrap.analyze(signal), 
                                       ax=axes[1,2], title="(F) Bootstrap")

plt.tight_layout()
plt.savefig('results/figures/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Custom Color Schemes

```python
# Define custom color scheme
colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'quaternary': '#d62728',
    'quinary': '#9467bd'
}

# Apply custom colors
fig, ax = plt.subplots(figsize=(12, 6))
time_series_plots.plot_time_series(
    signal,
    ax=ax,
    title="Custom Color Scheme",
    color=colors['primary'],
    linewidth=2
)
ax.grid(True, alpha=0.3, color=colors['secondary'])
plt.tight_layout()
plt.show()
```

## 📊 Statistical Summary Plots

### Results Summary Table

```python
import pandas as pd

# Create results summary
summary_data = {
    'Method': ['DFA', 'R/S', 'Spectral', 'Wavelet'],
    'Hurst Exponent': [results['DFA'], results['R/S'], results['Spectral'], results['Wavelet']],
    'R-squared': [0.95, 0.92, 0.88, 0.94],
    'Robustness': robustness,
    'Time (s)': time_values
}

summary_df = pd.DataFrame(summary_data)
print(summary_df)

# Create summary plot
fig, ax = plt.subplots(figsize=(10, 6))
summary_df.plot(x='Method', y=['Hurst Exponent', 'R-squared'], 
                kind='bar', ax=ax, color=[colors['primary'], colors['secondary']])
ax.set_title('Method Comparison Summary')
ax.set_ylabel('Value')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
```

### Parameter Sensitivity Analysis

```python
# Parameter sensitivity analysis
scales_range = [10, 20, 50, 100, 200]
hurst_values = []

for scale in scales_range:
    dfa_temp = DFAnalysis(max_scale=scale)
    result = dfa_temp.analyze(signal)
    hurst_values.append(result['hurst'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(scales_range, hurst_values, 'o-', color=colors['primary'], linewidth=2, markersize=8)
ax.set_xlabel('Maximum Scale')
ax.set_ylabel('Hurst Exponent')
ax.set_title('DFA Parameter Sensitivity')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## 📋 Report Generation

### Automated Report Creation

```python
from visualisation import results_visualisation

# Generate comprehensive report
report = results_visualisation.generate_analysis_report(
    signal=signal,
    results=results,
    validation_results=bootstrap.analyze(signal),
    save_path='results/reports/analysis_report.html'
)

print("Report generated successfully!")
```

### Custom Report Templates

```python
# Create custom report template
template_config = {
    'title': 'Long-Range Dependence Analysis Report',
    'author': 'Your Name',
    'date': '2025',
    'sections': [
        'Time Series Overview',
        'Analysis Results',
        'Method Comparison',
        'Validation Results',
        'Conclusions'
    ]
}

# Generate custom report
results_visualisation.generate_custom_report(
    signal=signal,
    results=results,
    template_config=template_config,
    save_path='results/reports/custom_report.html'
)
```

## 🎨 Advanced Customization

### Custom Plot Styles

```python
# Define custom style
custom_style = {
    'figure.figsize': (12, 8),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 14,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5
}

plt.rcParams.update(custom_style)

# Create plot with custom style
fig, ax = plt.subplots()
time_series_plots.plot_time_series(signal, ax=ax, title="Custom Style Plot")
plt.show()
```

### Interactive Plots (if using plotly)

```python
# Example of interactive plot (if plotly is available)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Create interactive time series plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode='lines', name='Time Series'))
    fig.update_layout(title='Interactive Time Series Plot', xaxis_title='Time', yaxis_title='Value')
    fig.show()
    
except ImportError:
    print("Plotly not available. Install with: pip install plotly")
```

## 📋 Best Practices

### 1. Figure Design
- Use consistent color schemes
- Ensure adequate contrast
- Include proper labels and titles
- Use appropriate figure sizes

### 2. Publication Standards
- Use vector formats (PDF, SVG) for publication
- Ensure high resolution (300 DPI minimum)
- Follow journal-specific requirements
- Include proper legends and captions

### 3. Data Visualization
- Choose appropriate plot types
- Avoid misleading visualizations
- Include uncertainty measures
- Use log scales when appropriate

### 4. Report Organization
- Structure reports logically
- Include executive summaries
- Provide clear conclusions
- Document methodology

## 🔧 Troubleshooting

### Common Issues and Solutions

**Issue 1: Plot not displaying**
```python
# Ensure matplotlib backend is set correctly
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' on some systems
```

**Issue 2: Figure too small/large**
```python
# Adjust figure size
plt.figure(figsize=(12, 8))
```

**Issue 3: Text overlapping**
```python
# Use tight layout
plt.tight_layout()
```

**Issue 4: Colors not visible**
```python
# Check color scheme
plt.style.use('default')  # Reset to default style
```

## 📚 Next Steps

- Review the [API Documentation](../docs/api_documentation.md)
- Explore the [Methodology Guide](../docs/methodology.md)
- Check out the [Analysis Protocol](../docs/analysis_protocol.md)

---

**You now have comprehensive knowledge of visualization and reporting!** 🎉

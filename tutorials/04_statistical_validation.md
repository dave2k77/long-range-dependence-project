# Statistical Validation Tutorial

## 🎯 Overview

This tutorial covers the statistical validation methods available in our framework: Bootstrap Analysis, Cross-validation, Monte Carlo Simulations, and Hypothesis Testing. These methods ensure the robustness and reliability of your long-range dependence analysis results.

## 🔄 Bootstrap Analysis

Bootstrap analysis provides confidence intervals and uncertainty quantification for your estimates.

### Basic Bootstrap Analysis

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

from analysis import DFAnalysis
from statistical_validation import BootstrapValidation
from data_processing import PureSignalGenerator

# Generate test data
generator = PureSignalGenerator(random_state=42)
signal = generator.generate_arfima(n=1000, d=0.3)

# Initialize bootstrap validator
bootstrap = BootstrapValidation(DFAnalysis())

# Run bootstrap analysis
result = bootstrap.analyze(signal, n_bootstrap=1000, confidence_level=0.95)

print(f"Original Hurst: {result['original_estimate']:.3f}")
print(f"Bootstrap mean: {result['bootstrap_mean']:.3f}")
print(f"Confidence interval: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
print(f"Standard error: {result['standard_error']:.3f}")
```

### Bootstrap with Different Methods

```python
from analysis import RSAnalysis, SpectralAnalysis

# Test multiple methods
methods = {
    'DFA': DFAnalysis(),
    'R/S': RSAnalysis(),
    'Spectral': SpectralAnalysis()
}

bootstrap_results = {}
for name, method in methods.items():
    bootstrap_validator = BootstrapValidation(method)
    result = bootstrap_validator.analyze(signal, n_bootstrap=500, confidence_level=0.95)
    bootstrap_results[name] = result
    
    print(f"\n{name} Results:")
    print(f"  Original: {result['original_estimate']:.3f}")
    print(f"  Bootstrap mean: {result['bootstrap_mean']:.3f}")
    print(f"  95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
```

### Bootstrap Visualization

```python
from visualisation import validation_plots

# Plot bootstrap results
validation_plots.plot_bootstrap_results(result, title="DFA Bootstrap Analysis")

# Plot bootstrap distribution
validation_plots.plot_bootstrap_distribution(result['bootstrap_samples'], 
                                            result['original_estimate'],
                                            title="Bootstrap Distribution")
```

## 🔀 Cross-Validation

Cross-validation assesses the robustness of your analysis across different data segments.

### Basic Cross-Validation

```python
from statistical_validation import CrossValidation

# Initialize cross-validation
cv = CrossValidation(DFAnalysis())

# Run cross-validation
result = cv.analyze(signal, n_folds=5, overlap_ratio=0.5)

print(f"Mean Hurst: {result['mean_estimate']:.3f}")
print(f"Standard deviation: {result['std_estimate']:.3f}")
print(f"Range: [{result['min_estimate']:.3f}, {result['max_estimate']:.3f}]")
print(f"CV score: {result['cv_score']:.3f}")
```

### Cross-Validation with Different Configurations

```python
# Test different fold configurations
fold_configs = [3, 5, 10]
overlap_configs = [0.0, 0.25, 0.5]

cv_results = {}
for folds in fold_configs:
    for overlap in overlap_configs:
        result = cv.analyze(signal, n_folds=folds, overlap_ratio=overlap)
        key = f"folds_{folds}_overlap_{overlap}"
        cv_results[key] = result
        
        print(f"\n{folds} folds, {overlap} overlap:")
        print(f"  Mean: {result['mean_estimate']:.3f}")
        print(f"  Std: {result['std_estimate']:.3f}")
        print(f"  CV score: {result['cv_score']:.3f}")
```

### Cross-Validation Visualization

```python
# Plot cross-validation results
validation_plots.plot_cross_validation_results(result, title="DFA Cross-Validation")

# Plot fold estimates
validation_plots.plot_fold_estimates(result['fold_estimates'], 
                                    result['mean_estimate'],
                                    title="Cross-Validation Fold Estimates")
```

## 🎲 Monte Carlo Simulations

Monte Carlo simulations evaluate method performance under controlled conditions.

### Basic Monte Carlo Analysis

```python
from statistical_validation import MonteCarloValidation

# Initialize Monte Carlo validator
mc = MonteCarloValidation(DFAnalysis())

# Run Monte Carlo analysis
result = mc.analyze(
    signal_length=1000,
    true_hurst=0.3,
    n_simulations=100,
    noise_level=0.1
)

print(f"True Hurst: {result['true_hurst']:.3f}")
print(f"Mean estimated Hurst: {result['mean_estimate']:.3f}")
print(f"Bias: {result['bias']:.3f}")
print(f"RMSE: {result['rmse']:.3f}")
print(f"Coverage rate: {result['coverage_rate']:.3f}")
```

### Monte Carlo with Different Conditions

```python
# Test different noise levels
noise_levels = [0.0, 0.05, 0.1, 0.2]
mc_results = {}

for noise in noise_levels:
    result = mc.analyze(
        signal_length=1000,
        true_hurst=0.3,
        n_simulations=100,
        noise_level=noise
    )
    mc_results[noise] = result
    
    print(f"\nNoise level {noise}:")
    print(f"  Bias: {result['bias']:.3f}")
    print(f"  RMSE: {result['rmse']:.3f}")
    print(f"  Coverage: {result['coverage_rate']:.3f}")
```

### Monte Carlo Visualization

```python
# Plot Monte Carlo results
validation_plots.plot_monte_carlo_results(result, title="DFA Monte Carlo Analysis")

# Plot bias and RMSE vs noise level
noise_levels = list(mc_results.keys())
biases = [mc_results[n]['bias'] for n in noise_levels]
rmses = [mc_results[n]['rmse'] for n in noise_levels]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(noise_levels, biases, 'o-')
ax1.set_xlabel('Noise Level')
ax1.set_ylabel('Bias')
ax1.set_title('Bias vs Noise Level')
ax1.grid(True)

ax2.plot(noise_levels, rmses, 'o-')
ax2.set_xlabel('Noise Level')
ax2.set_ylabel('RMSE')
ax2.set_title('RMSE vs Noise Level')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## 🧪 Hypothesis Testing

Hypothesis testing provides statistical significance for long-range dependence detection.

### Basic Hypothesis Testing

```python
from statistical_validation import HypothesisTesting

# Initialize hypothesis tester
ht = HypothesisTesting(DFAnalysis())

# Test for long-range dependence
result = ht.test_lrd(signal, alpha=0.05)

print(f"Hurst exponent: {result['hurst']:.3f}")
print(f"Test statistic: {result['test_statistic']:.3f}")
print(f"P-value: {result['p_value']:.3f}")
print(f"Significant LRD: {result['significant']}")
print(f"Effect size: {result['effect_size']:.3f}")
```

### Multiple Hypothesis Tests

```python
# Test different null hypotheses
test_configs = [
    {'null_hypothesis': 'random_walk', 'alpha': 0.05},
    {'null_hypothesis': 'short_range', 'alpha': 0.01},
    {'null_hypothesis': 'white_noise', 'alpha': 0.05}
]

ht_results = {}
for config in test_configs:
    result = ht.test_lrd(signal, **config)
    key = f"{config['null_hypothesis']}_{config['alpha']}"
    ht_results[key] = result
    
    print(f"\n{config['null_hypothesis']} test (α={config['alpha']}):")
    print(f"  P-value: {result['p_value']:.3f}")
    print(f"  Significant: {result['significant']}")
    print(f"  Effect size: {result['effect_size']:.3f}")
```

### Hypothesis Testing Visualization

```python
# Plot hypothesis test results
validation_plots.plot_hypothesis_test_results(result, title="LRD Hypothesis Test")

# Plot test statistic distribution
validation_plots.plot_test_statistic_distribution(result['null_distribution'],
                                                 result['test_statistic'],
                                                 title="Test Statistic Distribution")
```

## 🔄 Comprehensive Validation Pipeline

### Complete Validation Workflow

```python
from statistical_validation import ComprehensiveValidation

# Initialize comprehensive validator
validator = ComprehensiveValidation(DFAnalysis())

# Run complete validation
results = validator.validate_comprehensive(
    signal,
    bootstrap_config={'n_bootstrap': 1000, 'confidence_level': 0.95},
    cv_config={'n_folds': 5, 'overlap_ratio': 0.5},
    mc_config={'n_simulations': 100, 'noise_level': 0.1},
    ht_config={'alpha': 0.05}
)

print("Comprehensive Validation Results:")
print(f"Original estimate: {results['original_estimate']:.3f}")
print(f"Bootstrap CI: [{results['bootstrap']['ci_lower']:.3f}, {results['bootstrap']['ci_upper']:.3f}]")
print(f"CV mean ± std: {results['cross_validation']['mean_estimate']:.3f} ± {results['cross_validation']['std_estimate']:.3f}")
print(f"MC bias: {results['monte_carlo']['bias']:.3f}")
print(f"LRD significant: {results['hypothesis_test']['significant']}")
```

### Method Comparison with Validation

```python
# Compare multiple methods with validation
methods = {
    'DFA': DFAnalysis(),
    'R/S': RSAnalysis(),
    'Spectral': SpectralAnalysis()
}

comparison_results = {}
for name, method in methods.items():
    validator = ComprehensiveValidation(method)
    results = validator.validate_comprehensive(signal)
    comparison_results[name] = results
    
    print(f"\n{name} Validation Results:")
    print(f"  Estimate: {results['original_estimate']:.3f}")
    print(f"  Bootstrap CI: [{results['bootstrap']['ci_lower']:.3f}, {results['bootstrap']['ci_upper']:.3f}]")
    print(f"  CV stability: {results['cross_validation']['cv_score']:.3f}")
    print(f"  MC bias: {results['monte_carlo']['bias']:.3f}")
    print(f"  LRD significant: {results['hypothesis_test']['significant']}")
```

## 📊 Advanced Validation Visualizations

### Validation Summary Plot

```python
# Create comprehensive validation plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Bootstrap plot
validation_plots.plot_bootstrap_results(results['bootstrap'], ax=axes[0,0], title="Bootstrap")

# Cross-validation plot
validation_plots.plot_cross_validation_results(results['cross_validation'], 
                                              ax=axes[0,1], title="Cross-Validation")

# Monte Carlo plot
validation_plots.plot_monte_carlo_results(results['monte_carlo'], 
                                         ax=axes[1,0], title="Monte Carlo")

# Hypothesis test plot
validation_plots.plot_hypothesis_test_results(results['hypothesis_test'], 
                                             ax=axes[1,1], title="Hypothesis Test")

plt.tight_layout()
plt.show()
```

### Method Comparison Dashboard

```python
# Create method comparison dashboard
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Bootstrap comparison
bootstrap_means = [comparison_results[name]['bootstrap']['bootstrap_mean'] for name in methods.keys()]
bootstrap_cis = [(comparison_results[name]['bootstrap']['ci_lower'], 
                  comparison_results[name]['bootstrap']['ci_upper']) for name in methods.keys()]

axes[0,0].bar(methods.keys(), bootstrap_means, yerr=[(ci[1]-ci[0])/2 for ci in bootstrap_cis])
axes[0,0].set_title('Bootstrap Estimates')
axes[0,0].set_ylabel('Hurst Exponent')

# CV stability comparison
cv_scores = [comparison_results[name]['cross_validation']['cv_score'] for name in methods.keys()]
axes[0,1].bar(methods.keys(), cv_scores)
axes[0,1].set_title('Cross-Validation Stability')
axes[0,1].set_ylabel('CV Score')

# MC bias comparison
mc_biases = [comparison_results[name]['monte_carlo']['bias'] for name in methods.keys()]
axes[1,0].bar(methods.keys(), mc_biases)
axes[1,0].set_title('Monte Carlo Bias')
axes[1,0].set_ylabel('Bias')

# Significance comparison
significance = [comparison_results[name]['hypothesis_test']['significant'] for name in methods.keys()]
axes[1,1].bar(methods.keys(), significance)
axes[1,1].set_title('LRD Significance')
axes[1,1].set_ylabel('Significant')

plt.tight_layout()
plt.show()
```

## 📋 Best Practices

### 1. Bootstrap Analysis
- Use at least 1000 bootstrap samples for reliable confidence intervals
- Check bootstrap distribution for normality
- Consider bias-corrected confidence intervals for skewed distributions

### 2. Cross-Validation
- Use 5-10 folds for most applications
- Consider overlap between folds for time series data
- Monitor CV score for stability assessment

### 3. Monte Carlo Simulations
- Match simulation parameters to your data characteristics
- Use sufficient simulations (100+) for reliable estimates
- Test multiple noise levels and signal lengths

### 4. Hypothesis Testing
- Choose appropriate null hypothesis for your research question
- Consider multiple testing corrections for multiple comparisons
- Interpret effect sizes alongside p-values

### 5. Comprehensive Validation
- Always use multiple validation methods
- Document validation parameters and results
- Consider computational cost vs. validation depth

## 🔧 Troubleshooting

### Common Issues and Solutions

**Issue 1: Bootstrap convergence problems**
```python
# Increase bootstrap samples
result = bootstrap.analyze(signal, n_bootstrap=2000, confidence_level=0.95)
```

**Issue 2: Cross-validation instability**
```python
# Reduce overlap and increase folds
result = cv.analyze(signal, n_folds=10, overlap_ratio=0.25)
```

**Issue 3: Monte Carlo computational cost**
```python
# Use parallel processing
result = mc.analyze(signal_length=1000, n_simulations=100, n_jobs=-1)
```

**Issue 4: Hypothesis test power**
```python
# Increase sample size or adjust significance level
result = ht.test_lrd(signal, alpha=0.01, power_analysis=True)
```

## 📚 Next Steps

- **Tutorial 5**: [Visualization and Reporting](05_visualization.md)

---

**You now have comprehensive knowledge of statistical validation methods!** 🎉

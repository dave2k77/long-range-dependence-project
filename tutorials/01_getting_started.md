# Getting Started with Long-Range Dependence Analysis

## 🎯 Overview

This tutorial will guide you through setting up and running your first long-range dependence analysis using our comprehensive framework.

## 📋 Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- Git
- Basic understanding of time series analysis concepts

## 🛠 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/dave2k77/long-range-dependence-project.git
cd long-range-dependence-project
```

### Step 2: Create Virtual Environment

**On Windows:**
```bash
python -m venv fractal-env
fractal-env\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv fractal-env
source fractal-env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python scripts/demo_synthetic_data.py
```

If everything is set up correctly, you should see output showing synthetic data generation examples.

## 🚀 Your First Analysis

### Step 1: Generate Test Data

Let's start by generating some synthetic data to work with:

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

from data_processing.synthetic_generator import SyntheticDataGenerator

# Create a data generator
generator = SyntheticDataGenerator(random_state=42)

# Generate a simple ARFIMA process using the new convenience method
arfima_signal = generator.generate_arfima(n=1000, d=0.3)
print(f"Generated ARFIMA signal with {len(arfima_signal)} points")

# You can also generate other types of synthetic data
fbm_signal = generator.generate_fbm(n=1000, hurst=0.7)
fgn_signal = generator.generate_fgn(n=1000, hurst=0.6)
print(f"Generated fBm signal with {len(fbm_signal)} points")
print(f"Generated fGn signal with {len(fgn_signal)} points")
```

### Step 2: Run Basic Analysis

```python
from analysis.dfa_analysis import dfa
from analysis.rs_analysis import rs_analysis

# Analyze the signal using the updated function-based approach
scales, flucts, dfa_summary = dfa(arfima_signal, order=1)
scales_rs, rs_values, rs_summary = rs_analysis(arfima_signal)

print(f"DFA Hurst exponent: {dfa_summary.hurst:.3f}")
print(f"R/S Hurst exponent: {rs_summary.hurst:.3f}")
```

### Step 3: Visualize Results

```python
import matplotlib.pyplot as plt

# Plot the time series
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(arfima_signal[:200])
plt.title("ARFIMA Process (d=0.3) - First 200 points")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)

# Plot DFA results
plt.subplot(2, 2, 2)
plt.loglog(scales, flucts, 'o-', label='DFA')
plt.xlabel("Scale")
plt.ylabel("Fluctuation")
plt.title("DFA Analysis Results")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot R/S results
plt.subplot(2, 2, 3)
plt.loglog(scales_rs, rs_values, 's-', label='R/S')
plt.xlabel("Scale")
plt.ylabel("R/S Value")
plt.title("R/S Analysis Results")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 🔧 Advanced Usage

### Using the ARFIMA Model

```python
from analysis.arfima_modelling import ARFIMAModel

# Create and fit an ARFIMA model
model = ARFIMAModel(p=1, d=0.3, q=1, fast_mode=True)
fitted_model = model.fit(arfima_signal)

# Get estimates
hurst_est = fitted_model.estimate_hurst()
alpha_est = fitted_model.estimate_alpha()
conf_intervals = fitted_model.get_confidence_intervals()

print(f"Estimated Hurst exponent: {hurst_est:.3f}")
print(f"Estimated alpha: {alpha_est:.3f}")
print(f"Confidence intervals: {conf_intervals}")
```

### Running Multiple Analyses

```python
from analysis.mfdfa_analysis import mfdfa
from analysis.wavelet_analysis import wavelet_leaders_estimation

# Run multiple analysis methods
scales_mf, fq, mfdfa_summary = mfdfa(arfima_signal)
scales_wav, hq, wavelet_summary = wavelet_leaders_estimation(arfima_signal)

print(f"MFDFA Hurst: {mfdfa_summary.hurst:.3f}")
print(f"Wavelet Hurst: {wavelet_summary.hurst:.3f}")
```

## 📚 Next Steps

- **Tutorial 2**: Learn about synthetic data generation in detail
- **Tutorial 3**: Explore advanced analysis methods
- **Tutorial 4**: Understand statistical validation
- **Tutorial 5**: Create visualizations and plots
- **Tutorial 6**: Submit your own models and datasets

## 🆘 Troubleshooting

If you encounter issues:

1. **Import errors**: Make sure you've added `src` to your Python path
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Data generation issues**: Check that your random seed is set for reproducibility
4. **Analysis failures**: Ensure your data has sufficient length (recommend at least 500 points)

For more help, check the project documentation or create an issue on GitHub.

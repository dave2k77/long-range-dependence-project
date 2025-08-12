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

from data_processing import SyntheticDataGenerator

# Create a data generator
generator = SyntheticDataGenerator(random_state=42)

# Generate a simple ARFIMA process
arfima_signal = generator.generate_arfima(n=1000, d=0.3)
print(f"Generated ARFIMA signal with {len(arfima_signal)} points")
```

### Step 2: Run Basic Analysis

```python
from analysis import DFAnalysis, RSAnalysis

# Initialize analysis objects
dfa = DFAnalysis()
rs = RSAnalysis()

# Analyze the signal
dfa_result = dfa.analyze(arfima_signal)
rs_result = rs.analyze(arfima_signal)

print(f"DFA Hurst exponent: {dfa_result['hurst']:.3f}")
print(f"R/S Hurst exponent: {rs_result['hurst']:.3f}")
```

### Step 3: Visualize Results

```python
from visualisation import time_series_plots, fractal_plots

# Plot the time series
time_series_plots.plot_time_series(arfima_signal, title="ARFIMA Process (d=0.3)")

# Plot DFA results
fractal_plots.plot_dfa_results(dfa_result, title="DFA Analysis Results")
```

## 📊 Understanding the Results

### Hurst Exponent Interpretation

- **H = 0.5**: Random walk (no long-range dependence)
- **0.5 < H < 1**: Long-range dependence (persistent)
- **0 < H < 0.5**: Anti-persistent behavior

### What the Analysis Tells Us

1. **DFA (Detrended Fluctuation Analysis)**:
   - Robust to non-stationarities
   - Good for detecting long-range dependence
   - Works well with trends and seasonality

2. **R/S Analysis (Rescaled Range)**:
   - Classical method for Hurst exponent estimation
   - Sensitive to non-stationarities
   - Good baseline for comparison

## 🔧 Common Issues and Solutions

### Issue 1: Import Errors
**Problem**: `ModuleNotFoundError` when importing from `src`
**Solution**: Ensure you've added the src directory to your Python path:
```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))
```

### Issue 2: Memory Errors with Large Datasets
**Problem**: Out of memory when analyzing large time series
**Solution**: Use smaller segments or increase system memory:
```python
# Analyze in segments
segment_size = 1000
for i in range(0, len(data), segment_size):
    segment = data[i:i+segment_size]
    result = dfa.analyze(segment)
```

### Issue 3: Convergence Issues
**Problem**: Analysis doesn't converge or gives unrealistic results
**Solution**: Check data quality and preprocessing:
```python
# Check for NaN or infinite values
import numpy as np
print(f"NaN values: {np.isnan(data).sum()}")
print(f"Infinite values: {np.isinf(data).sum()}")
```

## 📈 Next Steps

Now that you've completed your first analysis, you can:

1. **Explore Different Methods**: Try Higuchi, MFDFA, or spectral analysis
2. **Generate More Data**: Use the synthetic data generator for different signal types
3. **Analyze Real Data**: Load and analyze your own time series data
4. **Run Validation**: Use bootstrap and Monte Carlo methods for robust results

## 📚 Additional Resources

- **Tutorial 2**: [Synthetic Data Generation](02_synthetic_data_generation.md)
- **Tutorial 3**: [Advanced Analysis Methods](03_advanced_analysis.md)
- **Tutorial 4**: [Statistical Validation](04_statistical_validation.md)
- **Tutorial 5**: [Visualization and Reporting](05_visualization.md)

## 🆘 Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/dave2k77/long-range-dependence-project/issues)
2. Review the [API Documentation](../docs/api_documentation.md)
3. Run the test suite: `python -m pytest tests/ -v`

---

**Congratulations!** You've successfully set up and run your first long-range dependence analysis. 🎉

# Long-Range Dependence Analysis Project

A comprehensive Python framework for analyzing long-range dependence in time series data, featuring advanced fractal analysis methods and synthetic data generation capabilities.

## 🎯 Overview

This project provides a complete toolkit for long-range dependence (LRD) analysis, including:

- **Multiple Analysis Methods**: DFA, R/S Analysis, Higuchi Method, MFDFA, Spectral Analysis, Wavelet Analysis
- **Synthetic Data Generation**: ARFIMA, Fractional Brownian Motion, Fractional Gaussian Noise with contamination
- **Statistical Validation**: Bootstrap, Cross-validation, Monte Carlo simulations, Hypothesis testing
- **Comprehensive Visualization**: Time series plots, fractal analysis results, validation plots
- **Data Management**: Automated data processing, quality assessment, and metadata management

## 🚀 Key Features

### Analysis Methods
- **Detrended Fluctuation Analysis (DFA)**: Robust method for estimating Hurst exponent
- **R/S Analysis**: Rescaled range analysis for long-range dependence detection
- **Higuchi Method**: Fractal dimension estimation for time series
- **Multifractal Detrended Fluctuation Analysis (MFDFA)**: Advanced multifractal analysis
- **Spectral Analysis**: Periodogram and Whittle estimation methods
- **Wavelet Analysis**: Continuous and discrete wavelet transforms

### Synthetic Data Generation
- **Pure Signal Generators**: ARFIMA(p,d,q), fBm, fGn processes
- **Data Contaminators**: Polynomial trends, periodicity, outliers, irregular sampling, heavy-tail fluctuations
- **Comprehensive Testing**: 34 different signal types for method validation

### Statistical Validation
- **Bootstrap Analysis**: Confidence intervals and uncertainty quantification
- **Cross-validation**: Robustness assessment across different data segments
- **Monte Carlo Simulations**: Performance evaluation under controlled conditions
- **Hypothesis Testing**: Statistical significance testing for LRD detection

## 📁 Project Structure

```
long-range-dependence-project/
├── src/                          # Source code
│   ├── analysis/                 # Analysis methods
│   ├── data_processing/          # Data handling and synthetic generation
│   └── visualisation/            # Plotting and visualization
├── scripts/                      # Execution scripts
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── data/                         # Data storage
│   ├── raw/                      # Raw data files
│   ├── processed/                # Processed data
│   └── metadata/                 # Data documentation
├── results/                      # Analysis results
│   ├── figures/                  # Generated plots
│   ├── tables/                   # Results tables
│   └── reports/                  # Analysis reports
├── config/                       # Configuration files
├── docs/                         # Documentation
└── manuscript/                   # Research manuscript
```

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dave2k77/long-range-dependence-project.git
   cd long-range-dependence-project
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv fractal-env
   source fractal-env/bin/activate  # On Windows: fractal-env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python scripts/demo_synthetic_data.py
   ```

## 📖 Usage

### Quick Start

```python
from src.data_processing import SyntheticDataGenerator
from src.analysis import DFAnalysis, RSAnalysis

# Generate synthetic test data
generator = SyntheticDataGenerator(random_state=42)
dataset = generator.generate_comprehensive_dataset(n=1000, save=True)

# Analyze data
dfa = DFAnalysis()
rs = RSAnalysis()

for name, signal in dataset['clean_signals'].items():
    dfa_result = dfa.analyze(signal)
    rs_result = rs.analyze(signal)
    print(f"{name}: DFA={dfa_result['hurst']:.3f}, RS={rs_result['hurst']:.3f}")
```

### Command Line Usage

**Generate synthetic data**:
```bash
# Generate comprehensive dataset
python scripts/generate_synthetic_data.py --n 1000

# Generate only clean signals
python scripts/generate_synthetic_data.py --clean-only --n 500
```

**Run analysis**:
```bash
# Run full analysis pipeline
python scripts/run_full_analysis.py

# Test specific methods
python scripts/test_analysis.py
```

**Demonstrations**:
```bash
# Synthetic data generation demo
python scripts/demo_synthetic_data.py

# ARFIMA modeling demo
python scripts/demo_arfima.py
```

### Jupyter Notebooks

- `notebooks/01_synthetic_data_generation.ipynb`: Synthetic data generation tutorial
- `notebooks/04_arfima_modelling.ipynb`: ARFIMA modeling examples
- `notebooks/05_dfa_analysis.ipynb`: DFA analysis tutorial

## 🔬 Analysis Methods

### Detrended Fluctuation Analysis (DFA)
```python
from src.analysis import DFAnalysis

dfa = DFAnalysis()
result = dfa.analyze(time_series)
print(f"Hurst exponent: {result['hurst']:.3f}")
```

### R/S Analysis
```python
from src.analysis import RSAnalysis

rs = RSAnalysis()
result = rs.analyze(time_series)
print(f"Hurst exponent: {result['hurst']:.3f}")
```

### Higuchi Method
```python
from src.analysis import HiguchiAnalysis

higuchi = HiguchiAnalysis()
result = higuchi.analyze(time_series)
print(f"Fractal dimension: {result['fractal_dimension']:.3f}")
```

### Multifractal Analysis
```python
from src.analysis import MFDFAAnalysis

mfdfa = MFDFAAnalysis()
result = mfdfa.analyze(time_series)
print(f"Multifractal spectrum: {result['spectrum']}")
```

## 📊 Synthetic Data Generation

### Pure Signal Generators
```python
from src.data_processing import PureSignalGenerator

generator = PureSignalGenerator(random_state=42)

# ARFIMA process
arfima_signal = generator.generate_arfima(n=1000, d=0.3)

# Fractional Brownian Motion
fbm_signal = generator.generate_fbm(n=1000, hurst=0.7)

# Fractional Gaussian Noise
fgn_signal = generator.generate_fgn(n=1000, hurst=0.5)
```

### Data Contamination
```python
from src.data_processing import DataContaminator

contaminator = DataContaminator(random_state=42)

# Add polynomial trend
trend_signal = contaminator.add_polynomial_trend(signal, degree=2, amplitude=0.1)

# Add periodicity
periodic_signal = contaminator.add_periodicity(signal, frequency=50, amplitude=0.2)

# Add outliers
outlier_signal = contaminator.add_outliers(signal, fraction=0.02, magnitude=4.0)
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
python -m pytest tests/ -v
```

Run specific test modules:
```bash
python -m pytest tests/test_synthetic_generator.py -v
python -m pytest tests/test_dfa.py -v
```

## 📈 Results and Visualization

The project automatically generates comprehensive visualizations:

- **Time Series Plots**: Raw and processed data visualization
- **Fractal Analysis Results**: DFA, R/S, Higuchi method plots
- **Validation Results**: Bootstrap, cross-validation, Monte Carlo plots
- **Synthetic Data Examples**: Pure and contaminated signal comparisons

Results are saved in the `results/` directory with organized subdirectories.

## 📚 Documentation

- `docs/synthetic_data_generation.md`: Synthetic data generation guide
- `docs/methodology.md`: Analysis method descriptions
- `docs/api_documentation.md`: API reference
- `docs/analysis_protocol.md`: Analysis workflow documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that allows for:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ✅ Patent use

**Requirements**: The only requirement is that the license and copyright notice be included in all copies or substantial portions of the software.

## 🏆 Acknowledgments

- **ARFIMA processes**: Granger & Joyeux (1980)
- **Fractional Brownian Motion**: Mandelbrot & Van Ness (1968)
- **DFA method**: Peng et al. (1994)
- **R/S Analysis**: Hurst (1951), Mandelbrot (1971)
- **Higuchi Method**: Higuchi (1988)

## 📞 Contact

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project is designed for research purposes and provides a comprehensive framework for long-range dependence analysis. The synthetic data generation capabilities make it particularly useful for method validation and benchmarking.


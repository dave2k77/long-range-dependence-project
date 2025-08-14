# Long-Range Dependence Analysis Project

A comprehensive Python framework for analyzing long-range dependence in time series data, featuring advanced fractal analysis methods, synthetic data generation capabilities, and JAX-accelerated parallel computation.

**Status: ✅ Production Ready**  
**Test Results: 255 passed, 0 failed**  
**Last Updated: December 2024**

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
- **ARFIMA Modeling**: Autoregressive Fractionally Integrated Moving Average models

### JAX-Accelerated Parallel Computation
- **GPU/TPU Support**: Hardware acceleration for large-scale computations
- **Vectorized Operations**: Batch processing of multiple datasets
- **JIT Compilation**: Just-in-time compilation for optimal performance
- **Memory Optimization**: Efficient memory usage for large datasets
- **Parallel Processing**: Multi-device computation support

### Synthetic Data Generation
- **Pure Signal Generators**: ARFIMA(p,d,q), fBm, fGn processes
- **Data Contaminators**: Polynomial trends, periodicity, outliers, irregular sampling, heavy-tail fluctuations
- **Comprehensive Testing**: 34 different signal types for method validation

### Statistical Validation
- **Bootstrap Analysis**: Confidence intervals and uncertainty quantification
- **Cross-validation**: Robustness assessment across different data segments
- **Monte Carlo Simulations**: Performance evaluation under controlled conditions
- **Hypothesis Testing**: Statistical significance testing for LRD detection

### Submission System
- **Model Submission**: Submit new estimator models with validation and testing
- **Dataset Submission**: Submit new datasets with quality assessment
- **Standards Compliance**: Automatic validation against quality standards
- **Integration Testing**: Full analysis pipeline integration
- **Registry Management**: Centralized model and dataset registry

### Development & CI/CD
- **Continuous Integration**: Automated testing with GitHub Actions
- **Code Quality**: Linting, formatting, and type checking
- **Security**: Automated vulnerability scanning
- **Containerization**: Docker support for development and deployment
- **Pre-commit Hooks**: Automated code quality enforcement
- **Test Coverage**: Comprehensive test suite with 255 passing tests


- **Data Management**: Load, view, and manage datasets
- **Analysis Pipeline**: Run different LRD analysis methods
- **Synthetic Data Generation**: Generate and test synthetic data
- **Submission System**: Submit new models and datasets
- **Results Visualization**: View and export analysis results
- **Configuration Management**: Manage project settings

## 📁 Project Structure

```
long-range-dependence-project/
├── src/                          # Source code
│   ├── analysis/                 # Analysis methods
│   ├── data_processing/          # Data handling and synthetic generation
│   ├── submission/               # Model and dataset submission system

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
├── tutorials/                    # Tutorial guides
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

4. **Install JAX (for GPU acceleration)**:
   ```bash
   # CPU-only version (included in requirements.txt)
   pip install jax jaxlib
   
   # GPU version (CUDA 11.8 or 12.1)
   # pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   # or
   # pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

5. **Verify installation**:
   ```bash
   python scripts/demo_synthetic_data.py
   python scripts/demo_jax_parallel.py
   ```

## 📖 Usage

### Quick Start

```python
from src.data_processing import SyntheticDataGenerator
from src.analysis import DFAnalysis, RSAnalysis
from src.analysis.jax_parallel_analysis import jax_parallel_analysis

# Generate synthetic test data
generator = SyntheticDataGenerator(random_state=42)
dataset = generator.generate_comprehensive_dataset(n=1000, save=True)

# Traditional analysis
dfa = DFAnalysis()
rs = RSAnalysis()

for name, signal in dataset['clean_signals'].items():
    dfa_result = dfa.analyze(signal)
    rs_result = rs.analyze(signal)
    print(f"{name}: DFA={dfa_result['hurst']:.3f}, RS={rs_result['hurst']:.3f}")

# JAX-accelerated parallel analysis
jax_results = jax_parallel_analysis(
    datasets=dataset['clean_signals'],
    methods=['dfa', 'higuchi'],
    config=None  # Uses default CPU configuration
)
print("JAX Results:", jax_results)
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

# JAX-accelerated parallel computation demo
python scripts/demo_jax_parallel.py

# Traditional parallel computation demo (Joblib - recommended)
python scripts/demo_joblib_parallel.py

# High-performance parallel computation demo (Numba)
python scripts/demo_numba_parallel.py
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

### Parallel Computation
```python
# Joblib-based parallel processing (recommended)
from src.analysis.joblib_parallel_analysis import joblib_parallel_analysis

datasets = {'data1': time_series1, 'data2': time_series2}
results = joblib_parallel_analysis(datasets, methods=['dfa', 'higuchi'])

# Numba-based high-performance processing
from src.analysis.numba_parallel_analysis import numba_parallel_analysis

results = numba_parallel_analysis(datasets, methods=['dfa', 'higuchi'])
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

## 🔄 Submission System

### Model Submission

Submit new estimator models to the benchmark:

```python
from src.submission import SubmissionManager, ModelMetadata

# Create model metadata
metadata = ModelMetadata(
    name="MyEstimator",
    version="1.0.0",
    author="Your Name",
    description="Description of your model",
    category="custom",
    parameters={"param1": 1.0},
    dependencies=["numpy", "pandas"]
)

# Submit model
manager = SubmissionManager()
result = manager.submit_model(
    model_file="path/to/your/model.py",
    metadata=metadata,
    run_full_analysis=True
)

print(f"Submission ID: {result.submission_id}")
print(f"Status: {result.status.value}")
```

### Dataset Submission

Submit new datasets for analysis:

```python
from src.submission import DatasetMetadata

# Create dataset metadata
metadata = DatasetMetadata(
    name="MyDataset",
    version="1.0.0",
    author="Your Name",
    description="Description of your dataset",
    category="financial",
    source="Data source",
    sampling_frequency="1 hour",
    units="USD",
    collection_date="2024-01-01"
)

# Submit dataset
result = manager.submit_dataset(
    file_path="path/to/your/dataset.csv",
    metadata=metadata,
    run_full_analysis=True
)
```

### Standards and Validation

The submission system automatically validates:

- **Model Standards**: Interface compliance, performance thresholds, documentation
- **Dataset Standards**: Format requirements, quality metrics, metadata completeness
- **Integration Testing**: Full analysis pipeline compatibility
- **Performance Benchmarking**: Comparison with existing methods

For detailed information, see the [Submission System Tutorial](tutorials/06_submission_system.md).

Results are saved in the `results/` directory with organized subdirectories.

## 🚀 Development & CI/CD

### Quick Start

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run all CI checks locally
make ci
```

### Available Commands

```bash
# Testing
make test              # Run all tests
make test-cov          # Run tests with coverage
make test-fast         # Run fast tests only

# Code Quality
make lint              # Run linting
make format            # Format code
make type-check        # Run type checking
make security          # Run security checks

# Development
make install-dev       # Install development dependencies
make clean             # Clean generated files
make docs              # Build documentation
```

### Docker Development

```bash
# Start all services
docker-compose up -d

# Run tests in container
docker-compose run test

# Start development shell
docker-compose run dev

# Start Jupyter notebook
docker-compose up jupyter
```

### CI/CD Pipeline

The project uses GitHub Actions for automated:

- **Multi-Python Testing**: Python 3.8, 3.9, 3.10, 3.11
- **Code Quality**: Linting, formatting, type checking
- **Security**: Vulnerability scanning
- **Coverage**: Test coverage reporting
- **Documentation**: Automated docs building

For detailed CI/CD information, see [CI/CD Guide](docs/ci_cd_guide.md).

## 📚 Documentation

- `docs/synthetic_data_generation.md`: Synthetic data generation guide
- `docs/methodology.md`: Analysis method descriptions
- `docs/api_documentation.md`: API reference
- `docs/analysis_protocol.md`: Analysis workflow documentation
- `docs/ci_cd_guide.md`: CI/CD and development guide

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


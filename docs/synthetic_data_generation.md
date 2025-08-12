# Synthetic Data Generation Module

## Overview

The synthetic data generation module provides comprehensive capabilities for generating test datasets for long-range dependence analysis. It includes pure signal generators and data contaminators to create realistic test scenarios.

## Components

### 1. Pure Signal Generators (`PureSignalGenerator`)

Generate clean signals without contamination for controlled experiments:

#### ARFIMA(p,d,q) Processes
```python
from data_processing import PureSignalGenerator

generator = PureSignalGenerator(random_state=42)
signal = generator.generate_arfima(n=1000, d=0.3)
```

**Parameters:**
- `n`: Length of time series
- `d`: Fractional differencing parameter (0 < d < 0.5)
- `ar_params`: AR parameters (optional)
- `ma_params`: MA parameters (optional)
- `sigma`: Standard deviation of innovations

#### Fractional Brownian Motion (fBm)
```python
signal = generator.generate_fbm(n=1000, hurst=0.7)
```

**Parameters:**
- `n`: Length of time series
- `hurst`: Hurst exponent (0 < H < 1)
- `sigma`: Standard deviation

#### Fractional Gaussian Noise (fGn)
```python
signal = generator.generate_fgn(n=1000, hurst=0.5)
```

**Parameters:**
- `n`: Length of time series
- `hurst`: Hurst exponent (0 < H < 1)
- `sigma`: Standard deviation

### 2. Data Contaminators (`DataContaminator`)

Add realistic artifacts to test robustness of analysis methods:

#### Polynomial Trends
```python
from data_processing import DataContaminator

contaminator = DataContaminator(random_state=42)
contaminated = contaminator.add_polynomial_trend(data, degree=2, amplitude=0.1)
```

#### Periodicity
```python
contaminated = contaminator.add_periodicity(data, frequency=50, amplitude=0.2)
```

#### Outliers
```python
contaminated = contaminator.add_outliers(data, fraction=0.02, magnitude=4.0)
```

#### Irregular Sampling
```python
sampled_data, time_indices = contaminator.add_irregular_sampling(data, missing_fraction=0.2)
```

#### Heavy-Tail Fluctuations
```python
contaminated = contaminator.add_heavy_tails(data, df=2.0, fraction=0.15)
```

### 3. Comprehensive Generator (`SyntheticDataGenerator`)

Combines pure signals and contaminants for complete dataset generation:

```python
from data_processing import SyntheticDataGenerator

generator = SyntheticDataGenerator(random_state=42)

# Generate clean signals only
clean_signals = generator.generate_clean_signals(n=1000, save=True)

# Generate contaminated signals
contaminated_signals = generator.generate_contaminated_signals(n=1000, save=True)

# Generate irregularly sampled signals
irregular_signals = generator.generate_irregular_sampled_signals(n=1000, save=True)

# Generate comprehensive dataset
dataset = generator.generate_comprehensive_dataset(n=1000, save=True)
```

## Usage Examples

### Basic Usage

```python
from data_processing import SyntheticDataGenerator

# Initialize generator
generator = SyntheticDataGenerator(random_state=42)

# Generate comprehensive dataset
dataset = generator.generate_comprehensive_dataset(n=1000, save=True)
```

### Command Line Usage

```bash
# Generate comprehensive dataset
python scripts/generate_synthetic_data.py --n 1000 --random-state 42

# Generate only clean signals
python scripts/generate_synthetic_data.py --clean-only --n 500

# Generate only contaminated signals
python scripts/generate_synthetic_data.py --contaminated-only --n 500

# Generate only irregular signals
python scripts/generate_synthetic_data.py --irregular-only --n 500
```

### Demonstration

```bash
# Run demonstration script
python scripts/demo_synthetic_data.py
```

## Data Storage

Generated data is automatically stored in the appropriate directories:

- **Raw Data**: `data/raw/` - CSV files with time series data
- **Metadata**: `data/metadata/` - JSON files with generation parameters and documentation

### Data Format

Each generated dataset includes:

1. **CSV File**: Two columns (`time`, `value`)
2. **Metadata File**: JSON with generation parameters, timestamps, and descriptions

### Metadata Structure

```json
{
    "name": "arfima_d03",
    "data_type": "synthetic",
    "description": "Synthetic ARFIMA signal with d=0.3",
    "parameters": {
        "n": 1000,
        "d": 0.3,
        "type": "clean"
    },
    "created_at": "2025-01-15T10:30:00",
    "file_path": "data/raw/arfima_d03.csv"
}
```

## Generated Datasets

The comprehensive generator creates the following datasets:

### Clean Signals
- ARFIMA with d = 0.1, 0.2, 0.3, 0.4
- fBm with H = 0.3, 0.5, 0.7
- fGn with H = 0.3, 0.5, 0.7

### Contaminated Signals
- Base signals with polynomial trends
- Base signals with periodicity
- Base signals with outliers
- Base signals with heavy-tail fluctuations
- Base signals with combined contamination

### Irregular Signals
- Base signals with 10%, 20%, 30% missing data

## Testing

Run the test suite to validate the module:

```bash
python -m pytest tests/test_synthetic_generator.py -v
```

## Integration

The synthetic data generation module integrates seamlessly with the existing analysis pipeline:

```python
from data_processing import SyntheticDataGenerator
from analysis import DFAnalysis, RSAnalysis

# Generate test data
generator = SyntheticDataGenerator()
dataset = generator.generate_comprehensive_dataset(n=1000, save=True)

# Analyze generated data
dfa = DFAnalysis()
rs = RSAnalysis()

for name, signal in dataset['clean_signals'].items():
    dfa_result = dfa.analyze(signal)
    rs_result = rs.analyze(signal)
    print(f"{name}: DFA={dfa_result['hurst']:.3f}, RS={rs_result['hurst']:.3f}")
```

## Quality Assurance

The module includes comprehensive testing and validation:

- **Unit Tests**: Test individual components
- **Integration Tests**: Test complete pipeline
- **Statistical Validation**: Verify generated data properties
- **Reproducibility**: Consistent results with fixed random seeds

## References

- ARFIMA processes: Granger & Joyeux (1980)
- Fractional Brownian Motion: Mandelbrot & Van Ness (1968)
- Fractional Gaussian Noise: Mandelbrot (1971)
- Davies-Harte method: Davies & Harte (1987)
- Circulant embedding: Dietrich & Newsam (1997)

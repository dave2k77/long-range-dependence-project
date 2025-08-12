# Long-Range Dependence Project: ARFIMA Implementation

This project provides a robust, custom implementation of ARFIMA (Autoregressive Fractionally Integrated Moving Average) models for time series analysis with long-range dependence.

## Overview

ARFIMA models extend traditional ARIMA models by allowing for fractional differencing, which captures long-range dependence in time series data. This implementation includes:

- **Fractional differencing and integration** with parameter d
- **Maximum likelihood estimation** of model parameters
- **Automatic model order selection** using information criteria
- **Forecasting capabilities** for future time steps
- **Comprehensive diagnostics** and model validation
- **Simulation tools** for generating ARFIMA processes

## Features

### Core ARFIMA Model
- **ARFIMAModel**: Main class for fitting and analyzing ARFIMA models
- **Parameter estimation**: Maximum likelihood estimation with constraints
- **Model diagnostics**: Residual analysis, Q-Q plots, ACF analysis
- **Forecasting**: Multi-step ahead predictions
- **Model comparison**: AIC/BIC-based model selection

### Utility Functions
- **estimate_arfima_order()**: Automatic order selection
- **arfima_simulation()**: Generate synthetic ARFIMA data
- **Fractional differencing/integration**: Core mathematical operations

### Key Capabilities
- ✅ Fractional differencing with parameter d (0 < d < 0.5)
- ✅ Autoregressive components (AR)
- ✅ Moving average components (MA)
- ✅ Maximum likelihood parameter estimation
- ✅ Constraint enforcement for stationarity/invertibility
- ✅ Model diagnostics and validation
- ✅ Forecasting with confidence intervals
- ✅ Comprehensive testing suite

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd long-range-dependence-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import numpy as np
from src.analysis.arfima_modelling import ARFIMAModel, arfima_simulation

# Generate synthetic ARFIMA data
data = arfima_simulation(
    n=1000,
    d=0.3,
    ar_params=np.array([0.5]),
    ma_params=np.array([0.3]),
    sigma=1.0,
    seed=42
)

# Fit ARFIMA model
model = ARFIMAModel(p=1, d=0.3, q=1)
fitted_model = model.fit(data)

# Generate forecasts
forecasts = fitted_model.forecast(steps=50)

# Get model summary
summary = fitted_model.summary()
print(f"Model: {summary['model']}")
print(f"Estimated d: {summary['parameters']['d']:.3f}")
print(f"AIC: {summary['fit_metrics']['aic']:.2f}")

# Plot diagnostics
fitted_model.plot_diagnostics()
```

### Automatic Model Selection

```python
from src.analysis.arfima_modelling import estimate_arfima_order

# Automatically estimate optimal model order
p, d, q = estimate_arfima_order(data, max_p=3, max_q=3)
print(f"Optimal order: ARFIMA({p},{d:.3f},{q})")

# Fit the optimal model
model = ARFIMAModel(p=p, d=d, q=q)
fitted_model = model.fit(data)
```

## API Reference

### ARFIMAModel

#### Constructor
```python
ARFIMAModel(p=1, d=0.5, q=1)
```

**Parameters:**
- `p` (int): Order of autoregressive component
- `d` (float): Fractional differencing parameter (0 < d < 0.5)
- `q` (int): Order of moving average component

#### Methods

##### fit(y, method='mle', initial_params=None, max_iter=1000, tol=1e-6)
Fit the ARFIMA model to time series data.

**Parameters:**
- `y`: Time series data (numpy array or pandas Series)
- `method`: Estimation method ('mle' for maximum likelihood)
- `initial_params`: Initial parameter values
- `max_iter`: Maximum optimization iterations
- `tol`: Optimization tolerance

**Returns:** Fitted model instance

##### forecast(steps, last_values=None)
Generate forecasts for future time steps.

**Parameters:**
- `steps` (int): Number of steps to forecast
- `last_values`: Last values of the series

**Returns:** Array of forecasted values

##### summary()
Generate comprehensive model summary.

**Returns:** Dictionary with model information

##### plot_diagnostics(figsize=(12, 8))
Plot model diagnostic plots.

##### predict(y)
Generate in-sample predictions.

### Utility Functions

#### estimate_arfima_order(y, max_p=3, max_q=3, d_values=None)
Estimate optimal ARFIMA model order using information criteria.

**Returns:** Tuple of (p, d, q) values

#### arfima_simulation(n, d, ar_params=None, ma_params=None, sigma=1.0, seed=None)
Simulate ARFIMA time series.

**Returns:** Simulated time series array

## Mathematical Background

### ARFIMA Model

The ARFIMA(p,d,q) model is defined as:

```
Φ(B)(1-B)^d X_t = Θ(B)ε_t
```

Where:
- `Φ(B)` is the AR polynomial of order p
- `Θ(B)` is the MA polynomial of order q
- `(1-B)^d` is the fractional differencing operator
- `ε_t` are i.i.d. innovations

### Fractional Differencing

The fractional differencing operator is defined as:

```
(1-B)^d = Σ_{k=0}^∞ (-1)^k (d choose k) B^k
```

Where the binomial coefficient is:

```
(d choose k) = d(d-1)...(d-k+1) / k!
```

## Examples

### Example 1: Financial Time Series Analysis

```python
import pandas as pd
from src.analysis.arfima_modelling import ARFIMAModel, estimate_arfima_order

# Load financial data
data = pd.read_csv('data/processed/financial_data.csv')
returns = data['returns'].values

# Estimate model order
p, d, q = estimate_arfima_order(returns, max_p=2, max_q=2)

# Fit model
model = ARFIMAModel(p=p, d=d, q=q)
fitted_model = model.fit(returns)

# Generate forecasts
forecasts = fitted_model.forecast(steps=30)

# Analyze results
summary = fitted_model.summary()
print(f"Long memory parameter d: {summary['parameters']['d']:.3f}")
```

### Example 2: Model Comparison

```python
import numpy as np
from src.analysis.arfima_modelling import ARFIMAModel

# Generate data
data = arfima_simulation(n=1000, d=0.3, ar_params=np.array([0.5]))

# Compare different models
models = []
for p in [0, 1, 2]:
    for q in [0, 1, 2]:
        try:
            model = ARFIMAModel(p=p, d=0.3, q=q)
            model.fit(data)
            models.append((model, model.aic))
        except:
            continue

# Find best model
best_model = min(models, key=lambda x: x[1])
print(f"Best model: ARFIMA({best_model[0].p},{best_model[0].d},{best_model[0].q})")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/analysis/arfima_modelling

# Run specific test file
pytest tests/test_arfima.py
```

## Demonstration

Run the demonstration script to see the ARFIMA implementation in action:

```bash
python scripts/demo_arfima.py
```

This will:
- Generate synthetic ARFIMA data
- Demonstrate model fitting and forecasting
- Show fractional differencing effects
- Compare different model orders
- Evaluate forecasting accuracy
- Generate diagnostic plots

## Performance Considerations

### Computational Complexity
- **Fractional differencing**: O(n²) for series of length n
- **Model fitting**: O(n × iterations) for optimization
- **Forecasting**: O(steps × max(p,q))

### Memory Usage
- **Model storage**: O(p + q) parameters
- **Fitted values**: O(n) for series of length n
- **Forecasts**: O(steps) for forecast horizon

### Optimization Tips
- Use smaller `max_p` and `max_q` for order estimation
- Consider using approximate methods for very long series
- Cache fractional differencing coefficients for repeated use

## Limitations and Assumptions

### Model Assumptions
- **Stationarity**: Series must be stationary after fractional differencing
- **Linearity**: Model assumes linear relationships
- **Gaussian innovations**: Assumes normally distributed errors
- **Constant parameters**: Parameters assumed constant over time

### Limitations
- **Computational cost**: Fractional differencing is computationally expensive
- **Parameter constraints**: d must be between 0 and 0.5 for stationarity
- **Model identification**: May be difficult to distinguish between similar models
- **Forecast uncertainty**: Does not provide confidence intervals

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Granger, C. W. J., & Joyeux, R. (1980). An introduction to long-memory time series models and fractional differencing. *Journal of Time Series Analysis*, 1(1), 15-29.

2. Hosking, J. R. M. (1981). Fractional differencing. *Biometrika*, 68(1), 165-176.

3. Sowell, F. (1992). Maximum likelihood estimation of stationary univariate fractionally integrated time series models. *Journal of Econometrics*, 53(1-3), 165-188.

4. Beran, J. (1994). *Statistics for long-memory processes*. Chapman & Hall.

## Contact

For questions or issues, please open an issue on the project repository.


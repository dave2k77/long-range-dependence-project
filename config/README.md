# Configuration System

This directory contains configuration files for the Long-Range Dependence Analysis project. The configuration system provides centralized management of all project settings, making it easy to customize behavior without modifying code.

## Configuration Files

### `data_config.yaml`
Contains settings for data loading, processing, and management:

- **Data Sources**: File formats, API settings, synthetic data generation
- **Processing**: Missing value handling, outlier detection, data transformation, stationarity testing
- **Quality Assessment**: Thresholds, distribution checks, time series validation
- **Storage**: Directory structure, file naming conventions, metadata management
- **Validation**: Schema validation, range checks, consistency validation
- **Performance**: Chunking, parallel processing, caching

### `analysis_config.yaml`
Contains settings for all analysis methods:

- **ARFIMA**: Model fitting parameters, model selection, diagnostics
- **DFA**: Scale configuration, detrending, fitting, quality checks
- **R/S**: Scale configuration, calculation methods, bias correction
- **MFDFA**: Scale configuration, moment orders, multifractal spectrum
- **Wavelet**: Wavelet type, scale generation, leaders, whittle estimation
- **Spectral**: Periodogram estimation, whittle MLE, periodogram regression
- **Performance**: Parallel processing, memory management, caching
- **Output**: Results storage, logging, validation

### `plot_config.yaml`
Contains settings for all visualization aspects:

- **General**: Style, fonts, colors, grid settings
- **Time Series**: Basic plots, multiple series, quality plots
- **Fractal**: DFA, R/S, MFDFA plot configurations
- **Wavelet**: Coefficients, leaders, whittle plots
- **Spectral**: Periodogram, whittle MLE, regression plots
- **Results**: Summary tables, comparison plots, correlation matrices
- **Layout**: Subplot spacing, legends, titles, axis labels
- **Output**: Save settings, display settings, quality settings
- **Accessibility**: Color blind friendly, high contrast, large fonts

## Usage

### Basic Usage

```python
from src.config_loader import get_config_value, get_dfa_config

# Get a specific configuration value
min_scale = get_config_value('analysis', 'dfa', 'scales', 'min_scale', default=10)

# Get entire configuration section
dfa_config = get_dfa_config()

# Use in analysis
from src.analysis.fractal_analysis import dfa
result = dfa(data, min_scale=min_scale, **dfa_config)
```

### Convenience Functions

The configuration loader provides several convenience functions:

```python
from src.config_loader import (
    get_data_config, get_analysis_config, get_plot_config,
    get_dfa_config, get_rs_config, get_wavelet_config,
    get_spectral_config, get_arfima_config,
    get_plot_style, get_figure_dpi, get_color_palette
)

# Get entire configurations
data_config = get_data_config()
analysis_config = get_analysis_config()
plot_config = get_plot_config()

# Get method-specific configurations
dfa_config = get_dfa_config()
rs_config = get_rs_config()
wavelet_config = get_wavelet_config()

# Get common plot settings
style = get_plot_style()
dpi = get_figure_dpi()
colors = get_color_palette()
```

### Nested Configuration Access

```python
from src.config_loader import get_config_value

# Access deeply nested values with defaults
freq_min = get_config_value('analysis', 'spectral', 'whittle_mle', 'freq_min', default=0.01)
freq_max = get_config_value('analysis', 'spectral', 'whittle_mle', 'freq_max', default=0.5)

# Access with type conversion
n_scales = get_config_value('analysis', 'dfa', 'scales', 'n_scales', default=20)
detrend_order = get_config_value('analysis', 'dfa', 'detrending', 'order', default=1)
```

## Configuration Validation

The system includes built-in validation:

```python
from src.config_loader import validate_all_configs

# Validate all configurations
validation_results = validate_all_configs()
print(validation_results)
# Output: {'data': True, 'analysis': True, 'plot': True}
```

## Reloading Configurations

To reload configurations after making changes:

```python
from src.config_loader import reload_configs

# Reload all configuration files
reload_configs()
```

## Example: Using Configuration in Analysis

```python
from src.config_loader import get_config_value
from src.analysis.fractal_analysis import dfa
import matplotlib.pyplot as plt

# Get DFA parameters from configuration
min_scale = get_config_value('analysis', 'dfa', 'scales', 'min_scale', default=10)
max_scale = get_config_value('analysis', 'dfa', 'scales', 'max_scale', default=None)
n_scales = get_config_value('analysis', 'dfa', 'scales', 'n_scales', default=20)
detrend_order = get_config_value('analysis', 'dfa', 'detrending', 'order', default=1)

# Run DFA analysis with configuration
result = dfa(data, min_scale=min_scale, max_scale=max_scale, 
             n_scales=n_scales, detrend_order=detrend_order)

# Get plot settings from configuration
figsize = get_config_value('plot', 'fractal', 'dfa', 'figsize', default=[12, 8])
scale_color = get_config_value('plot', 'fractal', 'dfa', 'scales', 'color', default='#1f77b4')
fit_color = get_config_value('plot', 'fractal', 'dfa', 'fitting', 'color', default='#ff7f0e')

# Create plot with configuration
plt.figure(figsize=figsize)
plt.loglog(result['scales'], result['fluctuations'], 'o', color=scale_color, label='Data')
plt.loglog(result['scales'], result['fit_line'], '--', color=fit_color, label='Fit')
plt.xlabel('Scale')
plt.ylabel('Fluctuation')
plt.legend()
plt.show()
```

## Example: Using Configuration in Data Processing

```python
from src.config_loader import get_config_value
from src.data_processing.preprocessing import TimeSeriesPreprocessor

# Get processing parameters from configuration
missing_method = get_config_value('data', 'processing', 'missing_values', 'method', default='interpolation')
outlier_method = get_config_value('data', 'processing', 'outliers', 'method', default='iqr')
outlier_threshold = get_config_value('data', 'processing', 'outliers', 'threshold', default=1.5)

# Create preprocessor with configuration
preprocessor = TimeSeriesPreprocessor(
    missing_values_method=missing_method,
    outlier_method=outlier_method,
    outlier_threshold=outlier_threshold
)

# Process data
cleaned_data = preprocessor.clean_time_series(data)
```

## Customizing Configurations

### Modifying Existing Settings

To modify a setting, simply edit the corresponding YAML file:

```yaml
# In analysis_config.yaml
dfa:
  scales:
    min_scale: 15  # Changed from 10
    n_scales: 25   # Changed from 20
```

### Adding New Settings

You can add new settings to any configuration file:

```yaml
# In data_config.yaml
processing:
  # ... existing settings ...
  new_feature:
    enabled: true
    parameter: 0.5
```

### Environment-Specific Configurations

For different environments (development, production, testing), you can:

1. Create environment-specific configuration files
2. Use environment variables to override settings
3. Use the `config_dir` parameter in `ConfigLoader`

```python
# Load environment-specific configuration
from src.config_loader import get_config_loader

if os.getenv('ENVIRONMENT') == 'production':
    config_loader = get_config_loader('config/production')
else:
    config_loader = get_config_loader('config/development')
```

## Best Practices

1. **Use Default Values**: Always provide sensible defaults when accessing configuration values
2. **Validate Configurations**: Use the validation functions to ensure configurations are correct
3. **Document Changes**: When modifying configurations, update this README
4. **Version Control**: Keep configuration files in version control
5. **Environment Separation**: Use different configurations for different environments
6. **Type Safety**: Be aware of data types when accessing configuration values

## Testing

Test the configuration system:

```bash
python scripts/test_config.py
```

This will run comprehensive tests on all configuration aspects and demonstrate practical usage.

## Dependencies

The configuration system requires:
- `PyYAML>=6.0` (for YAML file parsing)

Install with:
```bash
pip install PyYAML>=6.0
```

## Troubleshooting

### Common Issues

1. **Configuration Not Found**: Ensure the `config` directory exists and contains the YAML files
2. **YAML Syntax Errors**: Use a YAML validator to check syntax
3. **Missing Dependencies**: Install PyYAML if you get import errors
4. **Validation Failures**: Check that required keys are present in configuration files

### Debugging

Enable debug output by modifying the configuration loader:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When adding new configuration options:

1. Add the setting to the appropriate YAML file
2. Update the validation logic in `src/config_loader.py`
3. Add convenience functions if the setting is commonly used
4. Update this README with examples
5. Add tests to `scripts/test_config.py`

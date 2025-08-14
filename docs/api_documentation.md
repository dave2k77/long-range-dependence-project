# API Documentation

This document provides comprehensive API documentation for the Long-Range Dependence Analysis project, including all functions, classes, and modules.

**Status: ✅ Production Ready**  
**Last Updated: December 2024**

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Processing Module](#data-processing-module)
3. [Analysis Module](#analysis-module)
   - [Fractal Analysis](#fractal-analysis-srcanalysisfractal_analysispy)
   - [Higuchi Fractal Dimension Analysis](#higuchi-fractal-dimension-analysis-srcanalysishiguchi_analysispy)
   - [Wavelet Analysis](#wavelet-analysis-srcanalysiswavelet_analysispy)
   - [Spectral Analysis](#spectral-analysis-srcanalysisspectral_analysispy)
   - [Statistical Validation](#statistical-validation-srcanalysisstatistical_validationpy)
   - [Parallel Computation](#parallel-computation)
     - [JAX Parallel Analysis](#jax-parallel-analysis-srcanalysisjax_parallel_analysispy)
     - [Joblib Parallel Analysis](#joblib-parallel-analysis-srcanalysisjoblib_parallel_analysispy)
     - [Numba Parallel Analysis](#numba-parallel-analysis-srcanalysisnumba_parallel_analysispy)
4. [Statistical Validation Module](#statistical-validation-module)
5. [Visualization Module](#visualization-module)
   - [Time Series Plots](#time-series-plots-srcvisualisationtime_series_plotspy)
   - [Fractal Plots](#fractal-plots-srcvisualisationfractal_plotspy)
   - [Higuchi Plots](#higuchi-plots-srcvisualisationhiguchi_plotspy)
   - [Results Visualization](#results-visualization-srcvisualisationresults_visualisationpy)
   - [Validation Plots](#validation-plots-srcvisualisationvalidation_plotspy)
6. [Configuration Module](#configuration-module)
7. [Utility Functions](#utility-functions)
8. [Scripts](#scripts)
9. [Examples](#examples)

## Project Overview

The Long-Range Dependence Analysis project provides a comprehensive toolkit for analyzing time series data to detect and quantify long-range dependence properties. The project is organized into several key modules:

- **Data Processing**: Data loading, cleaning, and quality assessment
- **Analysis**: Implementation of various LRD analysis methods
- **Visualization**: Plotting and results presentation
- **Configuration**: Centralized settings management

## Data Processing Module

### Data Loader (`src/data_processing/data_loader.py`)

#### `DataLoader` Class

Main class for loading time series data from various sources.

```python
class DataLoader:
    """A comprehensive data loader for time series data."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress information, by default True
        """
```

##### Methods

###### `load_from_csv(file_path: str, **kwargs) -> pd.DataFrame`

Load data from a CSV file.

**Parameters:**
- `file_path` (str): Path to the CSV file
- `**kwargs`: Additional arguments passed to `pd.read_csv`

**Returns:**
- `pd.DataFrame`: Loaded data

**Example:**
```python
loader = DataLoader()
data = loader.load_from_csv('data/time_series.csv', index_col=0)
```

###### `load_from_excel(file_path: str, **kwargs) -> pd.DataFrame`

Load data from an Excel file.

**Parameters:**
- `file_path` (str): Path to the Excel file
- `**kwargs`: Additional arguments passed to `pd.read_excel`

**Returns:**
- `pd.DataFrame`: Loaded data

**Example:**
```python
data = loader.load_from_excel('data/time_series.xlsx', sheet_name='Sheet1')
```

###### `load_financial_data(symbols: List[str], start_date: str = None, end_date: str = None, period: str = "1y") -> Dict[str, pd.DataFrame]`

Load financial data using yfinance.

**Parameters:**
- `symbols` (List[str]): List of stock symbols
- `start_date` (str, optional): Start date in YYYY-MM-DD format
- `end_date` (str, optional): End date in YYYY-MM-DD format
- `period` (str, optional): Data period, by default "1y"

**Returns:**
- `Dict[str, pd.DataFrame]`: Dictionary mapping symbols to data

**Example:**
```python
data = loader.load_financial_data(['AAPL', 'GOOGL'], period='2y')
```

###### `load_synthetic_data(n: int = 1000, hurst: float = 0.7, seed: int = None) -> np.ndarray`

Generate synthetic fractional Brownian motion data.

**Parameters:**
- `n` (int, optional): Number of data points, by default 1000
- `hurst` (float, optional): Hurst exponent, by default 0.7
- `seed` (int, optional): Random seed, by default None

**Returns:**
- `np.ndarray`: Generated synthetic data

**Example:**
```python
data = loader.load_synthetic_data(n=2000, hurst=0.8, seed=42)
```

#### Convenience Functions

###### `load_data_from_file(file_path: str, **kwargs) -> pd.DataFrame`

Load data from a file, automatically detecting the format.

**Parameters:**
- `file_path` (str): Path to the data file
- `**kwargs`: Additional arguments for the loader

**Returns:**
- `pd.DataFrame`: Loaded data

**Example:**
```python
data = load_data_from_file('data/time_series.csv')
```

### Data Preprocessing (`src/data_processing/preprocessing.py`)

#### `TimeSeriesPreprocessor` Class

Class for cleaning and preprocessing time series data.

```python
class TimeSeriesPreprocessor:
    """A comprehensive preprocessor for time series data."""
    
    def __init__(self, missing_values_method: str = 'interpolation',
                 outlier_method: str = 'iqr', outlier_threshold: float = 1.5,
                 detrend_method: str = 'linear', detrend_order: int = 1):
        """
        Initialize the preprocessor.
        
        Parameters
        ----------
        missing_values_method : str, optional
            Method for handling missing values, by default 'interpolation'
        outlier_method : str, optional
            Method for outlier detection, by default 'iqr'
        outlier_threshold : float, optional
            Threshold for outlier detection, by default 1.5
        detrend_method : str, optional
            Method for detrending, by default 'linear'
        detrend_order : int, optional
            Order of polynomial detrending, by default 1
        """
```

##### Methods

###### `clean_time_series(data: Union[pd.Series, np.ndarray]) -> np.ndarray`

Clean time series data by handling missing values and outliers.

**Parameters:**
- `data` (Union[pd.Series, np.ndarray]): Input time series data

**Returns:**
- `np.ndarray`: Cleaned data

**Example:**
```python
preprocessor = TimeSeriesPreprocessor()
cleaned_data = preprocessor.clean_time_series(data)
```

###### `handle_missing_values(data: Union[pd.Series, np.ndarray]) -> np.ndarray`

Handle missing values in the data.

**Parameters:**
- `data` (Union[pd.Series, np.ndarray]): Input data with missing values

**Returns:**
- `np.ndarray`: Data with missing values handled

**Example:**
```python
data_no_missing = preprocessor.handle_missing_values(data)
```

###### `detect_outliers(data: Union[pd.Series, np.ndarray]) -> np.ndarray`

Detect outliers in the data.

**Parameters:**
- `data` (Union[pd.Series, np.ndarray]): Input data

**Returns:**
- `np.ndarray`: Boolean array indicating outliers

**Example:**
```python
outliers = preprocessor.detect_outliers(data)
print(f"Found {np.sum(outliers)} outliers")
```

###### `handle_outliers(data: Union[pd.Series, np.ndarray], method: str = 'winsorization') -> np.ndarray`

Handle outliers using specified method.

**Parameters:**
- `data` (Union[pd.Series, np.ndarray]): Input data
- `method` (str, optional): Method for handling outliers, by default 'winsorization'

**Returns:**
- `np.ndarray`: Data with outliers handled

**Example:**
```python
data_no_outliers = preprocessor.handle_outliers(data, method='removal')
```

###### `test_stationarity(data: Union[pd.Series, np.ndarray]) -> Dict[str, Dict[str, Any]]`

Test stationarity using multiple tests.

**Parameters:**
- `data` (Union[pd.Series, np.ndarray]): Input data

**Returns:**
- `Dict[str, Dict[str, Any]]`: Results from stationarity tests

**Example:**
```python
stationarity_results = preprocessor.test_stationarity(data)
for test, result in stationarity_results.items():
    print(f"{test}: {'Stationary' if result['stationary'] else 'Non-stationary'}")
```

###### `make_stationary(data: Union[pd.Series, np.ndarray], method: str = 'differencing') -> np.ndarray`

Make non-stationary data stationary.

**Parameters:**
- `data` (Union[pd.Series, np.ndarray]): Input data
- `method` (str, optional): Method for making data stationary, by default 'differencing'

**Returns:**
- `np.ndarray`: Stationary data

**Example:**
```python
stationary_data = preprocessor.make_stationary(data, method='differencing')
```

#### Convenience Functions

###### `clean_time_series(data: Union[pd.Series, np.ndarray], **kwargs) -> np.ndarray`

Convenience function for cleaning time series data.

**Parameters:**
- `data` (Union[pd.Series, np.ndarray]): Input data
- `**kwargs`: Arguments passed to TimeSeriesPreprocessor

**Returns:**
- `np.ndarray`: Cleaned data

**Example:**
```python
cleaned_data = clean_time_series(data, outlier_method='zscore', outlier_threshold=3.0)
```

### Data Quality Assessment (`src/data_processing/quality_check.py`)

#### `DataQualityChecker` Class

Class for assessing the quality of time series data.

```python
class DataQualityChecker:
    """A comprehensive data quality checker for time series data."""
    
    def __init__(self, missing_threshold: float = 0.1, outlier_threshold: float = 0.05):
        """
        Initialize the quality checker.
        
        Parameters
        ----------
        missing_threshold : float, optional
            Maximum allowed proportion of missing values, by default 0.1
        outlier_threshold : float, optional
            Maximum allowed proportion of outliers, by default 0.05
        """
```

##### Methods

###### `assess_data_quality(data: Union[pd.Series, np.ndarray]) -> Dict[str, Any]`

Assess the overall quality of the data.

**Parameters:**
- `data` (Union[pd.Series, np.ndarray]): Input data

**Returns:**
- `Dict[str, Any]`: Comprehensive quality assessment

**Example:**
```python
checker = DataQualityChecker()
quality_report = checker.assess_data_quality(data)
print(f"Overall quality score: {quality_report['overall_score']:.2f}")
```

###### `check_completeness(data: Union[pd.Series, np.ndarray]) -> Dict[str, Any]`

Check data completeness.

**Parameters:**
- `data` (Union[pd.Series, np.ndarray]): Input data

**Returns:**
- `Dict[str, Any]`: Completeness assessment

**Example:**
```python
completeness = checker.check_completeness(data)
print(f"Missing values: {completeness['missing_count']} ({completeness['missing_proportion']:.2%})")
```

###### `check_consistency(data: Union[pd.Series, np.ndarray]) -> Dict[str, Any]`

Check data consistency.

**Parameters:**
- `data` (Union[pd.Series, np.ndarray]): Input data

**Returns:**
- `Dict[str, Any]`: Consistency assessment

**Example:**
```python
consistency = checker.check_consistency(data)
print(f"Data type: {consistency['data_type']}")
print(f"Memory usage: {consistency['memory_usage']:.2f} MB")
```

#### Convenience Functions

###### `assess_data_quality(data: Union[pd.Series, np.ndarray], **kwargs) -> Dict[str, Any]`

Convenience function for assessing data quality.

**Parameters:**
- `data` (Union[pd.Series, np.ndarray]): Input data
- `**kwargs`: Arguments passed to DataQualityChecker

**Returns:**
- `Dict[str, Any]`: Quality assessment

**Example:**
```python
quality_report = assess_data_quality(data, missing_threshold=0.05)
```

### Data Management (`src/data_processing/data_manager.py`)

#### `DataManager` Class

Class for managing data storage, organization, and metadata.

```python
class DataManager:
    """A comprehensive data manager for organizing and saving time series data."""
    
    def __init__(self, data_root: str = "data", verbose: bool = True):
        """
        Initialize the data manager.
        
        Parameters
        ----------
        data_root : str, optional
            Root directory for data storage, by default "data"
        verbose : bool, optional
            Whether to print progress information, by default True
        """
```

##### Methods

###### `save_synthetic_data(data: Union[Dict[str, np.ndarray], pd.DataFrame, np.ndarray], name: str = None, data_type: str = "synthetic", description: str = None, parameters: Dict[str, Any] = None) -> str`

Save synthetic data with metadata.

**Parameters:**
- `data`: Data to save
- `name` (str, optional): Name for the dataset
- `data_type` (str, optional): Type of data, by default "synthetic"
- `description` (str, optional): Description of the data
- `parameters` (Dict[str, Any], optional): Generation parameters

**Returns:**
- `str`: Path to saved data file

**Example:**
```python
manager = DataManager()
file_path = manager.save_synthetic_data(
    data, 
    name="fractional_noise", 
    description="Synthetic fractional noise with H=0.7",
    parameters={'hurst': 0.7, 'n': 1000}
)
```

###### `save_processed_data(data: Union[pd.DataFrame, Dict[str, np.ndarray]], original_name: str, processing_steps: List[str], processing_parameters: Dict[str, Any] = None) -> str`

Save processed data with processing history.

**Parameters:**
- `data`: Processed data to save
- `original_name` (str): Name of the original dataset
- `processing_steps` (List[str]): List of processing steps applied
- `processing_parameters` (Dict[str, Any], optional): Processing parameters

**Returns:**
- `str`: Path to saved data file

**Example:**
```python
file_path = manager.save_processed_data(
    cleaned_data,
    original_name="raw_financial_data",
    processing_steps=["missing_value_handling", "outlier_removal", "detrending"],
    processing_parameters={'outlier_threshold': 1.5}
)
```

###### `setup_complete_dataset(n_synthetic: int = 1000, financial_symbols: List[str] = None, seed: int = 42) -> Dict[str, Any]`

Set up a complete dataset collection.

**Parameters:**
- `n_synthetic` (int, optional): Number of synthetic datasets, by default 1000
- `financial_symbols` (List[str], optional): Financial symbols to download, by default None
- `seed` (int, optional): Random seed, by default 42

**Returns:**
- `Dict[str, Any]`: Summary of dataset setup

**Example:**
```python
summary = manager.setup_complete_dataset(
    n_synthetic=2000,
    financial_symbols=['AAPL', 'GOOGL', 'MSFT'],
    seed=42
)
print(f"Created {summary['total_datasets']} datasets")
```

#### Convenience Functions

###### `setup_project_data(data_root: str = "data", n_synthetic: int = 1000, financial_symbols: List[str] = None, seed: int = 42) -> Dict[str, Any]`

Convenience function for setting up project data.

**Parameters:**
- `data_root` (str, optional): Root directory for data, by default "data"
- `n_synthetic` (int, optional): Number of synthetic datasets, by default 1000
- `financial_symbols` (List[str], optional): Financial symbols, by default None
- `seed` (int, optional): Random seed, by default 42

**Returns:**
- `Dict[str, Any]`: Dataset setup summary

**Example:**
```python
summary = setup_project_data(
    data_root="my_data",
    n_synthetic=1500,
    financial_symbols=['TSLA', 'AMZN']
)
```

## Analysis Module

The analysis module provides implementations of various long-range dependence analysis methods.

### Parallel Computation

The project includes three parallel computation implementations, with JAX providing the highest performance for GPU/TPU acceleration:

#### JAX Parallel Analysis (`src/analysis/jax_parallel_analysis.py`)

**For GPU/TPU acceleration and maximum performance** - Provides JAX-accelerated parallel computation with hardware acceleration.

```python
from src.analysis.jax_parallel_analysis import jax_parallel_analysis, create_jax_config

# Create configuration for GPU acceleration
config = create_jax_config(use_gpu=True, batch_size=64, num_parallel=8)

# High-performance parallel analysis with JAX
datasets = {'data1': time_series1, 'data2': time_series2}
results = jax_parallel_analysis(
    datasets=datasets,
    methods=['dfa', 'higuchi'],
    config=config
)
```

**Key Features:**
- ⚡ **Maximum performance** - GPU/TPU acceleration
- 🔄 **JIT compilation** - Automatic optimization
- 📊 **Vectorized operations** - Batch processing
- 🧠 **Automatic differentiation** - For optimization
- 🔧 **Memory efficient** - Optimized for large datasets

**Production Status:**
- ✅ **All tests passing** - 32/32 JAX tests
- ✅ **Error handling** - Comprehensive fallback mechanisms
- ✅ **Documentation** - Complete API and usage guides

#### Joblib Parallel Analysis (`src/analysis/joblib_parallel_analysis.py`)

**Recommended for most use cases** - Provides stable, reliable parallel processing with excellent error handling.

```python
from src.analysis.joblib_parallel_analysis import joblib_parallel_analysis, create_joblib_config

# Create configuration
config = create_joblib_config(n_jobs=4, verbose=1)

# Analyze multiple datasets in parallel
datasets = {'data1': time_series1, 'data2': time_series2}
results = joblib_parallel_analysis(
    datasets=datasets,
    methods=['dfa', 'higuchi', 'rs'],
    config=config
)
```

**Key Features:**
- ✅ **Stable and reliable** - No compilation issues
- ✅ **Easy debugging** - Excellent error handling
- ✅ **Progress tracking** - Built-in logging and progress bars
- ✅ **Automatic memory management** - No memory leaks
- ✅ **Production-ready** - Mature and well-tested

#### Numba Parallel Analysis (`src/analysis/numba_parallel_analysis.py`)

**For performance-critical applications** - Provides JIT-compiled high-performance computation.

```python
from src.analysis.numba_parallel_analysis import numba_parallel_analysis, create_numba_config

# Create configuration
config = create_numba_config(num_workers=4, use_jit=True)

# High-performance parallel analysis
results = numba_parallel_analysis(
    datasets=datasets,
    methods=['dfa', 'higuchi'],
    config=config
)
```

**Key Features:**
- ⚡ **High performance** - JIT compilation for speed
- ✅ **Stable** - Good error messages and debugging
- 🔧 **Easy to implement** - Simple @jit decorators
- 📈 **Scalable** - Good for CPU-bound tasks

**Comparison of Parallel Methods:**
- **JAX**: Maximum performance, GPU/TPU acceleration, production-ready
- **Joblib**: Stable, reliable, easy to debug, recommended for most users
- **Numba**: High performance, JIT compilation, good for CPU-bound tasks

### Fractal Analysis (`src/analysis/fractal_analysis.py`)

#### `DFAModel` Class

Class for Detrended Fluctuation Analysis.

```python
class DFAModel:
    """Detrended Fluctuation Analysis model."""
    
    def __init__(self, min_scale: int = 10, max_scale: int = None,
                 n_scales: int = 20, detrend_order: int = 1):
        """
        Initialize DFA model.
        
        Parameters
        ----------
        min_scale : int, optional
            Minimum scale for analysis, by default 10
        max_scale : int, optional
            Maximum scale for analysis, by default None
        n_scales : int, optional
            Number of scales to use, by default 20
        detrend_order : int, optional
            Order of polynomial detrending, by default 1
        """
```

##### Methods

###### `fit(data: np.ndarray) -> Dict[str, Any]`

Fit the DFA model to data.

**Parameters:**
- `data` (np.ndarray): Input time series data

**Returns:**
- `Dict[str, Any]`: DFA analysis results

**Example:**
```python
dfa_model = DFAModel(min_scale=10, n_scales=25, detrend_order=2)
results = dfa_model.fit(data)
print(f"Hurst exponent: {results['hurst']:.4f}")
```

#### `RSModel` Class

Class for Rescaled Range Analysis.

```python
class RSModel:
    """Rescaled Range Analysis model."""
    
    def __init__(self, min_scale: int = 10, max_scale: int = None,
                 n_scales: int = 20, bias_correction: bool = True):
        """
        Initialize R/S model.
        
        Parameters
        ----------
        min_scale : int, optional
            Minimum scale for analysis, by default 10
        max_scale : int, optional
            Maximum scale for analysis, by default None
        n_scales : int, optional
            Number of scales to use, by default 20
        bias_correction : bool, optional
            Whether to apply bias correction, by default True
        """
```

##### Methods

###### `fit(data: np.ndarray) -> Dict[str, Any]`

Fit the R/S model to data.

**Parameters:**
- `data` (np.ndarray): Input time series data

**Returns:**
- `Dict[str, Any]`: R/S analysis results

**Example:**
```python
rs_model = RSModel(min_scale=15, n_scales=30, bias_correction=True)
results = rs_model.fit(data)
print(f"Hurst exponent: {results['hurst']:.4f}")
```

#### `MFDFAModel` Class

Class for Multifractal Detrended Fluctuation Analysis.

```python
class MFDFAModel:
    """Multifractal Detrended Fluctuation Analysis model."""
    
    def __init__(self, min_scale: int = 10, max_scale: int = None,
                 n_scales: int = 20, q_min: float = -5, q_max: float = 5,
                 q_step: float = 0.5, detrend_order: int = 1):
        """
        Initialize MFDFA model.
        
        Parameters
        ----------
        min_scale : int, optional
            Minimum scale for analysis, by default 10
        max_scale : int, optional
            Maximum scale for analysis, by default None
        n_scales : int, optional
            Number of scales to use, by default 20
        q_min : float, optional
            Minimum moment order, by default -5
        q_max : float, optional
            Maximum moment order, by default 5
        q_step : float, optional
            Step size for moment orders, by default 0.5
        detrend_order : int, optional
            Order of polynomial detrending, by default 1
        """
```

##### Methods

###### `fit(data: np.ndarray) -> Dict[str, Any]`

Fit the MFDFA model to data.

**Parameters:**
- `data` (np.ndarray): Input time series data

**Returns:**
- `Dict[str, Any]`: MFDFA analysis results

**Example:**
```python
mfdfa_model = MFDFAModel(
    min_scale=10, 
    n_scales=25, 
    q_min=-3, 
    q_max=3, 
    q_step=0.5
)
results = mfdfa_model.fit(data)
print(f"Generalized Hurst exponents: {results['h_q']}")
```

#### Convenience Functions

###### `dfa(data: np.ndarray, **kwargs) -> Dict[str, Any]`

Convenience function for DFA analysis.

**Parameters:**
- `data` (np.ndarray): Input data
- `**kwargs`: Arguments passed to DFAModel

**Returns:**
- `Dict[str, Any]`: DFA results

**Example:**
```python
results = dfa(data, min_scale=10, n_scales=20, detrend_order=1)
```

###### `rescaled_range(data: np.ndarray, **kwargs) -> Dict[str, Any]`

Convenience function for R/S analysis.

**Parameters:**
- `data` (np.ndarray): Input data
- `**kwargs`: Arguments passed to RSModel

**Returns:**
- `Dict[str, Any]`: R/S results

**Example:**
```python
results = rescaled_range(data, min_scale=15, n_scales=25, bias_correction=True)
```

###### `mfdfa(data: np.ndarray, **kwargs) -> Dict[str, Any]`

Convenience function for MFDFA analysis.

**Parameters:**
- `data` (np.ndarray): Input data
- `**kwargs`: Arguments passed to MFDFAModel

**Returns:**
- `Dict[str, Any]`: MFDFA results

**Example:**
```python
results = mfdfa(data, min_scale=10, n_scales=20, q_min=-2, q_max=2)
```

### Higuchi Fractal Dimension Analysis (`src/analysis/higuchi_analysis.py`)

The Higuchi analysis module provides comprehensive fractal dimension estimation using Higuchi's method.

#### Functions

##### `higuchi_fractal_dimension(y, k_max=None, k_min=2, optimize_k=True, method='linear')`

Calculate Higuchi's fractal dimension of a time series.

**Parameters:**
- `y` (np.ndarray): Time series data (1D array)
- `k_max` (Optional[int]): Maximum value of k (time interval). If None, will be set to len(y)//4
- `k_min` (int): Minimum value of k (time interval). Must be >= 2
- `optimize_k` (bool): Whether to optimize the k range for better linearity
- `method` (str): Method for calculating the fractal dimension: 'linear' or 'robust'

**Returns:**
- `Tuple[np.ndarray, np.ndarray, HiguchiSummary]`: k_values, l_values, and summary object

**Example:**
```python
from analysis.higuchi_analysis import higuchi_fractal_dimension

# Analyze time series
k_values, l_values, summary = higuchi_fractal_dimension(
    time_series, k_max=100, k_min=2, optimize_k=True
)

print(f"Fractal Dimension: {summary.fractal_dimension:.4f}")
print(f"R²: {summary.r_squared:.4f}")
```

##### `estimate_higuchi_dimension(y, k_max=None, k_min=2, optimize_k=True)`

Convenience function to get just the fractal dimension estimate.

**Parameters:**
- `y` (np.ndarray): Time series data
- `k_max` (Optional[int]): Maximum k value
- `k_min` (int): Minimum k value
- `optimize_k` (bool): Whether to optimize k range

**Returns:**
- `float`: Estimated fractal dimension

##### `higuchi_analysis_batch(time_series_list, names=None, k_max=None, k_min=2, optimize_k=True)`

Perform Higuchi analysis on multiple time series.

**Parameters:**
- `time_series_list` (List[np.ndarray]): List of time series arrays
- `names` (Optional[List[str]]): Names for the time series
- `k_max` (Optional[int]): Maximum k value
- `k_min` (int): Minimum k value
- `optimize_k` (bool): Whether to optimize k range

**Returns:**
- `Dict[str, HiguchiSummary]`: Dictionary mapping names to HiguchiSummary objects

##### `validate_higuchi_results(summary, min_r2=0.8, max_std_error=0.1)`

Validate Higuchi analysis results.

**Parameters:**
- `summary` (HiguchiSummary): Higuchi analysis results
- `min_r2` (float): Minimum acceptable R² value
- `max_std_error` (float): Maximum acceptable standard error

**Returns:**
- `Dict[str, Any]`: Validation results with quality score and issues

#### HiguchiSummary Dataclass

Container for Higuchi fractal dimension analysis results.

**Attributes:**
- `fractal_dimension` (float): Estimated fractal dimension
- `std_error` (float): Standard error of the estimate
- `r_squared` (float): R² value of the regression
- `p_value` (float): P-value of the regression
- `k_values` (np.ndarray): Array of k values used
- `l_values` (np.ndarray): Array of corresponding L(k) values
- `log_k` (np.ndarray): Log of k values
- `log_l` (np.ndarray): Log of L(k) values
- `slope` (float): Slope of the regression line
- `intercept` (float): Intercept of the regression line
- `residuals` (np.ndarray): Regression residuals
- `k_range` (Tuple[int, int]): Range of k values used
- `n_points` (int): Number of points in the time series
- `method` (str): Analysis method used
- `additional_info` (Optional[Dict[str, Any]]): Additional information

### Wavelet Analysis (`src/analysis/wavelet_analysis.py`)

#### `WaveletModel` Class

Class for wavelet-based analysis.

```python
class WaveletModel:
    """Wavelet analysis model for long-range dependence."""
    
    def __init__(self, wavelet_type: str = 'db4', min_scale: int = 2,
                 max_scale: int = None, n_scales: int = 10):
        """
        Initialize wavelet model.
        
        Parameters
        ----------
        wavelet_type : str, optional
            Type of wavelet to use, by default 'db4'
        min_scale : int, optional
            Minimum scale for analysis, by default 2
        max_scale : int, optional
            Maximum scale for analysis, by default None
        n_scales : int, optional
            Number of scales to use, by default 10
        """
```

##### Methods

###### `fit(data: np.ndarray) -> Dict[str, Any]`

Fit the wavelet model to data.

**Parameters:**
- `data` (np.ndarray): Input time series data

**Returns:**
- `Dict[str, Any]`: Wavelet analysis results

**Example:**
```python
wavelet_model = WaveletModel(wavelet_type='db6', min_scale=2, n_scales=15)
results = wavelet_model.fit(data)
```

#### Convenience Functions

###### `wavelet_leaders_estimation(data: np.ndarray, **kwargs) -> Dict[str, Any]`

Estimate Hurst exponent using wavelet leaders.

**Parameters:**
- `data` (np.ndarray): Input data
- `**kwargs`: Arguments passed to WaveletModel

**Returns:**
- `Dict[str, Any]`: Wavelet leaders results

**Example:**
```python
results = wavelet_leaders_estimation(
    data, 
    wavelet_type='db4', 
    min_scale=2, 
    n_scales=12
)
```

###### `wavelet_whittle_estimation(data: np.ndarray, **kwargs) -> Dict[str, Any]`

Estimate Hurst exponent using wavelet Whittle method.

**Parameters:**
- `data` (np.ndarray): Input data
- `**kwargs`: Arguments passed to WaveletModel

**Returns:**
- `Dict[str, Any]`: Wavelet Whittle results

**Example:**
```python
results = wavelet_whittle_estimation(
    data, 
    wavelet_type='db4', 
    freq_min=0.01, 
    freq_max=0.5
)
```

### Spectral Analysis (`src/analysis/spectral_analysis.py`)

#### `SpectralModel` Class

Class for spectral analysis methods.

```python
class SpectralModel:
    """Spectral analysis model for long-range dependence."""
    
    def __init__(self, method: str = 'welch', window: str = 'hann',
                 nperseg: int = None, noverlap: int = None):
        """
        Initialize spectral model.
        
        Parameters
        ----------
        method : str, optional
            Method for periodogram estimation, by default 'welch'
        window : str, optional
            Window function to use, by default 'hann'
        nperseg : int, optional
            Length of each segment, by default None
        noverlap : int, optional
            Number of points to overlap, by default None
        """
```

##### Methods

###### `fit(data: np.ndarray) -> Dict[str, Any]`

Fit the spectral model to data.

**Parameters:**
- `data` (np.ndarray): Input time series data

**Returns:**
- `Dict[str, Any]`: Spectral analysis results

**Example:**
```python
spectral_model = SpectralModel(method='welch', window='hamming')
results = spectral_model.fit(data)
```

#### Convenience Functions

###### `periodogram_estimation(data: np.ndarray, **kwargs) -> Dict[str, Any]`

Estimate periodogram of the data.

**Parameters:**
- `data` (np.ndarray): Input data
- `**kwargs`: Arguments passed to SpectralModel

**Returns:**
- `Dict[str, Any]`: Periodogram results

**Example:**
```python
results = periodogram_estimation(
    data, 
    method='welch', 
    window='hann', 
    nperseg=256
)
```

###### `whittle_mle(data: np.ndarray, **kwargs) -> Dict[str, Any]`

Estimate Hurst exponent using Whittle maximum likelihood.

**Parameters:**
- `data` (np.ndarray): Input data
- `**kwargs`: Arguments for Whittle estimation

**Returns:**
- `Dict[str, Any]`: Whittle MLE results

**Example:**
```python
results = whittle_mle(
    data, 
    freq_min=0.01, 
    freq_max=0.5
)
```

### ARFIMA Analysis (`src/analysis/arfima_analysis.py`)

#### `ARFIMAModel` Class

Class for ARFIMA modeling.

```python
class ARFIMAModel:
    """ARFIMA model for long-range dependence."""
    
    def __init__(self, max_p: int = 3, max_d: float = 2, max_q: int = 3,
                 method: str = 'css-mle', optimizer: str = 'lbfgs'):
        """
        Initialize ARFIMA model.
        
        Parameters
        ----------
        max_p : int, optional
            Maximum AR order, by default 3
        max_d : float, optional
            Maximum differencing order, by default 2
        max_q : int, optional
            Maximum MA order, by default 3
        method : str, optional
            Estimation method, by default 'css-mle'
        optimizer : str, optional
            Optimization method, by default 'lbfgs'
        """
```

##### Methods

###### `fit(data: np.ndarray) -> Dict[str, Any]`

Fit the ARFIMA model to data.

**Parameters:**
- `data` (np.ndarray): Input time series data

**Returns:**
- `Dict[str, Any]`: ARFIMA model results

**Example:**
```python
arfima_model = ARFIMAModel(max_p=2, max_d=1.5, max_q=2)
results = arfima_model.fit(data)
print(f"Best model: ARFIMA{results['best_order']}")
```

###### `diagnose(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]`

Run diagnostic tests on fitted model.

**Parameters:**
- `results` (Dict[str, Any]): Results from model fitting

**Returns:**
- `Dict[str, Dict[str, Any]]`: Diagnostic test results

**Example:**
```python
diagnostics = arfima_model.diagnose(results)
for test, result in diagnostics.items():
    print(f"{test}: {'Pass' if result['passed'] else 'Fail'}")
```

## Statistical Validation Module

The Statistical Validation Module provides comprehensive tools for validating long-range dependence analysis results through hypothesis testing, bootstrap analysis, Monte Carlo significance tests, and cross-validation.

### Statistical Validator (`src/analysis/statistical_validation.py`)

#### `StatisticalValidator` Class

Main class for comprehensive statistical validation of long-range dependence analysis.

```python
class StatisticalValidator:
    """Comprehensive statistical validation for long-range dependence analysis."""
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the statistical validator.
        
        Parameters
        ----------
        random_state : Optional[int]
            Random seed for reproducibility
        """
```

##### Methods

###### `test_lrd_hypothesis(y: np.ndarray, method: str = 'dfa', alpha: float = 0.05, h0_hurst: float = 0.5, alternative: str = 'greater') -> HypothesisTestResult`

Test the hypothesis of long-range dependence vs. no long-range dependence.

**Parameters:**
- `y` (np.ndarray): Time series data
- `method` (str): Analysis method ('dfa', 'rs', 'wavelet', 'spectral')
- `alpha` (float): Significance level
- `h0_hurst` (float): Null hypothesis Hurst exponent (default: 0.5 for no LRD)
- `alternative` (str): Alternative hypothesis ('greater', 'less', 'two-sided')

**Returns:**
- `HypothesisTestResult`: Test results and decision

**Example:**
```python
validator = StatisticalValidator(random_state=42)
result = validator.test_lrd_hypothesis(
    data, 
    method='dfa', 
    alpha=0.05, 
    h0_hurst=0.5,
    alternative='greater'
)
print(f"Decision: {result.decision}")
print(f"p-value: {result.p_value:.4f}")
```

###### `bootstrap_validation(y: np.ndarray, method: str = 'dfa', n_bootstrap: int = 1000, confidence_level: float = 0.95, block_size: Optional[int] = None) -> BootstrapResult`

Perform bootstrap validation for Hurst exponent estimation.

**Parameters:**
- `y` (np.ndarray): Time series data
- `method` (str): Analysis method ('dfa', 'rs', 'wavelet', 'spectral')
- `n_bootstrap` (int): Number of bootstrap samples
- `confidence_level` (float): Confidence level for intervals
- `block_size` (Optional[int]): Block size for block bootstrap (None for iid bootstrap)

**Returns:**
- `BootstrapResult`: Bootstrap analysis results

**Example:**
```python
result = validator.bootstrap_validation(
    data, 
    method='dfa', 
    n_bootstrap=1000,
    confidence_level=0.95,
    block_size=50  # Block bootstrap for time series
)
print(f"CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
```

###### `monte_carlo_significance_test(y: np.ndarray, method: str = 'dfa', n_simulations: int = 1000, alpha: float = 0.05, null_model: str = 'iid') -> MonteCarloResult`

Perform Monte Carlo significance test for long-range dependence.

**Parameters:**
- `y` (np.ndarray): Time series data
- `method` (str): Analysis method ('dfa', 'rs', 'wavelet', 'spectral')
- `n_simulations` (int): Number of Monte Carlo simulations
- `alpha` (float): Significance level
- `null_model` (str): Null model type ('iid', 'ar1', 'ma1')

**Returns:**
- `MonteCarloResult`: Monte Carlo test results

**Example:**
```python
result = validator.monte_carlo_significance_test(
    data, 
    method='dfa', 
    n_simulations=1000,
    alpha=0.05,
    null_model='iid'
)
print(f"Decision: {result.decision}")
print(f"p-value: {result.p_value:.4f}")
```

###### `split_sample_validation(y: np.ndarray, method: str = 'dfa', n_folds: int = 5, test_size: float = 0.2) -> CrossValidationResult`

Perform split-sample cross-validation for Hurst exponent estimation.

**Parameters:**
- `y` (np.ndarray): Time series data
- `method` (str): Analysis method ('dfa', 'rs', 'wavelet', 'spectral')
- `n_folds` (int): Number of cross-validation folds
- `test_size` (float): Proportion of data for testing

**Returns:**
- `CrossValidationResult`: Cross-validation results

**Example:**
```python
result = validator.split_sample_validation(
    data, 
    method='dfa', 
    n_folds=5,
    test_size=0.2
)
print(f"Mean Hurst: {result.mean_hurst:.3f}")
print(f"Stability Score: {result.stability_score:.3f}")
```

###### `robustness_test(y: np.ndarray, methods: List[str] = None, parameter_ranges: Dict[str, List] = None) -> Dict[str, Any]`

Perform robustness testing across different methods and parameters.

**Parameters:**
- `y` (np.ndarray): Time series data
- `methods` (List[str]): List of analysis methods to test
- `parameter_ranges` (Dict[str, List]): Parameter ranges to test for each method

**Returns:**
- `Dict[str, Any]`: Robustness test results

**Example:**
```python
results = validator.robustness_test(
    data,
    methods=['dfa', 'rs'],
    parameter_ranges={
        'dfa': {'order': [1, 2, 3]},
        'rs': {'min_scale': [4, 8, 16]}
    }
)
```

### Convenience Functions

#### `test_lrd_hypothesis(y: np.ndarray, method: str = 'dfa', alpha: float = 0.05, h0_hurst: float = 0.5, alternative: str = 'greater', random_state: Optional[int] = None) -> HypothesisTestResult`

Convenience function for testing LRD hypothesis.

**Example:**
```python
result = test_lrd_hypothesis(
    data, 
    method='dfa', 
    alpha=0.05,
    random_state=42
)
```

#### `bootstrap_confidence_interval(y: np.ndarray, method: str = 'dfa', n_bootstrap: int = 1000, confidence_level: float = 0.95, block_size: Optional[int] = None, random_state: Optional[int] = None) -> BootstrapResult`

Convenience function for bootstrap confidence intervals.

**Example:**
```python
result = bootstrap_confidence_interval(
    data, 
    method='dfa', 
    n_bootstrap=1000,
    block_size=50,
    random_state=42
)
```

#### `monte_carlo_test(y: np.ndarray, method: str = 'dfa', n_simulations: int = 1000, alpha: float = 0.05, null_model: str = 'iid', random_state: Optional[int] = None) -> MonteCarloResult`

Convenience function for Monte Carlo significance test.

**Example:**
```python
result = monte_carlo_test(
    data, 
    method='dfa', 
    n_simulations=1000,
    null_model='iid',
    random_state=42
)
```

#### `cross_validate_lrd(y: np.ndarray, method: str = 'dfa', n_folds: int = 5, test_size: float = 0.2, random_state: Optional[int] = None) -> CrossValidationResult`

Convenience function for cross-validation.

**Example:**
```python
result = cross_validate_lrd(
    data, 
    method='dfa', 
    n_folds=5,
    random_state=42
)
```

#### `comprehensive_validation(y: np.ndarray, methods: List[str] = None, alpha: float = 0.05, n_bootstrap: int = 1000, n_simulations: int = 1000, random_state: Optional[int] = None) -> Dict[str, Any]`

Perform comprehensive statistical validation.

**Example:**
```python
results = comprehensive_validation(
    data,
    methods=['dfa', 'rs', 'wavelet', 'spectral'],
    alpha=0.05,
    n_bootstrap=1000,
    n_simulations=1000,
    random_state=42
)
```

### Result Classes

#### `HypothesisTestResult`

Container for hypothesis test results.

**Attributes:**
- `test_name` (str): Name of the test
- `null_hypothesis` (str): Null hypothesis description
- `alternative_hypothesis` (str): Alternative hypothesis description
- `test_statistic` (float): Test statistic value
- `p_value` (float): p-value
- `critical_value` (float): Critical value
- `significance_level` (float): Significance level
- `decision` (str): Test decision ('reject' or 'fail_to_reject')
- `effect_size` (Optional[float]): Effect size
- `confidence_interval` (Optional[Tuple[float, float]]): Confidence interval
- `additional_info` (Optional[Dict[str, Any]]): Additional information

#### `BootstrapResult`

Container for bootstrap analysis results.

**Attributes:**
- `original_estimate` (float): Original Hurst exponent estimate
- `bootstrap_estimates` (np.ndarray): Array of bootstrap estimates
- `mean_estimate` (float): Mean of bootstrap estimates
- `std_estimate` (float): Standard deviation of bootstrap estimates
- `confidence_interval` (Tuple[float, float]): Confidence interval
- `confidence_level` (float): Confidence level
- `bias` (float): Bootstrap bias
- `standard_error` (float): Standard error
- `additional_stats` (Optional[Dict[str, Any]]): Additional statistics

#### `MonteCarloResult`

Container for Monte Carlo significance test results.

**Attributes:**
- `test_statistic` (float): Observed test statistic
- `null_distribution` (np.ndarray): Null distribution
- `p_value` (float): p-value
- `significance_level` (float): Significance level
- `n_simulations` (int): Number of simulations
- `decision` (str): Test decision
- `effect_size` (float): Effect size
- `power` (Optional[float]): Statistical power

#### `CrossValidationResult`

Container for cross-validation results.

**Attributes:**
- `method` (str): Method name
- `n_folds` (int): Number of folds
- `hurst_estimates` (List[float]): Hurst estimates from each fold
- `confidence_intervals` (List[Tuple[float, float]]): Confidence intervals
- `mean_hurst` (float): Mean Hurst exponent
- `std_hurst` (float): Standard deviation of Hurst estimates
- `cv_score` (float): Coefficient of variation
- `stability_score` (float): Stability score
- `additional_metrics` (Optional[Dict[str, Any]]): Additional metrics

## Visualization Module

### Time Series Plots (`src/visualisation/time_series_plots.py`)

#### `plot_time_series(data: Union[np.ndarray, pd.Series], **kwargs) -> Tuple[plt.Figure, plt.Axes]`

Plot time series data.

**Parameters:**
- `data`: Time series data to plot
- `**kwargs`: Additional plotting arguments

**Returns:**
- `Tuple[plt.Figure, plt.Axes]`: Figure and axes objects

**Example:**
```python
fig, ax = plot_time_series(
    data, 
    title="Time Series Data",
    xlabel="Time",
    ylabel="Value",
    figsize=(12, 6)
)
plt.show()
```

#### `plot_multiple_time_series(data_dict: Dict[str, Union[np.ndarray, pd.Series]], **kwargs) -> plt.Figure`

Plot multiple time series on the same figure.

**Parameters:**
- `data_dict`: Dictionary mapping names to time series data
- `**kwargs`: Additional plotting arguments

**Returns:**
- `plt.Figure`: Figure object

**Example:**
```python
data_dict = {
    'Series 1': data1,
    'Series 2': data2,
    'Series 3': data3
}
fig = plot_multiple_time_series(data_dict, figsize=(14, 8))
plt.show()
```

### Fractal Plots (`src/visualisation/fractal_plots.py`)

#### `plot_dfa_results(results: Dict[str, Any], **kwargs) -> plt.Figure`

Plot DFA analysis results.

**Parameters:**
- `results`: DFA analysis results
- `**kwargs`: Additional plotting arguments

**Returns:**
- `plt.Figure`: Figure object

**Example:**
```python
fig = plot_dfa_results(
    dfa_results, 
    title="DFA Analysis Results",
    figsize=(12, 8)
)
plt.show()
```

#### `plot_rs_results(results: Dict[str, Any], **kwargs) -> plt.Figure`

Plot R/S analysis results.

**Parameters:**
- `results`: R/S analysis results
- `**kwargs`: Additional plotting arguments

**Returns:**
- `plt.Figure`: Figure object

**Example:**
```python
fig = plot_rs_results(
    rs_results, 
    title="R/S Analysis Results",
    figsize=(12, 8)
)
plt.show()
```

#### `plot_mfdfa_results(results: Dict[str, Any], **kwargs) -> plt.Figure`

Plot MFDFA analysis results.

**Parameters:**
- `results`: MFDFA analysis results

### Higuchi Plots (`src/visualisation/higuchi_plots.py`)

#### `plot_higuchi_analysis(k_values, l_values, summary, **kwargs) -> plt.Figure`

Create a comprehensive plot of Higuchi fractal dimension analysis.

**Parameters:**
- `k_values` (np.ndarray): Array of k values used in the analysis
- `l_values` (np.ndarray): Array of corresponding L(k) values
- `summary` (HiguchiSummary): Higuchi analysis results
- `**kwargs`: Additional plotting arguments

**Returns:**
- `plt.Figure`: The created figure

**Example:**
```python
from visualisation.higuchi_plots import plot_higuchi_analysis

fig = plot_higuchi_analysis(k_values, l_values, summary, figsize=(15, 10))
plt.show()
```

#### `plot_higuchi_comparison(summaries, **kwargs) -> plt.Figure`

Create comparison plots for multiple Higuchi analyses.

**Parameters:**
- `summaries` (Dict[str, HiguchiSummary]): Dictionary mapping names to HiguchiSummary objects
- `**kwargs`: Additional plotting arguments

**Returns:**
- `plt.Figure`: The created figure

**Example:**
```python
fig = plot_higuchi_comparison(results_dict, figsize=(16, 12))
plt.show()
```

#### `plot_higuchi_quality_assessment(summary, **kwargs) -> plt.Figure`

Create quality assessment plots for Higuchi analysis.

**Parameters:**
- `summary` (HiguchiSummary): Higuchi analysis results
- `**kwargs`: Additional plotting arguments

**Returns:**
- `plt.Figure`: The created figure

**Example:**
```python
fig = plot_higuchi_quality_assessment(summary, figsize=(12, 8))
plt.show()
```

#### `create_higuchi_report(summaries, save_dir="results/higuchi") -> str`

Create a comprehensive Higuchi analysis report.

**Parameters:**
- `summaries` (Dict[str, HiguchiSummary]): Dictionary mapping names to HiguchiSummary objects
- `save_dir` (str): Directory to save the report

**Returns:**
- `str`: Path to the saved report

**Example:**
```python
report_path = create_higuchi_report(results, save_dir="results/higuchi_analysis")
print(f"Report saved to: {report_path}")
```
- `**kwargs`: Additional plotting arguments

**Returns:**
- `plt.Figure`: Figure object

**Example:**
```python
fig = plot_mfdfa_results(
    mfdfa_results, 
    title="MFDFA Analysis Results",
    figsize=(15, 10)
)
plt.show()
```

### Results Visualization (`src/visualisation/results_visualisation.py`)

#### `create_summary_table(dfa_results: Dict[str, Dict[str, Any]] = None, rs_results: Dict[str, Dict[str, Any]] = None, wavelet_results: Dict[str, Dict[str, Any]] = None, spectral_results: Dict[str, Dict[str, Any]] = None, arfima_results: Dict[str, Dict[str, Any]] = None) -> pd.DataFrame`

Create a summary table of all analysis results.

**Parameters:**
- `dfa_results`: DFA results dictionary
- `rs_results`: R/S results dictionary
- `wavelet_results`: Wavelet results dictionary
- `spectral_results`: Spectral results dictionary
- `arfima_results`: ARFIMA results dictionary

**Returns:**
- `pd.DataFrame`: Summary table

**Example:**
```python
summary_table = create_summary_table(
    dfa_results=dfa_results,
    rs_results=rs_results,
    wavelet_results=wavelet_results,
    spectral_results=spectral_results,
    arfima_results=arfima_results
)
print(summary_table)
```

#### `plot_method_comparison(results_dict: Dict[str, Dict[str, Any]], **kwargs) -> plt.Figure`

Create comparison plots across different methods.

**Parameters:**
- `results_dict`: Dictionary mapping method names to results
- `**kwargs`: Additional plotting arguments

**Returns:**
- `plt.Figure`: Figure object

**Example:**
```python
methods_results = {
    'DFA': dfa_results,
    'R/S': rs_results,
    'Wavelet': wavelet_results
}
fig = plot_method_comparison(methods_results, figsize=(15, 10))
plt.show()
```

### Validation Plots (`src/visualisation/validation_plots.py`)

#### `plot_hypothesis_test_result(result: HypothesisTestResult, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure`

Plot hypothesis test results.

**Parameters:**
- `result` (HypothesisTestResult): Hypothesis test result
- `save_path` (Optional[str]): Path to save the plot
- `figsize` (Tuple[int, int]): Figure size

**Returns:**
- `plt.Figure`: The created figure

**Example:**
```python
result = test_lrd_hypothesis(data, method='dfa')
fig = plot_hypothesis_test_result(
    result, 
    save_path="results/hypothesis_test.png"
)
plt.show()
```

#### `plot_bootstrap_result(result: BootstrapResult, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure`

Plot bootstrap analysis results.

**Parameters:**
- `result` (BootstrapResult): Bootstrap analysis result
- `save_path` (Optional[str]): Path to save the plot
- `figsize` (Tuple[int, int]): Figure size

**Returns:**
- `plt.Figure`: The created figure

**Example:**
```python
result = bootstrap_confidence_interval(data, method='dfa')
fig = plot_bootstrap_result(
    result, 
    save_path="results/bootstrap_analysis.png"
)
plt.show()
```

#### `plot_monte_carlo_result(result: MonteCarloResult, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure`

Plot Monte Carlo significance test results.

**Parameters:**
- `result` (MonteCarloResult): Monte Carlo test result
- `save_path` (Optional[str]): Path to save the plot
- `figsize` (Tuple[int, int]): Figure size

**Returns:**
- `plt.Figure`: The created figure

**Example:**
```python
result = monte_carlo_test(data, method='dfa')
fig = plot_monte_carlo_result(
    result, 
    save_path="results/monte_carlo_test.png"
)
plt.show()
```

#### `plot_cross_validation_result(result: CrossValidationResult, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure`

Plot cross-validation results.

**Parameters:**
- `result` (CrossValidationResult): Cross-validation result
- `save_path` (Optional[str]): Path to save the plot
- `figsize` (Tuple[int, int]): Figure size

**Returns:**
- `plt.Figure`: The created figure

**Example:**
```python
result = cross_validate_lrd(data, method='dfa')
fig = plot_cross_validation_result(
    result, 
    save_path="results/cross_validation.png"
)
plt.show()
```

#### `plot_comprehensive_validation_summary(validation_results: Dict[str, Any], save_path: Optional[str] = None, figsize: Tuple[int, int] = (16, 12)) -> plt.Figure`

Plot comprehensive validation summary across all methods.

**Parameters:**
- `validation_results` (Dict[str, Any]): Comprehensive validation results
- `save_path` (Optional[str]): Path to save the plot
- `figsize` (Tuple[int, int]): Figure size

**Returns:**
- `plt.Figure`: The created figure

**Example:**
```python
results = comprehensive_validation(data)
fig = plot_comprehensive_validation_summary(
    results, 
    save_path="results/comprehensive_summary.png"
)
plt.show()
```

#### `create_validation_report(validation_results: Dict[str, Any], save_dir: str = "results/validation") -> str`

Create a comprehensive validation report with all plots.

**Parameters:**
- `validation_results` (Dict[str, Any]): Comprehensive validation results
- `save_dir` (str): Directory to save the report

**Returns:**
- `str`: Path to the saved report

**Example:**
```python
results = comprehensive_validation(data)
report_path = create_validation_report(
    results, 
    save_dir="results/validation_report"
)
print(f"Report saved to: {report_path}")
```

#### `plot_validation_result(result: Union[HypothesisTestResult, BootstrapResult, MonteCarloResult, CrossValidationResult], save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure`

Convenience function to plot any validation result.

**Parameters:**
- `result`: Validation result to plot
- `save_path` (Optional[str]): Path to save the plot
- `figsize` (Tuple[int, int]): Figure size

**Returns:**
- `plt.Figure`: The created figure

**Example:**
```python
# Works with any validation result type
fig = plot_validation_result(result, save_path="results/validation_plot.png")
plt.show()
```

## Configuration Module

### Configuration Loader (`src/config_loader.py`)

#### `ConfigLoader` Class

Class for loading and managing configuration files.

```python
class ConfigLoader:
    """A configuration loader for managing project settings."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration loader.
        
        Parameters
        ----------
        config_dir : str, optional
            Directory containing configuration files, by default "config"
        """
```

##### Methods

###### `get_config(config_name: str) -> Dict[str, Any]`

Get a specific configuration.

**Parameters:**
- `config_name` (str): Name of the configuration ('data', 'analysis', 'plot')

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

**Example:**
```python
config = ConfigLoader()
data_config = config.get_config('data')
print(f"Data formats: {data_config['data_sources']['file_formats']}")
```

###### `get_nested_config(config_name: str, *keys: str, default: Any = None) -> Any`

Get a nested configuration value.

**Parameters:**
- `config_name` (str): Name of the configuration
- `*keys`: Nested keys to access
- `default`: Default value if key not found

**Returns:**
- `Any`: Configuration value

**Example:**
```python
min_scale = config.get_nested_config('analysis', 'dfa', 'scales', 'min_scale', default=10)
print(f"DFA min scale: {min_scale}")
```

#### Convenience Functions

###### `get_data_config() -> Dict[str, Any]`

Get data configuration.

**Returns:**
- `Dict[str, Any]`: Data configuration

**Example:**
```python
data_config = get_data_config()
print(f"Storage directories: {data_config['storage']['directories']}")
```

###### `get_analysis_config() -> Dict[str, Any]`

Get analysis configuration.

**Returns:**
- `Dict[str, Any]`: Analysis configuration

**Example:**
```python
analysis_config = get_analysis_config()
print(f"DFA settings: {analysis_config['dfa']}")
```

###### `get_plot_config() -> Dict[str, Any]`

Get plotting configuration.

**Returns:**
- `Dict[str, Any]`: Plotting configuration

**Example:**
```python
plot_config = get_plot_config()
print(f"Figure DPI: {plot_config['general']['figure']['dpi']}")
```

## Utility Functions

### Statistical Functions

#### `calculate_hurst_exponent(alpha: float) -> float`

Calculate Hurst exponent from DFA alpha.

**Parameters:**
- `alpha` (float): DFA alpha value

**Returns:**
- `float`: Hurst exponent

**Example:**
```python
hurst = calculate_hurst_exponent(1.7)
print(f"Hurst exponent: {hurst:.4f}")
```

#### `calculate_alpha_from_hurst(hurst: float) -> float`

Calculate DFA alpha from Hurst exponent.

**Parameters:**
- `hurst` (float): Hurst exponent

**Returns:**
- `float`: DFA alpha value

**Example:**
```python
alpha = calculate_alpha_from_hurst(0.7)
print(f"DFA alpha: {alpha:.4f}")
```

### Data Generation Functions

#### `generate_fractional_noise(n: int, hurst: float, seed: int = None) -> np.ndarray`

Generate fractional Gaussian noise.

**Parameters:**
- `n` (int): Number of data points
- `hurst` (float): Hurst exponent
- `seed` (int, optional): Random seed

**Returns:**
- `np.ndarray`: Generated fractional noise

**Example:**
```python
noise = generate_fractional_noise(1000, hurst=0.8, seed=42)
print(f"Generated {len(noise)} points with H={0.8}")
```

## Scripts

### Main Analysis Script (`scripts/run_full_analysis.py`)

Main script for running complete analysis pipeline.

**Usage:**
```bash
python scripts/run_full_analysis.py
```

**Features:**
- Loads data from organized data folders
- Runs all analysis methods
- Generates comprehensive visualizations
- Saves results and tables

### Data Setup Script (`scripts/setup_data.py`)

Script for setting up project data collection.

**Usage:**
```bash
python scripts/setup_data.py
```

**Features:**
- Generates synthetic datasets
- Downloads financial data
- Organizes data structure
- Creates metadata

### Configuration Test Script (`scripts/test_config.py`)

Script for testing configuration system.

**Usage:**
```bash
python scripts/test_config.py
```

**Features:**
- Tests configuration loading
- Validates configuration files
- Demonstrates usage examples

## Examples

### Basic Analysis Example

```python
import numpy as np
from src.analysis.fractal_analysis import dfa, rescaled_range
from src.visualisation.fractal_plots import plot_dfa_results, plot_rs_results

# Generate synthetic data
np.random.seed(42)
data = np.random.randn(1000)

# Run DFA analysis
dfa_results = dfa(data, min_scale=10, n_scales=20)
print(f"DFA Hurst exponent: {dfa_results['hurst']:.4f}")

# Run R/S analysis
rs_results = rescaled_range(data, min_scale=10, n_scales=20)
print(f"R/S Hurst exponent: {rs_results['hurst']:.4f}")

# Plot results
fig1 = plot_dfa_results(dfa_results)
fig2 = plot_rs_results(rs_results)
plt.show()
```

### Complete Pipeline Example

```python
from src.data_processing.data_loader import DataLoader
from src.data_processing.preprocessing import TimeSeriesPreprocessor
from src.analysis.fractal_analysis import dfa, rescaled_range, mfdfa
from src.visualisation.results_visualisation import create_summary_table

# Load and preprocess data
loader = DataLoader()
data = loader.load_from_csv('data/time_series.csv')

preprocessor = TimeSeriesPreprocessor()
cleaned_data = preprocessor.clean_time_series(data)

# Run multiple analyses
dfa_results = dfa(cleaned_data)
rs_results = rescaled_range(cleaned_data)
mfdfa_results = mfdfa(cleaned_data)

# Create summary table
summary_table = create_summary_table(
    dfa_results=dfa_results,
    rs_results=rs_results,
    mfdfa_results=mfdfa_results
)

print("Analysis Summary:")
print(summary_table)
```

### Configuration-Based Analysis

```python
from src.config_loader import get_config_value
from src.analysis.fractal_analysis import dfa

# Get configuration values
min_scale = get_config_value('analysis', 'dfa', 'scales', 'min_scale', default=10)
n_scales = get_config_value('analysis', 'dfa', 'scales', 'n_scales', default=20)
detrend_order = get_config_value('analysis', 'dfa', 'detrending', 'order', default=1)

# Run analysis with configuration
results = dfa(data, min_scale=min_scale, n_scales=n_scales, detrend_order=detrend_order)
print(f"Results using configuration: {results}")
```

---

*This API documentation provides comprehensive information about all functions, classes, and modules in the project. For additional examples and usage patterns, refer to the analysis protocol and methodology documents.*

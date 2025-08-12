"""
Data Preprocessing Module for Long-Range Dependence Analysis

This module provides functions to preprocess and clean time series data:
- Missing value handling
- Outlier detection and treatment
- Data transformation (differencing, detrending, etc.)
- Stationarity testing
- Data normalization
- Time series specific operations
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, List, Tuple, Any, Callable
import warnings
from scipy import stats
from scipy.signal import detrend
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


class TimeSeriesPreprocessor:
    """
    A comprehensive preprocessor for time series data.
    
    Handles common preprocessing tasks including cleaning, transformation,
    and preparation for long-range dependence analysis.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print informative messages during preprocessing
        """
        self.verbose = verbose
        self.scalers = {}
        self.transformations = []
    
    def clean_data(self, df: pd.DataFrame, 
                   handle_missing: str = 'interpolate',
                   handle_outliers: str = 'iqr',
                   outlier_threshold: float = 1.5,
                   min_length: int = 50) -> pd.DataFrame:
        """
        Clean time series data by handling missing values and outliers.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input time series data
        handle_missing : str
            Method for handling missing values ('drop', 'interpolate', 'forward_fill', 'backward_fill')
        handle_outliers : str
            Method for handling outliers ('iqr', 'zscore', 'winsorize', 'remove')
        outlier_threshold : float
            Threshold for outlier detection
        min_length : int
            Minimum required length after cleaning
            
        Returns:
        --------
        pd.DataFrame
            Cleaned time series data
        """
        if self.verbose:
            print(f"Cleaning data with shape: {df.shape}")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean, method=handle_missing)
        
        # Handle outliers
        df_clean = self._handle_outliers(df_clean, method=handle_outliers, 
                                       threshold=outlier_threshold)
        
        # Check minimum length
        if len(df_clean) < min_length:
            warnings.warn(f"Data length ({len(df_clean)}) is below minimum ({min_length})")
        
        if self.verbose:
            print(f"Cleaned data shape: {df_clean.shape}")
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """Handle missing values in the time series."""
        if df.isnull().sum().sum() == 0:
            return df
        
        if self.verbose:
            print(f"Handling {df.isnull().sum().sum()} missing values using {method}")
        
        if method == 'drop':
            df_clean = df.dropna()
        elif method == 'interpolate':
            df_clean = df.interpolate(method='time' if df.index.dtype == 'datetime64[ns]' else 'linear')
        elif method == 'forward_fill':
            df_clean = df.fillna(method='ffill')
        elif method == 'backward_fill':
            df_clean = df.fillna(method='bfill')
        else:
            raise ValueError(f"Unknown missing value handling method: {method}")
        
        # Drop any remaining NaN values
        df_clean = df_clean.dropna()
        
        return df_clean
    
    def _handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                        threshold: float = 1.5) -> pd.DataFrame:
        """Handle outliers in the time series."""
        df_clean = df.copy()
        
        for column in df_clean.columns:
            if df_clean[column].dtype in ['float64', 'int64']:
                outliers = self._detect_outliers(df_clean[column], method, threshold)
                
                if outliers.sum() > 0:
                    if self.verbose:
                        print(f"Found {outliers.sum()} outliers in column {column}")
                    
                    if method == 'remove':
                        df_clean = df_clean[~outliers]
                    elif method == 'winsorize':
                        df_clean[column] = self._winsorize(df_clean[column], threshold)
                    elif method in ['iqr', 'zscore']:
                        # Replace outliers with median
                        median_val = df_clean[column].median()
                        df_clean.loc[outliers, column] = median_val
        
        return df_clean
    
    def _detect_outliers(self, series: pd.Series, method: str, threshold: float) -> pd.Series:
        """Detect outliers using various methods."""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            return z_scores > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def _winsorize(self, series: pd.Series, threshold: float) -> pd.Series:
        """Winsorize data to handle outliers."""
        from scipy.stats.mstats import winsorize
        return pd.Series(winsorize(series, limits=[threshold, threshold]), 
                        index=series.index)
    
    def transform_data(self, df: pd.DataFrame, 
                      transformations: List[str] = None,
                      **kwargs) -> pd.DataFrame:
        """
        Apply various transformations to the time series data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input time series data
        transformations : List[str]
            List of transformations to apply ('differencing', 'detrending', 'normalize', 'log', 'sqrt')
        **kwargs : dict
            Additional parameters for specific transformations
            
        Returns:
        --------
        pd.DataFrame
            Transformed time series data
        """
        if transformations is None:
            transformations = []
        
        if self.verbose:
            print(f"Applying transformations: {transformations}")
        
        df_transformed = df.copy()
        
        for transform in transformations:
            if transform == 'differencing':
                df_transformed = self._apply_differencing(df_transformed, **kwargs)
            elif transform == 'detrending':
                df_transformed = self._apply_detrending(df_transformed, **kwargs)
            elif transform == 'normalize':
                df_transformed = self._apply_normalization(df_transformed, **kwargs)
            elif transform == 'log':
                df_transformed = self._apply_log_transform(df_transformed, **kwargs)
            elif transform == 'sqrt':
                df_transformed = self._apply_sqrt_transform(df_transformed, **kwargs)
            else:
                warnings.warn(f"Unknown transformation: {transform}")
        
        self.transformations.extend(transformations)
        return df_transformed
    
    def _apply_differencing(self, df: pd.DataFrame, order: int = 1, 
                           seasonal: bool = False, period: int = None) -> pd.DataFrame:
        """Apply differencing to make series stationary."""
        df_diff = df.copy()
        
        for column in df_diff.columns:
            if df_diff[column].dtype in ['float64', 'int64']:
                if seasonal and period:
                    # Seasonal differencing
                    df_diff[column] = df_diff[column].diff(period)
                else:
                    # Regular differencing
                    df_diff[column] = df_diff[column].diff(order)
        
        # Remove NaN values from differencing
        df_diff = df_diff.dropna()
        
        if self.verbose:
            print(f"Applied differencing (order={order})")
        
        return df_diff
    
    def _apply_detrending(self, df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """Remove trend from time series."""
        df_detrended = df.copy()
        
        for column in df_detrended.columns:
            if df_detrended[column].dtype in ['float64', 'int64']:
                if method == 'linear':
                    df_detrended[column] = detrend(df_detrended[column], type='linear')
                elif method == 'constant':
                    df_detrended[column] = detrend(df_detrended[column], type='constant')
                else:
                    raise ValueError(f"Unknown detrending method: {method}")
        
        if self.verbose:
            print(f"Applied detrending (method={method})")
        
        return df_detrended
    
    def _apply_normalization(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Normalize time series data."""
        df_normalized = df.copy()
        
        for column in df_normalized.columns:
            if df_normalized[column].dtype in ['float64', 'int64']:
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'robust':
                    scaler = RobustScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown normalization method: {method}")
                
                # Fit and transform
                values = df_normalized[column].values.reshape(-1, 1)
                df_normalized[column] = scaler.fit_transform(values).flatten()
                
                # Store scaler for potential inverse transformation
                self.scalers[column] = scaler
        
        if self.verbose:
            print(f"Applied normalization (method={method})")
        
        return df_normalized
    
    def _apply_log_transform(self, df: pd.DataFrame, offset: float = 1.0) -> pd.DataFrame:
        """Apply log transformation to time series."""
        df_log = df.copy()
        
        for column in df_log.columns:
            if df_log[column].dtype in ['float64', 'int64']:
                # Add offset to handle zero/negative values
                df_log[column] = np.log(df_log[column] + offset)
        
        if self.verbose:
            print(f"Applied log transformation (offset={offset})")
        
        return df_log
    
    def _apply_sqrt_transform(self, df: pd.DataFrame, offset: float = 0.0) -> pd.DataFrame:
        """Apply square root transformation to time series."""
        df_sqrt = df.copy()
        
        for column in df_sqrt.columns:
            if df_sqrt[column].dtype in ['float64', 'int64']:
                # Add offset to handle negative values
                df_sqrt[column] = np.sqrt(df_sqrt[column] + offset)
        
        if self.verbose:
            print(f"Applied sqrt transformation (offset={offset})")
        
        return df_sqrt
    
    def test_stationarity(self, df: pd.DataFrame, method: str = 'adf') -> Dict[str, Any]:
        """
        Test stationarity of time series data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Time series data to test
        method : str
            Stationarity test method ('adf', 'kpss', 'pp')
            
        Returns:
        --------
        Dict[str, Any]
            Test results including p-value and stationarity conclusion
        """
        results = {}
        
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:
                series = df[column].dropna()
                
                if method == 'adf':
                    from statsmodels.tsa.stattools import adfuller
                    test_result = adfuller(series)
                    results[column] = {
                        'test_statistic': test_result[0],
                        'p_value': test_result[1],
                        'critical_values': test_result[4],
                        'is_stationary': test_result[1] < 0.05
                    }
                
                elif method == 'kpss':
                    from statsmodels.tsa.stattools import kpss
                    test_result = kpss(series, regression='c')
                    results[column] = {
                        'test_statistic': test_result[0],
                        'p_value': test_result[1],
                        'critical_values': test_result[3],
                        'is_stationary': test_result[1] > 0.05
                    }
                
                elif method == 'pp':
                    from statsmodels.tsa.stattools import PhillipsPerron
                    test_result = PhillipsPerron(series)
                    results[column] = {
                        'test_statistic': test_result.stat,
                        'p_value': test_result.pvalue,
                        'critical_values': test_result.critical_values,
                        'is_stationary': test_result.pvalue < 0.05
                    }
                
                else:
                    raise ValueError(f"Unknown stationarity test method: {method}")
        
        if self.verbose:
            for column, result in results.items():
                status = "stationary" if result['is_stationary'] else "non-stationary"
                print(f"{column}: {status} (p-value: {result['p_value']:.4f})")
        
        return results
    
    def prepare_for_analysis(self, df: pd.DataFrame, 
                           target_column: Optional[str] = None,
                           min_length: int = 100,
                           ensure_stationary: bool = False) -> pd.DataFrame:
        """
        Prepare time series data for long-range dependence analysis.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input time series data
        target_column : Optional[str]
            Specific column to analyze (if None, uses first numeric column)
        min_length : int
            Minimum required length for analysis
        ensure_stationary : bool
            Whether to apply transformations to ensure stationarity
            
        Returns:
        --------
        pd.DataFrame
            Prepared data ready for analysis
        """
        if self.verbose:
            print("Preparing data for long-range dependence analysis...")
        
        # Select target column
        if target_column is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found in data")
            target_column = numeric_columns[0]
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Extract target series
        series = df[target_column].copy()
        
        # Clean data
        series = self._handle_missing_values(series.to_frame())[target_column]
        series = self._handle_outliers(series.to_frame())[target_column]
        
        # Check length
        if len(series) < min_length:
            warnings.warn(f"Series length ({len(series)}) is below recommended minimum ({min_length})")
        
        # Ensure stationarity if requested
        if ensure_stationary:
            stationarity_results = self.test_stationarity(series.to_frame())
            if not stationarity_results[target_column]['is_stationary']:
                if self.verbose:
                    print("Series is non-stationary, applying differencing...")
                series = self._apply_differencing(series.to_frame())[target_column]
        
        # Convert to numpy array for analysis
        data_array = series.values
        
        if self.verbose:
            print(f"Prepared data: {len(data_array)} points")
        
        return data_array
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of applied preprocessing steps."""
        return {
            'transformations': self.transformations,
            'scalers': list(self.scalers.keys()),
            'total_steps': len(self.transformations)
        }


# Convenience functions
def clean_time_series(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Clean time series data with default settings."""
    preprocessor = TimeSeriesPreprocessor()
    return preprocessor.clean_data(df, **kwargs)


def make_stationary(df: pd.DataFrame, method: str = 'differencing', **kwargs) -> pd.DataFrame:
    """Make time series stationary using specified method."""
    preprocessor = TimeSeriesPreprocessor()
    return preprocessor.transform_data(df, transformations=[method], **kwargs)


def normalize_time_series(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """Normalize time series data."""
    preprocessor = TimeSeriesPreprocessor()
    return preprocessor.transform_data(df, transformations=['normalize'], method=method)


def test_stationarity(df: pd.DataFrame, method: str = 'adf') -> Dict[str, Any]:
    """Test stationarity of time series data."""
    preprocessor = TimeSeriesPreprocessor()
    return preprocessor.test_stationarity(df, method=method)


def prepare_for_lrd_analysis(df: pd.DataFrame, **kwargs) -> np.ndarray:
    """Prepare time series data for long-range dependence analysis."""
    preprocessor = TimeSeriesPreprocessor()
    return preprocessor.prepare_for_analysis(df, **kwargs)

"""
Data Loading Module for Long-Range Dependence Analysis

This module provides functions to load time series data from various sources:
- CSV files
- Excel files
- JSON files
- NumPy arrays
- Pandas DataFrames
- Text files
- API data sources
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any
import warnings
import requests
from datetime import datetime, timedelta
import yfinance as yf


class DataLoader:
    """
    A comprehensive data loader for time series analysis.
    
    Supports multiple file formats and data sources with automatic
    format detection and data validation.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print informative messages during loading
        """
        self.verbose = verbose
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.txt', '.npy', '.npz']
    
    def load_data(self, source: Union[str, Path, np.ndarray, pd.DataFrame], 
                  **kwargs) -> pd.DataFrame:
        """
        Load data from various sources with automatic format detection.
        
        Parameters:
        -----------
        source : Union[str, Path, np.ndarray, pd.DataFrame]
            Data source (file path, array, or DataFrame)
        **kwargs : dict
            Additional arguments passed to specific loaders
            
        Returns:
        --------
        pd.DataFrame
            Loaded data as a pandas DataFrame
        """
        if isinstance(source, (str, Path)):
            return self._load_from_file(source, **kwargs)
        elif isinstance(source, np.ndarray):
            return self._load_from_array(source, **kwargs)
        elif isinstance(source, pd.DataFrame):
            return self._load_from_dataframe(source, **kwargs)
        else:
            raise ValueError(f"Unsupported data source type: {type(source)}")
    
    def _load_from_file(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from a file with automatic format detection."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            return self._load_csv(file_path, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            return self._load_excel(file_path, **kwargs)
        elif file_extension == '.json':
            return self._load_json(file_path, **kwargs)
        elif file_extension == '.txt':
            return self._load_text(file_path, **kwargs)
        elif file_extension == '.npy':
            return self._load_numpy(file_path, **kwargs)
        elif file_extension == '.npz':
            return self._load_numpy_compressed(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from CSV file."""
        if self.verbose:
            print(f"Loading CSV file: {file_path}")
        
        # Set default parameters for time series data
        default_kwargs = {
            'index_col': 0,
            'parse_dates': True,
            'infer_datetime_format': True
        }
        default_kwargs.update(kwargs)
        
        try:
            df = pd.read_csv(file_path, **default_kwargs)
            if self.verbose:
                print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            # Try without date parsing if it fails
            if 'parse_dates' in default_kwargs:
                del default_kwargs['parse_dates']
                del default_kwargs['infer_datetime_format']
                df = pd.read_csv(file_path, **default_kwargs)
                if self.verbose:
                    print(f"Loaded without date parsing: {len(df)} rows and {len(df.columns)} columns")
                return df
            else:
                raise e
    
    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from Excel file."""
        if self.verbose:
            print(f"Loading Excel file: {file_path}")
        
        # Set default parameters
        default_kwargs = {
            'index_col': 0,
            'parse_dates': True
        }
        default_kwargs.update(kwargs)
        
        try:
            df = pd.read_excel(file_path, **default_kwargs)
            if self.verbose:
                print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            # Try without date parsing if it fails
            if 'parse_dates' in default_kwargs:
                del default_kwargs['parse_dates']
                df = pd.read_excel(file_path, **default_kwargs)
                if self.verbose:
                    print(f"Loaded without date parsing: {len(df)} rows and {len(df.columns)} columns")
                return df
            else:
                raise e
    
    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from JSON file."""
        if self.verbose:
            print(f"Loading JSON file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
            
            if self.verbose:
                print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {e}")
    
    def _load_text(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from text file."""
        if self.verbose:
            print(f"Loading text file: {file_path}")
        
        # Set default parameters for space/tab separated files
        default_kwargs = {
            'sep': None,
            'engine': 'python',
            'index_col': 0
        }
        default_kwargs.update(kwargs)
        
        try:
            df = pd.read_csv(file_path, **default_kwargs)
            if self.verbose:
                print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading text file: {e}")
    
    def _load_numpy(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from NumPy .npy file."""
        if self.verbose:
            print(f"Loading NumPy file: {file_path}")
        
        try:
            array = np.load(file_path)
            df = pd.DataFrame(array)
            
            if self.verbose:
                print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading NumPy file: {e}")
    
    def _load_numpy_compressed(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from NumPy .npz file."""
        if self.verbose:
            print(f"Loading compressed NumPy file: {file_path}")
        
        try:
            data = np.load(file_path)
            
            # If there's only one array, load it directly
            if len(data.files) == 1:
                array = data[data.files[0]]
                df = pd.DataFrame(array)
            else:
                # If multiple arrays, create DataFrame with all arrays
                arrays = {name: data[name] for name in data.files}
                df = pd.DataFrame(arrays)
            
            if self.verbose:
                print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading compressed NumPy file: {e}")
    
    def _load_from_array(self, array: np.ndarray, **kwargs) -> pd.DataFrame:
        """Load data from NumPy array."""
        if self.verbose:
            print(f"Loading NumPy array with shape: {array.shape}")
        
        df = pd.DataFrame(array)
        
        # Set column names if provided
        if 'columns' in kwargs:
            df.columns = kwargs['columns']
        
        if self.verbose:
            print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def _load_from_dataframe(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Load data from pandas DataFrame."""
        if self.verbose:
            print(f"Loading DataFrame with shape: {df.shape}")
        
        # Return a copy to avoid modifying the original
        return df.copy()


def load_financial_data(symbol: str, start_date: str = None, end_date: str = None,
                       period: str = "1y") -> pd.DataFrame:
    """
    Load financial time series data using yfinance.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    period : str, optional
        Data period if start/end dates not provided ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
    Returns:
    --------
    pd.DataFrame
        Financial data with OHLCV columns
    """
    try:
        ticker = yf.Ticker(symbol)
        
        if start_date and end_date:
            data = ticker.history(start=start_date, end=end_date)
        else:
            data = ticker.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        print(f"Loaded {len(data)} days of data for {symbol}")
        return data
        
    except Exception as e:
        raise ValueError(f"Error loading financial data for {symbol}: {e}")


def load_synthetic_data(data_type: str = "fractional_noise", n: int = 1000, 
                       **kwargs) -> pd.DataFrame:
    """
    Generate synthetic time series data for testing.
    
    Parameters:
    -----------
    data_type : str
        Type of synthetic data ('fractional_noise', 'random_walk', 'white_noise', 'trend', 'seasonal')
    n : int
        Number of data points
    **kwargs : dict
        Additional parameters for specific data types
        
    Returns:
    --------
    pd.DataFrame
        Synthetic time series data
    """
    np.random.seed(kwargs.get('seed', 42))
    
    if data_type == "fractional_noise":
        d = kwargs.get('d', 0.3)
        data = np.zeros(n)
        for i in range(1, n):
            data[i] = data[i-1] + np.random.normal(0, 1) * (i ** (-d))
        name = f"fractional_noise_d{d}"
        
    elif data_type == "random_walk":
        data = np.cumsum(np.random.randn(n))
        name = "random_walk"
        
    elif data_type == "white_noise":
        data = np.random.randn(n)
        name = "white_noise"
        
    elif data_type == "trend":
        t = np.arange(n)
        trend = kwargs.get('trend', 0.01)
        noise = kwargs.get('noise', 0.1)
        data = trend * t + np.random.normal(0, noise, n)
        name = f"trend_{trend}"
        
    elif data_type == "seasonal":
        t = np.arange(n)
        amplitude = kwargs.get('amplitude', 0.5)
        frequency = kwargs.get('frequency', 100)
        noise = kwargs.get('noise', 0.1)
        data = amplitude * np.sin(2 * np.pi * t / frequency) + np.random.normal(0, noise, n)
        name = f"seasonal_f{frequency}"
        
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2020-01-01', periods=n, freq='D'),
        'value': data
    })
    df.set_index('timestamp', inplace=True)
    df.name = name
    
    print(f"Generated {n} points of {data_type} data")
    return df


def load_multiple_files(file_paths: List[Union[str, Path]], 
                       loader: Optional[DataLoader] = None) -> Dict[str, pd.DataFrame]:
    """
    Load multiple files and return a dictionary of DataFrames.
    
    Parameters:
    -----------
    file_paths : List[Union[str, Path]]
        List of file paths to load
    loader : Optional[DataLoader]
        DataLoader instance (creates new one if None)
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary mapping file names to DataFrames
    """
    if loader is None:
        loader = DataLoader()
    
    data_dict = {}
    
    for file_path in file_paths:
        file_path = Path(file_path)
        try:
            df = loader.load_data(file_path)
            data_dict[file_path.stem] = df
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return data_dict


# Convenience functions for common use cases
def load_csv_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load CSV data with sensible defaults for time series."""
    loader = DataLoader()
    return loader._load_csv(Path(file_path), **kwargs)


def load_excel_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load Excel data with sensible defaults for time series."""
    loader = DataLoader()
    return loader._load_excel(Path(file_path), **kwargs)


def load_json_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load JSON data."""
    loader = DataLoader()
    return loader._load_json(Path(file_path), **kwargs)

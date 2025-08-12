"""
Data Management Module for Long-Range Dependence Analysis

This module provides comprehensive data management capabilities:
- Saving synthetic data to appropriate locations
- Saving processed realistic data
- Managing data metadata and documentation
- Data versioning and organization
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any
from datetime import datetime
import warnings

from .data_loader import load_synthetic_data, load_financial_data
from .preprocessing import TimeSeriesPreprocessor, clean_time_series, make_stationary
from .quality_check import DataQualityChecker, assess_data_quality


class DataManager:
    """
    A comprehensive data manager for organizing and saving time series data.
    
    Manages the data folder structure and ensures proper organization
    of synthetic and processed data for long-range dependence analysis.
    """
    
    def __init__(self, data_root: str = "data", verbose: bool = True):
        """
        Initialize the data manager.
        
        Parameters:
        -----------
        data_root : str
            Root directory for data storage
        verbose : bool
            Whether to print informative messages
        """
        self.data_root = Path(data_root)
        self.verbose = verbose
        
        # Ensure data directory structure exists
        self._setup_directory_structure()
        
        # Initialize subdirectories
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        self.metadata_dir = self.data_root / "metadata"
        
    def _setup_directory_structure(self) -> None:
        """Create the necessary directory structure."""
        directories = [
            self.data_root,
            self.data_root / "raw",
            self.data_root / "processed", 
            self.data_root / "metadata"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"Ensured directory exists: {directory}")
    
    def save_synthetic_data(self, data: Union[Dict[str, np.ndarray], pd.DataFrame, np.ndarray],
                           name: str = None,
                           data_type: str = "synthetic",
                           description: str = None,
                           parameters: Dict[str, Any] = None) -> str:
        """
        Save synthetic data to the raw data directory.
        
        Parameters:
        -----------
        data : Union[Dict[str, np.ndarray], pd.DataFrame, np.ndarray]
            Data to save
        name : str, optional
            Name for the dataset
        data_type : str
            Type of synthetic data ('fractional_noise', 'random_walk', etc.)
        description : str, optional
            Description of the data
        parameters : Dict[str, Any], optional
            Parameters used to generate the data
            
        Returns:
        --------
        str
            Path to the saved file
        """
        # Generate filename if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{data_type}_{timestamp}"
        
        # Convert data to DataFrame if needed
        if isinstance(data, np.ndarray):
            df = pd.DataFrame({
                'timestamp': pd.date_range(start='2020-01-01', periods=len(data), freq='D'),
                'value': data
            })
            df.set_index('timestamp', inplace=True)
        elif isinstance(data, dict):
            # Handle multiple series
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Save data
        file_path = self.raw_dir / f"{name}.csv"
        df.to_csv(file_path)
        
        # Save metadata
        metadata = {
            'name': name,
            'data_type': data_type,
            'description': description or f"Synthetic {data_type} data",
            'parameters': parameters or {},
            'created_at': datetime.now().isoformat(),
            'shape': df.shape,
            'columns': list(df.columns),
            'index_type': str(type(df.index)),
            'file_path': str(file_path)
        }
        
        metadata_path = self.metadata_dir / f"{name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"Saved synthetic data: {file_path}")
            print(f"Saved metadata: {metadata_path}")
        
        return str(file_path)
    
    def generate_and_save_synthetic_datasets(self, n: int = 1000, seed: int = 42) -> Dict[str, str]:
        """
        Generate and save a comprehensive set of synthetic datasets.
        
        Parameters:
        -----------
        n : int
            Number of data points per series
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping dataset names to file paths
        """
        np.random.seed(seed)
        
        datasets = {}
        
        # Generate different types of synthetic data
        synthetic_types = [
            ("fractional_noise_d03", "fractional_noise", {"d": 0.3}),
            ("fractional_noise_d05", "fractional_noise", {"d": 0.5}),
            ("random_walk", "random_walk", {}),
            ("white_noise", "white_noise", {}),
            ("trend_data", "trend", {"trend": 0.01, "noise": 0.1}),
            ("seasonal_data", "seasonal", {"amplitude": 0.5, "frequency": 100, "noise": 0.1}),
            ("arfima_lrd", "fractional_noise", {"d": 0.3}),  # Long-range dependent
            ("arfima_srd", "fractional_noise", {"d": 0.1}),  # Short-range dependent
        ]
        
        for name, data_type, params in synthetic_types:
            try:
                # Generate data
                df = load_synthetic_data(data_type=data_type, n=n, seed=seed, **params)
                
                # Save data
                file_path = self.save_synthetic_data(
                    data=df,
                    name=name,
                    data_type=data_type,
                    description=f"Synthetic {data_type} data with parameters {params}",
                    parameters=params
                )
                
                datasets[name] = file_path
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to generate {name}: {e}")
        
        if self.verbose:
            print(f"Generated and saved {len(datasets)} synthetic datasets")
        
        return datasets
    
    def save_processed_data(self, data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                           original_name: str,
                           processing_steps: List[str],
                           processing_parameters: Dict[str, Any] = None) -> str:
        """
        Save processed data to the processed data directory.
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, Dict[str, np.ndarray]]
            Processed data to save
        original_name : str
            Name of the original dataset
        processing_steps : List[str]
            List of processing steps applied
        processing_parameters : Dict[str, Any], optional
            Parameters used in processing
            
        Returns:
        --------
        str
            Path to the saved file
        """
        # Generate processed filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_name = f"processed_{original_name}_{timestamp}"
        
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Save processed data
        file_path = self.processed_dir / f"{processed_name}.csv"
        df.to_csv(file_path)
        
        # Save processing metadata
        metadata = {
            'original_name': original_name,
            'processed_name': processed_name,
            'processing_steps': processing_steps,
            'processing_parameters': processing_parameters or {},
            'processed_at': datetime.now().isoformat(),
            'shape': df.shape,
            'columns': list(df.columns),
            'file_path': str(file_path)
        }
        
        metadata_path = self.metadata_dir / f"{processed_name}_processing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"Saved processed data: {file_path}")
            print(f"Saved processing metadata: {metadata_path}")
        
        return str(file_path)
    
    def load_and_process_realistic_data(self, symbols: List[str] = None,
                                      start_date: str = "2020-01-01",
                                      end_date: str = None) -> Dict[str, str]:
        """
        Load and process realistic financial data.
        
        Parameters:
        -----------
        symbols : List[str], optional
            List of stock symbols to load (default: ['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
        start_date : str
            Start date for data collection
        end_date : str, optional
            End date for data collection (default: today)
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping dataset names to file paths
        """
        if symbols is None:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        processed_datasets = {}
        
        for symbol in symbols:
            try:
                # Load financial data
                df = load_financial_data(symbol, start_date=start_date, end_date=end_date)
                
                # Save raw data
                raw_file_path = self.save_synthetic_data(
                    data=df,
                    name=f"financial_{symbol}",
                    data_type="financial",
                    description=f"Financial data for {symbol}",
                    parameters={"symbol": symbol, "start_date": start_date, "end_date": end_date}
                )
                
                # Process the data
                preprocessor = TimeSeriesPreprocessor()
                
                # Clean the data
                cleaned_df = preprocessor.clean_data(df)
                
                # Make stationary (use returns)
                if 'Close' in cleaned_df.columns:
                    returns = cleaned_df['Close'].pct_change().dropna()
                    stationary_df = pd.DataFrame({'returns': returns})
                else:
                    # Use first column if Close not available
                    first_col = cleaned_df.columns[0]
                    returns = cleaned_df[first_col].pct_change().dropna()
                    stationary_df = pd.DataFrame({'returns': returns})
                
                # Save processed data
                processed_file_path = self.save_processed_data(
                    data=stationary_df,
                    original_name=f"financial_{symbol}",
                    processing_steps=["cleaning", "stationarity_transformation"],
                    processing_parameters={
                        "cleaning_method": "interpolation",
                        "stationarity_method": "returns",
                        "remove_outliers": True
                    }
                )
                
                processed_datasets[symbol] = processed_file_path
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to process {symbol}: {e}")
        
        if self.verbose:
            print(f"Processed {len(processed_datasets)} financial datasets")
        
        return processed_datasets
    
    def create_data_dictionary(self) -> pd.DataFrame:
        """
        Create a comprehensive data dictionary for all datasets.
        
        Returns:
        --------
        pd.DataFrame
            Data dictionary with information about all datasets
        """
        data_info = []
        
        # Process raw data files
        for file_path in self.raw_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
                metadata_path = self.metadata_dir / f"{file_path.stem}_metadata.json"
                
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                data_info.append({
                    'name': file_path.stem,
                    'type': 'raw',
                    'file_path': str(file_path),
                    'rows': len(df),
                    'columns': len(df.columns),
                    'data_type': metadata.get('data_type', 'unknown'),
                    'description': metadata.get('description', ''),
                    'created_at': metadata.get('created_at', ''),
                    'parameters': str(metadata.get('parameters', {}))
                })
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not process {file_path}: {e}")
        
        # Process processed data files
        for file_path in self.processed_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
                metadata_path = self.metadata_dir / f"{file_path.stem}_processing_metadata.json"
                
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                data_info.append({
                    'name': file_path.stem,
                    'type': 'processed',
                    'file_path': str(file_path),
                    'rows': len(df),
                    'columns': len(df.columns),
                    'data_type': 'processed',
                    'description': f"Processed version of {metadata.get('original_name', 'unknown')}",
                    'created_at': metadata.get('processed_at', ''),
                    'parameters': str(metadata.get('processing_parameters', {}))
                })
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not process {file_path}: {e}")
        
        # Create DataFrame and save
        data_dict_df = pd.DataFrame(data_info)
        data_dict_path = self.metadata_dir / "data_dictionary.csv"
        data_dict_df.to_csv(data_dict_path, index=False)
        
        if self.verbose:
            print(f"Created data dictionary: {data_dict_path}")
            print(f"Found {len(data_dict_df)} datasets")
        
        return data_dict_df
    
    def create_collection_protocol(self) -> str:
        """
        Create a data collection protocol document.
        
        Returns:
        --------
        str
            Path to the protocol document
        """
        protocol_content = f"""# Data Collection Protocol

## Overview
This document describes the data collection and management protocol for the Long-Range Dependence Analysis project.

## Data Structure
- **Raw Data**: `data/raw/` - Original datasets before processing
- **Processed Data**: `data/processed/` - Cleaned and transformed datasets
- **Metadata**: `data/metadata/` - Documentation and information about datasets

## Synthetic Data Generation
Synthetic datasets are generated using the following methods:
1. **Fractional Noise**: ARFIMA-like processes with different d parameters
2. **Random Walk**: Cumulative sum of random normal variables
3. **White Noise**: Independent and identically distributed random variables
4. **Trend Data**: Linear trend with added noise
5. **Seasonal Data**: Sinusoidal patterns with noise

## Realistic Data Collection
Financial data is collected using the yfinance API:
- Stock price data for major companies
- Daily frequency
- Automatic cleaning and preprocessing
- Conversion to returns for stationarity

## Data Processing Pipeline
1. **Loading**: Data is loaded from various sources
2. **Cleaning**: Missing values, outliers, and inconsistencies are handled
3. **Transformation**: Data is made stationary and normalized
4. **Quality Assessment**: Data quality is evaluated
5. **Saving**: Processed data is saved with metadata

## Metadata Standards
Each dataset includes:
- Name and description
- Data type and parameters
- Creation timestamp
- Processing history (for processed data)
- Quality metrics

## Version Control
- All datasets are versioned with timestamps
- Original data is preserved in raw directory
- Processing steps are documented
- Data dictionary is automatically updated

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        protocol_path = self.metadata_dir / "collection_protocol.md"
        with open(protocol_path, 'w') as f:
            f.write(protocol_content)
        
        if self.verbose:
            print(f"Created collection protocol: {protocol_path}")
        
        return str(protocol_path)
    
    def setup_complete_dataset(self, n_synthetic: int = 1000, 
                              financial_symbols: List[str] = None,
                              seed: int = 42) -> Dict[str, Any]:
        """
        Set up a complete dataset collection for the project.
        
        Parameters:
        -----------
        n_synthetic : int
            Number of points for synthetic datasets
        financial_symbols : List[str], optional
            List of financial symbols to process
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        Dict[str, Any]
            Summary of created datasets
        """
        if self.verbose:
            print("Setting up complete dataset collection...")
        
        # Generate synthetic datasets
        synthetic_datasets = self.generate_and_save_synthetic_datasets(n=n_synthetic, seed=seed)
        
        # Process realistic data
        realistic_datasets = self.load_and_process_realistic_data(symbols=financial_symbols)
        
        # Create documentation
        data_dict = self.create_data_dictionary()
        protocol_path = self.create_collection_protocol()
        
        summary = {
            'synthetic_datasets': len(synthetic_datasets),
            'realistic_datasets': len(realistic_datasets),
            'total_datasets': len(synthetic_datasets) + len(realistic_datasets),
            'data_dictionary_path': str(self.metadata_dir / "data_dictionary.csv"),
            'protocol_path': protocol_path,
            'synthetic_files': synthetic_datasets,
            'realistic_files': realistic_datasets
        }
        
        if self.verbose:
            print(f"\nDataset setup complete!")
            print(f"- Synthetic datasets: {summary['synthetic_datasets']}")
            print(f"- Realistic datasets: {summary['realistic_datasets']}")
            print(f"- Total datasets: {summary['total_datasets']}")
            print(f"- Data dictionary: {summary['data_dictionary_path']}")
            print(f"- Protocol: {summary['protocol_path']}")
        
        return summary


# Convenience functions
def setup_project_data(data_root: str = "data", 
                      n_synthetic: int = 1000,
                      financial_symbols: List[str] = None,
                      seed: int = 42) -> Dict[str, Any]:
    """
    Convenience function to set up complete project data.
    
    Parameters:
    -----------
    data_root : str
        Root directory for data storage
    n_synthetic : int
        Number of points for synthetic datasets
    financial_symbols : List[str], optional
        List of financial symbols to process
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dict[str, Any]
        Summary of created datasets
    """
    manager = DataManager(data_root=data_root)
    return manager.setup_complete_dataset(
        n_synthetic=n_synthetic,
        financial_symbols=financial_symbols,
        seed=seed
    )


def save_synthetic_data(data: Union[Dict[str, np.ndarray], pd.DataFrame, np.ndarray],
                       name: str = None,
                       data_root: str = "data",
                       **kwargs) -> str:
    """
    Convenience function to save synthetic data.
    
    Parameters:
    -----------
    data : Union[Dict[str, np.ndarray], pd.DataFrame, np.ndarray]
        Data to save
    name : str, optional
        Name for the dataset
    data_root : str
        Root directory for data storage
    **kwargs : dict
        Additional arguments passed to DataManager.save_synthetic_data
        
    Returns:
    --------
    str
        Path to the saved file
    """
    manager = DataManager(data_root=data_root)
    return manager.save_synthetic_data(data=data, name=name, **kwargs)


def save_processed_data(data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                       original_name: str,
                       data_root: str = "data",
                       **kwargs) -> str:
    """
    Convenience function to save processed data.
    
    Parameters:
    -----------
    data : Union[pd.DataFrame, Dict[str, np.ndarray]]
        Processed data to save
    original_name : str
        Name of the original dataset
    data_root : str
        Root directory for data storage
    **kwargs : dict
        Additional arguments passed to DataManager.save_processed_data
        
    Returns:
    --------
    str
        Path to the saved file
    """
    manager = DataManager(data_root=data_root)
    return manager.save_processed_data(data=data, original_name=original_name, **kwargs)

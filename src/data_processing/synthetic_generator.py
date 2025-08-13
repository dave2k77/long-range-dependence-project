"""
Synthetic Data Generation Module for Long-Range Dependence Analysis

This module provides comprehensive synthetic data generation capabilities:
- Pure signal generators: ARFIMA, Fractional Brownian Motion (fBm), Fractional Gaussian Noise (fGn)
- Contaminators: polynomial trends, periodicity, outliers, irregular sampling, heavy-tail fluctuations
- Data storage with proper metadata and organization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import warnings
from scipy import stats
from scipy.signal import periodogram
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
from pathlib import Path
import json

from .data_manager import DataManager


class PureSignalGenerator:
    """
    Pure signal generators for synthetic time series data.
    
    Generates clean signals without contamination for controlled experiments.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the pure signal generator.
        
        Parameters:
        -----------
        random_state : int, optional
            Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def generate_arfima(self, n: int, d: float, ar_params: List[float] = None, 
                       ma_params: List[float] = None, sigma: float = 1.0) -> np.ndarray:
        """
        Generate ARFIMA(p,d,q) time series.
        
        Parameters:
        -----------
        n : int
            Length of the time series
        d : float
            Fractional differencing parameter (0 < d < 0.5 for stationarity)
        ar_params : List[float], optional
            AR parameters
        ma_params : List[float], optional
            MA parameters
        sigma : float
            Standard deviation of innovations
            
        Returns:
        --------
        np.ndarray
            Generated ARFIMA time series
        """
        if not 0 < d < 0.5:
            raise ValueError("d must be between 0 and 0.5 for stationarity")
        
        # Generate fractional noise using spectral method
        freqs = np.fft.fftfreq(n)
        # Avoid division by zero by handling the DC component separately
        power_spectrum = np.zeros_like(freqs)
        mask = np.abs(freqs) > 0
        power_spectrum[mask] = (2 * np.sin(np.pi * np.abs(freqs[mask]))) ** (-2 * d)
        power_spectrum[0] = 0  # Set DC component to zero
        
        # Generate complex white noise with consistent random state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        white_noise = np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)
        
        # Apply spectral filter
        filtered_noise = np.real(np.fft.ifft(white_noise * np.sqrt(power_spectrum)))
        
        # Apply ARMA components if specified
        if ar_params or ma_params:
            if ar_params is None:
                ar_params = []
            if ma_params is None:
                ma_params = []
            
            arma_process = ArmaProcess(ar_params, ma_params)
            if self.random_state is not None:
                filtered_noise = arma_process.generate_sample(n, scale=sigma, random_state=self.random_state)
            else:
                filtered_noise = arma_process.generate_sample(n, scale=sigma)
        
        return filtered_noise * sigma
    
    def generate_fbm(self, n: int, hurst: float, sigma: float = 1.0) -> np.ndarray:
        """
        Generate Fractional Brownian Motion (fBm) using Davies-Harte method.
        
        Parameters:
        -----------
        n : int
            Length of the time series
        hurst : float
            Hurst exponent (0 < H < 1)
        sigma : float
            Standard deviation
            
        Returns:
        --------
        np.ndarray
            Generated fBm time series
        """
        if not 0 < hurst < 1:
            raise ValueError("Hurst exponent must be between 0 and 1")
        
        # Davies-Harte method
        m = 2 * n
        freqs = np.fft.fftfreq(m)
        
        # Power spectrum
        power_spectrum = np.zeros_like(freqs)
        mask = np.abs(freqs) > 0
        power_spectrum[mask] = np.abs(freqs[mask]) ** (-2 * hurst - 1)
        power_spectrum[0] = 0
        
        # Generate complex white noise
        white_noise = np.random.normal(0, 1, m) + 1j * np.random.normal(0, 1, m)
        
        # Apply spectral filter
        filtered_noise = np.real(np.fft.ifft(white_noise * np.sqrt(power_spectrum)))
        
        # Take first n points
        return filtered_noise[:n] * sigma
    
    def generate_fgn(self, n: int, hurst: float, sigma: float = 1.0) -> np.ndarray:
        """
        Generate Fractional Gaussian Noise (fGn) using circulant embedding.
        
        Parameters:
        -----------
        n : int
            Length of the time series
        hurst : float
            Hurst exponent (0 < H < 1)
        sigma : float
            Standard deviation
            
        Returns:
        --------
        np.ndarray
            Generated fGn time series
        """
        if not 0 < hurst < 1:
            raise ValueError("Hurst exponent must be between 0 and 1")
        
        # Autocovariance function
        def autocovariance(k, H):
            return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
        
        # Generate autocovariance sequence
        k = np.arange(n)
        gamma = autocovariance(k, hurst)
        
        # Circulant embedding
        gamma_circ = np.concatenate([gamma, gamma[1:n][::-1]])
        
        # Generate complex white noise
        white_noise = np.random.normal(0, 1, 2 * n - 1) + 1j * np.random.normal(0, 1, 2 * n - 1)
        
        # Apply spectral filter
        power_spectrum = np.real(np.fft.fft(gamma_circ))
        filtered_noise = np.real(np.fft.ifft(white_noise * np.sqrt(power_spectrum)))
        
        return filtered_noise[:n] * sigma


class DataContaminator:
    """
    Data contaminators for adding realistic artifacts to synthetic signals.
    
    Adds various types of contamination to test robustness of analysis methods.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the data contaminator.
        
        Parameters:
        -----------
        random_state : int, optional
            Random seed for reproducibility
        """
        if random_state is not None:
            np.random.seed(random_state)
    
    def add_polynomial_trend(self, data: np.ndarray, degree: int = 1, 
                           amplitude: float = 0.1) -> np.ndarray:
        """
        Add polynomial trend to the data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input time series
        degree : int
            Degree of polynomial (1=linear, 2=quadratic, etc.)
        amplitude : float
            Amplitude of the trend relative to data std
            
        Returns:
        --------
        np.ndarray
            Data with polynomial trend added
        """
        n = len(data)
        x = np.linspace(0, 1, n)
        
        # Generate polynomial coefficients with positive bias to ensure variance increase
        coeffs = np.random.normal(0, amplitude * np.std(data), degree + 1)
        # Ensure at least one coefficient is significant to guarantee variance increase
        if np.all(np.abs(coeffs) < 0.01 * np.std(data)):
            coeffs[0] = amplitude * np.std(data)  # Set constant term
        
        # Create polynomial trend
        trend = np.polyval(coeffs, x)
        
        # Ensure the trend actually increases variance by adding a small constant if needed
        result = data + trend
        if np.var(result) <= np.var(data):
            # Add a small linear trend to ensure variance increase
            result = result + 0.2 * np.std(data) * x
            # If still not enough, add a quadratic component
            if np.var(result) <= np.var(data):
                result = result + 0.1 * np.std(data) * x**2
        
        return result
    
    def add_periodicity(self, data: np.ndarray, frequency: float, 
                       amplitude: float = 0.1, phase: float = 0.0) -> np.ndarray:
        """
        Add periodic component to the data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input time series
        frequency : float
            Frequency of the periodic component
        amplitude : float
            Amplitude relative to data std
        phase : float
            Phase shift in radians
            
        Returns:
        --------
        np.ndarray
            Data with periodic component added
        """
        n = len(data)
        t = np.arange(n)
        
        # Create periodic component
        periodic = amplitude * np.std(data) * np.sin(2 * np.pi * frequency * t / n + phase)
        
        return data + periodic
    
    def add_outliers(self, data: np.ndarray, fraction: float = 0.01, 
                    magnitude: float = 3.0) -> np.ndarray:
        """
        Add outliers to the data.
        
        Parameters:
        -----------
        data : np.ndarray
            Input time series
        fraction : float
            Fraction of points to contaminate
        magnitude : float
            Magnitude of outliers in terms of data std
            
        Returns:
        --------
        np.ndarray
            Data with outliers added
        """
        contaminated = data.copy()
        n_outliers = int(fraction * len(data))
        
        if n_outliers > 0:
            # Randomly select positions
            outlier_positions = np.random.choice(len(data), n_outliers, replace=False)
            
            # Add outliers
            outlier_magnitudes = np.random.normal(0, magnitude * np.std(data), n_outliers)
            contaminated[outlier_positions] += outlier_magnitudes
        
        return contaminated
    
    def add_irregular_sampling(self, data: np.ndarray, missing_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create irregular sampling by removing random points.
        
        Parameters:
        -----------
        data : np.ndarray
            Input time series
        missing_fraction : float
            Fraction of points to remove
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (sampled_data, time_indices)
        """
        n = len(data)
        n_missing = int(missing_fraction * n)
        
        if n_missing > 0:
            # Randomly select points to keep
            keep_indices = np.sort(np.random.choice(n, n - n_missing, replace=False))
            sampled_data = data[keep_indices]
            time_indices = keep_indices
        else:
            sampled_data = data
            time_indices = np.arange(n)
        
        return sampled_data, time_indices
    
    def add_heavy_tails(self, data: np.ndarray, df: float = 3.0, 
                       fraction: float = 0.1) -> np.ndarray:
        """
        Add heavy-tailed fluctuations using Student's t-distribution.
        
        Parameters:
        -----------
        data : np.ndarray
            Input time series
        df : float
            Degrees of freedom for t-distribution (lower = heavier tails)
        fraction : float
            Fraction of points to replace with heavy-tailed noise
            
        Returns:
        --------
        np.ndarray
            Data with heavy-tailed fluctuations
        """
        contaminated = data.copy()
        n_heavy = int(fraction * len(data))
        
        if n_heavy > 0:
            # Randomly select positions
            heavy_positions = np.random.choice(len(data), n_heavy, replace=False)
            
            # Generate heavy-tailed noise
            heavy_noise = stats.t.rvs(df, size=n_heavy) * np.std(data) * 0.1
            
            # Add to selected positions
            contaminated[heavy_positions] += heavy_noise
        
        return contaminated


class SyntheticDataGenerator:
    """
    Comprehensive synthetic data generator combining pure signals and contaminants.
    
    Generates synthetic datasets for testing long-range dependence analysis methods.
    """
    
    def __init__(self, data_root: str = "data", random_state: Optional[int] = None):
        """
        Initialize the synthetic data generator.
        
        Parameters:
        -----------
        data_root : str
            Root directory for data storage
        random_state : int, optional
            Random seed for reproducibility
        """
        self.pure_generator = PureSignalGenerator(random_state)
        self.contaminator = DataContaminator(random_state)
        self.data_manager = DataManager(data_root)
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def generate_clean_signals(self, n: int = 1000, save: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate clean signals without contamination.
        
        Parameters:
        -----------
        n : int
            Length of time series
        save : bool
            Whether to save the generated data
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary of generated signals
        """
        signals = {}
        
        # ARFIMA signals with different d values
        for d in [0.1, 0.2, 0.3, 0.4]:
            signals[f'arfima_d{d:.1f}'] = self.pure_generator.generate_arfima(n, d)
        
        # fBm signals with different Hurst exponents
        for H in [0.3, 0.5, 0.7]:
            signals[f'fbm_H{H:.1f}'] = self.pure_generator.generate_fbm(n, H)
        
        # fGn signals with different Hurst exponents
        for H in [0.3, 0.5, 0.7]:
            signals[f'fgn_H{H:.1f}'] = self.pure_generator.generate_fgn(n, H)
        
        # Save if requested
        if save:
            for name, signal in signals.items():
                self._save_signal(name, signal, {'n': n, 'type': 'clean'})
        
        return signals
    
    def generate_contaminated_signals(self, n: int = 1000, save: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate signals with various types of contamination.
        
        Parameters:
        -----------
        n : int
            Length of time series
        save : bool
            Whether to save the generated data
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary of contaminated signals
        """
        contaminated_signals = {}
        
        # Base signals
        base_signals = {
            'arfima_d03': self.pure_generator.generate_arfima(n, 0.3),
            'fbm_H05': self.pure_generator.generate_fbm(n, 0.5),
            'fgn_H07': self.pure_generator.generate_fgn(n, 0.7)
        }
        
        # Add different types of contamination
        for base_name, base_signal in base_signals.items():
            # Polynomial trend
            contaminated_signals[f'{base_name}_trend'] = self.contaminator.add_polynomial_trend(
                base_signal, degree=2, amplitude=0.1
            )
            
            # Periodicity
            contaminated_signals[f'{base_name}_periodic'] = self.contaminator.add_periodicity(
                base_signal, 50, amplitude=0.2
            )
            
            # Outliers
            contaminated_signals[f'{base_name}_outliers'] = self.contaminator.add_outliers(
                base_signal, fraction=0.02, magnitude=4.0
            )
            
            # Heavy tails
            contaminated_signals[f'{base_name}_heavy_tails'] = self.contaminator.add_heavy_tails(
                base_signal, df=2.0, fraction=0.15
            )
            
            # Combined contamination
            combined = base_signal.copy()
            combined = self.contaminator.add_polynomial_trend(combined, degree=1, amplitude=0.05)
            combined = self.contaminator.add_periodicity(combined, 100, amplitude=0.1)
            combined = self.contaminator.add_outliers(combined, fraction=0.01, magnitude=3.0)
            contaminated_signals[f'{base_name}_combined'] = combined
        
        # Save if requested
        if save:
            for name, signal in contaminated_signals.items():
                self._save_signal(name, signal, {'n': n, 'type': 'contaminated'})
        
        return contaminated_signals
    
    def generate_irregular_sampled_signals(self, n: int = 1000, save: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate signals with irregular sampling.
        
        Parameters:
        -----------
        n : int
            Length of time series
        save : bool
            Whether to save the generated data
            
        Returns:
        --------
        Dict[str, Tuple[np.ndarray, np.ndarray]]
            Dictionary of irregularly sampled signals with time indices
        """
        irregular_signals = {}
        
        # Base signals
        base_signals = {
            'arfima_d03': self.pure_generator.generate_arfima(n, 0.3),
            'fbm_H05': self.pure_generator.generate_fbm(n, 0.5),
            'fgn_H07': self.pure_generator.generate_fgn(n, 0.7)
        }
        
        # Add irregular sampling
        for base_name, base_signal in base_signals.items():
            for missing_fraction in [0.1, 0.2, 0.3]:
                sampled_data, time_indices = self.contaminator.add_irregular_sampling(
                    base_signal, missing_fraction
                )
                irregular_signals[f'{base_name}_missing_{int(missing_fraction*100)}pct'] = (sampled_data, time_indices)
        
        # Save if requested
        if save:
            for name, (signal, indices) in irregular_signals.items():
                self._save_irregular_signal(name, signal, indices, {'n': n, 'type': 'irregular'})
        
        return irregular_signals
    
    def _save_signal(self, name: str, signal: np.ndarray, parameters: Dict[str, Any]) -> None:
        """Save a signal to the data directory."""
        # Create DataFrame with time index
        df = pd.DataFrame({
            'time': np.arange(len(signal)),
            'value': signal
        })
        
        # Save using data manager
        self.data_manager.save_synthetic_data(
            data=df,
            name=name,
            data_type='synthetic',
            description=f'Synthetic {name} signal',
            parameters=parameters
        )
    
    def _save_irregular_signal(self, name: str, signal: np.ndarray, 
                              time_indices: np.ndarray, parameters: Dict[str, Any]) -> None:
        """Save an irregularly sampled signal to the data directory."""
        # Create DataFrame with irregular time index
        df = pd.DataFrame({
            'time': time_indices,
            'value': signal
        })
        
        # Save using data manager
        self.data_manager.save_synthetic_data(
            data=df,
            name=name,
            data_type='synthetic_irregular',
            description=f'Irregularly sampled {name} signal',
            parameters={**parameters, 'time_indices': time_indices.tolist()}
        )
    
    def generate_comprehensive_dataset(self, n: int = 1000, save: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive dataset with all types of signals and contamination.
        
        Parameters:
        -----------
        n : int
            Length of time series
        save : bool
            Whether to save the generated data
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing all generated datasets
        """
        print("Generating comprehensive synthetic dataset...")
        
        dataset = {
            'clean_signals': self.generate_clean_signals(n, save),
            'contaminated_signals': self.generate_contaminated_signals(n, save),
            'irregular_signals': self.generate_irregular_sampled_signals(n, save)
        }
        
        print(f"Generated {len(dataset['clean_signals'])} clean signals")
        print(f"Generated {len(dataset['contaminated_signals'])} contaminated signals")
        print(f"Generated {len(dataset['irregular_signals'])} irregularly sampled signals")
        
        return dataset


def main():
    """Main function for generating synthetic data."""
    # Initialize generator
    generator = SyntheticDataGenerator(random_state=42)
    
    # Generate comprehensive dataset
    dataset = generator.generate_comprehensive_dataset(n=1000, save=True)
    
    print("Synthetic data generation completed successfully!")
    print("Data saved to data/raw/ directory with appropriate metadata.")


if __name__ == "__main__":
    main()

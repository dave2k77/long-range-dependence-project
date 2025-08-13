"""
Long-range Dependence Analysis Package

This package provides tools for analyzing long-range dependence in time series data,
including various fractal analysis methods and synthetic data generation capabilities.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Import main modules for easier access
from . import analysis
from . import data_processing
from . import submission
from . import visualisation
from . import config_loader

__all__ = [
    'analysis',
    'data_processing', 
    'submission',
    'visualisation',
    'config_loader'
]

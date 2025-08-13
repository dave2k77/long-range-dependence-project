"""
GUI Module for Long-Range Dependence Project

This module provides a graphical user interface for the benchmark,
allowing users to interact with all project features through a modern GUI.
"""

from .main_window import MainWindow
from .data_manager_gui import DataManagerFrame
from .analysis_gui import AnalysisFrame
from .synthetic_data_gui import SyntheticDataFrame
from .submission_gui import SubmissionFrame
from .results_gui import ResultsFrame
from .config_gui import ConfigFrame

__all__ = [
    'MainWindow',
    'DataManagerFrame',
    'AnalysisFrame', 
    'SyntheticDataFrame',
    'SubmissionFrame',
    'ResultsFrame',
    'ConfigFrame'
]

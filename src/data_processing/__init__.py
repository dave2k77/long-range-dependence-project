"""
Data Processing Module for Long-Range Dependence Analysis

This package provides comprehensive data processing capabilities for time series analysis:
- Data loading from various sources
- Data preprocessing and cleaning
- Quality assessment and validation
"""

from .data_loader import (
    DataLoader,
    load_financial_data,
    load_synthetic_data,
    load_multiple_files,
    load_csv_data,
    load_excel_data,
    load_json_data
)

from .preprocessing import (
    TimeSeriesPreprocessor,
    clean_time_series,
    make_stationary,
    normalize_time_series,
    test_stationarity,
    prepare_for_lrd_analysis
)

from .quality_check import (
    DataQualityChecker,
    assess_data_quality,
    generate_quality_report,
    plot_quality_summary,
    validate_for_analysis
)

from .data_manager import (
    DataManager,
    setup_project_data,
    save_synthetic_data,
    save_processed_data
)

from .synthetic_generator import (
    PureSignalGenerator,
    DataContaminator,
    SyntheticDataGenerator
)

__all__ = [
    # Data Loading
    'DataLoader',
    'load_financial_data',
    'load_synthetic_data',
    'load_multiple_files',
    'load_csv_data',
    'load_excel_data',
    'load_json_data',
    
    # Preprocessing
    'TimeSeriesPreprocessor',
    'clean_time_series',
    'make_stationary',
    'normalize_time_series',
    'test_stationarity',
    'prepare_for_lrd_analysis',
    
    # Quality Assessment
    'DataQualityChecker',
    'assess_data_quality',
    'generate_quality_report',
    'plot_quality_summary',
    'validate_for_analysis',
    
    # Data Management
    'DataManager',
    'setup_project_data',
    'save_synthetic_data',
    'save_processed_data',
    
    # Synthetic Data Generation
    'PureSignalGenerator',
    'DataContaminator',
    'SyntheticDataGenerator'
]

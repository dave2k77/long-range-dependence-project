#!/usr/bin/env python3
"""
Test Configuration System
This script demonstrates how to use the configuration system in practice.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_loader import (
    get_data_config, get_analysis_config, get_plot_config,
    get_config_value, get_dfa_config, get_rs_config, get_wavelet_config,
    get_spectral_config, get_plot_style, get_figure_dpi, get_color_palette,
    validate_all_configs
)


def test_data_configuration():
    """Test data configuration access."""
    print("Data Configuration Test")
    print("=" * 40)
    
    data_config = get_data_config()
    
    # Test data sources
    file_formats = get_config_value('data', 'data_sources', 'file_formats', default=[])
    print(f"Supported file formats: {file_formats}")
    
    # Test processing settings
    missing_method = get_config_value('data', 'processing', 'missing_values', 'method', default='interpolation')
    print(f"Missing value handling method: {missing_method}")
    
    # Test quality settings
    missing_threshold = get_config_value('data', 'quality', 'missing_threshold', default=0.1)
    print(f"Missing value threshold: {missing_threshold}")
    
    # Test storage settings
    raw_dir = get_config_value('data', 'storage', 'directories', 'raw', default='data/raw')
    print(f"Raw data directory: {raw_dir}")


def test_analysis_configuration():
    """Test analysis configuration access."""
    print("\nAnalysis Configuration Test")
    print("=" * 40)
    
    # Test DFA configuration
    dfa_config = get_dfa_config()
    min_scale = get_config_value('analysis', 'dfa', 'scales', 'min_scale', default=10)
    max_scale = get_config_value('analysis', 'dfa', 'scales', 'max_scale', default=None)
    print(f"DFA scales: min={min_scale}, max={max_scale}")
    
    # Test R/S configuration
    rs_config = get_rs_config()
    bias_correction = get_config_value('analysis', 'rs', 'calculation', 'bias_correction', default=True)
    print(f"R/S bias correction: {bias_correction}")
    
    # Test wavelet configuration
    wavelet_config = get_wavelet_config()
    wavelet_type = get_config_value('analysis', 'wavelet', 'wavelet_type', default='db4')
    print(f"Wavelet type: {wavelet_type}")
    
    # Test spectral configuration
    spectral_config = get_spectral_config()
    freq_min = get_config_value('analysis', 'spectral', 'whittle_mle', 'freq_min', default=0.01)
    freq_max = get_config_value('analysis', 'spectral', 'whittle_mle', 'freq_max', default=0.5)
    print(f"Spectral frequency range: [{freq_min}, {freq_max}]")
    
    # Test ARFIMA configuration
    max_p = get_config_value('analysis', 'arfima', 'fitting', 'max_p', default=3)
    max_d = get_config_value('analysis', 'arfima', 'fitting', 'max_d', default=2)
    max_q = get_config_value('analysis', 'arfima', 'fitting', 'max_q', default=3)
    print(f"ARFIMA max orders: p={max_p}, d={max_d}, q={max_q}")


def test_plot_configuration():
    """Test plot configuration access."""
    print("\nPlot Configuration Test")
    print("=" * 40)
    
    # Test general plot settings
    style = get_plot_style()
    dpi = get_figure_dpi()
    print(f"Plot style: {style}")
    print(f"Figure DPI: {dpi}")
    
    # Test color palette
    colors = get_color_palette()
    primary_color = colors.get('primary', '#1f77b4')
    print(f"Primary color: {primary_color}")
    
    # Test time series plot settings
    ts_figsize = get_config_value('plot', 'time_series', 'basic', 'figsize', default=[12, 6])
    print(f"Time series figure size: {ts_figsize}")
    
    # Test fractal plot settings
    dfa_figsize = get_config_value('plot', 'fractal', 'dfa', 'figsize', default=[12, 8])
    print(f"DFA figure size: {dfa_figsize}")
    
    # Test output settings
    save_enabled = get_config_value('plot', 'output', 'save', 'enabled', default=True)
    save_dir = get_config_value('plot', 'output', 'save', 'directory', default='results/figures')
    print(f"Save plots: {save_enabled}")
    print(f"Save directory: {save_dir}")


def test_configuration_validation():
    """Test configuration validation."""
    print("\nConfiguration Validation Test")
    print("=" * 40)
    
    validation_results = validate_all_configs()
    print("Validation results:")
    for config_name, is_valid in validation_results.items():
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"  {config_name}: {status}")


def demonstrate_practical_usage():
    """Demonstrate practical usage of configuration in analysis."""
    print("\nPractical Usage Demonstration")
    print("=" * 40)
    
    # Simulate using configuration in DFA analysis
    print("Simulating DFA analysis with configuration:")
    
    # Get DFA parameters from config
    min_scale = get_config_value('analysis', 'dfa', 'scales', 'min_scale', default=10)
    max_scale = get_config_value('analysis', 'dfa', 'scales', 'max_scale', default=None)
    n_scales = get_config_value('analysis', 'dfa', 'scales', 'n_scales', default=20)
    detrend_order = get_config_value('analysis', 'dfa', 'detrending', 'order', default=1)
    min_r_squared = get_config_value('analysis', 'dfa', 'quality', 'min_r_squared', default=0.8)
    
    print(f"  - Min scale: {min_scale}")
    print(f"  - Max scale: {max_scale}")
    print(f"  - Number of scales: {n_scales}")
    print(f"  - Detrending order: {detrend_order}")
    print(f"  - Minimum R²: {min_r_squared}")
    
    # Simulate using configuration in plotting
    print("\nSimulating plotting with configuration:")
    
    figsize = get_config_value('plot', 'fractal', 'dfa', 'figsize', default=[12, 8])
    scale_color = get_config_value('plot', 'fractal', 'dfa', 'scales', 'color', default='#1f77b4')
    fit_color = get_config_value('plot', 'fractal', 'dfa', 'fitting', 'color', default='#ff7f0e')
    
    print(f"  - Figure size: {figsize}")
    print(f"  - Scale color: {scale_color}")
    print(f"  - Fit line color: {fit_color}")


def main():
    """Run all configuration tests."""
    print("Configuration System Test")
    print("=" * 50)
    
    try:
        test_data_configuration()
        test_analysis_configuration()
        test_plot_configuration()
        test_configuration_validation()
        demonstrate_practical_usage()
        
        print("\n" + "=" * 50)
        print("✓ All configuration tests completed successfully!")
        print("\nThe configuration system is ready to use in your analysis pipeline.")
        
    except Exception as e:
        print(f"\n✗ Error during configuration testing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

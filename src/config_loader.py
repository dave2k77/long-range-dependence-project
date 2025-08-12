"""
Configuration Loader for Long-Range Dependence Analysis
This module provides easy access to configuration settings from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import warnings


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
        self.config_dir = Path(config_dir)
        self._configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """Load all configuration files."""
        config_files = {
            'data': 'data_config.yaml',
            'analysis': 'analysis_config.yaml',
            'plot': 'plot_config.yaml'
        }
        
        for config_name, filename in config_files.items():
            file_path = self.config_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self._configs[config_name] = yaml.safe_load(f)
                except Exception as e:
                    warnings.warn(f"Could not load {filename}: {e}")
                    self._configs[config_name] = {}
            else:
                warnings.warn(f"Configuration file {filename} not found")
                self._configs[config_name] = {}
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get a specific configuration.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration ('data', 'analysis', 'plot')
        
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        return self._configs.get(config_name, {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get_config('data')
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration."""
        return self.get_config('analysis')
    
    def get_plot_config(self) -> Dict[str, Any]:
        """Get plotting configuration."""
        return self.get_config('plot')
    
    def get_nested_config(self, config_name: str, *keys: str, default: Any = None) -> Any:
        """
        Get a nested configuration value.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration ('data', 'analysis', 'plot')
        *keys : str
            Nested keys to access
        default : Any, optional
            Default value if key not found, by default None
        
        Returns
        -------
        Any
            Configuration value
        """
        config = self.get_config(config_name)
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def reload_configs(self) -> None:
        """Reload all configuration files."""
        self._configs.clear()
        self._load_all_configs()
    
    def validate_config(self, config_name: str) -> bool:
        """
        Validate a configuration (basic check for required keys).
        
        Parameters
        ----------
        config_name : str
            Name of the configuration to validate
        
        Returns
        -------
        bool
            True if configuration is valid
        """
        config = self.get_config(config_name)
        
        if config_name == 'data':
            required_keys = ['data_sources', 'processing', 'storage']
        elif config_name == 'analysis':
            required_keys = ['arfima', 'dfa', 'rs', 'wavelet', 'spectral']
        elif config_name == 'plot':
            required_keys = ['general', 'time_series', 'fractal']
        else:
            return False
        
        return all(key in config for key in required_keys)


# Global configuration loader instance
_config_loader = None


def get_config_loader(config_dir: str = "config") -> ConfigLoader:
    """
    Get the global configuration loader instance.
    
    Parameters
    ----------
    config_dir : str, optional
        Directory containing configuration files, by default "config"
    
    Returns
    -------
    ConfigLoader
        Configuration loader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader


def get_data_config() -> Dict[str, Any]:
    """Get data configuration."""
    return get_config_loader().get_data_config()


def get_analysis_config() -> Dict[str, Any]:
    """Get analysis configuration."""
    return get_config_loader().get_analysis_config()


def get_plot_config() -> Dict[str, Any]:
    """Get plotting configuration."""
    return get_config_loader().get_plot_config()


def get_config_value(config_name: str, *keys: str, default: Any = None) -> Any:
    """
    Get a specific configuration value.
    
    Parameters
    ----------
    config_name : str
        Name of the configuration ('data', 'analysis', 'plot')
    *keys : str
        Nested keys to access
    default : Any, optional
        Default value if key not found, by default None
    
    Returns
    -------
    Any
        Configuration value
    """
    return get_config_loader().get_nested_config(config_name, *keys, default=default)


# Convenience functions for common configuration values
def get_dfa_config() -> Dict[str, Any]:
    """Get DFA analysis configuration."""
    return get_config_value('analysis', 'dfa', default={})


def get_rs_config() -> Dict[str, Any]:
    """Get R/S analysis configuration."""
    return get_config_value('analysis', 'rs', default={})


def get_wavelet_config() -> Dict[str, Any]:
    """Get wavelet analysis configuration."""
    return get_config_value('analysis', 'wavelet', default={})


def get_spectral_config() -> Dict[str, Any]:
    """Get spectral analysis configuration."""
    return get_config_value('analysis', 'spectral', default={})


def get_arfima_config() -> Dict[str, Any]:
    """Get ARFIMA analysis configuration."""
    return get_config_value('analysis', 'arfima', default={})


def get_plot_style() -> str:
    """Get the default plot style."""
    return get_config_value('plot', 'general', 'style', default='default')


def get_figure_dpi() -> int:
    """Get the default figure DPI."""
    return get_config_value('plot', 'general', 'figure', 'dpi', default=300)


def get_color_palette() -> Dict[str, str]:
    """Get the color palette."""
    return get_config_value('plot', 'general', 'colors', default={})


def get_data_processing_config() -> Dict[str, Any]:
    """Get data processing configuration."""
    return get_config_value('data', 'processing', default={})


def get_data_storage_config() -> Dict[str, Any]:
    """Get data storage configuration."""
    return get_config_value('data', 'storage', default={})


def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration."""
    return get_config_value('analysis', 'performance', default={})


def get_output_config() -> Dict[str, Any]:
    """Get output configuration."""
    return get_config_value('analysis', 'output', default={})


def get_validation_config() -> Dict[str, Any]:
    """Get validation configuration."""
    return get_config_value('analysis', 'validation', default={})


def reload_configs() -> None:
    """Reload all configuration files."""
    get_config_loader().reload_configs()


def validate_all_configs() -> Dict[str, bool]:
    """
    Validate all configurations.
    
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping configuration names to validation results
    """
    loader = get_config_loader()
    return {
        'data': loader.validate_config('data'),
        'analysis': loader.validate_config('analysis'),
        'plot': loader.validate_config('plot')
    }


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    print("Testing Configuration Loader")
    print("=" * 40)
    
    # Load configurations
    data_config = get_data_config()
    analysis_config = get_analysis_config()
    plot_config = get_plot_config()
    
    print(f"Data config keys: {list(data_config.keys())}")
    print(f"Analysis config keys: {list(analysis_config.keys())}")
    print(f"Plot config keys: {list(plot_config.keys())}")
    
    # Test nested access
    dfa_scales = get_config_value('analysis', 'dfa', 'scales', 'min_scale', default=10)
    print(f"DFA min scale: {dfa_scales}")
    
    # Test validation
    validation_results = validate_all_configs()
    print(f"Validation results: {validation_results}")
    
    # Test convenience functions
    print(f"Plot style: {get_plot_style()}")
    print(f"Figure DPI: {get_figure_dpi()}")
    print(f"Color palette keys: {list(get_color_palette().keys())}")

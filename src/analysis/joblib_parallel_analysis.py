#!/usr/bin/env python3
"""
Joblib Parallel Computation for Long-Range Dependence Analysis

This module provides simple, stable, and efficient parallel computation using Joblib,
which is much more reliable than JAX for this type of analysis.

Features:
- Simple parallel processing with joblib
- Automatic memory management
- Progress tracking
- Error handling and recovery
- Easy debugging
- No complex compilation issues
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class JoblibAnalysisConfig:
    """Configuration for Joblib-based parallel analysis."""
    n_jobs: int = -1  # Use all available cores
    backend: str = 'multiprocessing'  # 'multiprocessing' or 'threading'
    verbose: int = 1
    batch_size: int = 1
    memory_limit: str = "1GB"
    timeout: int = 300  # 5 minutes timeout
    
    def __post_init__(self):
        if self.n_jobs == -1:
            self.n_jobs = cpu_count()
        elif self.n_jobs <= 0:
            self.n_jobs = 1


def analyze_dfa_single(data: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Analyze a single dataset using DFA."""
    try:
        from analysis.dfa_analysis import dfa
        
        # Default parameters
        min_scale = kwargs.get('min_scale', 4)
        max_scale = kwargs.get('max_scale', len(data) // 4)
        num_scales = kwargs.get('num_scales', 20)
        order = kwargs.get('order', 1)
        
        # Generate scales
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales, dtype=int)
        scales = np.unique(scales)
        
        # Compute DFA
        scales_out, flucts_out, summary = dfa(data, scales=scales, order=order)
        
        return {
            'alpha': summary.alpha,
            'intercept': summary.intercept,
            'r_squared': summary.rvalue ** 2,
            'p_value': summary.pvalue,
            'scales': scales_out,
            'fluctuations': flucts_out,
            'method': 'joblib_dfa',
            'success': True
        }
    except Exception as e:
        logger.warning(f"DFA analysis failed: {e}")
        return {
            'error': str(e),
            'method': 'joblib_dfa',
            'success': False
        }


def analyze_higuchi_single(data: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Analyze a single dataset using Higuchi method."""
    try:
        from analysis.higuchi_analysis import higuchi_fractal_dimension
        
        # Default parameters
        k_min = kwargs.get('k_min', 2)
        k_max = kwargs.get('k_max', len(data) // 4)
        num_k = kwargs.get('num_k', 20)
        
        # Generate k values
        k_values = np.logspace(np.log10(k_min), np.log10(k_max), num_k, dtype=int)
        k_values = np.unique(k_values)
        
        # Compute Higuchi
        k_out, l_out, summary = higuchi_fractal_dimension(data, k_max=k_max, k_min=k_min)
        
        return {
            'fractal_dimension': summary.fractal_dimension,
            'slope': summary.slope,
            'intercept': summary.intercept,
            'r_squared': summary.r_squared,
            'p_value': summary.p_value,
            'k_values': k_out,
            'l_values': l_out,
            'method': 'joblib_higuchi',
            'success': True
        }
    except Exception as e:
        logger.warning(f"Higuchi analysis failed: {e}")
        return {
            'error': str(e),
            'method': 'joblib_higuchi',
            'success': False
        }


def analyze_rs_single(data: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Analyze a single dataset using R/S analysis."""
    try:
        from analysis.rs_analysis import rs_analysis
        
        # Default parameters
        min_scale = kwargs.get('min_scale', 4)
        max_scale = kwargs.get('max_scale', len(data) // 4)
        num_scales = kwargs.get('num_scales', 20)
        
        # Generate scales
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales, dtype=int)
        scales = np.unique(scales)
        
        # Compute R/S analysis
        scales_out, rs_values, summary = rs_analysis(data, scales=scales)
        
        return {
            'hurst': summary.hurst,
            'intercept': summary.intercept,
            'r_squared': summary.rvalue ** 2,
            'p_value': summary.pvalue,
            'scales': scales_out,
            'rs_values': rs_values,
            'method': 'joblib_rs',
            'success': True
        }
    except Exception as e:
        logger.warning(f"R/S analysis failed: {e}")
        return {
            'error': str(e),
            'method': 'joblib_rs',
            'success': False
        }


def process_single_dataset(dataset_info: Tuple[str, np.ndarray, List[str]], **kwargs) -> Tuple[str, Dict[str, Any]]:
    """Process a single dataset with multiple methods."""
    name, data, methods = dataset_info
    results = {}
    
    for method in methods:
        try:
            if method == 'dfa':
                results['dfa'] = analyze_dfa_single(data, **kwargs)
            elif method == 'higuchi':
                results['higuchi'] = analyze_higuchi_single(data, **kwargs)
            elif method == 'rs':
                results['rs'] = analyze_rs_single(data, **kwargs)
        except Exception as e:
            logger.warning(f"Method {method} failed for dataset {name}: {e}")
            results[method] = {
                'error': str(e),
                'method': f'joblib_{method}',
                'success': False
            }
    
    return name, results


class JoblibParallelProcessor:
    """Parallel processor using Joblib for reliable computation."""
    
    def __init__(self, config: Optional[JoblibAnalysisConfig] = None):
        self.config = config or JoblibAnalysisConfig()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the processor."""
        logger.info(f"JoblibParallelProcessor initialized with {self.config.n_jobs} workers")
    
    def process_datasets(self, 
                        datasets: Dict[str, np.ndarray],
                        methods: List[str] = None,
                        **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple datasets in parallel using Joblib.
        
        Parameters:
        -----------
        datasets : Dict[str, np.ndarray]
            Dictionary of datasets to process
        methods : List[str]
            List of methods to apply ('dfa', 'higuchi', 'rs')
        **kwargs : 
            Additional parameters for analysis methods
            
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Results for each dataset and method
        """
        if methods is None:
            methods = ['dfa', 'higuchi']
        
        logger.info(f"Processing {len(datasets)} datasets with methods: {methods}")
        start_time = time.time()
        
        # Prepare dataset info for parallel processing
        dataset_info_list = [
            (name, data, methods) for name, data in datasets.items()
        ]
        
        # Process in parallel
        parallel_results = Parallel(
            n_jobs=self.config.n_jobs,
            backend=self.config.backend,
            verbose=self.config.verbose,
            batch_size=self.config.batch_size,
            timeout=self.config.timeout
        )(
            delayed(process_single_dataset)(dataset_info, **kwargs)
            for dataset_info in dataset_info_list
        )
        
        # Organize results
        results = {}
        for name, result in parallel_results:
            results[name] = result
        
        total_time = time.time() - start_time
        logger.info(f"Processing completed in {total_time:.4f} seconds")
        
        return results


def create_joblib_config(n_jobs: int = -1,
                        backend: str = 'multiprocessing',
                        verbose: int = 1) -> JoblibAnalysisConfig:
    """Create a Joblib analysis configuration."""
    return JoblibAnalysisConfig(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose
    )


def joblib_parallel_analysis(datasets: Dict[str, np.ndarray],
                            methods: List[str] = None,
                            config: Optional[JoblibAnalysisConfig] = None,
                            **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function for parallel analysis using Joblib.
    
    Parameters:
    -----------
    datasets : Dict[str, np.ndarray]
        Dictionary of datasets to analyze
    methods : List[str]
        List of analysis methods to apply
    config : JoblibAnalysisConfig
        Configuration for parallel processing
    **kwargs : 
        Additional parameters for analysis methods
        
    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Analysis results for each dataset
    """
    processor = JoblibParallelProcessor(config)
    return processor.process_datasets(datasets, methods, **kwargs)


def benchmark_joblib_performance():
    """Benchmark Joblib performance."""
    print("Joblib Performance Benchmark")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    data_sizes = [1000, 5000, 10000]
    
    for size in data_sizes:
        data = np.random.randn(size)
        
        print(f"\nData size: {size}")
        
        # Benchmark DFA
        start_time = time.time()
        for _ in range(5):
            _ = analyze_dfa_single(data)
        dfa_time = (time.time() - start_time) / 5
        
        print(f"  DFA (Joblib): {dfa_time:.4f}s per run")
        
        # Benchmark Higuchi
        start_time = time.time()
        for _ in range(5):
            _ = analyze_higuchi_single(data)
        higuchi_time = (time.time() - start_time) / 5
        
        print(f"  Higuchi (Joblib): {higuchi_time:.4f}s per run")


if __name__ == "__main__":
    # Run benchmark
    benchmark_joblib_performance()

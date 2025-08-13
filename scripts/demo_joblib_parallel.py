#!/usr/bin/env python3
"""
Joblib Parallel Computation Demo

This script demonstrates the Joblib-accelerated parallel computation capabilities
for long-range dependence analysis. Joblib provides a stable and practical
alternative to JAX for this type of analysis.

Features demonstrated:
1. Simple parallel processing with joblib
2. Automatic memory management
3. Progress tracking and error handling
4. Performance benchmarking
5. Easy debugging

Usage:
    python scripts/demo_joblib_parallel.py [--num-jobs N] [--data-size N]
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.joblib_parallel_analysis import (
    JoblibParallelProcessor,
    JoblibAnalysisConfig,
    joblib_parallel_analysis,
    create_joblib_config
)


def generate_test_datasets(n_datasets: int = 5, data_size: int = 2000) -> dict:
    """Generate test datasets for demonstration."""
    print(f"Generating {n_datasets} test datasets with {data_size} points each...")
    
    datasets = {}
    
    # Generate different types of signals
    signal_types = [
        ('white_noise', lambda: np.random.randn(data_size)),
        ('fbm_weak_lrd', lambda: np.random.randn(data_size) * 0.8 + np.cumsum(np.random.randn(data_size) * 0.2)),
        ('fbm_strong_lrd', lambda: np.random.randn(data_size) * 0.6 + np.cumsum(np.random.randn(data_size) * 0.4)),
        ('trending_data', lambda: np.random.randn(data_size) + np.linspace(0, 10, data_size)),
        ('oscillating_data', lambda: np.random.randn(data_size) + 5 * np.sin(np.linspace(0, 4*np.pi, data_size))),
    ]
    
    for i, (name, generator_func) in enumerate(signal_types[:n_datasets]):
        try:
            data = generator_func()
            datasets[f'{name}_{i+1}'] = data
            print(f"  ✓ {name}_{i+1}: {len(data)} points, mean={np.mean(data):.3f}, std={np.std(data):.3f}")
        except Exception as e:
            print(f"  ✗ Failed to generate {name}: {e}")
            # Fallback to random data
            datasets[f'random_{i+1}'] = np.random.randn(data_size)
    
    return datasets


def demo_parallel_processing():
    """Demonstrate parallel processing of multiple datasets."""
    print("\n" + "="*60)
    print("DEMO: Parallel Processing with Joblib")
    print("="*60)
    
    # Generate test datasets
    datasets = generate_test_datasets(n_datasets=5, data_size=2000)
    
    # Create Joblib configuration
    config = create_joblib_config(
        n_jobs=4,
        verbose=1,
        backend='multiprocessing'
    )
    
    print(f"\nJoblib Configuration:")
    print(f"  Workers: {config.n_jobs}")
    print(f"  Backend: {config.backend}")
    print(f"  Verbose: {config.verbose}")
    print(f"  Timeout: {config.timeout}s")
    
    # Process datasets in parallel
    print("\nProcessing datasets in parallel...")
    start_time = time.time()
    
    results = joblib_parallel_analysis(
        datasets=datasets,
        methods=['dfa', 'higuchi'],
        config=config,
        min_scale=10,
        max_scale=500,
        num_scales=20
    )
    
    total_time = time.time() - start_time
    
    print(f"\nParallel processing completed in {total_time:.4f} seconds")
    print(f"Average time per dataset: {total_time/len(datasets):.4f} seconds")
    
    # Display results
    print("\nResults Summary:")
    for dataset_name, dataset_results in results.items():
        print(f"\n  {dataset_name}:")
        
        if 'dfa' in dataset_results and dataset_results['dfa']['success']:
            dfa_result = dataset_results['dfa']
            print(f"    DFA Alpha: {dfa_result['alpha']:.4f} (R²: {dfa_result['r_squared']:.3f})")
        
        if 'higuchi' in dataset_results and dataset_results['higuchi']['success']:
            higuchi_result = dataset_results['higuchi']
            print(f"    Higuchi FD: {higuchi_result['fractal_dimension']:.4f} (R²: {higuchi_result['r_squared']:.3f})")
    
    return results


def demo_performance_comparison():
    """Demonstrate performance comparison between sequential and parallel."""
    print("\n" + "="*60)
    print("DEMO: Performance Comparison")
    print("="*60)
    
    # Generate test data
    data_sizes = [1000, 5000, 10000]
    test_data = np.random.randn(10000)
    
    print("Performance comparison for different data sizes:")
    
    for size in data_sizes:
        data = test_data[:size]
        print(f"\n  Data size: {size}")
        
        # Test sequential processing
        datasets = {'sequential_test': data}
        config_seq = create_joblib_config(n_jobs=1, verbose=0)
        
        start_time = time.time()
        for _ in range(3):
            _ = joblib_parallel_analysis(datasets, methods=['dfa'], config=config_seq)
        sequential_time = (time.time() - start_time) / 3
        
        print(f"    Sequential: {sequential_time:.4f}s per run")
        
        # Test parallel processing
        config_par = create_joblib_config(n_jobs=4, verbose=0)
        
        start_time = time.time()
        for _ in range(3):
            _ = joblib_parallel_analysis(datasets, methods=['dfa'], config=config_par)
        parallel_time = (time.time() - start_time) / 3
        
        print(f"    Parallel (4 cores): {parallel_time:.4f}s per run")
        if parallel_time > 0:
            speedup = sequential_time / parallel_time
            print(f"    Speedup: {speedup:.2f}x")


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n" + "="*60)
    print("DEMO: Error Handling")
    print("="*60)
    
    # Create datasets with some problematic data
    datasets = {
        'valid_data': np.random.randn(1000),
        'short_data': np.random.randn(10),  # Too short for analysis
        'nan_data': np.full(1000, np.nan),  # Contains NaN values
        'empty_data': np.array([]),  # Empty array
    }
    
    config = create_joblib_config(n_jobs=2, verbose=1)
    
    print("Testing error handling with problematic datasets...")
    results = joblib_parallel_analysis(
        datasets=datasets,
        methods=['dfa', 'higuchi'],
        config=config
    )
    
    print("\nError handling results:")
    for name, result in results.items():
        print(f"\n  {name}:")
        for method, method_result in result.items():
            if method_result['success']:
                print(f"    {method}: SUCCESS")
            else:
                print(f"    {method}: FAILED - {method_result['error']}")


def create_visualization(results):
    """Create visualization of results."""
    print("\n" + "="*60)
    print("DEMO: Results Visualization")
    print("="*60)
    
    # Extract results for plotting
    dataset_names = []
    dfa_alphas = []
    higuchi_fds = []
    
    for name, dataset_results in results.items():
        dataset_names.append(name)
        
        # DFA results
        if 'dfa' in dataset_results and dataset_results['dfa']['success']:
            dfa_alphas.append(dataset_results['dfa']['alpha'])
        else:
            dfa_alphas.append(np.nan)
        
        # Higuchi results
        if 'higuchi' in dataset_results and dataset_results['higuchi']['success']:
            higuchi_fds.append(dataset_results['higuchi']['fractal_dimension'])
        else:
            higuchi_fds.append(np.nan)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # DFA results
    valid_dfa = ~np.isnan(dfa_alphas)
    if np.any(valid_dfa):
        ax1.bar(range(len(dataset_names)), dfa_alphas, alpha=0.7, color='skyblue')
        ax1.set_title('DFA Alpha Values')
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Alpha')
        ax1.set_xticks(range(len(dataset_names)))
        ax1.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random walk')
        ax1.legend()
    
    # Higuchi results
    valid_higuchi = ~np.isnan(higuchi_fds)
    if np.any(valid_higuchi):
        ax2.bar(range(len(dataset_names)), higuchi_fds, alpha=0.7, color='lightcoral')
        ax2.set_title('Higuchi Fractal Dimensions')
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Fractal Dimension')
        ax2.set_xticks(range(len(dataset_names)))
        ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Random walk')
        ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("results/figures/joblib_parallel_demo.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Visualization saved to: {output_path}")
    
    plt.show()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Joblib Parallel Computation Demo')
    parser.add_argument('--num-jobs', type=int, default=4, help='Number of parallel jobs')
    parser.add_argument('--data-size', type=int, default=2000, help='Size of test datasets')
    parser.add_argument('--backend', type=str, default='multiprocessing', 
                       choices=['multiprocessing', 'threading'], help='Joblib backend')
    
    args = parser.parse_args()
    
    print("Joblib Parallel Computation Demo")
    print("="*60)
    print(f"Jobs: {args.num_jobs}")
    print(f"Data size: {args.data_size}")
    print(f"Backend: {args.backend}")
    
    try:
        # Demo 1: Parallel processing
        print("\n1. Testing parallel processing...")
        results = demo_parallel_processing()
        
        # Demo 2: Performance comparison
        print("\n2. Testing performance comparison...")
        demo_performance_comparison()
        
        # Demo 3: Error handling
        print("\n3. Testing error handling...")
        demo_error_handling()
        
        # Demo 4: Visualization
        print("\n4. Creating visualization...")
        create_visualization(results)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✓ Parallel processing with Joblib")
        print("✓ Performance optimization")
        print("✓ Error handling and recovery")
        print("✓ Results visualization")
        print("✓ Stable and reliable computation")
        print("\nJoblib provides a much more stable alternative to JAX!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Numba Parallel Computation Demo

This script demonstrates the Numba-accelerated parallel computation capabilities
for long-range dependence analysis. Numba provides a much more stable and
practical alternative to JAX for this type of analysis.

Features demonstrated:
1. High-performance JIT compilation
2. Parallel processing of multiple datasets
3. Memory-efficient computation
4. Easy debugging and error handling
5. Performance benchmarking

Usage:
    python scripts/demo_numba_parallel.py [--num-workers N] [--data-size N]
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.numba_parallel_analysis import (
    NumbaParallelProcessor,
    NumbaAnalysisConfig,
    numba_parallel_analysis,
    create_numba_config,
    benchmark_numba_performance
)
from data_processing.synthetic_generator import SyntheticDataGenerator


def generate_test_datasets(n_datasets: int = 5, data_size: int = 2000) -> dict:
    """Generate test datasets for demonstration."""
    print(f"Generating {n_datasets} test datasets with {data_size} points each...")
    
    generator = SyntheticDataGenerator(random_state=42)
    datasets = {}
    
    # Generate different types of signals
    signal_types = [
        ('white_noise', lambda: np.random.randn(data_size)),
        ('fbm_weak_lrd', lambda: generator.generate_fbm(n=data_size, hurst=0.6)),
        ('fbm_strong_lrd', lambda: generator.generate_fbm(n=data_size, hurst=0.8)),
        ('arfima_weak', lambda: generator.generate_arfima(n=data_size, d=0.2)),
        ('arfima_strong', lambda: generator.generate_arfima(n=data_size, d=0.4)),
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
    print("DEMO: Parallel Processing with Numba")
    print("="*60)
    
    # Generate test datasets
    datasets = generate_test_datasets(n_datasets=5, data_size=2000)
    
    # Create Numba configuration
    config = create_numba_config(
        num_workers=4,
        use_jit=True,
        use_parallel=True
    )
    
    print(f"\nNumba Configuration:")
    print(f"  Workers: {config.num_workers}")
    print(f"  JIT compilation: {config.use_jit}")
    print(f"  Parallel processing: {config.use_parallel}")
    print(f"  Chunk size: {config.chunk_size}")
    
    # Process datasets in parallel
    print("\nProcessing datasets in parallel...")
    start_time = time.time()
    
    results = numba_parallel_analysis(
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
        
        if 'dfa' in dataset_results and 'error' not in dataset_results['dfa']:
            dfa_result = dataset_results['dfa']
            print(f"    DFA Alpha: {dfa_result['alpha']:.4f} (R²: {dfa_result['r_squared']:.3f})")
        
        if 'higuchi' in dataset_results and 'error' not in dataset_results['higuchi']:
            higuchi_result = dataset_results['higuchi']
            print(f"    Higuchi FD: {higuchi_result['fractal_dimension']:.4f} (R²: {higuchi_result['r_squared']:.3f})")
    
    return results


def demo_performance_comparison():
    """Demonstrate performance comparison between different approaches."""
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
        
        # Test Numba DFA
        scales = np.logspace(1, np.log10(size//4), 20, dtype=int)
        
        start_time = time.time()
        for _ in range(5):
            from analysis.numba_parallel_analysis import _numba_dfa_single
            _ = _numba_dfa_single(data, scales, 1)
        numba_time = (time.time() - start_time) / 5
        
        print(f"    Numba DFA: {numba_time:.4f}s per run")
        
        # Test standard NumPy DFA (if available)
        try:
            from analysis.dfa_analysis import dfa
            start_time = time.time()
            for _ in range(5):
                _ = dfa(data, scales=scales, order=1)
            numpy_time = (time.time() - start_time) / 5
            print(f"    NumPy DFA: {numpy_time:.4f}s per run")
            print(f"    Speedup: {numpy_time/numba_time:.2f}x")
        except Exception as e:
            print(f"    NumPy DFA: Not available ({e})")


def demo_memory_efficiency():
    """Demonstrate memory efficiency of Numba implementation."""
    print("\n" + "="*60)
    print("DEMO: Memory Efficiency")
    print("="*60)
    
    import psutil
    import os
    
    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    # Generate large dataset
    print("Testing memory efficiency with large dataset...")
    initial_memory = get_memory_usage()
    print(f"  Initial memory: {initial_memory:.1f} MB")
    
    # Generate large dataset
    large_data = np.random.randn(50000)
    after_data_memory = get_memory_usage()
    print(f"  After data generation: {after_data_memory:.1f} MB")
    print(f"  Data memory usage: {after_data_memory - initial_memory:.1f} MB")
    
    # Process with Numba
    config = create_numba_config(num_workers=2, use_parallel=False)
    processor = NumbaParallelProcessor(config)
    
    start_time = time.time()
    results = processor.process_datasets(
        {'large_dataset': large_data},
        methods=['dfa', 'higuchi']
    )
    processing_time = time.time() - start_time
    
    after_processing_memory = get_memory_usage()
    print(f"  After processing: {after_processing_memory:.1f} MB")
    print(f"  Processing memory overhead: {after_processing_memory - after_data_memory:.1f} MB")
    print(f"  Processing time: {processing_time:.4f} seconds")
    
    # Display results
    if 'large_dataset' in results:
        dataset_results = results['large_dataset']
        if 'dfa' in dataset_results and 'error' not in dataset_results['dfa']:
            print(f"  DFA Alpha: {dataset_results['dfa']['alpha']:.4f}")
        if 'higuchi' in dataset_results and 'error' not in dataset_results['higuchi']:
            print(f"  Higuchi FD: {dataset_results['higuchi']['fractal_dimension']:.4f}")


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
        if 'dfa' in dataset_results and 'error' not in dataset_results['dfa']:
            dfa_alphas.append(dataset_results['dfa']['alpha'])
        else:
            dfa_alphas.append(np.nan)
        
        # Higuchi results
        if 'higuchi' in dataset_results and 'error' not in dataset_results['higuchi']:
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
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("results/figures/numba_parallel_demo.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Visualization saved to: {output_path}")
    
    plt.show()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Numba Parallel Computation Demo')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--data-size', type=int, default=2000, help='Size of test datasets')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    print("Numba Parallel Computation Demo")
    print("="*60)
    print(f"Workers: {args.num_workers}")
    print(f"Data size: {args.data_size}")
    
    try:
        # Demo 1: Parallel processing
        print("\n1. Testing parallel processing...")
        results = demo_parallel_processing()
        
        # Demo 2: Performance comparison
        print("\n2. Testing performance comparison...")
        demo_performance_comparison()
        
        # Demo 3: Memory efficiency
        print("\n3. Testing memory efficiency...")
        demo_memory_efficiency()
        
        # Demo 4: Visualization
        print("\n4. Creating visualization...")
        create_visualization(results)
        
        # Optional: Run benchmark
        if args.benchmark:
            print("\n5. Running performance benchmark...")
            benchmark_numba_performance()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✓ Parallel processing with Numba")
        print("✓ Performance optimization")
        print("✓ Memory efficiency")
        print("✓ Results visualization")
        print("✓ Stable and reliable computation")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

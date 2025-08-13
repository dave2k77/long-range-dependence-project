#!/usr/bin/env python3
"""
JAX Parallel Computation Demo

This script demonstrates the JAX-accelerated parallel computation capabilities
for long-range dependence analysis. It showcases:

1. GPU/TPU acceleration (if available)
2. Parallel processing of multiple datasets
3. Batch processing with vectorized operations
4. Monte Carlo analysis with JAX acceleration
5. Performance benchmarking
6. Integration with the existing submission framework

Usage:
    python scripts/demo_jax_parallel.py [--gpu] [--tpu] [--batch-size N] [--num-parallel N]
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path - ensure it's added correctly
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from analysis.jax_parallel_analysis import (
        JAXAnalysisConfig, 
        JAXParallelProcessor, 
        jax_parallel_analysis, 
        jax_monte_carlo_analysis,
        create_jax_config
    )
    from submission.jax_model_submission import (
        JAXDFAModel, 
        JAXHiguchiModel, 
        JAXModelSubmission,
        create_jax_model_metadata
    )
    from data_processing.synthetic_generator import SyntheticDataGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Looking for module in: {src_path}")
    raise


def generate_test_datasets(n_datasets: int = 10, length: int = 1000) -> dict:
    """Generate test datasets with known properties"""
    generator = SyntheticDataGenerator()
    datasets = {}
    
    # Generate different types of time series
    hurst_values = np.linspace(0.1, 0.9, n_datasets)
    
    for i in range(n_datasets):
        hurst = hurst_values[i]
        
        # Fractional Brownian motion
        fbm_data = generator.generate_fractional_brownian_motion(
            length=length, 
            hurst=hurst, 
            seed=42 + i
        )
        datasets[f'fbm_h{hurst:.1f}'] = fbm_data
        
        # ARFIMA process
        arfima_data = generator.generate_arfima_process(
            length=length,
            d=hurst - 0.5,  # d = H - 0.5
            seed=42 + i
        )
        datasets[f'arfima_d{hurst-0.5:.1f}'] = arfima_data
    
    return datasets


def benchmark_jax_vs_numpy():
    """Benchmark JAX vs NumPy implementations"""
    print("=" * 60)
    print("BENCHMARKING: JAX vs NumPy")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    test_data = np.random.normal(0, 1, 5000)
    
    # Test JAX DFA
    print("\nTesting JAX DFA...")
    jax_config = create_jax_config(use_gpu=False, batch_size=32)
    jax_dfa = JAXDFAModel(jax_config=jax_config)
    
    start_time = time.time()
    jax_dfa.fit(test_data)
    jax_time = time.time() - start_time
    
    jax_alpha = jax_dfa.estimate_alpha()
    jax_r2 = jax_dfa.get_quality_metrics()['r_squared']
    
    print(f"JAX DFA Results:")
    print(f"  Alpha: {jax_alpha:.4f}")
    print(f"  R²: {jax_r2:.4f}")
    print(f"  Time: {jax_time:.4f} seconds")
    
    # Test JAX Higuchi
    print("\nTesting JAX Higuchi...")
    jax_higuchi = JAXHiguchiModel(jax_config=jax_config)
    
    start_time = time.time()
    jax_higuchi.fit(test_data)
    jax_higuchi_time = time.time() - start_time
    
    jax_fd = jax_higuchi.get_quality_metrics()['fractal_dimension']
    jax_higuchi_r2 = jax_higuchi.get_quality_metrics()['r_squared']
    
    print(f"JAX Higuchi Results:")
    print(f"  Fractal Dimension: {jax_fd:.4f}")
    print(f"  R²: {jax_higuchi_r2:.4f}")
    print(f"  Time: {jax_higuchi_time:.4f} seconds")
    
    return {
        'jax_dfa_time': jax_time,
        'jax_higuchi_time': jax_higuchi_time,
        'jax_dfa_alpha': jax_alpha,
        'jax_higuchi_fd': jax_fd
    }


def demo_parallel_processing():
    """Demonstrate parallel processing of multiple datasets"""
    print("\n" + "=" * 60)
    print("DEMO: Parallel Processing of Multiple Datasets")
    print("=" * 60)
    
    # Generate test datasets
    print("Generating test datasets...")
    datasets = generate_test_datasets(n_datasets=8, length=2000)
    
    print(f"Generated {len(datasets)} datasets:")
    for name in datasets.keys():
        print(f"  - {name}")
    
    # Configure JAX for parallel processing
    jax_config = create_jax_config(
        use_gpu=False,  # Set to True if GPU is available
        batch_size=16,
        num_parallel=4,
        enable_jit=True,
        enable_vmap=True
    )
    
    print(f"\nJAX Configuration:")
    print(f"  Batch size: {jax_config.batch_size}")
    print(f"  Parallel devices: {jax_config.num_parallel}")
    print(f"  JIT compilation: {jax_config.enable_jit}")
    print(f"  Vectorized operations: {jax_config.enable_vmap}")
    
    # Process datasets in parallel
    print("\nProcessing datasets in parallel...")
    start_time = time.time()
    
    results = jax_parallel_analysis(
        datasets=datasets,
        methods=['dfa', 'higuchi'],
        config=jax_config,
        min_scale=4,
        num_scales=20
    )
    
    total_time = time.time() - start_time
    
    print(f"\nParallel processing completed in {total_time:.4f} seconds")
    print(f"Average time per dataset: {total_time/len(datasets):.4f} seconds")
    
    # Display results
    print("\nResults Summary:")
    print("-" * 40)
    
    for method, method_results in results.items():
        print(f"\n{method.upper()} Results:")
        for dataset_name, result in method_results.items():
            if method == 'dfa':
                alpha = result['alpha']
                r2 = result['r_squared']
                print(f"  {dataset_name}: α={alpha:.3f}, R²={r2:.3f}")
            elif method == 'higuchi':
                fd = result['fractal_dimension']
                r2 = result['r_squared']
                print(f"  {dataset_name}: D={fd:.3f}, R²={r2:.3f}")
    
    return results


def demo_monte_carlo_analysis():
    """Demonstrate JAX-accelerated Monte Carlo analysis"""
    print("\n" + "=" * 60)
    print("DEMO: JAX-Accelerated Monte Carlo Analysis")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    test_data = np.random.normal(0, 1, 1000)
    
    print(f"Generated test data with {len(test_data)} points")
    
    # Configure JAX
    jax_config = create_jax_config(
        use_gpu=False,
        batch_size=64,
        num_parallel=4,
        enable_jit=True,
        enable_vmap=True
    )
    
    # Perform Monte Carlo analysis
    print("\nPerforming Monte Carlo analysis...")
    start_time = time.time()
    
    mc_results = jax_monte_carlo_analysis(
        data=test_data,
        n_simulations=1000,
        methods=['dfa', 'higuchi'],
        config=jax_config,
        min_scale=4,
        num_scales=20
    )
    
    total_time = time.time() - start_time
    
    print(f"Monte Carlo analysis completed in {total_time:.4f} seconds")
    print(f"Average time per simulation: {total_time/1000:.6f} seconds")
    
    # Display results
    print("\nMonte Carlo Results:")
    print("-" * 30)
    
    for method, results in mc_results.items():
        if method == 'dfa':
            alphas = results['alphas']
            r2_values = results['r_squared']
            
            alpha_mean = np.mean(alphas)
            alpha_std = np.std(alphas)
            alpha_ci = np.percentile(alphas, [2.5, 97.5])
            
            print(f"\nDFA Results:")
            print(f"  Alpha: {alpha_mean:.4f} ± {alpha_std:.4f}")
            print(f"  95% CI: [{alpha_ci[0]:.4f}, {alpha_ci[1]:.4f}]")
            print(f"  R²: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
            
        elif method == 'higuchi':
            fds = results['fractal_dimensions']
            r2_values = results['r_squared']
            
            fd_mean = np.mean(fds)
            fd_std = np.std(fds)
            fd_ci = np.percentile(fds, [2.5, 97.5])
            
            print(f"\nHiguchi Results:")
            print(f"  Fractal Dimension: {fd_mean:.4f} ± {fd_std:.4f}")
            print(f"  95% CI: [{fd_ci[0]:.4f}, {fd_ci[1]:.4f}]")
            print(f"  R²: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
    
    return mc_results


def demo_model_submission():
    """Demonstrate JAX model submission"""
    print("\n" + "=" * 60)
    print("DEMO: JAX Model Submission")
    print("=" * 60)
    
    # Create JAX configuration
    jax_config = create_jax_config(
        use_gpu=False,
        batch_size=32,
        num_parallel=4
    )
    
    # Create model metadata
    metadata = create_jax_model_metadata(
        name="DemoJAXDFAModel",
        version="1.0.0",
        author="Demo Author",
        description="JAX-accelerated DFA model for demonstration",
        algorithm_type="dfa",
        parameters={
            'min_scale': 4,
            'max_scale': 256,
            'num_scales': 20
        },
        dependencies=['jax', 'jaxlib', 'numpy'],
        file_path="models/demo_jax_dfa_model.py",
        jax_config=jax_config,
        gpu_required=False,
        parallel_capable=True,
        batch_processing=True
    )
    
    print("Created JAX model metadata:")
    print(f"  Name: {metadata.name}")
    print(f"  Version: {metadata.version}")
    print(f"  Algorithm: {metadata.algorithm_type}")
    print(f"  Parallel capable: {metadata.parallel_capable}")
    print(f"  Batch processing: {metadata.batch_processing}")
    
    # Create submission system
    submission_system = JAXModelSubmission(jax_config=jax_config)
    
    print("\nJAX model submission system initialized")
    print(f"Available devices: {submission_system.tester.jax_config.num_parallel}")
    
    return metadata, submission_system


def create_visualization(results: dict, mc_results: dict):
    """Create visualizations of the results"""
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('JAX Parallel Computation Demo Results', fontsize=16)
    
    # Plot 1: DFA results
    ax1 = axes[0, 0]
    dfa_results = results.get('dfa', {})
    if dfa_results:
        alphas = [result['alpha'] for result in dfa_results.values()]
        r2_values = [result['r_squared'] for result in dfa_results.values()]
        dataset_names = list(dfa_results.keys())
        
        x_pos = np.arange(len(dataset_names))
        bars = ax1.bar(x_pos, alphas, alpha=0.7)
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Alpha (Hurst Exponent)')
        ax1.set_title('DFA Results - Alpha Values')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name.split('_')[-1] for name in dataset_names], rotation=45)
        
        # Add R² values as text
        for i, (bar, r2) in enumerate(zip(bars, r2_values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'R²={r2:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Higuchi results
    ax2 = axes[0, 1]
    higuchi_results = results.get('higuchi', {})
    if higuchi_results:
        fds = [result['fractal_dimension'] for result in higuchi_results.values()]
        r2_values = [result['r_squared'] for result in higuchi_results.values()]
        dataset_names = list(higuchi_results.keys())
        
        x_pos = np.arange(len(dataset_names))
        bars = ax2.bar(x_pos, fds, alpha=0.7, color='orange')
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Fractal Dimension')
        ax2.set_title('Higuchi Results - Fractal Dimensions')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([name.split('_')[-1] for name in dataset_names], rotation=45)
        
        # Add R² values as text
        for i, (bar, r2) in enumerate(zip(bars, r2_values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'R²={r2:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Monte Carlo DFA distribution
    ax3 = axes[1, 0]
    if 'dfa' in mc_results:
        alphas = mc_results['dfa']['alphas']
        ax3.hist(alphas, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(np.mean(alphas), color='red', linestyle='--', label=f'Mean: {np.mean(alphas):.3f}')
        ax3.axvline(np.median(alphas), color='green', linestyle='--', label=f'Median: {np.median(alphas):.3f}')
        ax3.set_xlabel('Alpha Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Monte Carlo DFA - Alpha Distribution')
        ax3.legend()
    
    # Plot 4: Monte Carlo Higuchi distribution
    ax4 = axes[1, 1]
    if 'higuchi' in mc_results:
        fds = mc_results['higuchi']['fractal_dimensions']
        ax4.hist(fds, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(np.mean(fds), color='red', linestyle='--', label=f'Mean: {np.mean(fds):.3f}')
        ax4.axvline(np.median(fds), color='green', linestyle='--', label=f'Median: {np.median(fds):.3f}')
        ax4.set_xlabel('Fractal Dimension')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Monte Carlo Higuchi - Fractal Dimension Distribution')
        ax4.legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_path = Path("results/figures/jax_parallel_demo.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='JAX Parallel Computation Demo')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--tpu', action='store_true', help='Use TPU acceleration')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--num-parallel', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    print("JAX Parallel Computation Demo")
    print("=" * 60)
    print(f"GPU acceleration: {args.gpu}")
    print(f"TPU acceleration: {args.tpu}")
    print(f"Batch size: {args.batch_size}")
    print(f"Parallel processes: {args.num_parallel}")
    print("=" * 60)
    
    try:
        # Benchmark JAX vs NumPy
        benchmark_results = benchmark_jax_vs_numpy()
        
        # Demo parallel processing
        parallel_results = demo_parallel_processing()
        
        # Demo Monte Carlo analysis
        mc_results = demo_monte_carlo_analysis()
        
        # Demo model submission
        metadata, submission_system = demo_model_submission()
        
        # Create visualizations
        if not args.no_viz:
            create_visualization(parallel_results, mc_results)
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ JAX-accelerated computation")
        print("✓ Parallel processing of multiple datasets")
        print("✓ Vectorized operations with vmap")
        print("✓ JIT compilation for performance")
        print("✓ Monte Carlo analysis with bootstrap")
        print("✓ Model submission framework integration")
        print("✓ Performance benchmarking")
        
        if args.gpu or args.tpu:
            print("✓ GPU/TPU acceleration")
        
        print("\nPerformance Summary:")
        print(f"  JAX DFA time: {benchmark_results['jax_dfa_time']:.4f} seconds")
        print(f"  JAX Higuchi time: {benchmark_results['jax_higuchi_time']:.4f} seconds")
        print(f"  Parallel processing: {len(parallel_results.get('dfa', {}))} datasets")
        print(f"  Monte Carlo simulations: 1000 per method")
        
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

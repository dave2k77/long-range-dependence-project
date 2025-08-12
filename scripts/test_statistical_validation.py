#!/usr/bin/env python3
"""
Test Statistical Validation Script for Long-Range Dependence Project

This script demonstrates the comprehensive statistical validation functionality:
- Hypothesis testing for long-range dependence
- Bootstrap confidence intervals
- Monte Carlo significance tests
- Cross-validation analysis
- Robustness testing
- Comprehensive validation reporting
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.statistical_validation import (
    StatisticalValidator,
    test_lrd_hypothesis,
    bootstrap_confidence_interval,
    monte_carlo_test,
    cross_validate_lrd,
    comprehensive_validation
)
from analysis.arfima_modelling import arfima_simulation
from visualisation.validation_plots import (
    plot_hypothesis_test_result,
    plot_bootstrap_result,
    plot_monte_carlo_result,
    plot_cross_validation_result,
    plot_comprehensive_validation_summary,
    create_validation_report
)


def generate_test_data():
    """Generate test data with known long-range dependence."""
    print("Generating test data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate ARFIMA process with long-range dependence (d=0.3, H=0.8)
    lrd_data = arfima_simulation(1000, d=0.3, ar_params=np.array([0.5]), ma_params=np.array([0.3]))
    
    # Generate random walk (H=0.5, no long-range dependence)
    random_walk = np.cumsum(np.random.normal(0, 1, 1000))
    
    # Generate white noise (H=0.5, no long-range dependence)
    white_noise = np.random.normal(0, 1, 1000)
    
    # Generate ARFIMA process with strong long-range dependence (d=0.4, H=0.9)
    strong_lrd_data = arfima_simulation(1000, d=0.4, ar_params=np.array([0.3]), ma_params=np.array([0.2]))
    
    return {
        'lrd_data': lrd_data,
        'random_walk': random_walk,
        'white_noise': white_noise,
        'strong_lrd_data': strong_lrd_data
    }


def test_hypothesis_testing():
    """Test hypothesis testing functionality."""
    print("\n" + "="*60)
    print("TESTING HYPOTHESIS TESTING")
    print("="*60)
    
    data = generate_test_data()
    methods = ['dfa', 'rs', 'wavelet', 'spectral']
    
    for dataset_name, dataset in data.items():
        print(f"\nTesting {dataset_name}:")
        print("-" * 40)
        
        for method in methods:
            try:
                result = test_lrd_hypothesis(
                    dataset, 
                    method=method, 
                    alpha=0.05, 
                    h0_hurst=0.5,
                    alternative='greater',
                    random_state=42
                )
                
                print(f"  {method.upper()}:")
                print(f"    Decision: {result.decision.upper()}")
                print(f"    p-value: {result.p_value:.4f}")
                print(f"    Hurst estimate: {result.additional_info['hurst_estimate']:.3f}")
                print(f"    Effect size: {result.effect_size:.3f}")
                
            except Exception as e:
                print(f"  {method.upper()}: Error - {e}")


def test_bootstrap_analysis():
    """Test bootstrap analysis functionality."""
    print("\n" + "="*60)
    print("TESTING BOOTSTRAP ANALYSIS")
    print("="*60)
    
    data = generate_test_data()
    methods = ['dfa', 'rs']
    
    for dataset_name, dataset in data.items():
        print(f"\nTesting {dataset_name}:")
        print("-" * 40)
        
        for method in methods:
            try:
                result = bootstrap_confidence_interval(
                    dataset,
                    method=method,
                    n_bootstrap=500,  # Reduced for faster testing
                    confidence_level=0.95,
                    block_size=50,  # Block bootstrap for time series
                    random_state=42
                )
                
                print(f"  {method.upper()}:")
                print(f"    Original estimate: {result.original_estimate:.3f}")
                print(f"    Bootstrap mean: {result.mean_estimate:.3f}")
                print(f"    Bootstrap std: {result.std_estimate:.3f}")
                print(f"    Bias: {result.bias:.4f}")
                print(f"    CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
                
            except Exception as e:
                print(f"  {method.upper()}: Error - {e}")


def test_monte_carlo_tests():
    """Test Monte Carlo significance tests."""
    print("\n" + "="*60)
    print("TESTING MONTE CARLO SIGNIFICANCE TESTS")
    print("="*60)
    
    data = generate_test_data()
    methods = ['dfa', 'rs']
    null_models = ['iid', 'ar1', 'ma1']
    
    for dataset_name, dataset in data.items():
        print(f"\nTesting {dataset_name}:")
        print("-" * 40)
        
        for method in methods:
            for null_model in null_models:
                try:
                    result = monte_carlo_test(
                        dataset,
                        method=method,
                        n_simulations=500,  # Reduced for faster testing
                        alpha=0.05,
                        null_model=null_model,
                        random_state=42
                    )
                    
                    print(f"  {method.upper()} vs {null_model.upper()}:")
                    print(f"    Decision: {result.decision.upper()}")
                    print(f"    p-value: {result.p_value:.4f}")
                    print(f"    Effect size: {result.effect_size:.3f}")
                    print(f"    Observed statistic: {result.test_statistic:.3f}")
                    
                except Exception as e:
                    print(f"  {method.upper()} vs {null_model.upper()}: Error - {e}")


def test_cross_validation():
    """Test cross-validation functionality."""
    print("\n" + "="*60)
    print("TESTING CROSS-VALIDATION")
    print("="*60)
    
    data = generate_test_data()
    methods = ['dfa', 'rs']
    
    for dataset_name, dataset in data.items():
        print(f"\nTesting {dataset_name}:")
        print("-" * 40)
        
        for method in methods:
            try:
                result = cross_validate_lrd(
                    dataset,
                    method=method,
                    n_folds=5,
                    test_size=0.2,
                    random_state=42
                )
                
                print(f"  {method.upper()}:")
                print(f"    Mean Hurst: {result.mean_hurst:.3f}")
                print(f"    Std Hurst: {result.std_hurst:.3f}")
                print(f"    CV Score: {result.cv_score:.3f}")
                print(f"    Stability Score: {result.stability_score:.3f}")
                print(f"    Range: {result.additional_metrics['range']:.3f}")
                
            except Exception as e:
                print(f"  {method.upper()}: Error - {e}")


def test_comprehensive_validation():
    """Test comprehensive validation functionality."""
    print("\n" + "="*60)
    print("TESTING COMPREHENSIVE VALIDATION")
    print("="*60)
    
    data = generate_test_data()
    
    # Test with LRD data
    print("\nTesting comprehensive validation on LRD data:")
    print("-" * 50)
    
    try:
        results = comprehensive_validation(
            data['lrd_data'],
            methods=['dfa', 'rs', 'wavelet', 'spectral'],
            alpha=0.05,
            n_bootstrap=500,  # Reduced for faster testing
            n_simulations=500,  # Reduced for faster testing
            random_state=42
        )
        
        print("Comprehensive validation completed successfully!")
        print(f"Methods tested: {list(results['hypothesis_tests'].keys())}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        for method in results['hypothesis_tests'].keys():
            print(f"\n{method.upper()}:")
            print(f"  Hypothesis test: {results['hypothesis_tests'][method].decision}")
            print(f"  Bootstrap bias: {results['bootstrap_analyses'][method].bias:.4f}")
            print(f"  CV stability: {results['cross_validation'][method].stability_score:.3f}")
            print(f"  Monte Carlo: {results['monte_carlo_tests'][method].decision}")
        
        return results
        
    except Exception as e:
        print(f"Comprehensive validation failed: {e}")
        return None


def test_visualization():
    """Test validation visualization functionality."""
    print("\n" + "="*60)
    print("TESTING VALIDATION VISUALIZATION")
    print("="*60)
    
    data = generate_test_data()
    
    # Create results directory
    results_dir = Path("results/validation_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Test individual plots
    print("\nTesting individual validation plots...")
    
    # Hypothesis test plot
    try:
        result = test_lrd_hypothesis(data['lrd_data'], method='dfa', random_state=42)
        fig = plot_hypothesis_test_result(
            result, 
            save_path=str(results_dir / "hypothesis_test_example.png")
        )
        plt.close(fig)
        print("✓ Hypothesis test plot created")
    except Exception as e:
        print(f"✗ Hypothesis test plot failed: {e}")
    
    # Bootstrap plot
    try:
        result = bootstrap_confidence_interval(data['lrd_data'], method='dfa', n_bootstrap=500, random_state=42)
        fig = plot_bootstrap_result(
            result, 
            save_path=str(results_dir / "bootstrap_example.png")
        )
        plt.close(fig)
        print("✓ Bootstrap plot created")
    except Exception as e:
        print(f"✗ Bootstrap plot failed: {e}")
    
    # Monte Carlo plot
    try:
        result = monte_carlo_test(data['lrd_data'], method='dfa', n_simulations=500, random_state=42)
        fig = plot_monte_carlo_result(
            result, 
            save_path=str(results_dir / "monte_carlo_example.png")
        )
        plt.close(fig)
        print("✓ Monte Carlo plot created")
    except Exception as e:
        print(f"✗ Monte Carlo plot failed: {e}")
    
    # Cross-validation plot
    try:
        result = cross_validate_lrd(data['lrd_data'], method='dfa', random_state=42)
        fig = plot_cross_validation_result(
            result, 
            save_path=str(results_dir / "cross_validation_example.png")
        )
        plt.close(fig)
        print("✓ Cross-validation plot created")
    except Exception as e:
        print(f"✗ Cross-validation plot failed: {e}")


def test_comprehensive_report():
    """Test comprehensive validation report generation."""
    print("\n" + "="*60)
    print("TESTING COMPREHENSIVE VALIDATION REPORT")
    print("="*60)
    
    data = generate_test_data()
    
    try:
        # Run comprehensive validation
        results = comprehensive_validation(
            data['lrd_data'],
            methods=['dfa', 'rs'],  # Reduced for faster testing
            alpha=0.05,
            n_bootstrap=500,
            n_simulations=500,
            random_state=42
        )
        
        if results:
            # Create comprehensive report
            report_path = create_validation_report(
                results, 
                save_dir="results/validation_comprehensive"
            )
            print(f"✓ Comprehensive validation report created at: {report_path}")
            
            # Create summary plot
            fig = plot_comprehensive_validation_summary(
                results,
                save_path="results/validation_comprehensive/summary_plot.png"
            )
            plt.close(fig)
            print("✓ Comprehensive summary plot created")
            
        else:
            print("✗ Comprehensive validation failed")
            
    except Exception as e:
        print(f"✗ Comprehensive report generation failed: {e}")


def main():
    """Run all statistical validation tests."""
    print("Statistical Validation Test Suite")
    print("=" * 60)
    print("This script tests the comprehensive statistical validation functionality")
    print("for long-range dependence analysis.")
    
    # Test individual components
    test_hypothesis_testing()
    test_bootstrap_analysis()
    test_monte_carlo_tests()
    test_cross_validation()
    
    # Test comprehensive validation
    test_comprehensive_validation()
    
    # Test visualization
    test_visualization()
    
    # Test comprehensive report
    test_comprehensive_report()
    
    print("\n" + "="*60)
    print("STATISTICAL VALIDATION TESTING COMPLETED")
    print("="*60)
    print("Check the 'results/validation_test' and 'results/validation_comprehensive'")
    print("directories for generated plots and reports.")
    print("\nKey features tested:")
    print("- Hypothesis testing for LRD vs. no LRD")
    print("- Bootstrap confidence intervals (IID and block bootstrap)")
    print("- Monte Carlo significance tests (IID, AR1, MA1 null models)")
    print("- Cross-validation with stability assessment")
    print("- Comprehensive validation across multiple methods")
    print("- Visualization of all validation results")
    print("- Automated report generation")


if __name__ == "__main__":
    main()

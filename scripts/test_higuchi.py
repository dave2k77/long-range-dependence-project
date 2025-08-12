#!/usr/bin/env python3
"""
Test Script for Higuchi Fractal Dimension Analysis

This script demonstrates the Higuchi fractal dimension analysis functionality:
- Basic Higuchi analysis on synthetic data
- Parameter optimization
- Quality assessment
- Visualization
- Batch analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
import sys
sys.path.append('src')

from analysis.higuchi_analysis import (
    higuchi_fractal_dimension,
    estimate_higuchi_dimension,
    higuchi_analysis_batch,
    validate_higuchi_results
)
from visualisation.higuchi_plots import (
    plot_higuchi_analysis,
    plot_higuchi_comparison,
    plot_higuchi_quality_assessment,
    create_higuchi_report
)


def generate_test_data():
    """Generate test time series data."""
    np.random.seed(42)
    n_points = 1000
    
    # 1. Fractional Brownian motion (H = 0.3, strong LRD)
    t = np.linspace(0, 1, n_points)
    h_fbm = 0.3
    fbm = np.zeros(n_points)
    for i in range(1, n_points):
        fbm[i] = fbm[i-1] + np.random.normal(0, 1) * (t[i] - t[i-1])**h_fbm
    
    # 2. Random walk (H = 0.5, no LRD)
    random_walk = np.cumsum(np.random.normal(0, 1, n_points))
    
    # 3. White noise (H = 0.5, no LRD)
    white_noise = np.random.normal(0, 1, n_points)
    
    # 4. Strong LRD process (H = 0.8)
    h_strong = 0.8
    strong_lrd = np.zeros(n_points)
    for i in range(1, n_points):
        strong_lrd[i] = strong_lrd[i-1] + np.random.normal(0, 1) * (t[i] - t[i-1])**h_strong
    
    return {
        'fractional_brownian': fbm,
        'random_walk': random_walk,
        'white_noise': white_noise,
        'strong_lrd': strong_lrd
    }


def test_basic_higuchi():
    """Test basic Higuchi analysis."""
    print("="*60)
    print("TESTING BASIC HIGUCHI ANALYSIS")
    print("="*60)
    
    data = generate_test_data()
    
    for name, series in data.items():
        print(f"\nAnalyzing {name}:")
        print("-" * 40)
        
        try:
            # Basic analysis
            k_values, l_values, summary = higuchi_fractal_dimension(
                series, k_max=100, k_min=2, optimize_k=True
            )
            
            print(f"  Fractal Dimension: {summary.fractal_dimension:.4f}")
            print(f"  Standard Error: {summary.std_error:.4f}")
            print(f"  R²: {summary.r_squared:.4f}")
            print(f"  P-value: {summary.p_value:.4f}")
            print(f"  k range: [{summary.k_range[0]}, {summary.k_range[1]}]")
            print(f"  n_k values: {len(k_values)}")
            
            # Quality validation
            validation = validate_higuchi_results(summary)
            print(f"  Quality Score: {validation['quality_score']:.3f}")
            if validation['warnings']:
                print(f"  Warnings: {', '.join(validation['warnings'])}")
            if validation['issues']:
                print(f"  Issues: {', '.join(validation['issues'])}")
                
        except Exception as e:
            print(f"  Error: {e}")


def test_parameter_optimization():
    """Test different parameter settings."""
    print("\n" + "="*60)
    print("TESTING PARAMETER OPTIMIZATION")
    print("="*60)
    
    data = generate_test_data()
    series = data['fractional_brownian']
    
    print("\nTesting different k_max values:")
    print("-" * 40)
    
    k_max_values = [50, 100, 200, 300]
    
    for k_max in k_max_values:
        try:
            k_values, l_values, summary = higuchi_fractal_dimension(
                series, k_max=k_max, k_min=2, optimize_k=False
            )
            
            print(f"  k_max = {k_max}: D = {summary.fractal_dimension:.4f}, R² = {summary.r_squared:.4f}")
            
        except Exception as e:
            print(f"  k_max = {k_max}: Error - {e}")
    
    print("\nTesting with and without k optimization:")
    print("-" * 40)
    
    # Without optimization
    k_values1, l_values1, summary1 = higuchi_fractal_dimension(
        series, k_max=100, k_min=2, optimize_k=False
    )
    
    # With optimization
    k_values2, l_values2, summary2 = higuchi_fractal_dimension(
        series, k_max=100, k_min=2, optimize_k=True
    )
    
    print(f"  Without optimization: D = {summary1.fractal_dimension:.4f}, R² = {summary1.r_squared:.4f}")
    print(f"  With optimization: D = {summary2.fractal_dimension:.4f}, R² = {summary2.r_squared:.4f}")
    print(f"  k range without opt: [{summary1.k_range[0]}, {summary1.k_range[1]}]")
    print(f"  k range with opt: [{summary2.k_range[0]}, {summary2.k_range[1]}]")


def test_batch_analysis():
    """Test batch analysis functionality."""
    print("\n" + "="*60)
    print("TESTING BATCH ANALYSIS")
    print("="*60)
    
    data = generate_test_data()
    
    try:
        # Convert to list format for batch analysis
        series_list = list(data.values())
        names = list(data.keys())
        
        results = higuchi_analysis_batch(
            series_list, names, k_max=100, k_min=2, optimize_k=True
        )
        
        print(f"Successfully analyzed {len(results)} time series:")
        print("-" * 40)
        
        for name, summary in results.items():
            print(f"  {name}: D = {summary.fractal_dimension:.4f}, R² = {summary.r_squared:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Batch analysis failed: {e}")
        return {}


def test_visualization():
    """Test visualization functionality."""
    print("\n" + "="*60)
    print("TESTING VISUALIZATION")
    print("="*60)
    
    data = generate_test_data()
    series = data['fractional_brownian']
    
    try:
        # Basic analysis plot
        k_values, l_values, summary = higuchi_fractal_dimension(
            series, k_max=100, k_min=2, optimize_k=True
        )
        
        print("Creating individual analysis plot...")
        fig1 = plot_higuchi_analysis(k_values, l_values, summary)
        print("✓ Individual analysis plot created")
        
        print("Creating quality assessment plot...")
        fig2 = plot_higuchi_quality_assessment(summary)
        print("✓ Quality assessment plot created")
        
        # Close figures to free memory
        plt.close(fig1)
        plt.close(fig2)
        
    except Exception as e:
        print(f"Visualization failed: {e}")


def test_comprehensive_report():
    """Test comprehensive report generation."""
    print("\n" + "="*60)
    print("TESTING COMPREHENSIVE REPORT")
    print("="*60)
    
    data = generate_test_data()
    
    try:
        # Analyze all series
        series_list = list(data.values())
        names = list(data.keys())
        
        results = higuchi_analysis_batch(
            series_list, names, k_max=100, k_min=2, optimize_k=True
        )
        
        if results:
            print("Creating comprehensive report...")
            report_path = create_higuchi_report(results, save_dir="results/higuchi_test")
            print(f"✓ Comprehensive report created at: {report_path}")
            
            # Test comparison plot
            print("Creating comparison plot...")
            fig = plot_higuchi_comparison(results)
            plt.close(fig)
            print("✓ Comparison plot created")
            
        else:
            print("No results to create report from")
            
    except Exception as e:
        print(f"Report generation failed: {e}")


def main():
    """Run all tests."""
    print("Higuchi Fractal Dimension Analysis Test Suite")
    print("="*60)
    print("This script tests the comprehensive Higuchi fractal dimension")
    print("analysis functionality for long-range dependence analysis.")
    print("="*60)
    
    # Create results directory
    Path("results/higuchi_test").mkdir(parents=True, exist_ok=True)
    
    # Run tests
    test_basic_higuchi()
    test_parameter_optimization()
    test_batch_analysis()
    test_visualization()
    test_comprehensive_report()
    
    print("\n" + "="*60)
    print("HIGUCHI ANALYSIS TESTING COMPLETED")
    print("="*60)
    print("Check the 'results/higuchi_test' directory for generated plots and reports.")
    
    print("\nKey features tested:")
    print("- Basic Higuchi fractal dimension calculation")
    print("- Parameter optimization and k range selection")
    print("- Quality assessment and validation")
    print("- Batch analysis of multiple time series")
    print("- Comprehensive visualization capabilities")
    print("- Automated report generation")


if __name__ == "__main__":
    main()

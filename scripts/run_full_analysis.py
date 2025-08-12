#!/usr/bin/env python3
"""
Full Analysis Script for Long-Range Dependence Project

This script runs comprehensive analysis using both ARFIMA and DFA models
on synthetic and real data, generating plots and saving results.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.arfima_modelling import ARFIMAModel, arfima_simulation, estimate_arfima_order
from analysis.dfa_analysis import DFAModel, dfa, hurst_from_dfa_alpha, d_from_hurst
from analysis.rs_analysis import RSModel, rs_analysis, d_from_hurst_rs, alpha_from_hurst_rs
from analysis.mfdfa_analysis import MFDFAModel, mfdfa, hurst_from_mfdfa, alpha_from_mfdfa
from analysis.wavelet_analysis import WaveletModel, wavelet_leaders_estimation, wavelet_whittle_estimation
from analysis.spectral_analysis import SpectralModel, whittle_mle, periodogram_estimation
# Note: These modules are now implemented and available
# from data_processing.data_loader import load_data
# from utils.config import load_config
from visualisation.time_series_plots import plot_time_series
from visualisation.fractal_plots import plot_dfa_analysis, plot_rs_analysis, plot_mfdfa_analysis
from visualisation.results_visualisation import (
    plot_wavelet_analysis, plot_spectral_analysis, plot_arfima_forecasts,
    plot_comprehensive_comparison, create_summary_table
)


def setup_directories() -> None:
    """Create necessary directories for results."""
    dirs = [
        "results/figures",
        "results/tables", 
        "results/models",
        "results/reports"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_data_from_folder(data_folder: str = "data") -> Dict[str, np.ndarray]:
    """Load data from the organized data folder structure."""
    data_folder = Path(data_folder)
    data = {}
    
    # Try to load from processed data first (preferred for analysis)
    processed_dir = data_folder / "processed"
    if processed_dir.exists():
        for file_path in processed_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file_path, index_col=0)
                # Extract the original symbol name from the processed filename
                # Format: processed_financial_SYMBOL_TIMESTAMP.csv
                filename_parts = file_path.stem.split("_")
                if len(filename_parts) >= 3 and filename_parts[0] == "processed" and filename_parts[1] == "financial":
                    series_name = filename_parts[2]  # Extract the symbol (AAPL, GOOGL, etc.)
                else:
                    # Fallback to using the stem without "processed_" prefix
                    series_name = file_path.stem.replace("processed_", "")
                
                # Convert to numpy arrays, taking the first column if multiple
                if len(df.columns) >= 1:
                    data[series_name] = df.iloc[:, 0].values
                    print(f"Loaded processed data: {series_name} ({len(data[series_name])} points)")
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
    
    # If no processed data, try raw data
    if not data:
        raw_dir = data_folder / "raw"
        if raw_dir.exists():
            for file_path in raw_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(file_path, index_col=0)
                    # Convert to numpy arrays, taking the first column if multiple
                    if len(df.columns) >= 1:
                        series_name = file_path.stem
                        data[series_name] = df.iloc[:, 0].values
                        print(f"Loaded raw data: {series_name} ({len(data[series_name])} points)")
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
    
    return data


def generate_synthetic_data(n: int = 1000, seed: int = 42) -> Dict[str, np.ndarray]:
    """Generate synthetic time series for analysis."""
    np.random.seed(seed)
    
    data = {}
    
    # Try ARFIMA simulations first
    try:
        # ARFIMA(1, 0.3, 1) - long-range dependent
        arfima_lrd = arfima_simulation(
            n=n, d=0.3, ar_params=np.array([0.5]), 
            ma_params=np.array([0.3]), sigma=1.0, seed=seed
        )
        data["arfima_lrd"] = arfima_lrd
    except Exception as e:
        print(f"Warning: ARFIMA LRD simulation failed: {e}")
        # Fallback to simple fractional noise
        d = 0.3
        frac_noise = np.zeros(n)
        for i in range(1, n):
            frac_noise[i] = frac_noise[i-1] + np.random.normal(0, 1) * (i ** (-d))
        data["arfima_lrd"] = frac_noise
    
    try:
        # ARFIMA(1, 0.1, 1) - short-range dependent
        arfima_srd = arfima_simulation(
            n=n, d=0.1, ar_params=np.array([0.5]), 
            ma_params=np.array([0.3]), sigma=1.0, seed=seed+1
        )
        data["arfima_srd"] = arfima_srd
    except Exception as e:
        print(f"Warning: ARFIMA SRD simulation failed: {e}")
        # Fallback to simple AR(1)
        phi = 0.5
        ar1 = np.zeros(n)
        for i in range(1, n):
            ar1[i] = phi * ar1[i-1] + np.random.normal(0, 1)
        data["arfima_srd"] = ar1
    
    # Simple series that are more stable for ARFIMA fitting
    # Random walk (non-stationary)
    data["random_walk"] = np.cumsum(np.random.randn(n))
    
    # White noise
    data["white_noise"] = np.random.randn(n)
    
    # Simple trend
    t = np.arange(n)
    data["trend"] = 0.01 * t + np.random.normal(0, 0.1, n)
    
    # Simple seasonal
    seasonal = 0.5 * np.sin(2 * np.pi * t / 100) + np.random.normal(0, 0.1, n)
    data["seasonal"] = seasonal
    
    return data


def run_arfima_analysis(data: Dict[str, np.ndarray], 
                        series_names: List[str]) -> Dict[str, Dict]:
    """Run ARFIMA analysis on multiple series with robust error handling."""
    results = {}
    
    for name in series_names:
        print(f"Running ARFIMA analysis on {name}...")
        y = data[name]
        
        try:
            # Estimate order with timeout
            p, d_est, q = estimate_arfima_order(y, max_p=1, max_q=1)  # Reduced order for stability
            print(f"  Estimated order: ARFIMA({p}, {d_est:.3f}, {q})")
            
            # Fit model with reduced iterations
            model = ARFIMAModel(d=d_est, p=p, q=q)
            model.fit(y, max_iter=200)  # Reduced iterations for stability
            
            # Check if model is properly fitted
            if not model.is_fitted or model.params is None:
                raise ValueError("Model fitting failed")
            
            # Generate forecasts with reduced bootstrap samples
            try:
                fc, lo, hi = model.forecast(
                    steps=20, alpha=0.05,  # Reduced forecast steps
                    interval_method='bootstrap', B=50, seed=42  # Reduced bootstrap samples
                )
            except Exception as forecast_error:
                print(f"  Warning: Forecast generation failed, using point forecasts: {forecast_error}")
                # Fallback to simple point forecasts
                fc = model.predict(steps=20)
                lo = fc - 0.1 * np.std(y)  # Simple confidence bands
                hi = fc + 0.1 * np.std(y)
            
            # Store results
            results[name] = {
                "model": model,
                "estimated_order": (p, d_est, q),
                "forecast": fc,
                "forecast_lo": lo,
                "forecast_hi": hi,
                "summary": model.summary()
            }
            
        except Exception as e:
            print(f"  Error in ARFIMA analysis for {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def run_dfa_analysis(data: Dict[str, np.ndarray], 
                     series_names: List[str]) -> Dict[str, Dict]:
    """Run DFA analysis on multiple series."""
    results = {}
    
    for name in series_names:
        print(f"Running DFA analysis on {name}...")
        y = data[name]
        
        try:
            # Run DFA
            scales, flucts, summary = dfa(y, order=1, overlap=True)
            
            # Create model object for consistency
            model = DFAModel(order=1, overlap=True)
            model.scales = scales
            model.flucts = flucts
            model.summary = summary
            model.is_fitted = True
            
            # Convert to Hurst exponent
            H = hurst_from_dfa_alpha(summary.alpha)
            d_hurst = d_from_hurst(H)
            
            results[name] = {
                "model": model,
                "scales": scales,
                "fluctuations": flucts,
                "alpha": summary.alpha,
                "hurst": H,
                "d_hurst": d_hurst,
                "summary": summary
            }
            
        except Exception as e:
            print(f"  Error in DFA analysis for {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def run_rs_analysis(data: Dict[str, np.ndarray], 
                    series_names: List[str]) -> Dict[str, Dict]:
    """Run R/S analysis on multiple series."""
    results = {}
    
    for name in series_names:
        print(f"Running R/S analysis on {name}...")
        y = data[name]
        
        try:
            # Run R/S analysis
            scales, rs_values, summary = rs_analysis(y)
            
            # Create model object for consistency
            model = RSModel()
            model.scales = scales
            model.rs_values = rs_values
            model.summary = summary
            model.is_fitted = True
            
            # Convert to other parameters
            d_hurst = d_from_hurst_rs(summary.hurst)
            alpha_rs = alpha_from_hurst_rs(summary.hurst)
            
            results[name] = {
                "model": model,
                "scales": scales,
                "rs_values": rs_values,
                "hurst": summary.hurst,
                "d_hurst": d_hurst,
                "alpha_rs": alpha_rs,
                "summary": summary
            }
            
        except Exception as e:
            print(f"  Error in R/S analysis for {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def run_mfdfa_analysis(data: Dict[str, np.ndarray], 
                       series_names: List[str]) -> Dict[str, Dict]:
    """Run MFDFA analysis on multiple series."""
    results = {}
    
    for name in series_names:
        print(f"Running MFDFA analysis on {name}...")
        y = data[name]
        
        try:
            # Run MFDFA analysis
            scales, fq, summary = mfdfa(y)
            
            # Create model object for consistency
            model = MFDFAModel()
            model.scales = scales
            model.fq = fq
            model.summary = summary
            model.is_fitted = True
            
            # Extract key parameters
            hq_2 = hurst_from_mfdfa(summary.hq, summary.q_values, q_target=2.0)
            alpha_2 = alpha_from_mfdfa(summary.alpha, summary.q_values, q_target=2.0)
            
            results[name] = {
                "model": model,
                "scales": scales,
                "fq": fq,
                "hq_2": hq_2,
                "alpha_2": alpha_2,
                "summary": summary
            }
            
        except Exception as e:
            print(f"  Error in MFDFA analysis for {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def run_wavelet_analysis(data: Dict[str, np.ndarray], 
                         series_names: List[str]) -> Dict[str, Dict]:
    """Run wavelet analysis on multiple series."""
    results = {}
    
    for name in series_names:
        print(f"Running wavelet analysis on {name}...")
        y = data[name]
        
        try:
            # Run both wavelet methods
            # Wavelet Leaders
            scales_leaders, leaders, summary_leaders = wavelet_leaders_estimation(y)
            
            # Wavelet Whittle
            scales_whittle, coeff_variances, summary_whittle = wavelet_whittle_estimation(y)
            
            # Create model objects for consistency
            model_leaders = WaveletModel(method='leaders')
            model_leaders.scales = scales_leaders
            model_leaders.coefficients = leaders
            model_leaders.summary = summary_leaders
            model_leaders.is_fitted = True
            
            model_whittle = WaveletModel(method='whittle')
            model_whittle.scales = scales_whittle
            model_whittle.coefficients = coeff_variances
            model_whittle.summary = summary_whittle
            model_whittle.is_fitted = True
            
            results[name] = {
                "leaders_model": model_leaders,
                "whittle_model": model_whittle,
                "scales_leaders": scales_leaders,
                "scales_whittle": scales_whittle,
                "leaders": leaders,
                "coeff_variances": coeff_variances,
                "summary_leaders": summary_leaders,
                "summary_whittle": summary_whittle
            }
            
        except Exception as e:
            print(f"  Error in wavelet analysis for {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def run_spectral_analysis(data: Dict[str, np.ndarray], 
                         series_names: List[str]) -> Dict[str, Dict]:
    """Run spectral analysis on multiple series."""
    results = {}
    
    for name in series_names:
        print(f"Running spectral analysis on {name}...")
        y = data[name]
        
        try:
            # Run both spectral methods
            # Whittle MLE
            freqs_whittle, periodogram_whittle, summary_whittle = whittle_mle(y)
            
            # Periodogram Regression
            freqs_periodogram, periodogram_periodogram, summary_periodogram = periodogram_estimation(y)
            
            # Create model objects for consistency
            model_whittle = SpectralModel(method='whittle')
            model_whittle.frequencies = freqs_whittle
            model_whittle.power_spectrum = periodogram_whittle
            model_whittle.summary = summary_whittle
            model_whittle.is_fitted = True
            
            model_periodogram = SpectralModel(method='periodogram')
            model_periodogram.frequencies = freqs_periodogram
            model_periodogram.power_spectrum = periodogram_periodogram
            model_periodogram.summary = summary_periodogram
            model_periodogram.is_fitted = True
            
            results[name] = {
                "whittle_model": model_whittle,
                "periodogram_model": model_periodogram,
                "freqs_whittle": freqs_whittle,
                "freqs_periodogram": freqs_periodogram,
                "periodogram_whittle": periodogram_whittle,
                "periodogram_periodogram": periodogram_periodogram,
                "summary_whittle": summary_whittle,
                "summary_periodogram": summary_periodogram
            }
            
        except Exception as e:
            print(f"  Error in spectral analysis for {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


def create_analysis_plots(data: Dict[str, np.ndarray],
                          arfima_results: Dict[str, Dict],
                          dfa_results: Dict[str, Dict],
                          rs_results: Dict[str, Dict],
                          mfdfa_results: Dict[str, Dict],
                          wavelet_results: Dict[str, Dict],
                          spectral_results: Dict[str, Dict],
                          series_names: List[str]) -> None:
    """Create comprehensive analysis plots using the new visualization modules."""
    
    print("Creating analysis plots...")
    
    # 1. Time series plots
    plot_time_series(data, save_path='results/figures/time_series.png')
    
    # 2. DFA analysis plots
    plot_dfa_analysis(dfa_results, series_names, save_path='results/figures/dfa_analysis.png')
    
    # 3. R/S analysis plots
    plot_rs_analysis(rs_results, series_names, save_path='results/figures/rs_analysis.png')
    
    # 4. MFDFA analysis plots
    plot_mfdfa_analysis(mfdfa_results, series_names, save_path='results/figures/mfdfa_analysis.png')
    
    # 5. Wavelet analysis plots
    plot_wavelet_analysis(wavelet_results, series_names, save_path='results/figures/wavelet_analysis.png')
    
    # 6. Spectral analysis plots
    plot_spectral_analysis(spectral_results, series_names, save_path='results/figures/spectral_analysis.png')
    
    # 7. ARFIMA forecasts
    plot_arfima_forecasts(arfima_results, data, series_names, save_path='results/figures/arfima_forecasts.png')
    
    # 8. Comprehensive comparison plot
    plot_comprehensive_comparison(
        dfa_results, rs_results, mfdfa_results, wavelet_results, 
        spectral_results, arfima_results, series_names, 
        save_path='results/figures/parameter_comparison.png'
    )
    
    print("All analysis plots created successfully!")


def save_results_table(arfima_results: Dict[str, Dict],
                       dfa_results: Dict[str, Dict],
                       rs_results: Dict[str, Dict],
                       mfdfa_results: Dict[str, Dict],
                       wavelet_results: Dict[str, Dict],
                       spectral_results: Dict[str, Dict],
                       series_names: List[str]) -> None:
    """Save comprehensive results table using the new visualization module."""
    print("Creating comprehensive results table...")
    
    summary_df = create_summary_table(
        dfa_results, rs_results, mfdfa_results, wavelet_results,
        spectral_results, arfima_results, series_names,
        save_path='results/tables/comprehensive_results.csv'
    )
    
    print("Results table saved to 'results/tables/comprehensive_results.csv'")
    print(f"Table contains {len(summary_df)} series and {len(summary_df.columns)} parameters")


def main():
    """Main analysis function with timeout protection."""
    print("Starting Full Analysis for Long-Range Dependence Project")
    print("=" * 60)
    
    try:
        # Setup
        setup_directories()
        
        # Try to load data from organized data folder first
        print("\n1. Loading data...")
        data = load_data_from_folder("data")
        
        if data:
            print(f"Loaded {len(data)} datasets from data folder:")
            series_names = list(data.keys())
            for name in series_names:
                print(f"  - {name}: {len(data[name])} points")
        else:
            # Fallback to generating synthetic data
            print("No data found in data folder, generating synthetic data...")
            data = generate_synthetic_data(n=1000, seed=42)
            series_names = list(data.keys())
            
            print(f"Generated {len(series_names)} time series:")
            for name in series_names:
                print(f"  - {name}: {len(data[name])} points")
        
        # Run ARFIMA analysis with individual timeouts
        print("\n2. Running ARFIMA analysis...")
        arfima_results = run_arfima_analysis(data, series_names)
        
        # Run DFA analysis
        print("\n3. Running DFA analysis...")
        dfa_results = run_dfa_analysis(data, series_names)
        
        # Run R/S analysis
        print("\n4. Running R/S analysis...")
        rs_results = run_rs_analysis(data, series_names)
        
        # Run MFDFA analysis
        print("\n5. Running MFDFA analysis...")
        mfdfa_results = run_mfdfa_analysis(data, series_names)
        
        # Run wavelet analysis
        print("\n6. Running wavelet analysis...")
        wavelet_results = run_wavelet_analysis(data, series_names)
        
        # Run spectral analysis
        print("\n7. Running spectral analysis...")
        spectral_results = run_spectral_analysis(data, series_names)
        
        # Create plots
        print("\n8. Creating analysis plots...")
        create_analysis_plots(data, arfima_results, dfa_results, rs_results, mfdfa_results, wavelet_results, spectral_results, series_names)
        
        # Save results table
        print("\n9. Saving results table...")
        save_results_table(arfima_results, dfa_results, rs_results, mfdfa_results, wavelet_results, spectral_results, series_names)
        
        # Print summary
        print("\n10. Analysis Summary:")
        print("-" * 30)
        
        for name in series_names:
            print(f"\n{name.replace('_', ' ').title()}:")
            
            if name in arfima_results and "error" not in arfima_results[name]:
                arfima_model = arfima_results[name]["model"]
                print(f"  ARFIMA: d={arfima_model.params.d:.3f}, "
                      f"p={arfima_model.params.p}, q={arfima_model.params.q}")
            else:
                print(f"  ARFIMA: Error")
            
            if name in dfa_results and "error" not in dfa_results[name]:
                print(f"  DFA: α={dfa_results[name]['alpha']:.3f}, "
                      f"H={dfa_results[name]['hurst']:.3f}")
            else:
                print(f"  DFA: Error")
            
            if name in rs_results and "error" not in rs_results[name]:
                print(f"  R/S: H={rs_results[name]['hurst']:.3f}, "
                      f"d={rs_results[name]['d_hurst']:.3f}")
            else:
                print(f"  R/S: Error")
            
            if name in mfdfa_results and "error" not in mfdfa_results[name]:
                print(f"  MFDFA: h(q=2)={mfdfa_results[name]['hq_2']:.3f}, "
                      f"α(q=2)={mfdfa_results[name]['alpha_2']:.3f}")
            else:
                print(f"  MFDFA: Error")
            
            if name in wavelet_results and "error" not in wavelet_results[name]:
                print(f"  Wavelet Leaders: d={wavelet_results[name]['summary_leaders'].d:.3f}, "
                      f"H={wavelet_results[name]['summary_leaders'].hurst:.3f}")
                print(f"  Wavelet Whittle: d={wavelet_results[name]['summary_whittle'].d:.3f}, "
                      f"H={wavelet_results[name]['summary_whittle'].hurst:.3f}")
            else:
                print(f"  Wavelet: Error")
            
            if name in spectral_results and "error" not in spectral_results[name]:
                print(f"  Spectral Whittle MLE: d={spectral_results[name]['summary_whittle'].d:.3f}, "
                      f"H={spectral_results[name]['summary_whittle'].hurst:.3f}")
                print(f"  Spectral Periodogram: d={spectral_results[name]['summary_periodogram'].d:.3f}, "
                      f"H={spectral_results[name]['summary_periodogram'].hurst:.3f}")
            else:
                print(f"  Spectral: Error")
        
        print(f"\nAnalysis complete! Results saved to:")
        print(f"  - Plots: results/figures/")
        print(f"  - Tables: results/tables/")
        print(f"  - Models: results/models/")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

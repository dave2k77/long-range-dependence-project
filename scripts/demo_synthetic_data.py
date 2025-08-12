#!/usr/bin/env python3
"""
Demonstration script for synthetic data generation capabilities.

This script showcases the comprehensive synthetic data generation module
with examples of pure signals, contamination, and data storage.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.synthetic_generator import (
    PureSignalGenerator, 
    DataContaminator, 
    SyntheticDataGenerator
)


def demo_pure_signals():
    """Demonstrate pure signal generation."""
    print("=" * 60)
    print("DEMONSTRATION: Pure Signal Generation")
    print("=" * 60)
    
    # Initialize generator
    generator = PureSignalGenerator(random_state=42)
    n = 500
    
    # Generate ARFIMA signals
    print("\nGenerating ARFIMA signals...")
    arfima_signals = {}
    for d in [0.1, 0.2, 0.3, 0.4]:
        signal = generator.generate_arfima(n, d)
        arfima_signals[f'd={d}'] = signal
        print(f"  - ARFIMA(d={d}): mean={np.mean(signal):.4f}, std={np.std(signal):.4f}")
    
    # Generate fBm signals
    print("\nGenerating fBm signals...")
    fbm_signals = {}
    for H in [0.3, 0.5, 0.7]:
        signal = generator.generate_fbm(n, H)
        fbm_signals[f'H={H}'] = signal
        print(f"  - fBm(H={H}): mean={np.mean(signal):.4f}, std={np.std(signal):.4f}")
    
    # Generate fGn signals
    print("\nGenerating fGn signals...")
    fgn_signals = {}
    for H in [0.3, 0.5, 0.7]:
        signal = generator.generate_fgn(n, H)
        fgn_signals[f'H={H}'] = signal
        print(f"  - fGn(H={H}): mean={np.mean(signal):.4f}, std={np.std(signal):.4f}")
    
    return arfima_signals, fbm_signals, fgn_signals


def demo_contamination():
    """Demonstrate data contamination."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Data Contamination")
    print("=" * 60)
    
    # Initialize generators
    pure_generator = PureSignalGenerator(random_state=42)
    contaminator = DataContaminator(random_state=42)
    
    # Generate base signal
    base_signal = pure_generator.generate_arfima(500, 0.3)
    print(f"\nBase signal: mean={np.mean(base_signal):.4f}, std={np.std(base_signal):.4f}")
    
    # Apply different contaminations
    contaminated = {}
    
    # Polynomial trend
    trend_signal = contaminator.add_polynomial_trend(base_signal, degree=2, amplitude=0.1)
    contaminated['Polynomial Trend'] = trend_signal
    print(f"Polynomial trend: mean={np.mean(trend_signal):.4f}, std={np.std(trend_signal):.4f}")
    
    # Periodicity
    periodic_signal = contaminator.add_periodicity(base_signal, frequency=50, amplitude=0.2)
    contaminated['Periodicity'] = periodic_signal
    print(f"Periodicity: mean={np.mean(periodic_signal):.4f}, std={np.std(periodic_signal):.4f}")
    
    # Outliers
    outlier_signal = contaminator.add_outliers(base_signal, fraction=0.02, magnitude=4.0)
    contaminated['Outliers'] = outlier_signal
    print(f"Outliers: mean={np.mean(outlier_signal):.4f}, std={np.std(outlier_signal):.4f}")
    
    # Heavy tails
    heavy_tail_signal = contaminator.add_heavy_tails(base_signal, df=2.0, fraction=0.15)
    contaminated['Heavy Tails'] = heavy_tail_signal
    print(f"Heavy tails: mean={np.mean(heavy_tail_signal):.4f}, std={np.std(heavy_tail_signal):.4f}")
    
    return base_signal, contaminated


def demo_irregular_sampling():
    """Demonstrate irregular sampling."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Irregular Sampling")
    print("=" * 60)
    
    # Initialize generators
    pure_generator = PureSignalGenerator(random_state=42)
    contaminator = DataContaminator(random_state=42)
    
    # Generate base signal
    base_signal = pure_generator.generate_arfima(500, 0.3)
    
    # Apply irregular sampling
    irregular_signals = {}
    for missing_fraction in [0.1, 0.2, 0.3]:
        sampled_data, time_indices = contaminator.add_irregular_sampling(
            base_signal, missing_fraction
        )
        irregular_signals[f'Missing {int(missing_fraction*100)}%'] = (sampled_data, time_indices)
        print(f"Missing {int(missing_fraction*100)}%: {len(sampled_data)} points (original: {len(base_signal)})")
    
    return irregular_signals


def demo_comprehensive_generation():
    """Demonstrate comprehensive dataset generation."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Comprehensive Dataset Generation")
    print("=" * 60)
    
    # Initialize comprehensive generator
    generator = SyntheticDataGenerator(random_state=42)
    
    # Generate comprehensive dataset (without saving for demo)
    print("\nGenerating comprehensive dataset...")
    dataset = generator.generate_comprehensive_dataset(n=500, save=False)
    
    print(f"\nDataset Summary:")
    print(f"  - Clean signals: {len(dataset['clean_signals'])}")
    print(f"  - Contaminated signals: {len(dataset['contaminated_signals'])}")
    print(f"  - Irregular signals: {len(dataset['irregular_signals'])}")
    print(f"  - Total signals: {len(dataset['clean_signals']) + len(dataset['contaminated_signals']) + len(dataset['irregular_signals'])}")
    
    # Show some examples
    print(f"\nClean signal examples:")
    for i, name in enumerate(list(dataset['clean_signals'].keys())[:3]):
        signal = dataset['clean_signals'][name]
        print(f"  - {name}: mean={np.mean(signal):.4f}, std={np.std(signal):.4f}")
    
    print(f"\nContaminated signal examples:")
    for i, name in enumerate(list(dataset['contaminated_signals'].keys())[:3]):
        signal = dataset['contaminated_signals'][name]
        print(f"  - {name}: mean={np.mean(signal):.4f}, std={np.std(signal):.4f}")
    
    return dataset


def demo_data_saving():
    """Demonstrate data saving functionality."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Data Saving")
    print("=" * 60)
    
    # Initialize generator with temporary directory
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    generator = SyntheticDataGenerator(data_root=temp_dir, random_state=42)
    
    try:
        # Generate and save a small dataset
        print(f"\nGenerating and saving dataset to: {temp_dir}")
        dataset = generator.generate_comprehensive_dataset(n=200, save=True)
        
        # Check what was created
        from pathlib import Path
        temp_path = Path(temp_dir)
        
        raw_files = list(temp_path.glob("raw/*.csv"))
        metadata_files = list(temp_path.glob("metadata/*.json"))
        
        print(f"\nFiles created:")
        print(f"  - Raw data files: {len(raw_files)}")
        print(f"  - Metadata files: {len(metadata_files)}")
        
        if raw_files:
            print(f"\nExample raw file: {raw_files[0].name}")
            # Read and show first few lines
            import pandas as pd
            df = pd.read_csv(raw_files[0])
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  First few values: {df['value'].head().tolist()}")
        
        if metadata_files:
            print(f"\nExample metadata file: {metadata_files[0].name}")
            # Read and show metadata
            import json
            with open(metadata_files[0], 'r') as f:
                metadata = json.load(f)
            print(f"  Name: {metadata.get('name', 'N/A')}")
            print(f"  Type: {metadata.get('data_type', 'N/A')}")
            print(f"  Parameters: {metadata.get('parameters', 'N/A')}")
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")


def main():
    """Run all demonstrations."""
    print("SYNTHETIC DATA GENERATION DEMONSTRATION")
    print("=" * 80)
    print("This script demonstrates the comprehensive synthetic data generation")
    print("capabilities for long-range dependence analysis.")
    print("=" * 80)
    
    # Run demonstrations
    arfima_signals, fbm_signals, fgn_signals = demo_pure_signals()
    base_signal, contaminated = demo_contamination()
    irregular_signals = demo_irregular_sampling()
    dataset = demo_comprehensive_generation()
    demo_data_saving()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nThe synthetic data generation module provides:")
    print("✓ Pure signal generators (ARFIMA, fBm, fGn)")
    print("✓ Data contaminators (trends, periodicity, outliers, etc.)")
    print("✓ Irregular sampling capabilities")
    print("✓ Comprehensive dataset generation")
    print("✓ Proper data storage with metadata")
    print("\nAll generated data conforms to the project's data standards.")


if __name__ == "__main__":
    main()

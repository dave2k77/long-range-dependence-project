#!/usr/bin/env python3
"""
Data Setup Script for Long-Range Dependence Project

This script sets up the complete dataset collection including:
- Synthetic datasets for testing and validation
- Realistic financial datasets for analysis
- Proper data organization and metadata
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.data_manager import setup_project_data, DataManager


def main():
    """Set up complete dataset collection for the project."""
    print("Setting up Data Collection for Long-Range Dependence Project")
    print("=" * 60)
    
    try:
        # Set up complete dataset collection
        print("\n1. Setting up complete dataset collection...")
        
        # Define financial symbols to process
        financial_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
        
        # Set up the data
        summary = setup_project_data(
            data_root="data",
            n_synthetic=1000,  # 1000 points per synthetic dataset
            financial_symbols=financial_symbols,
            seed=42  # For reproducibility
        )
        
        print("\n2. Dataset Setup Summary:")
        print(f"   - Synthetic datasets: {summary['synthetic_datasets']}")
        print(f"   - Realistic datasets: {summary['realistic_datasets']}")
        print(f"   - Total datasets: {summary['total_datasets']}")
        print(f"   - Data dictionary: {summary['data_dictionary_path']}")
        print(f"   - Protocol: {summary['protocol_path']}")
        
        # Show synthetic datasets
        print("\n3. Synthetic Datasets Created:")
        for name, path in summary['synthetic_files'].items():
            print(f"   - {name}: {path}")
        
        # Show realistic datasets
        print("\n4. Realistic Datasets Created:")
        for symbol, path in summary['realistic_files'].items():
            print(f"   - {symbol}: {path}")
        
        print("\n5. Data Organization:")
        print("   - Raw data: data/raw/")
        print("   - Processed data: data/processed/")
        print("   - Metadata: data/metadata/")
        
        print("\n✓ Data setup completed successfully!")
        print("\nYou can now use these datasets for analysis with:")
        print("   python scripts/run_full_analysis.py")
        
    except Exception as e:
        print(f"\n✗ Error during data setup: {e}")
        print("Please check your internet connection for financial data download.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

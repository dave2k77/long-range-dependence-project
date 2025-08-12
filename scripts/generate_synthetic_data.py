#!/usr/bin/env python3
"""
Synthetic Data Generation Script

This script generates comprehensive synthetic datasets for long-range dependence analysis,
including pure signals (ARFIMA, fBm, fGn) and contaminated signals with various artifacts.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.synthetic_generator import SyntheticDataGenerator
import argparse


def main():
    """Main function for generating synthetic data."""
    parser = argparse.ArgumentParser(description="Generate synthetic data for long-range dependence analysis")
    parser.add_argument("--n", type=int, default=1000, help="Length of time series (default: 1000)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--data-root", type=str, default="data", help="Data root directory (default: data)")
    parser.add_argument("--clean-only", action="store_true", help="Generate only clean signals")
    parser.add_argument("--contaminated-only", action="store_true", help="Generate only contaminated signals")
    parser.add_argument("--irregular-only", action="store_true", help="Generate only irregularly sampled signals")
    
    args = parser.parse_args()
    
    print("Initializing synthetic data generator...")
    generator = SyntheticDataGenerator(data_root=args.data_root, random_state=args.random_state)
    
    if args.clean_only:
        print("Generating clean signals only...")
        clean_signals = generator.generate_clean_signals(n=args.n, save=True)
        print(f"Generated {len(clean_signals)} clean signals")
        
    elif args.contaminated_only:
        print("Generating contaminated signals only...")
        contaminated_signals = generator.generate_contaminated_signals(n=args.n, save=True)
        print(f"Generated {len(contaminated_signals)} contaminated signals")
        
    elif args.irregular_only:
        print("Generating irregularly sampled signals only...")
        irregular_signals = generator.generate_irregular_sampled_signals(n=args.n, save=True)
        print(f"Generated {len(irregular_signals)} irregularly sampled signals")
        
    else:
        print("Generating comprehensive synthetic dataset...")
        dataset = generator.generate_comprehensive_dataset(n=args.n, save=True)
        print(f"Generated {len(dataset['clean_signals'])} clean signals")
        print(f"Generated {len(dataset['contaminated_signals'])} contaminated signals")
        print(f"Generated {len(dataset['irregular_signals'])} irregularly sampled signals")
    
    print("Synthetic data generation completed successfully!")
    print(f"Data saved to {args.data_root}/raw/ directory with appropriate metadata.")


if __name__ == "__main__":
    main()

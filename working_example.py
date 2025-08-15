#!/usr/bin/env python3
"""
Working Example Script

This script demonstrates how to properly import and use the long-range dependence analysis tools.
Run this script from the project root directory.
"""

import sys
import os
from pathlib import Path

def setup_imports():
    """Set up the Python path to include the src directory"""
    # Get the current working directory
    current_dir = Path.cwd()
    
    # Find the project root (where src directory is located)
    project_root = None
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "src").exists() and (parent / "src" / "data_processing").exists():
            project_root = parent
            break
    
    if project_root is None:
        print("❌ Could not find project root with src directory")
        print(f"Current directory: {current_dir}")
        print("Make sure you're running this script from the project root directory")
        return False
    
    # Add src to Python path
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"✓ Added {src_path} to Python path")
    
    print(f"✓ Project root: {project_root}")
    return True

def main():
    """Main function demonstrating the basic functionality"""
    print("=" * 60)
    print("LONG-RANGE DEPENDENCE ANALYSIS - WORKING EXAMPLE")
    print("=" * 60)
    
    # First, fix the imports
    if not setup_imports():
        return
    
    try:
        # Now import the modules
        print("\n📦 Importing modules...")
        from data_processing.synthetic_generator import SyntheticDataGenerator
        print("✓ SyntheticDataGenerator imported successfully")
        
        # Create a data generator
        print("\n🔧 Creating data generator...")
        generator = SyntheticDataGenerator(random_state=42)
        print("✓ Data generator created successfully")
        
        # Generate some test data
        print("\n📊 Generating synthetic data...")
        
        # Generate ARFIMA process
        arfima_signal = generator.generate_arfima(n=1000, d=0.3)
        print(f"✓ Generated ARFIMA signal with {len(arfima_signal)} points (d=0.3)")
        
        # Generate Fractional Brownian Motion
        fbm_signal = generator.generate_fbm(n=1000, hurst=0.7)
        print(f"✓ Generated fBm signal with {len(fbm_signal)} points (H=0.7)")
        
        # Generate Fractional Gaussian Noise
        fgn_signal = generator.generate_fgn(n=1000, hurst=0.6)
        print(f"✓ Generated fGn signal with {len(fgn_signal)} points (H=0.6)")
        
        # Basic statistics
        print("\n📈 Basic statistics:")
        print(f"ARFIMA - Mean: {arfima_signal.mean():.4f}, Std: {arfima_signal.std():.4f}")
        print(f"fBm - Mean: {fbm_signal.mean():.4f}, Std: {fbm_signal.std():.4f}")
        print(f"fGn - Mean: {fgn_signal.mean():.4f}, Std: {fgn_signal.std():.4f}")
        
        print("\n✅ SUCCESS! All imports and basic functionality working.")
        print("\nYou can now use these modules in your own scripts:")
        print("from data_processing.synthetic_generator import SyntheticDataGenerator")
        print("from analysis.higuchi_analysis import HiguchiAnalysis")
        print("from analysis.dfa_analysis import DFAAnalysis")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
My Analysis Template

This is a template script showing how to properly import and use the long-range dependence analysis tools.
Copy this file and modify it for your own analysis.
"""

# Option 1: Use the import helper (recommended)
from src.import_helper import setup_project_imports
setup_project_imports()

# Option 2: Manual import path setup (alternative)
# import sys
# from pathlib import Path
# current_dir = Path.cwd()
# project_root = None
# for parent in [current_dir] + list(current_dir.parents):
#     if (parent / "src").exists() and (parent / "src" / "data_processing").exists():
#         project_root = parent
#         break
# if project_root:
#     src_path = str(project_root / "src")
#     if src_path not in sys.path:
#         sys.path.insert(0, src_path)

# Now you can import the modules
from data_processing.synthetic_generator import SyntheticDataGenerator
from analysis.higuchi_analysis import higuchi_fractal_dimension
from analysis.dfa_analysis import dfa

def main():
    """Your main analysis function"""
    print("🚀 Starting my analysis...")
    
    # Create a data generator
    generator = SyntheticDataGenerator(random_state=42)
    
    # Generate some data
    data = generator.generate_arfima(n=1000, d=0.3)
    print(f"Generated data with {len(data)} points")
    
    # Perform Higuchi analysis
    print("📊 Performing Higuchi analysis...")
    k_values, l_values, higuchi_result = higuchi_fractal_dimension(data)
    print(f"Higuchi fractal dimension: {higuchi_result.fractal_dimension:.3f}")
    print(f"R-squared: {higuchi_result.r_squared:.3f}")
    
    # Perform DFA analysis
    print("📊 Performing DFA analysis...")
    dfa_scales, dfa_flucts, dfa_result = dfa(data)
    print(f"DFA alpha: {dfa_result.alpha:.3f}")
    print(f"R-value: {dfa_result.rvalue:.3f}")
    
    print("✅ Analysis complete!")
    
    # You can access more detailed results
    print(f"\n📈 Detailed results:")
    print(f"Higuchi - k range: {higuchi_result.k_range}, n points: {higuchi_result.n_points}")
    print(f"DFA - scales: {len(dfa_result.scales)}, flucts: {len(dfa_result.flucts)}")

if __name__ == "__main__":
    main()

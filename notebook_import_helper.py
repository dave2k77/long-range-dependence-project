#!/usr/bin/env python3
"""
Notebook Import Helper

This script provides a simple way to set up imports for notebooks.
Copy and paste the code below into your notebook cell to fix import issues.
"""

import sys
from pathlib import Path

def setup_notebook_imports():
    """Set up Python paths for notebook usage"""
    current_dir = Path.cwd()
    project_root = None
    
    # Walk up the directory tree to find the project root
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "src").exists() and (parent / "src" / "data_processing").exists():
            project_root = parent
            break
    
    if project_root is None:
        print("❌ Could not find project root with src directory")
        return False
    
    src_path = str(project_root / "src")
    
    # Add src to Python path if not already there
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    print(f"✓ Added {src_path} to Python path")
    print(f"✓ Project root: {project_root}")
    print(f"✓ Current working directory: {current_dir}")
    
    # Test the import
    try:
        from data_processing.synthetic_generator import SyntheticDataGenerator
        print("✓ Successfully imported SyntheticDataGenerator")
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

if __name__ == "__main__":
    setup_notebook_imports()
else:
    # When imported, just provide the function
    pass

# Code to copy into notebook:
print("=" * 60)
print("COPY THIS CODE INTO YOUR NOTEBOOK CELL:")
print("=" * 60)
print("""
# Fix import issues in notebook
import sys
from pathlib import Path

# Find project root and add src to path
current_dir = Path.cwd()
project_root = None

for parent in [current_dir] + list(current_dir.parents):
    if (parent / "src").exists() and (parent / "src" / "data_processing").exists():
        project_root = parent
        break

if project_root:
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    print(f"✓ Added {src_path} to Python path")
    
    # Now you can import
    from data_processing.synthetic_generator import SyntheticDataGenerator
    print("✓ Import successful!")
else:
    print("❌ Could not find project root")
""")
print("=" * 60)

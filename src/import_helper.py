"""
Import Helper Module

This module provides utilities to fix import path issues when running scripts
from different directories. Import this at the top of your scripts to ensure
all modules can be found.
"""

import sys
from pathlib import Path

def setup_project_imports():
    """
    Set up the Python path to include the project's src directory.
    
    This function should be called at the beginning of any script that needs
    to import from the project modules.
    
    Returns:
        bool: True if successful, False otherwise
    """
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
        print("Make sure you're running the script from the project root directory")
        return False
    
    # Add src to Python path
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"✓ Added {src_path} to Python path")
    
    return True

def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path: Path to the project root, or None if not found
    """
    current_dir = Path.cwd()
    
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "src").exists() and (parent / "src" / "data_processing").exists():
            return parent
    
    return None

# Auto-setup when module is imported
if __name__ != "__main__":
    setup_project_imports()

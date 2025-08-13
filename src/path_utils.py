"""
Path utilities for the long-range dependence analysis package.

This module provides functions to properly set up Python paths for importing
modules from the src directory, whether running from scripts or notebooks.
"""

import sys
from pathlib import Path


def setup_project_paths():
    """
    Set up Python paths to include the project's src directory.
    
    This function detects whether it's running from a script or notebook
    and sets up the appropriate paths.
    
    Returns:
        str: The path to the src directory that was added
    """
    # Try to get the file path (works for scripts)
    try:
        if '__file__' in globals():
            # Running from a script
            current_file = Path(__file__)
            project_root = current_file.parent.parent
        else:
            # Running from notebook or interactive session
            # Try to find the project root by looking for src directory
            current_dir = Path.cwd()
            project_root = None
            
            # Walk up the directory tree to find the project root
            for parent in [current_dir] + list(current_dir.parents):
                if (parent / "src").exists() and (parent / "src" / "data_processing").exists():
                    project_root = parent
                    break
            
            if project_root is None:
                # Fallback: assume we're in the project root
                project_root = current_dir
    
    except Exception:
        # Fallback: use current working directory
        project_root = Path.cwd()
    
    src_path = str(project_root / "src")
    
    # Add src to Python path if not already there
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    return src_path


def verify_imports():
    """
    Verify that key modules can be imported after setting up paths.
    
    Returns:
        bool: True if all imports succeed, False otherwise
    """
    try:
        from data_processing.synthetic_generator import SyntheticDataGenerator
        return True
    except ImportError as e:
        print(f"Import verification failed: {e}")
        return False


def print_path_info():
    """
    Print information about the current path setup.
    """
    src_path = setup_project_paths()
    print(f"Project paths set up:")
    print(f"  Project root: {Path(src_path).parent}")
    print(f"  Src directory: {src_path}")
    print(f"  Current working directory: {Path.cwd()}")
    print(f"  Python path starts with: {sys.path[:3]}")
    
    if verify_imports():
        print("✓ All key modules can be imported successfully")
    else:
        print("❌ Some modules cannot be imported")


# Note: Call setup_project_paths() explicitly when needed

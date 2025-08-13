#!/usr/bin/env python3
"""
GUI Launcher Script for Long-Range Dependence Project

This script launches the graphical user interface for the benchmark.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    """Main entry point for the GUI application."""
    try:
        # Import and run the GUI
        from gui.main_window import MainWindow
        
        print("Starting Long-Range Dependence Analysis Benchmark GUI...")
        print("Loading interface...")
        
        # Create and run the main window
        app = MainWindow()
        app.run()
        
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

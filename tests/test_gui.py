"""
Tests for GUI modules

This module tests that the GUI components can be imported and initialized correctly.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestGUIImports(unittest.TestCase):
    """Test that GUI modules can be imported correctly."""
    
    def test_gui_module_import(self):
        """Test that the main GUI module can be imported."""
        try:
            from gui import MainWindow
            self.assertTrue(True, "GUI module imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import GUI module: {e}")
    
    def test_gui_components_import(self):
        """Test that individual GUI components can be imported."""
        try:
            from gui.data_manager_gui import DataManagerFrame
            from gui.analysis_gui import AnalysisFrame
            from gui.synthetic_data_gui import SyntheticDataFrame
            from gui.submission_gui import SubmissionFrame
            from gui.results_gui import ResultsFrame
            from gui.config_gui import ConfigFrame
            self.assertTrue(True, "All GUI components imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import GUI components: {e}")
    
    def test_gui_dependencies(self):
        """Test that GUI dependencies are available."""
        try:
            import tkinter as tk
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import yaml
            self.assertTrue(True, "All GUI dependencies available")
        except ImportError as e:
            self.fail(f"Missing GUI dependency: {e}")


if __name__ == "__main__":
    unittest.main()

"""
Test suite for synthetic data generation module.

Tests the PureSignalGenerator, DataContaminator, and SyntheticDataGenerator classes.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.synthetic_generator import (
    PureSignalGenerator, 
    DataContaminator, 
    SyntheticDataGenerator
)


class TestPureSignalGenerator(unittest.TestCase):
    """Test cases for PureSignalGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = PureSignalGenerator(random_state=42)
        self.n = 100
    
    def test_generate_arfima_valid_d(self):
        """Test ARFIMA generation with valid d parameter."""
        d = 0.3
        signal = self.generator.generate_arfima(self.n, d)
        
        self.assertEqual(len(signal), self.n)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(np.all(np.isfinite(signal)))
    
    def test_generate_arfima_invalid_d(self):
        """Test ARFIMA generation with invalid d parameter."""
        with self.assertRaises(ValueError):
            self.generator.generate_arfima(self.n, 0.6)  # d > 0.5
        
        with self.assertRaises(ValueError):
            self.generator.generate_arfima(self.n, -0.1)  # d < 0
    
    def test_generate_fbm_valid_hurst(self):
        """Test fBm generation with valid Hurst exponent."""
        H = 0.7
        signal = self.generator.generate_fbm(self.n, H)
        
        self.assertEqual(len(signal), self.n)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(np.all(np.isfinite(signal)))
    
    def test_generate_fbm_invalid_hurst(self):
        """Test fBm generation with invalid Hurst exponent."""
        with self.assertRaises(ValueError):
            self.generator.generate_fbm(self.n, 1.2)  # H > 1
        
        with self.assertRaises(ValueError):
            self.generator.generate_fbm(self.n, -0.1)  # H < 0
    
    def test_generate_fgn_valid_hurst(self):
        """Test fGn generation with valid Hurst exponent."""
        H = 0.5
        signal = self.generator.generate_fgn(self.n, H)
        
        self.assertEqual(len(signal), self.n)
        self.assertIsInstance(signal, np.ndarray)
        self.assertTrue(np.all(np.isfinite(signal)))
    
    def test_generate_fgn_invalid_hurst(self):
        """Test fGn generation with invalid Hurst exponent."""
        with self.assertRaises(ValueError):
            self.generator.generate_fgn(self.n, 1.1)  # H > 1
        
        with self.assertRaises(ValueError):
            self.generator.generate_fgn(self.n, -0.1)  # H < 0
    
    def test_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        generator1 = PureSignalGenerator(random_state=42)
        generator2 = PureSignalGenerator(random_state=42)
        
        signal1 = generator1.generate_arfima(self.n, 0.3)
        signal2 = generator2.generate_arfima(self.n, 0.3)
        
        np.testing.assert_array_equal(signal1, signal2)


class TestDataContaminator(unittest.TestCase):
    """Test cases for DataContaminator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.contaminator = DataContaminator(random_state=42)
        self.test_data = np.random.normal(0, 1, 100)
    
    def test_add_polynomial_trend(self):
        """Test polynomial trend addition."""
        contaminated = self.contaminator.add_polynomial_trend(
            self.test_data, degree=2, amplitude=0.1
        )
        
        self.assertEqual(len(contaminated), len(self.test_data))
        self.assertIsInstance(contaminated, np.ndarray)
        self.assertTrue(np.all(np.isfinite(contaminated)))
        
        # Check that trend was actually added (variance should increase)
        self.assertGreater(np.var(contaminated), np.var(self.test_data))
    
    def test_add_periodicity(self):
        """Test periodic component addition."""
        contaminated = self.contaminator.add_periodicity(
            self.test_data, frequency=10, amplitude=0.2
        )
        
        self.assertEqual(len(contaminated), len(self.test_data))
        self.assertIsInstance(contaminated, np.ndarray)
        self.assertTrue(np.all(np.isfinite(contaminated)))
    
    def test_add_outliers(self):
        """Test outlier addition."""
        contaminated = self.contaminator.add_outliers(
            self.test_data, fraction=0.05, magnitude=3.0
        )
        
        self.assertEqual(len(contaminated), len(self.test_data))
        self.assertIsInstance(contaminated, np.ndarray)
        self.assertTrue(np.all(np.isfinite(contaminated)))
        
        # Check that outliers were added (max absolute value should increase)
        self.assertGreater(
            np.max(np.abs(contaminated)), 
            np.max(np.abs(self.test_data))
        )
    
    def test_add_irregular_sampling(self):
        """Test irregular sampling."""
        sampled_data, time_indices = self.contaminator.add_irregular_sampling(
            self.test_data, missing_fraction=0.2
        )
        
        self.assertLess(len(sampled_data), len(self.test_data))
        self.assertEqual(len(sampled_data), len(time_indices))
        self.assertTrue(np.all(np.isfinite(sampled_data)))
        self.assertTrue(np.all(time_indices >= 0))
        self.assertTrue(np.all(time_indices < len(self.test_data)))
    
    def test_add_heavy_tails(self):
        """Test heavy-tailed fluctuations addition."""
        contaminated = self.contaminator.add_heavy_tails(
            self.test_data, df=2.0, fraction=0.1
        )
        
        self.assertEqual(len(contaminated), len(self.test_data))
        self.assertIsInstance(contaminated, np.ndarray)
        self.assertTrue(np.all(np.isfinite(contaminated)))


class TestSyntheticDataGenerator(unittest.TestCase):
    """Test cases for SyntheticDataGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.generator = SyntheticDataGenerator(data_root=self.test_dir, random_state=42)
        self.n = 100
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator.pure_generator)
        self.assertIsNotNone(self.generator.contaminator)
        self.assertIsNotNone(self.generator.data_manager)
    
    def test_generate_clean_signals(self):
        """Test clean signal generation."""
        signals = self.generator.generate_clean_signals(n=self.n, save=False)
        
        self.assertIsInstance(signals, dict)
        self.assertGreater(len(signals), 0)
        
        # Check that all signals have correct length
        for name, signal in signals.items():
            self.assertEqual(len(signal), self.n)
            self.assertIsInstance(signal, np.ndarray)
            self.assertTrue(np.all(np.isfinite(signal)))
    
    def test_generate_contaminated_signals(self):
        """Test contaminated signal generation."""
        signals = self.generator.generate_contaminated_signals(n=self.n, save=False)
        
        self.assertIsInstance(signals, dict)
        self.assertGreater(len(signals), 0)
        
        # Check that all signals have correct length
        for name, signal in signals.items():
            self.assertEqual(len(signal), self.n)
            self.assertIsInstance(signal, np.ndarray)
            self.assertTrue(np.all(np.isfinite(signal)))
    
    def test_generate_irregular_sampled_signals(self):
        """Test irregularly sampled signal generation."""
        signals = self.generator.generate_irregular_sampled_signals(n=self.n, save=False)
        
        self.assertIsInstance(signals, dict)
        self.assertGreater(len(signals), 0)
        
        # Check that all signals have correct structure
        for name, (signal, indices) in signals.items():
            self.assertIsInstance(signal, np.ndarray)
            self.assertIsInstance(indices, np.ndarray)
            self.assertEqual(len(signal), len(indices))
            self.assertTrue(np.all(np.isfinite(signal)))
    
    def test_generate_comprehensive_dataset(self):
        """Test comprehensive dataset generation."""
        dataset = self.generator.generate_comprehensive_dataset(n=self.n, save=False)
        
        self.assertIn('clean_signals', dataset)
        self.assertIn('contaminated_signals', dataset)
        self.assertIn('irregular_signals', dataset)
        
        self.assertGreater(len(dataset['clean_signals']), 0)
        self.assertGreater(len(dataset['contaminated_signals']), 0)
        self.assertGreater(len(dataset['irregular_signals']), 0)
    
    def test_save_functionality(self):
        """Test that data saving works correctly."""
        # Generate a small dataset and save it
        signals = self.generator.generate_clean_signals(n=50, save=True)
        
        # Check that files were created
        raw_dir = Path(self.test_dir) / "raw"
        self.assertTrue(raw_dir.exists())
        
        # Check that metadata was created
        metadata_dir = Path(self.test_dir) / "metadata"
        self.assertTrue(metadata_dir.exists())


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete synthetic data generation pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.generator = SyntheticDataGenerator(data_root=self.test_dir, random_state=42)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_end_to_end_generation(self):
        """Test complete end-to-end synthetic data generation."""
        # Generate comprehensive dataset
        dataset = self.generator.generate_comprehensive_dataset(n=200, save=True)
        
        # Verify all components are present
        self.assertIn('clean_signals', dataset)
        self.assertIn('contaminated_signals', dataset)
        self.assertIn('irregular_signals', dataset)
        
        # Check that data files were created
        raw_dir = Path(self.test_dir) / "raw"
        metadata_dir = Path(self.test_dir) / "metadata"
        
        self.assertTrue(raw_dir.exists())
        self.assertTrue(metadata_dir.exists())
        
        # Count generated files
        csv_files = list(raw_dir.glob("*.csv"))
        json_files = list(metadata_dir.glob("*.json"))
        
        self.assertGreater(len(csv_files), 0)
        self.assertGreater(len(json_files), 0)
    
    def test_data_quality(self):
        """Test that generated data has reasonable statistical properties."""
        signals = self.generator.generate_clean_signals(n=500, save=False)
        
        for name, signal in signals.items():
            # Check basic statistics
            self.assertGreater(np.std(signal), 0)
            self.assertTrue(np.all(np.isfinite(signal)))
            
            # Check that signal is not constant
            self.assertGreater(np.var(signal), 0)


if __name__ == "__main__":
    unittest.main()

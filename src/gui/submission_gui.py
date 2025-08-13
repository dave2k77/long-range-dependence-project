"""
Submission GUI Frame

This module provides a GUI interface for submitting new models and datasets
to the benchmark with validation and testing.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
from pathlib import Path
import threading
import json
from typing import Dict, Any, Optional
import queue

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from submission import (
    SubmissionManager, ModelMetadata, DatasetMetadata, BaseEstimatorModel
)


class DemoEstimatorModel(BaseEstimatorModel):
    """Example estimator model for GUI demonstration."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hurst_estimate = None
        self.alpha_estimate = None
        self.confidence_intervals = {}
        self.quality_metrics = {}
    
    def fit(self, data):
        """Fit the model to the data."""
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        
        data = self.preprocess_data(data)
        
        # Simple variance-time plot analysis
        import numpy as np
        n = len(data)
        scales = [2**i for i in range(2, int(np.log2(n/4)))]
        variances = []
        
        for scale in scales:
            if scale < n:
                chunks = [data[i:i+scale] for i in range(0, n-scale+1, scale)]
                chunk_vars = [chunk.var() for chunk in chunks if len(chunk) == scale]
                if chunk_vars:
                    variances.append(np.mean(chunk_vars))
                else:
                    variances.append(np.nan)
            else:
                variances.append(np.nan)
        
        # Remove NaN values
        valid_scales = [s for s, v in zip(scales, variances) if not np.isnan(v)]
        valid_variances = [v for v in variances if not np.isnan(v)]
        
        if len(valid_scales) < 3:
            raise ValueError("Insufficient valid scales for analysis")
        
        # Fit power law: variance ~ scale^beta
        log_scales = np.log(valid_scales)
        log_variances = np.log(valid_variances)
        
        # Simple linear regression
        coeffs = np.polyfit(log_scales, log_variances, 1)
        beta = coeffs[0]
        
        self.hurst_estimate = 1 - beta / 2
        self.alpha_estimate = 2 * self.hurst_estimate
        
        # Calculate confidence intervals (simplified)
        std_error = np.sqrt(np.mean((log_variances - np.polyval(coeffs, log_scales))**2))
        self.confidence_intervals = {
            "hurst": (max(0, self.hurst_estimate - 2*std_error), 
                     min(1, self.hurst_estimate + 2*std_error)),
            "alpha": (max(0, self.alpha_estimate - 2*std_error), 
                     min(2, self.alpha_estimate + 2*std_error))
        }
        
        # Calculate quality metrics
        r_squared = 1 - np.sum((log_variances - np.polyval(coeffs, log_scales))**2) / np.sum((log_variances - np.mean(log_variances))**2)
        self.quality_metrics = {
            "r_squared": r_squared,
            "std_error": std_error,
            "n_scales": len(valid_scales)
        }
        
        self.fitted = True
        return self
    
    def estimate_hurst(self):
        """Estimate the Hurst exponent."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.hurst_estimate
    
    def estimate_alpha(self):
        """Estimate the alpha parameter."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.alpha_estimate
    
    def get_confidence_intervals(self):
        """Get confidence intervals."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.confidence_intervals
    
    def get_quality_metrics(self):
        """Get quality metrics."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.quality_metrics


class SubmissionFrame(ttk.Frame):
    """Submission tab for submitting models and datasets."""
    
    def __init__(self, parent, shared_data: Dict[str, Any], message_queue: queue.Queue):
        """Initialize the submission frame."""
        super().__init__(parent)
        self.shared_data = shared_data
        self.message_queue = message_queue
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the user interface."""
        # Configure grid weights
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Left panel - Controls
        self._setup_controls_panel()
        
        # Right panel - Results display
        self._setup_results_panel()
        
    def _setup_controls_panel(self):
        """Setup the controls panel."""
        controls_frame = ttk.LabelFrame(self, text="Submission Controls", padding="10")
        controls_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        
        # Submission type selection
        type_frame = ttk.LabelFrame(controls_frame, text="Submission Type", padding="5")
        type_frame.pack(fill="x", pady=(0, 10))
        
        self.submission_type = tk.StringVar(value="model")
        ttk.Radiobutton(type_frame, text="Model Submission", 
                       variable=self.submission_type, value="model",
                       command=self._on_type_changed).pack(anchor="w", pady=2)
        ttk.Radiobutton(type_frame, text="Dataset Submission", 
                       variable=self.submission_type, value="dataset",
                       command=self._on_type_changed).pack(anchor="w", pady=2)
        
        # Model submission section
        self.model_frame = ttk.LabelFrame(controls_frame, text="Model Submission", padding="5")
        self.model_frame.pack(fill="x", pady=(0, 10))
        
        # Model file selection
        model_file_frame = ttk.Frame(self.model_frame)
        model_file_frame.pack(fill="x", pady=2)
        ttk.Label(model_file_frame, text="Model File:").pack(side="left")
        self.model_file_var = tk.StringVar()
        ttk.Entry(model_file_frame, textvariable=self.model_file_var, width=20).pack(side="left", padx=(5, 5))
        ttk.Button(model_file_frame, text="Browse", 
                  command=self._browse_model_file).pack(side="right")
        
        # Model metadata
        model_meta_frame = ttk.Frame(self.model_frame)
        model_meta_frame.pack(fill="x", pady=2)
        ttk.Label(model_meta_frame, text="Model Name:").pack(side="left")
        self.model_name_var = tk.StringVar(value="DemoModel")
        ttk.Entry(model_meta_frame, textvariable=self.model_name_var, width=20).pack(side="right")
        
        model_version_frame = ttk.Frame(self.model_frame)
        model_version_frame.pack(fill="x", pady=2)
        ttk.Label(model_version_frame, text="Version:").pack(side="left")
        self.model_version_var = tk.StringVar(value="1.0.0")
        ttk.Entry(model_version_frame, textvariable=self.model_version_var, width=20).pack(side="right")
        
        model_author_frame = ttk.Frame(self.model_frame)
        model_author_frame.pack(fill="x", pady=2)
        ttk.Label(model_author_frame, text="Author:").pack(side="left")
        self.model_author_var = tk.StringVar(value="Demo Author")
        ttk.Entry(model_author_frame, textvariable=self.model_author_var, width=20).pack(side="right")
        
        model_desc_frame = ttk.Frame(self.model_frame)
        model_desc_frame.pack(fill="x", pady=2)
        ttk.Label(model_desc_frame, text="Description:").pack(side="left")
        self.model_desc_var = tk.StringVar(value="Demo estimator model")
        ttk.Entry(model_desc_frame, textvariable=self.model_desc_var, width=20).pack(side="right")
        
        # Dataset submission section
        self.dataset_frame = ttk.LabelFrame(controls_frame, text="Dataset Submission", padding="5")
        self.dataset_frame.pack(fill="x", pady=(0, 10))
        
        # Dataset file selection
        dataset_file_frame = ttk.Frame(self.dataset_frame)
        dataset_file_frame.pack(fill="x", pady=2)
        ttk.Label(dataset_file_frame, text="Dataset File:").pack(side="left")
        self.dataset_file_var = tk.StringVar()
        ttk.Entry(dataset_file_frame, textvariable=self.dataset_file_var, width=20).pack(side="left", padx=(5, 5))
        ttk.Button(dataset_file_frame, text="Browse", 
                  command=self._browse_dataset_file).pack(side="right")
        
        # Dataset metadata
        dataset_meta_frame = ttk.Frame(self.dataset_frame)
        dataset_meta_frame.pack(fill="x", pady=2)
        ttk.Label(dataset_meta_frame, text="Dataset Name:").pack(side="left")
        self.dataset_name_var = tk.StringVar(value="DemoDataset")
        ttk.Entry(dataset_meta_frame, textvariable=self.dataset_name_var, width=20).pack(side="right")
        
        dataset_source_frame = ttk.Frame(self.dataset_frame)
        dataset_source_frame.pack(fill="x", pady=2)
        ttk.Label(dataset_source_frame, text="Source:").pack(side="left")
        self.dataset_source_var = tk.StringVar(value="Synthetic")
        ttk.Entry(dataset_source_frame, textvariable=self.dataset_source_var, width=20).pack(side="right")
        
        dataset_desc_frame = ttk.Frame(self.dataset_frame)
        dataset_desc_frame.pack(fill="x", pady=2)
        ttk.Label(dataset_desc_frame, text="Description:").pack(side="left")
        self.dataset_desc_var = tk.StringVar(value="Demo dataset for testing")
        ttk.Entry(dataset_desc_frame, textvariable=self.dataset_desc_var, width=20).pack(side="right")
        
        # Action buttons
        actions_frame = ttk.LabelFrame(controls_frame, text="Actions", padding="5")
        actions_frame.pack(fill="x")
        
        ttk.Button(actions_frame, text="Submit Model", 
                  command=self._submit_model).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Submit Dataset", 
                  command=self._submit_dataset).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Create Demo Model", 
                  command=self._create_demo_model).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Create Demo Dataset", 
                  command=self._create_demo_dataset).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Clear Results", 
                  command=self._clear_results).pack(fill="x", pady=2)
        
        # Initialize visibility
        self._on_type_changed()
        
    def _setup_results_panel(self):
        """Setup the results display panel."""
        results_frame = ttk.LabelFrame(self, text="Submission Results", padding="10")
        results_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        results_frame.grid_rowconfigure(1, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Results controls
        results_controls = ttk.Frame(results_frame)
        results_controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Button(results_controls, text="Refresh", 
                  command=self._refresh_results).pack(side="right")
        
        # Results area
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, font=("Courier", 9))
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")
        
        # Initialize empty results
        self.results_text.insert(tk.END, "No submission results yet.\nSubmit a model or dataset to see results.")
        
    def _on_type_changed(self):
        """Handle submission type change."""
        submission_type = self.submission_type.get()
        
        if submission_type == "model":
            self.model_frame.pack(fill="x", pady=(0, 10))
            self.dataset_frame.pack_forget()
        else:
            self.model_frame.pack_forget()
            self.dataset_frame.pack(fill="x", pady=(0, 10))
            
    def _browse_model_file(self):
        """Browse for model file."""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("Python files", "*.py"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.model_file_var.set(file_path)
            
    def _browse_dataset_file(self):
        """Browse for dataset file."""
        file_path = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx;*.xls"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.dataset_file_var.set(file_path)
            
    def _submit_model(self):
        """Submit a model to the benchmark."""
        # Validate inputs
        if not self.model_file_var.get():
            messagebox.showwarning("Warning", "Please select a model file.")
            return
            
        if not self.model_name_var.get():
            messagebox.showwarning("Warning", "Please enter a model name.")
            return
            
        # Run submission in background thread
        def submission_thread():
            try:
                self.message_queue.put({'type': 'status', 'text': 'Submitting model...'})
                self.message_queue.put({'type': 'progress_start'})
                
                # Create model metadata
                metadata = ModelMetadata(
                    name=self.model_name_var.get(),
                    version=self.model_version_var.get(),
                    author=self.model_author_var.get(),
                    description=self.model_desc_var.get(),
                    file_path=self.model_file_var.get()
                )
                
                # Create submission manager
                manager = SubmissionManager()
                
                # Submit model
                result = manager.submit_model(metadata)
                
                # Update results in main thread
                self.after(0, lambda: self._display_submission_result(result, "Model"))
                
            except Exception as e:
                self.message_queue.put({'type': 'error', 'text': f'Model submission error: {e}'})
            finally:
                self.message_queue.put({'type': 'progress_stop'})
                self.message_queue.put({'type': 'status', 'text': 'Ready'})
                
        threading.Thread(target=submission_thread, daemon=True).start()
        
    def _submit_dataset(self):
        """Submit a dataset to the benchmark."""
        # Validate inputs
        if not self.dataset_file_var.get():
            messagebox.showwarning("Warning", "Please select a dataset file.")
            return
            
        if not self.dataset_name_var.get():
            messagebox.showwarning("Warning", "Please enter a dataset name.")
            return
            
        # Run submission in background thread
        def submission_thread():
            try:
                self.message_queue.put({'type': 'status', 'text': 'Submitting dataset...'})
                self.message_queue.put({'type': 'progress_start'})
                
                # Create dataset metadata
                metadata = DatasetMetadata(
                    name=self.dataset_name_var.get(),
                    source=self.dataset_source_var.get(),
                    description=self.dataset_desc_var.get(),
                    file_path=self.dataset_file_var.get(),
                    sampling_frequency=1.0,
                    units="arbitrary",
                    collection_date="2024-01-01"
                )
                
                # Create submission manager
                manager = SubmissionManager()
                
                # Submit dataset
                result = manager.submit_dataset(metadata)
                
                # Update results in main thread
                self.after(0, lambda: self._display_submission_result(result, "Dataset"))
                
            except Exception as e:
                self.message_queue.put({'type': 'error', 'text': f'Dataset submission error: {e}'})
            finally:
                self.message_queue.put({'type': 'progress_stop'})
                self.message_queue.put({'type': 'status', 'text': 'Ready'})
                
        threading.Thread(target=submission_thread, daemon=True).start()
        
    def _create_demo_model(self):
        """Create a demo model file."""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Demo Model",
                defaultextension=".py",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")]
            )
            
            if file_path:
                # Create demo model content
                demo_content = '''"""
Demo Estimator Model for Long-Range Dependence Analysis

This is a demonstration model that implements the BaseEstimatorModel interface.
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from submission.model_submission import BaseEstimatorModel


class DemoEstimatorModel(BaseEstimatorModel):
    """Demo estimator model for demonstration purposes."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hurst_estimate = None
        self.alpha_estimate = None
        self.confidence_intervals = {}
        self.quality_metrics = {}
    
    def fit(self, data):
        """Fit the model to the data."""
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        
        data = self.preprocess_data(data)
        
        # Simple variance-time plot analysis
        n = len(data)
        scales = [2**i for i in range(2, int(np.log2(n/4)))]
        variances = []
        
        for scale in scales:
            if scale < n:
                chunks = [data[i:i+scale] for i in range(0, n-scale+1, scale)]
                chunk_vars = [chunk.var() for chunk in chunks if len(chunk) == scale]
                if chunk_vars:
                    variances.append(np.mean(chunk_vars))
                else:
                    variances.append(np.nan)
            else:
                variances.append(np.nan)
        
        # Remove NaN values
        valid_scales = [s for s, v in zip(scales, variances) if not np.isnan(v)]
        valid_variances = [v for v in variances if not np.isnan(v)]
        
        if len(valid_scales) < 3:
            raise ValueError("Insufficient valid scales for analysis")
        
        # Fit power law: variance ~ scale^beta
        log_scales = np.log(valid_scales)
        log_variances = np.log(valid_variances)
        
        # Simple linear regression
        coeffs = np.polyfit(log_scales, log_variances, 1)
        beta = coeffs[0]
        
        self.hurst_estimate = 1 - beta / 2
        self.alpha_estimate = 2 * self.hurst_estimate
        
        # Calculate confidence intervals (simplified)
        std_error = np.sqrt(np.mean((log_variances - np.polyval(coeffs, log_scales))**2))
        self.confidence_intervals = {
            "hurst": (max(0, self.hurst_estimate - 2*std_error), 
                     min(1, self.hurst_estimate + 2*std_error)),
            "alpha": (max(0, self.alpha_estimate - 2*std_error), 
                     min(2, self.alpha_estimate + 2*std_error))
        }
        
        # Calculate quality metrics
        r_squared = 1 - np.sum((log_variances - np.polyval(coeffs, log_scales))**2) / np.sum((log_variances - np.mean(log_variances))**2)
        self.quality_metrics = {
            "r_squared": r_squared,
            "std_error": std_error,
            "n_scales": len(valid_scales)
        }
        
        self.fitted = True
        return self
    
    def estimate_hurst(self):
        """Estimate the Hurst exponent."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.hurst_estimate
    
    def estimate_alpha(self):
        """Estimate the alpha parameter."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.alpha_estimate
    
    def get_confidence_intervals(self):
        """Get confidence intervals."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.confidence_intervals
    
    def get_quality_metrics(self):
        """Get quality metrics."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.quality_metrics


# Example usage
if __name__ == "__main__":
    # Create demo model instance
    model = DemoEstimatorModel()
    
    # Generate some test data
    np.random.seed(42)
    test_data = np.random.randn(1000)
    
    # Fit the model
    model.fit(test_data)
    
    # Get results
    hurst = model.estimate_hurst()
    alpha = model.estimate_alpha()
    confidence_intervals = model.get_confidence_intervals()
    quality_metrics = model.get_quality_metrics()
    
    print(f"Hurst exponent: {hurst:.4f}")
    print(f"Alpha parameter: {alpha:.4f}")
    print(f"Confidence intervals: {confidence_intervals}")
    print(f"Quality metrics: {quality_metrics}")
'''
                
                # Write to file
                with open(file_path, 'w') as f:
                    f.write(demo_content)
                    
                self.model_file_var.set(file_path)
                self.message_queue.put({'type': 'info', 'text': f'Demo model created: {file_path}'})
                
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f'Error creating demo model: {e}'})
            
    def _create_demo_dataset(self):
        """Create a demo dataset file."""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Demo Dataset",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            ]
            
            if file_path:
                # Generate demo data
                import numpy as np
                import pandas as pd
                
                np.random.seed(42)
                n = 1000
                
                # Generate fractional Brownian motion with H=0.7
                hurst = 0.7
                t = np.arange(n)
                increments = np.random.randn(n-1)
                
                # Create fBm using cumulative sum of increments
                fbm = np.zeros(n)
                for i in range(1, n):
                    fbm[i] = fbm[i-1] + increments[i-1] * (i ** hurst - (i-1) ** hurst)
                
                # Create DataFrame and save
                df = pd.DataFrame({'value': fbm})
                df.to_csv(file_path, index=False)
                
                self.dataset_file_var.set(file_path)
                self.message_queue.put({'type': 'info', 'text': f'Demo dataset created: {file_path}'})
                
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f'Error creating demo dataset: {e}'})
            
    def _display_submission_result(self, result, submission_type: str):
        """Display submission result."""
        self.results_text.delete(1.0, tk.END)
        
        result_text = f"{submission_type} Submission Result\n"
        result_text += "=" * 50 + "\n\n"
        
        result_text += f"Status: {result.status.value}\n"
        result_text += f"Valid: {result.is_valid}\n\n"
        
        if result.validation_results:
            result_text += "Validation Results:\n"
            result_text += "-" * 20 + "\n"
            for i, validation in enumerate(result.validation_results, 1):
                result_text += f"{i}. {validation.status.value}\n"
                if validation.errors:
                    result_text += f"   Errors: {validation.errors}\n"
                if validation.warnings:
                    result_text += f"   Warnings: {validation.warnings}\n"
                result_text += "\n"
                
        if result.test_results:
            result_text += "Test Results:\n"
            result_text += "-" * 15 + "\n"
            for test_name, test_result in result.test_results.items():
                result_text += f"{test_name}: {test_result}\n"
            result_text += "\n"
            
        if result.integration_results:
            result_text += "Integration Results:\n"
            result_text += "-" * 20 + "\n"
            for integration_name, integration_result in result.integration_results.items():
                result_text += f"{integration_name}: {integration_result}\n"
            result_text += "\n"
            
        if result.message:
            result_text += f"Message: {result.message}\n"
            
        self.results_text.insert(tk.END, result_text)
        
    def _refresh_results(self):
        """Refresh the results display."""
        # This could be extended to reload results from storage
        pass
        
    def _clear_results(self):
        """Clear submission results."""
        if messagebox.askyesno("Confirm", "Clear submission results?"):
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No submission results yet.\nSubmit a model or dataset to see results.")
            
    def submit_model(self):
        """Public method to trigger model submission."""
        self._submit_model()
        
    def submit_dataset(self):
        """Public method to trigger dataset submission."""
        self._submit_dataset()

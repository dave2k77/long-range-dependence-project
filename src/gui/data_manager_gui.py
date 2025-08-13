"""
Data Manager GUI Frame

This module provides a GUI interface for managing datasets,
including loading, viewing, and preprocessing data.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
from pathlib import Path
import threading
from typing import Dict, Any, Optional
import queue

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_processing.data_loader import load_data
from data_processing.preprocessing import preprocess_data
from data_processing.quality_check import perform_quality_checks


class DataManagerFrame(ttk.Frame):
    """Data Manager tab for loading and managing datasets."""
    
    def __init__(self, parent, shared_data: Dict[str, Any], message_queue: queue.Queue):
        """Initialize the data manager frame."""
        super().__init__(parent)
        self.shared_data = shared_data
        self.message_queue = message_queue
        
        self._setup_ui()
        self._load_existing_datasets()
        
    def _setup_ui(self):
        """Setup the user interface."""
        # Configure grid weights
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Left panel - Controls
        self._setup_controls_panel()
        
        # Right panel - Data display
        self._setup_data_panel()
        
    def _setup_controls_panel(self):
        """Setup the controls panel."""
        controls_frame = ttk.LabelFrame(self, text="Data Controls", padding="10")
        controls_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        
        # Load data section
        load_frame = ttk.LabelFrame(controls_frame, text="Load Data", padding="5")
        load_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(load_frame, text="Load CSV File", 
                  command=self._load_csv_file).pack(fill="x", pady=2)
        ttk.Button(load_frame, text="Load Excel File", 
                  command=self._load_excel_file).pack(fill="x", pady=2)
        ttk.Button(load_frame, text="Load from Data Folder", 
                  command=self._load_from_data_folder).pack(fill="x", pady=2)
        
        # Dataset list section
        dataset_frame = ttk.LabelFrame(controls_frame, text="Loaded Datasets", padding="5")
        dataset_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Dataset listbox
        self.dataset_listbox = tk.Listbox(dataset_frame, height=8)
        self.dataset_listbox.pack(fill="both", expand=True, pady=2)
        self.dataset_listbox.bind('<<ListboxSelect>>', self._on_dataset_selected)
        
        # Dataset controls
        dataset_controls = ttk.Frame(dataset_frame)
        dataset_controls.pack(fill="x", pady=2)
        
        ttk.Button(dataset_controls, text="View", 
                  command=self._view_selected_dataset).pack(side="left", padx=2)
        ttk.Button(dataset_controls, text="Remove", 
                  command=self._remove_selected_dataset).pack(side="left", padx=2)
        ttk.Button(dataset_controls, text="Preprocess", 
                  command=self._preprocess_selected_dataset).pack(side="left", padx=2)
        
        # Data info section
        info_frame = ttk.LabelFrame(controls_frame, text="Dataset Info", padding="5")
        info_frame.pack(fill="x")
        
        self.info_text = tk.Text(info_frame, height=6, width=30)
        self.info_text.pack(fill="both", expand=True)
        
    def _setup_data_panel(self):
        """Setup the data display panel."""
        data_frame = ttk.LabelFrame(self, text="Data Visualization", padding="10")
        data_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        data_frame.grid_rowconfigure(1, weight=1)
        data_frame.grid_columnconfigure(0, weight=1)
        
        # Data controls
        data_controls = ttk.Frame(data_frame)
        data_controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(data_controls, text="Display:").pack(side="left")
        self.display_var = tk.StringVar(value="time_series")
        display_combo = ttk.Combobox(data_controls, textvariable=self.display_var, 
                                   values=["time_series", "histogram", "autocorrelation", "statistics"],
                                   state="readonly", width=15)
        display_combo.pack(side="left", padx=(5, 0))
        display_combo.bind('<<ComboboxSelected>>', self._update_display)
        
        ttk.Button(data_controls, text="Refresh", 
                  command=self._refresh_display).pack(side="right")
        
        # Plot area
        self.plot_frame = ttk.Frame(data_frame)
        self.plot_frame.grid(row=1, column=0, sticky="nsew")
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        
        # Initialize empty plot
        self._create_empty_plot()
        
    def _create_empty_plot(self):
        """Create an empty plot area."""
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.text(0.5, 0.5, 'No data loaded\nLoad a dataset to view visualization', 
                    ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
    def _load_csv_file(self):
        """Load a CSV file."""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self._load_dataset_from_file(file_path)
            
    def _load_excel_file(self):
        """Load an Excel file."""
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[
                ("Excel files", "*.xlsx;*.xls"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self._load_dataset_from_file(file_path)
            
    def _load_from_data_folder(self):
        """Load datasets from the data folder."""
        data_folder = Path(__file__).parent.parent.parent / "data"
        
        if not data_folder.exists():
            messagebox.showwarning("Warning", "Data folder not found. Please create a 'data' folder.")
            return
            
        # Load from processed data first
        processed_dir = data_folder / "processed"
        if processed_dir.exists():
            for file_path in processed_dir.glob("*.csv"):
                self._load_dataset_from_file(str(file_path))
                
        # Load from raw data
        raw_dir = data_folder / "raw"
        if raw_dir.exists():
            for file_path in raw_dir.glob("*.csv"):
                self._load_dataset_from_file(str(file_path))
                
        self.message_queue.put({'type': 'info', 'text': 'Loaded datasets from data folder'})
        
    def load_dataset_from_file(self, file_path: str):
        """Load a dataset from file path."""
        try:
            self.message_queue.put({'type': 'status', 'text': f'Loading {file_path}...'})
            
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, index_col=0)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, index_col=0)
            else:
                raise ValueError("Unsupported file format")
                
            # Extract dataset name from filename
            dataset_name = Path(file_path).stem
            
            # Convert to numpy array (take first column if multiple)
            if len(df.columns) >= 1:
                data = df.iloc[:, 0].values
            else:
                data = df.values.flatten()
                
            # Store in shared data
            self.shared_data['datasets'][dataset_name] = {
                'data': data,
                'file_path': file_path,
                'name': dataset_name,
                'length': len(data),
                'loaded': True
            }
            
            # Update dataset list
            self._update_dataset_list()
            
            # Select the newly loaded dataset
            self._select_dataset(dataset_name)
            
            self.message_queue.put({'type': 'status', 'text': f'Loaded {dataset_name} ({len(data)} points)'})
            
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f'Error loading file: {e}'})
            
    def _update_dataset_list(self):
        """Update the dataset listbox."""
        self.dataset_listbox.delete(0, tk.END)
        for name in self.shared_data['datasets'].keys():
            self.dataset_listbox.insert(tk.END, name)
            
    def _on_dataset_selected(self, event):
        """Handle dataset selection."""
        selection = self.dataset_listbox.curselection()
        if selection:
            dataset_name = self.dataset_listbox.get(selection[0])
            self._select_dataset(dataset_name)
            
    def _select_dataset(self, dataset_name: str):
        """Select a dataset and update display."""
        if dataset_name in self.shared_data['datasets']:
            self.shared_data['current_dataset'] = dataset_name
            self._update_info_display()
            self._update_data_display()
            
    def _update_info_display(self):
        """Update the dataset info display."""
        dataset_name = self.shared_data.get('current_dataset')
        if not dataset_name or dataset_name not in self.shared_data['datasets']:
            self.info_text.delete(1.0, tk.END)
            return
            
        dataset = self.shared_data['datasets'][dataset_name]
        data = dataset['data']
        
        # Calculate basic statistics
        info_text = f"Dataset: {dataset_name}\n"
        info_text += f"Length: {len(data):,}\n"
        info_text += f"Mean: {np.mean(data):.4f}\n"
        info_text += f"Std: {np.std(data):.4f}\n"
        info_text += f"Min: {np.min(data):.4f}\n"
        info_text += f"Max: {np.max(data):.4f}\n"
        info_text += f"File: {Path(dataset['file_path']).name}\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        
    def _update_data_display(self):
        """Update the data visualization."""
        dataset_name = self.shared_data.get('current_dataset')
        if not dataset_name or dataset_name not in self.shared_data['datasets']:
            return
            
        dataset = self.shared_data['datasets'][dataset_name]
        data = dataset['data']
        display_type = self.display_var.get()
        
        # Clear previous plot
        self.ax.clear()
        
        if display_type == "time_series":
            self._plot_time_series(data, dataset_name)
        elif display_type == "histogram":
            self._plot_histogram(data, dataset_name)
        elif display_type == "autocorrelation":
            self._plot_autocorrelation(data, dataset_name)
        elif display_type == "statistics":
            self._plot_statistics(data, dataset_name)
            
        self.canvas.draw()
        
    def _plot_time_series(self, data: np.ndarray, title: str):
        """Plot time series data."""
        self.ax.plot(data, linewidth=0.8)
        self.ax.set_title(f"Time Series: {title}")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, alpha=0.3)
        
    def _plot_histogram(self, data: np.ndarray, title: str):
        """Plot histogram of data."""
        self.ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        self.ax.set_title(f"Histogram: {title}")
        self.ax.set_xlabel("Value")
        self.ax.set_ylabel("Frequency")
        self.ax.grid(True, alpha=0.3)
        
    def _plot_autocorrelation(self, data: np.ndarray, title: str):
        """Plot autocorrelation function."""
        from scipy import signal
        
        # Calculate autocorrelation
        autocorr = signal.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Plot only first 100 lags for clarity
        max_lag = min(100, len(autocorr))
        lags = np.arange(max_lag)
        
        self.ax.plot(lags, autocorr[:max_lag])
        self.ax.set_title(f"Autocorrelation: {title}")
        self.ax.set_xlabel("Lag")
        self.ax.set_ylabel("Autocorrelation")
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
    def _plot_statistics(self, data: np.ndarray, title: str):
        """Plot statistical summary."""
        # Create a text-based statistical summary
        stats_text = f"Statistical Summary: {title}\n\n"
        stats_text += f"Count: {len(data):,}\n"
        stats_text += f"Mean: {np.mean(data):.4f}\n"
        stats_text += f"Std: {np.std(data):.4f}\n"
        stats_text += f"Min: {np.min(data):.4f}\n"
        stats_text += f"25%: {np.percentile(data, 25):.4f}\n"
        stats_text += f"50%: {np.percentile(data, 50):.4f}\n"
        stats_text += f"75%: {np.percentile(data, 75):.4f}\n"
        stats_text += f"Max: {np.max(data):.4f}\n"
        stats_text += f"Skewness: {self._calculate_skewness(data):.4f}\n"
        stats_text += f"Kurtosis: {self._calculate_kurtosis(data):.4f}\n"
        
        self.ax.text(0.1, 0.9, stats_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title(f"Statistics: {title}")
        self.ax.axis('off')
        
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
        
    def _update_display(self, event=None):
        """Update the display when display type changes."""
        self._update_data_display()
        
    def _refresh_display(self):
        """Refresh the current display."""
        self._update_data_display()
        
    def _view_selected_dataset(self):
        """View the selected dataset."""
        selection = self.dataset_listbox.curselection()
        if selection:
            dataset_name = self.dataset_listbox.get(selection[0])
            self._select_dataset(dataset_name)
            
    def _remove_selected_dataset(self):
        """Remove the selected dataset."""
        selection = self.dataset_listbox.curselection()
        if selection:
            dataset_name = self.dataset_listbox.get(selection[0])
            
            if messagebox.askyesno("Confirm", f"Remove dataset '{dataset_name}'?"):
                # Remove from shared data
                if dataset_name in self.shared_data['datasets']:
                    del self.shared_data['datasets'][dataset_name]
                    
                # Clear current dataset if it was the removed one
                if self.shared_data.get('current_dataset') == dataset_name:
                    self.shared_data['current_dataset'] = None
                    
                # Update displays
                self._update_dataset_list()
                self._update_info_display()
                self._create_empty_plot()
                
    def _preprocess_selected_dataset(self):
        """Preprocess the selected dataset."""
        selection = self.dataset_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a dataset to preprocess.")
            return
            
        dataset_name = self.dataset_listbox.get(selection[0])
        if dataset_name not in self.shared_data['datasets']:
            return
            
        # Run preprocessing in background thread
        def preprocess_thread():
            try:
                self.message_queue.put({'type': 'status', 'text': f'Preprocessing {dataset_name}...'})
                self.message_queue.put({'type': 'progress_start'})
                
                dataset = self.shared_data['datasets'][dataset_name]
                data = dataset['data']
                
                # Apply preprocessing
                processed_data = preprocess_data(data)
                
                # Store processed data
                processed_name = f"{dataset_name}_processed"
                self.shared_data['datasets'][processed_name] = {
                    'data': processed_data,
                    'file_path': dataset['file_path'],
                    'name': processed_name,
                    'length': len(processed_data),
                    'loaded': True,
                    'processed': True
                }
                
                # Update GUI in main thread
                self.after(0, self._update_dataset_list)
                self.after(0, lambda: self._select_dataset(processed_name))
                self.after(0, lambda: self.message_queue.put({
                    'type': 'info', 
                    'text': f'Preprocessed {dataset_name} -> {processed_name}'
                }))
                
            except Exception as e:
                self.message_queue.put({'type': 'error', 'text': f'Preprocessing error: {e}'})
            finally:
                self.message_queue.put({'type': 'progress_stop'})
                self.message_queue.put({'type': 'status', 'text': 'Ready'})
                
        threading.Thread(target=preprocess_thread, daemon=True).start()
        
    def _load_existing_datasets(self):
        """Load existing datasets from the data folder."""
        # This will be called on initialization to load any existing datasets
        pass

"""
Synthetic Data GUI Frame

This module provides a GUI interface for generating synthetic time series data
with various long-range dependence properties and contamination types.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
from pathlib import Path
import threading
import pandas as pd
from typing import Dict, Any, List, Optional
import queue

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_processing.synthetic_generator import (
    PureSignalGenerator, DataContaminator, SyntheticDataGenerator
)


class SyntheticDataFrame(ttk.Frame):
    """Synthetic Data tab for generating and testing synthetic time series."""
    
    def __init__(self, parent, shared_data: Dict[str, Any], message_queue: queue.Queue):
        """Initialize the synthetic data frame."""
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
        
        # Right panel - Data display
        self._setup_data_panel()
        
    def _setup_controls_panel(self):
        """Setup the controls panel."""
        controls_frame = ttk.LabelFrame(self, text="Synthetic Data Controls", padding="10")
        controls_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        
        # Data generation parameters
        params_frame = ttk.LabelFrame(controls_frame, text="Generation Parameters", padding="5")
        params_frame.pack(fill="x", pady=(0, 10))
        
        # Data length
        length_frame = ttk.Frame(params_frame)
        length_frame.pack(fill="x", pady=2)
        ttk.Label(length_frame, text="Data Length:").pack(side="left")
        self.length_var = tk.StringVar(value="1000")
        ttk.Entry(length_frame, textvariable=self.length_var, width=10).pack(side="right")
        
        # Random seed
        seed_frame = ttk.Frame(params_frame)
        seed_frame.pack(fill="x", pady=2)
        ttk.Label(seed_frame, text="Random Seed:").pack(side="left")
        self.seed_var = tk.StringVar(value="42")
        ttk.Entry(seed_frame, textvariable=self.seed_var, width=10).pack(side="right")
        
        # Pure signal types
        signal_frame = ttk.LabelFrame(controls_frame, text="Pure Signal Types", padding="5")
        signal_frame.pack(fill="x", pady=(0, 10))
        
        self.signal_vars = {}
        signal_types = [
            ("fBm", "Fractional Brownian Motion"),
            ("fGn", "Fractional Gaussian Noise"),
            ("ARFIMA", "ARFIMA Process")
        ]
        
        for signal_type, description in signal_types:
            var = tk.BooleanVar()
            self.signal_vars[signal_type] = var
            cb = ttk.Checkbutton(signal_frame, text=description, variable=var)
            cb.pack(anchor="w", pady=1)
            
        # Signal parameters
        signal_params_frame = ttk.LabelFrame(controls_frame, text="Signal Parameters", padding="5")
        signal_params_frame.pack(fill="x", pady=(0, 10))
        
        # Hurst exponent
        hurst_frame = ttk.Frame(signal_params_frame)
        hurst_frame.pack(fill="x", pady=2)
        ttk.Label(hurst_frame, text="Hurst Exponent:").pack(side="left")
        self.hurst_var = tk.StringVar(value="0.7")
        ttk.Entry(hurst_frame, textvariable=self.hurst_var, width=10).pack(side="right")
        
        # ARFIMA parameters
        arfima_frame = ttk.Frame(signal_params_frame)
        arfima_frame.pack(fill="x", pady=2)
        ttk.Label(arfima_frame, text="ARFIMA d:").pack(side="left")
        self.arfima_d_var = tk.StringVar(value="0.3")
        ttk.Entry(arfima_frame, textvariable=self.arfima_d_var, width=10).pack(side="right")
        
        # Contamination types
        contamination_frame = ttk.LabelFrame(controls_frame, text="Contamination Types", padding="5")
        contamination_frame.pack(fill="x", pady=(0, 10))
        
        self.contamination_vars = {}
        contamination_types = [
            ("trend", "Polynomial Trend"),
            ("periodicity", "Periodicity"),
            ("outliers", "Outliers"),
            ("irregular", "Irregular Sampling"),
            ("heavy_tails", "Heavy Tails")
        ]
        
        for cont_type, description in contamination_types:
            var = tk.BooleanVar()
            self.contamination_vars[cont_type] = var
            cb = ttk.Checkbutton(contamination_frame, text=description, variable=var)
            cb.pack(anchor="w", pady=1)
            
        # Contamination parameters
        cont_params_frame = ttk.LabelFrame(controls_frame, text="Contamination Parameters", padding="5")
        cont_params_frame.pack(fill="x", pady=(0, 10))
        
        # Trend degree
        trend_frame = ttk.Frame(cont_params_frame)
        trend_frame.pack(fill="x", pady=2)
        ttk.Label(trend_frame, text="Trend Degree:").pack(side="left")
        self.trend_degree_var = tk.StringVar(value="2")
        ttk.Entry(trend_frame, textvariable=self.trend_degree_var, width=10).pack(side="right")
        
        # Outlier fraction
        outlier_frame = ttk.Frame(cont_params_frame)
        outlier_frame.pack(fill="x", pady=2)
        ttk.Label(outlier_frame, text="Outlier Fraction:").pack(side="left")
        self.outlier_fraction_var = tk.StringVar(value="0.05")
        ttk.Entry(outlier_frame, textvariable=self.outlier_fraction_var, width=10).pack(side="right")
        
        # Action buttons
        actions_frame = ttk.LabelFrame(controls_frame, text="Actions", padding="5")
        actions_frame.pack(fill="x")
        
        ttk.Button(actions_frame, text="Generate Data", 
                  command=self._generate_data).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Generate All Types", 
                  command=self._generate_all_types).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Save Generated Data", 
                  command=self._save_generated_data).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Clear Generated Data", 
                  command=self._clear_generated_data).pack(fill="x", pady=2)
        
    def _setup_data_panel(self):
        """Setup the data display panel."""
        data_frame = ttk.LabelFrame(self, text="Generated Data", padding="10")
        data_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        data_frame.grid_rowconfigure(1, weight=1)
        data_frame.grid_columnconfigure(0, weight=1)
        
        # Data controls
        data_controls = ttk.Frame(data_frame)
        data_controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(data_controls, text="Dataset:").pack(side="left")
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(data_controls, textvariable=self.dataset_var, 
                                         state="readonly", width=20)
        self.dataset_combo.pack(side="left", padx=(5, 0))
        self.dataset_combo.bind('<<ComboboxSelected>>', self._on_dataset_changed)
        
        ttk.Label(data_controls, text="Display:").pack(side="left", padx=(10, 0))
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
        self.ax.text(0.5, 0.5, 'No synthetic data generated\nGenerate data to view visualization', 
                    ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
    def _generate_data(self):
        """Generate synthetic data based on selected parameters."""
        # Get parameters
        try:
            length = int(self.length_var.get())
            seed = int(self.seed_var.get())
            hurst = float(self.hurst_var.get())
            arfima_d = float(self.arfima_d_var.get())
            trend_degree = int(self.trend_degree_var.get())
            outlier_fraction = float(self.outlier_fraction_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {e}")
            return
            
        # Check if any signal types are selected
        selected_signals = [signal for signal, var in self.signal_vars.items() if var.get()]
        if not selected_signals:
            messagebox.showwarning("Warning", "Please select at least one signal type.")
            return
            
        # Run generation in background thread
        def generation_thread():
            try:
                self.message_queue.put({'type': 'status', 'text': 'Generating synthetic data...'})
                self.message_queue.put({'type': 'progress_start'})
                
                # Initialize generators
                pure_generator = PureSignalGenerator(random_state=seed)
                contaminator = DataContaminator(random_state=seed)
                
                generated_data = {}
                
                for signal_type in selected_signals:
                    self.message_queue.put({'type': 'status', 'text': f'Generating {signal_type}...'})
                    
                    # Generate pure signal
                    if signal_type == "fBm":
                        pure_data = pure_generator.generate_fbm(length, hurst)
                        signal_name = f"fBm_H{hurst}"
                    elif signal_type == "fGn":
                        pure_data = pure_generator.generate_fgn(length, hurst)
                        signal_name = f"fGn_H{hurst}"
                    elif signal_type == "ARFIMA":
                        pure_data = pure_generator.generate_arfima(length, d=arfima_d)
                        signal_name = f"ARFIMA_d{arfima_d}"
                        
                    # Store pure signal
                    generated_data[signal_name] = {
                        'data': pure_data,
                        'type': signal_type,
                        'parameters': {'hurst': hurst, 'length': length},
                        'contaminated': False
                    }
                    
                    # Apply contamination if selected
                    selected_contaminations = [cont for cont, var in self.contamination_vars.items() if var.get()]
                    
                    for cont_type in selected_contaminations:
                        self.message_queue.put({'type': 'status', 'text': f'Applying {cont_type} contamination...'})
                        
                        if cont_type == "trend":
                            contaminated_data = contaminator.add_polynomial_trend(
                                pure_data, degree=trend_degree
                            )
                        elif cont_type == "periodicity":
                            contaminated_data = contaminator.add_periodicity(pure_data)
                        elif cont_type == "outliers":
                            contaminated_data = contaminator.add_outliers(
                                pure_data, fraction=outlier_fraction
                            )
                        elif cont_type == "irregular":
                            contaminated_data = contaminator.add_irregular_sampling(pure_data)
                        elif cont_type == "heavy_tails":
                            contaminated_data = contaminator.add_heavy_tails(pure_data)
                            
                        # Store contaminated signal
                        cont_name = f"{signal_name}_{cont_type}"
                        generated_data[cont_name] = {
                            'data': contaminated_data,
                            'type': signal_type,
                            'parameters': {'hurst': hurst, 'length': length},
                            'contaminated': True,
                            'contamination': cont_type
                        }
                        
                # Store in shared data
                self.shared_data['synthetic_data'].update(generated_data)
                
                # Update GUI in main thread
                self.after(0, self._update_dataset_list)
                if generated_data:
                    first_key = list(generated_data.keys())[0]
                    self.after(0, lambda: self._select_dataset(first_key))
                self.after(0, lambda: self.message_queue.put({
                    'type': 'info', 
                    'text': f'Generated {len(generated_data)} synthetic datasets'
                }))
                
            except Exception as e:
                self.message_queue.put({'type': 'error', 'text': f'Generation error: {e}'})
            finally:
                self.message_queue.put({'type': 'progress_stop'})
                self.message_queue.put({'type': 'status', 'text': 'Ready'})
                
        threading.Thread(target=generation_thread, daemon=True).start()
        
    def _generate_all_types(self):
        """Generate all types of synthetic data."""
        # Select all signal types
        for var in self.signal_vars.values():
            var.set(True)
            
        # Select all contamination types
        for var in self.contamination_vars.values():
            var.set(True)
            
        # Generate data
        self._generate_data()
        
    def _update_dataset_list(self):
        """Update the dataset dropdown."""
        datasets = list(self.shared_data['synthetic_data'].keys())
        self.dataset_combo['values'] = datasets
        
        if datasets and not self.dataset_var.get():
            self.dataset_var.set(datasets[0])
            
    def _on_dataset_changed(self, event=None):
        """Handle dataset selection change."""
        selected_dataset = self.dataset_var.get()
        if selected_dataset in self.shared_data['synthetic_data']:
            self._select_dataset(selected_dataset)
            
    def _select_dataset(self, dataset_name: str):
        """Select a dataset and update display."""
        if dataset_name in self.shared_data['synthetic_data']:
            self.dataset_var.set(dataset_name)
            self._update_data_display()
            
    def _update_data_display(self):
        """Update the data visualization."""
        dataset_name = self.dataset_var.get()
        if not dataset_name or dataset_name not in self.shared_data['synthetic_data']:
            return
            
        dataset = self.shared_data['synthetic_data'][dataset_name]
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
        
    def _save_generated_data(self):
        """Save generated synthetic data to files."""
        if not self.shared_data['synthetic_data']:
            messagebox.showwarning("Warning", "No synthetic data to save.")
            return
            
        # Ask for directory
        directory = filedialog.askdirectory(title="Select Directory to Save Synthetic Data")
        if not directory:
            return
            
        try:
            saved_count = 0
            for name, dataset in self.shared_data['synthetic_data'].items():
                file_path = Path(directory) / f"{name}.csv"
                df = pd.DataFrame(dataset['data'], columns=['value'])
                df.to_csv(file_path, index=False)
                saved_count += 1
                
            self.message_queue.put({'type': 'info', 'text': f'Saved {saved_count} datasets to {directory}'})
            
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f'Error saving data: {e}'})
            
    def _clear_generated_data(self):
        """Clear generated synthetic data."""
        if messagebox.askyesno("Confirm", "Clear all generated synthetic data?"):
            self.shared_data['synthetic_data'].clear()
            self._update_dataset_list()
            self._create_empty_plot()

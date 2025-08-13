"""
Results GUI Frame

This module provides a GUI interface for viewing and managing analysis results,
including visualization and export capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
from pathlib import Path
import json
import pandas as pd
from typing import Dict, Any, List, Optional
import queue

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class ResultsFrame(ttk.Frame):
    """Results tab for viewing and managing analysis results."""
    
    def __init__(self, parent, shared_data: Dict[str, Any], message_queue: queue.Queue):
        """Initialize the results frame."""
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
        controls_frame = ttk.LabelFrame(self, text="Results Controls", padding="10")
        controls_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        
        # Dataset selection
        dataset_frame = ttk.LabelFrame(controls_frame, text="Dataset Selection", padding="5")
        dataset_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(dataset_frame, text="Dataset:").pack(anchor="w")
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.dataset_var, 
                                         state="readonly", width=25)
        self.dataset_combo.pack(fill="x", pady=2)
        self.dataset_combo.bind('<<ComboboxSelected>>', self._on_dataset_changed)
        
        # Analysis method selection
        method_frame = ttk.LabelFrame(controls_frame, text="Analysis Method", padding="5")
        method_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(method_frame, text="Method:").pack(anchor="w")
        self.method_var = tk.StringVar()
        self.method_combo = ttk.Combobox(method_frame, textvariable=self.method_var, 
                                        state="readonly", width=25)
        self.method_combo.pack(fill="x", pady=2)
        self.method_combo.bind('<<ComboboxSelected>>', self._on_method_changed)
        
        # Display options
        display_frame = ttk.LabelFrame(controls_frame, text="Display Options", padding="5")
        display_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(display_frame, text="Display:").pack(anchor="w")
        self.display_var = tk.StringVar(value="summary")
        display_combo = ttk.Combobox(display_frame, textvariable=self.display_var, 
                                   values=["summary", "plots", "detailed", "comparison"],
                                   state="readonly", width=25)
        display_combo.pack(fill="x", pady=2)
        display_combo.bind('<<ComboboxSelected>>', self._update_display)
        
        # Results summary
        summary_frame = ttk.LabelFrame(controls_frame, text="Results Summary", padding="5")
        summary_frame.pack(fill="x", pady=(0, 10))
        
        self.summary_text = tk.Text(summary_frame, height=8, width=30, font=("Courier", 8))
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.summary_text.pack(side="left", fill="both", expand=True)
        summary_scrollbar.pack(side="right", fill="y")
        
        # Action buttons
        actions_frame = ttk.LabelFrame(controls_frame, text="Actions", padding="5")
        actions_frame.pack(fill="x")
        
        ttk.Button(actions_frame, text="Refresh Results", 
                  command=self._refresh_results).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Export Results", 
                  command=self._export_results).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Save Plot", 
                  command=self._save_plot).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Clear Results", 
                  command=self._clear_results).pack(fill="x", pady=2)
        
    def _setup_results_panel(self):
        """Setup the results display panel."""
        results_frame = ttk.LabelFrame(self, text="Results Visualization", padding="10")
        results_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        results_frame.grid_rowconfigure(1, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Results controls
        results_controls = ttk.Frame(results_frame)
        results_controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Button(results_controls, text="Refresh", 
                  command=self._refresh_display).pack(side="right")
        
        # Plot area
        self.plot_frame = ttk.Frame(results_frame)
        self.plot_frame.grid(row=1, column=0, sticky="nsew")
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        
        # Initialize empty plot
        self._create_empty_plot()
        
    def _create_empty_plot(self):
        """Create an empty plot area."""
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.text(0.5, 0.5, 'No results to display\nRun analysis to view results', 
                    ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
    def _update_dataset_list(self):
        """Update the dataset dropdown."""
        datasets = list(self.shared_data['analysis_results'].keys())
        self.dataset_combo['values'] = datasets
        
        if datasets and not self.dataset_var.get():
            self.dataset_var.set(datasets[0])
            
    def _update_method_list(self):
        """Update the method dropdown."""
        dataset_name = self.dataset_var.get()
        if dataset_name and dataset_name in self.shared_data['analysis_results']:
            results = self.shared_data['analysis_results'][dataset_name]
            methods = list(results.keys())
            self.method_combo['values'] = methods
            
            if methods and not self.method_var.get():
                self.method_var.set(methods[0])
        else:
            self.method_combo['values'] = []
            self.method_var.set("")
            
    def _on_dataset_changed(self, event=None):
        """Handle dataset selection change."""
        self._update_method_list()
        self._update_summary()
        self._update_display()
        
    def _on_method_changed(self, event=None):
        """Handle method selection change."""
        self._update_summary()
        self._update_display()
        
    def _update_summary(self):
        """Update the results summary."""
        self.summary_text.delete(1.0, tk.END)
        
        dataset_name = self.dataset_var.get()
        method_name = self.method_var.get()
        
        if not dataset_name or not method_name:
            return
            
        if dataset_name not in self.shared_data['analysis_results']:
            return
            
        results = self.shared_data['analysis_results'][dataset_name]
        if method_name not in results:
            return
            
        result = results[method_name]
        
        # Create summary text
        summary_text = f"Dataset: {dataset_name}\n"
        summary_text += f"Method: {method_name}\n"
        summary_text += f"Status: {'Success' if 'error' not in result else 'Error'}\n\n"
        
        if 'error' in result:
            summary_text += f"Error: {result['error']}\n"
        else:
            if 'hurst' in result:
                summary_text += f"Hurst: {result['hurst']:.4f}\n"
            if 'alpha' in result:
                summary_text += f"Alpha: {result['alpha']:.4f}\n"
            if 'parameters' in result:
                summary_text += f"Parameters: {result['parameters']}\n"
                
        self.summary_text.insert(tk.END, summary_text)
        
    def _update_display(self, event=None):
        """Update the results display."""
        dataset_name = self.dataset_var.get()
        method_name = self.method_var.get()
        display_type = self.display_var.get()
        
        if not dataset_name or not method_name:
            return
            
        if dataset_name not in self.shared_data['analysis_results']:
            return
            
        results = self.shared_data['analysis_results'][dataset_name]
        if method_name not in results:
            return
            
        result = results[method_name]
        
        # Clear previous display
        self.ax.clear()
        
        if display_type == "summary":
            self._display_summary(result, dataset_name, method_name)
        elif display_type == "plots":
            self._display_plots(result, dataset_name, method_name)
        elif display_type == "detailed":
            self._display_detailed(result, dataset_name, method_name)
        elif display_type == "comparison":
            self._display_comparison(dataset_name)
            
        self.canvas.draw()
        
    def _display_summary(self, result: Dict[str, Any], dataset_name: str, method_name: str):
        """Display results summary."""
        summary_text = f"Results Summary\n"
        summary_text += f"Dataset: {dataset_name}\n"
        summary_text += f"Method: {method_name}\n\n"
        
        if 'error' in result:
            summary_text += f"Status: Error\n"
            summary_text += f"Error: {result['error']}\n"
        else:
            summary_text += f"Status: Success\n"
            if 'hurst' in result:
                summary_text += f"Hurst Exponent: {result['hurst']:.4f}\n"
            if 'alpha' in result:
                summary_text += f"Alpha Parameter: {result['alpha']:.4f}\n"
            if 'confidence_intervals' in result:
                summary_text += f"Confidence Intervals: {result['confidence_intervals']}\n"
            if 'quality_metrics' in result:
                summary_text += f"Quality Metrics: {result['quality_metrics']}\n"
                
        self.ax.text(0.1, 0.9, summary_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title(f"Summary: {method_name} - {dataset_name}")
        self.ax.axis('off')
        
    def _display_plots(self, result: Dict[str, Any], dataset_name: str, method_name: str):
        """Display results plots."""
        if 'error' in result:
            self.ax.text(0.5, 0.5, f"Error: {result['error']}", 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.ax.set_title(f"Error: {method_name} - {dataset_name}")
            return
            
        if method_name == "DFA" and 'scales' in result and 'fluctuations' in result:
            self.ax.loglog(result['scales'], result['fluctuations'], 'o-')
            self.ax.set_title(f"DFA Analysis: {dataset_name}")
            self.ax.set_xlabel("Scale")
            self.ax.set_ylabel("Fluctuation")
            self.ax.grid(True, alpha=0.3)
            
        elif method_name == "RS" and 'lags' in result and 'rs_values' in result:
            self.ax.loglog(result['lags'], result['rs_values'], 'o-')
            self.ax.set_title(f"R/S Analysis: {dataset_name}")
            self.ax.set_xlabel("Lag")
            self.ax.set_ylabel("R/S")
            self.ax.grid(True, alpha=0.3)
            
        elif method_name == "Spectral" and 'frequencies' in result and 'power_spectrum' in result:
            self.ax.loglog(result['frequencies'], result['power_spectrum'])
            self.ax.set_title(f"Spectral Analysis: {dataset_name}")
            self.ax.set_xlabel("Frequency")
            self.ax.set_ylabel("Power Spectrum")
            self.ax.grid(True, alpha=0.3)
            
        elif method_name == "ARFIMA":
            # Display ARFIMA results as text
            arfima_text = f"ARFIMA Results: {dataset_name}\n\n"
            if 'hurst' in result:
                arfima_text += f"Hurst: {result['hurst']:.4f}\n"
            if 'alpha' in result:
                arfima_text += f"Alpha: {result['alpha']:.4f}\n"
            if 'confidence_intervals' in result:
                arfima_text += f"Confidence Intervals: {result['confidence_intervals']}\n"
                
            self.ax.text(0.1, 0.9, arfima_text, transform=self.ax.transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.set_title(f"ARFIMA: {dataset_name}")
            self.ax.axis('off')
            
        else:
            # Generic display for other methods
            self.ax.text(0.5, 0.5, f"No specific plot for {method_name}\nCheck detailed view for results", 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.ax.set_title(f"{method_name}: {dataset_name}")
            
    def _display_detailed(self, result: Dict[str, Any], dataset_name: str, method_name: str):
        """Display detailed results."""
        detailed_text = f"Detailed Results\n"
        detailed_text += f"Dataset: {dataset_name}\n"
        detailed_text += f"Method: {method_name}\n"
        detailed_text += "=" * 50 + "\n\n"
        
        for key, value in result.items():
            if key != 'method':
                if isinstance(value, dict):
                    detailed_text += f"{key}:\n"
                    for k, v in value.items():
                        detailed_text += f"  {k}: {v}\n"
                elif isinstance(value, list):
                    detailed_text += f"{key}: [{len(value)} values]\n"
                    if len(value) <= 10:
                        detailed_text += f"  Values: {value}\n"
                else:
                    detailed_text += f"{key}: {value}\n"
            detailed_text += "\n"
            
        self.ax.text(0.02, 0.98, detailed_text, transform=self.ax.transAxes, 
                    fontsize=8, verticalalignment='top', fontfamily='monospace')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title(f"Detailed: {method_name} - {dataset_name}")
        self.ax.axis('off')
        
    def _display_comparison(self, dataset_name: str):
        """Display comparison of different methods."""
        if dataset_name not in self.shared_data['analysis_results']:
            return
            
        results = self.shared_data['analysis_results'][dataset_name]
        
        # Extract Hurst exponents for comparison
        hurst_values = {}
        for method, result in results.items():
            if 'error' not in result and 'hurst' in result:
                hurst_values[method] = result['hurst']
                
        if not hurst_values:
            self.ax.text(0.5, 0.5, "No Hurst exponents available for comparison", 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.ax.set_title(f"Comparison: {dataset_name}")
            return
            
        # Create bar plot
        methods = list(hurst_values.keys())
        values = list(hurst_values.values())
        
        bars = self.ax.bar(methods, values, alpha=0.7)
        self.ax.set_title(f"Method Comparison: {dataset_name}")
        self.ax.set_ylabel("Hurst Exponent")
        self.ax.set_xlabel("Method")
        self.ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
        # Add reference line at H=0.5 (no long-range dependence)
        self.ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='H=0.5 (No LRD)')
        self.ax.legend()
        
    def _refresh_display(self):
        """Refresh the current display."""
        self._update_display()
        
    def _refresh_results(self):
        """Refresh all results displays."""
        self._update_dataset_list()
        self._update_method_list()
        self._update_summary()
        self._update_display()
        
    def _export_results(self):
        """Export results to file."""
        if not self.shared_data['analysis_results']:
            messagebox.showwarning("Warning", "No results to export.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(self.shared_data['analysis_results'], f, indent=2)
                elif file_path.endswith('.csv'):
                    # Create a summary CSV
                    summary_data = []
                    for dataset_name, results in self.shared_data['analysis_results'].items():
                        for method_name, result in results.items():
                            row = {
                                'dataset': dataset_name,
                                'method': method_name,
                                'status': 'error' if 'error' in result else 'success'
                            }
                            if 'error' not in result:
                                if 'hurst' in result:
                                    row['hurst'] = result['hurst']
                                if 'alpha' in result:
                                    row['alpha'] = result['alpha']
                            summary_data.append(row)
                    
                    df = pd.DataFrame(summary_data)
                    df.to_csv(file_path, index=False)
                    
                self.message_queue.put({'type': 'info', 'text': f'Results exported to {file_path}'})
                
            except Exception as e:
                self.message_queue.put({'type': 'error', 'text': f'Error exporting results: {e}'})
                
    def _save_plot(self):
        """Save the current plot."""
        file_path = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.message_queue.put({'type': 'info', 'text': f'Plot saved to {file_path}'})
            except Exception as e:
                self.message_queue.put({'type': 'error', 'text': f'Error saving plot: {e}'})
                
    def _clear_results(self):
        """Clear analysis results."""
        if messagebox.askyesno("Confirm", "Clear all analysis results?"):
            self.shared_data['analysis_results'].clear()
            self._update_dataset_list()
            self._update_method_list()
            self._update_summary()
            self._create_empty_plot()
            
    def refresh_results(self):
        """Public method to refresh results."""
        self._refresh_results()
        
    def save_results_to_file(self, file_path: str):
        """Public method to save results to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.shared_data['analysis_results'], f, indent=2)
            self.message_queue.put({'type': 'info', 'text': f'Results saved to {file_path}'})
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f'Error saving results: {e}'})

"""
Analysis GUI Frame

This module provides a GUI interface for running long-range dependence
analysis methods on datasets.
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
import json
from typing import Dict, Any, List, Optional
import queue

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from analysis.dfa_analysis import DFAModel, dfa, hurst_from_dfa_alpha
from analysis.rs_analysis import RSModel, rs_analysis, d_from_hurst_rs
from analysis.mfdfa_analysis import MFDFAModel, mfdfa, hurst_from_mfdfa
from analysis.wavelet_analysis import WaveletModel, wavelet_leaders_estimation
from analysis.spectral_analysis import SpectralModel, whittle_mle, periodogram_estimation
from analysis.arfima_modelling import ARFIMAModel, arfima_simulation
from visualisation.fractal_plots import plot_dfa_analysis, plot_rs_analysis, plot_mfdfa_analysis
from visualisation.results_visualisation import plot_wavelet_analysis, plot_spectral_analysis


class AnalysisFrame(ttk.Frame):
    """Analysis tab for running LRD analysis methods."""
    
    def __init__(self, parent, shared_data: Dict[str, Any], message_queue: queue.Queue):
        """Initialize the analysis frame."""
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
        controls_frame = ttk.LabelFrame(self, text="Analysis Controls", padding="10")
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
        
        # Analysis methods
        methods_frame = ttk.LabelFrame(controls_frame, text="Analysis Methods", padding="5")
        methods_frame.pack(fill="x", pady=(0, 10))
        
        self.method_vars = {}
        methods = [
            ("DFA", "Detrended Fluctuation Analysis"),
            ("RS", "R/S Analysis"),
            ("MFDFA", "Multifractal DFA"),
            ("Wavelet", "Wavelet Analysis"),
            ("Spectral", "Spectral Analysis"),
            ("ARFIMA", "ARFIMA Modelling")
        ]
        
        for method, description in methods:
            var = tk.BooleanVar()
            self.method_vars[method] = var
            cb = ttk.Checkbutton(methods_frame, text=description, variable=var)
            cb.pack(anchor="w", pady=1)
            
        # Analysis parameters
        params_frame = ttk.LabelFrame(controls_frame, text="Parameters", padding="5")
        params_frame.pack(fill="x", pady=(0, 10))
        
        # DFA parameters
        dfa_frame = ttk.Frame(params_frame)
        dfa_frame.pack(fill="x", pady=2)
        ttk.Label(dfa_frame, text="DFA min scale:").pack(side="left")
        self.dfa_min_scale = tk.StringVar(value="10")
        ttk.Entry(dfa_frame, textvariable=self.dfa_min_scale, width=8).pack(side="right")
        
        dfa_max_frame = ttk.Frame(params_frame)
        dfa_max_frame.pack(fill="x", pady=2)
        ttk.Label(dfa_max_frame, text="DFA max scale:").pack(side="left")
        self.dfa_max_scale = tk.StringVar(value="100")
        ttk.Entry(dfa_max_frame, textvariable=self.dfa_max_scale, width=8).pack(side="right")
        
        # R/S parameters
        rs_frame = ttk.Frame(params_frame)
        rs_frame.pack(fill="x", pady=2)
        ttk.Label(rs_frame, text="R/S min lag:").pack(side="left")
        self.rs_min_lag = tk.StringVar(value="10")
        ttk.Entry(rs_frame, textvariable=self.rs_min_lag, width=8).pack(side="right")
        
        rs_max_frame = ttk.Frame(params_frame)
        rs_max_frame.pack(fill="x", pady=2)
        ttk.Label(rs_max_frame, text="R/S max lag:").pack(side="left")
        self.rs_max_lag = tk.StringVar(value="100")
        ttk.Entry(rs_max_frame, textvariable=self.rs_max_lag, width=8).pack(side="right")
        
        # Action buttons
        actions_frame = ttk.LabelFrame(controls_frame, text="Actions", padding="5")
        actions_frame.pack(fill="x")
        
        ttk.Button(actions_frame, text="Run Selected Analysis", 
                  command=self._run_selected_analysis).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Run Full Analysis", 
                  command=self._run_full_analysis).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Clear Results", 
                  command=self._clear_results).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Save Results", 
                  command=self._save_results).pack(fill="x", pady=2)
        
    def _setup_results_panel(self):
        """Setup the results display panel."""
        results_frame = ttk.LabelFrame(self, text="Analysis Results", padding="10")
        results_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        results_frame.grid_rowconfigure(1, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Results controls
        results_controls = ttk.Frame(results_frame)
        results_controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(results_controls, text="Display:").pack(side="left")
        self.display_var = tk.StringVar(value="summary")
        display_combo = ttk.Combobox(results_controls, textvariable=self.display_var, 
                                   values=["summary", "plots", "detailed"],
                                   state="readonly", width=15)
        display_combo.pack(side="left", padx=(5, 0))
        display_combo.bind('<<ComboboxSelected>>', self._update_display)
        
        ttk.Button(results_controls, text="Refresh", 
                  command=self._refresh_display).pack(side="right")
        
        # Results area
        self.results_frame = ttk.Frame(results_frame)
        self.results_frame.grid(row=1, column=0, sticky="nsew")
        self.results_frame.grid_rowconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        # Initialize empty results
        self._create_empty_results()
        
    def _create_empty_results(self):
        """Create an empty results display."""
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.text(0.5, 0.5, 'No analysis results\nRun analysis to view results', 
                    ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.results_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
    def _update_dataset_list(self):
        """Update the dataset dropdown."""
        datasets = list(self.shared_data['datasets'].keys())
        self.dataset_combo['values'] = datasets
        
        if datasets and not self.dataset_var.get():
            self.dataset_var.set(datasets[0])
            
    def _on_dataset_changed(self, event=None):
        """Handle dataset selection change."""
        # Update current dataset in shared data
        selected_dataset = self.dataset_var.get()
        if selected_dataset in self.shared_data['datasets']:
            self.shared_data['current_dataset'] = selected_dataset
            
    def _run_selected_analysis(self):
        """Run selected analysis methods."""
        # Check if dataset is selected
        dataset_name = self.dataset_var.get()
        if not dataset_name or dataset_name not in self.shared_data['datasets']:
            messagebox.showwarning("Warning", "Please select a dataset first.")
            return
            
        # Check if any methods are selected
        selected_methods = [method for method, var in self.method_vars.items() if var.get()]
        if not selected_methods:
            messagebox.showwarning("Warning", "Please select at least one analysis method.")
            return
            
        # Run analysis in background thread
        def analysis_thread():
            try:
                self.message_queue.put({'type': 'status', 'text': 'Running analysis...'})
                self.message_queue.put({'type': 'progress_start'})
                
                dataset = self.shared_data['datasets'][dataset_name]
                data = dataset['data']
                
                results = {}
                
                for method in selected_methods:
                    self.message_queue.put({'type': 'status', 'text': f'Running {method}...'})
                    
                    if method == "DFA":
                        results[method] = self._run_dfa_analysis(data)
                    elif method == "RS":
                        results[method] = self._run_rs_analysis(data)
                    elif method == "MFDFA":
                        results[method] = self._run_mfdfa_analysis(data)
                    elif method == "Wavelet":
                        results[method] = self._run_wavelet_analysis(data)
                    elif method == "Spectral":
                        results[method] = self._run_spectral_analysis(data)
                    elif method == "ARFIMA":
                        results[method] = self._run_arfima_analysis(data)
                        
                # Store results
                self.shared_data['analysis_results'][dataset_name] = results
                
                # Update display in main thread
                self.after(0, self._update_results_display)
                self.after(0, lambda: self.message_queue.put({
                    'type': 'info', 
                    'text': f'Analysis completed for {dataset_name}'
                }))
                
            except Exception as e:
                self.message_queue.put({'type': 'error', 'text': f'Analysis error: {e}'})
            finally:
                self.message_queue.put({'type': 'progress_stop'})
                self.message_queue.put({'type': 'status', 'text': 'Ready'})
                
        threading.Thread(target=analysis_thread, daemon=True).start()
        
    def _run_full_analysis(self):
        """Run full analysis pipeline."""
        # Select all methods
        for var in self.method_vars.values():
            var.set(True)
            
        # Run selected analysis
        self._run_selected_analysis()
        
    def _run_dfa_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Run DFA analysis."""
        try:
            min_scale = int(self.dfa_min_scale.get())
            max_scale = int(self.dfa_max_scale.get())
            
            # Run DFA
            scales, fluctuations = dfa(data, min_scale=min_scale, max_scale=max_scale)
            
            # Calculate Hurst exponent
            hurst = hurst_from_dfa_alpha(scales, fluctuations)
            
            return {
                'method': 'DFA',
                'scales': scales.tolist(),
                'fluctuations': fluctuations.tolist(),
                'hurst': float(hurst),
                'parameters': {
                    'min_scale': min_scale,
                    'max_scale': max_scale
                }
            }
        except Exception as e:
            return {'method': 'DFA', 'error': str(e)}
            
    def _run_rs_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Run R/S analysis."""
        try:
            min_lag = int(self.rs_min_lag.get())
            max_lag = int(self.rs_max_lag.get())
            
            # Run R/S analysis
            lags, rs_values = rs_analysis(data, min_lag=min_lag, max_lag=max_lag)
            
            # Calculate Hurst exponent
            hurst = d_from_hurst_rs(lags, rs_values)
            
            return {
                'method': 'RS',
                'lags': lags.tolist(),
                'rs_values': rs_values.tolist(),
                'hurst': float(hurst),
                'parameters': {
                    'min_lag': min_lag,
                    'max_lag': max_lag
                }
            }
        except Exception as e:
            return {'method': 'RS', 'error': str(e)}
            
    def _run_mfdfa_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Run MFDFA analysis."""
        try:
            # Run MFDFA
            scales, fluctuations, q_values = mfdfa(data)
            
            # Calculate Hurst exponent for q=2
            hurst = hurst_from_mfdfa(scales, fluctuations, q_values, q=2)
            
            return {
                'method': 'MFDFA',
                'scales': scales.tolist(),
                'fluctuations': fluctuations.tolist(),
                'q_values': q_values.tolist(),
                'hurst': float(hurst),
                'parameters': {}
            }
        except Exception as e:
            return {'method': 'MFDFA', 'error': str(e)}
            
    def _run_wavelet_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Run wavelet analysis."""
        try:
            # Run wavelet analysis
            scales, coefficients = wavelet_leaders_estimation(data)
            
            return {
                'method': 'Wavelet',
                'scales': scales.tolist(),
                'coefficients': coefficients.tolist(),
                'parameters': {}
            }
        except Exception as e:
            return {'method': 'Wavelet', 'error': str(e)}
            
    def _run_spectral_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Run spectral analysis."""
        try:
            # Run spectral analysis
            frequencies, power_spectrum = periodogram_estimation(data)
            
            return {
                'method': 'Spectral',
                'frequencies': frequencies.tolist(),
                'power_spectrum': power_spectrum.tolist(),
                'parameters': {}
            }
        except Exception as e:
            return {'method': 'Spectral', 'error': str(e)}
            
    def _run_arfima_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Run ARFIMA analysis."""
        try:
            # Run ARFIMA analysis
            model = ARFIMAModel()
            model.fit(data)
            
            hurst = model.estimate_hurst()
            alpha = model.estimate_alpha()
            confidence_intervals = model.get_confidence_intervals()
            
            return {
                'method': 'ARFIMA',
                'hurst': float(hurst),
                'alpha': float(alpha),
                'confidence_intervals': confidence_intervals,
                'parameters': {}
            }
        except Exception as e:
            return {'method': 'ARFIMA', 'error': str(e)}
            
    def _update_results_display(self):
        """Update the results display."""
        dataset_name = self.shared_data.get('current_dataset')
        if not dataset_name or dataset_name not in self.shared_data['analysis_results']:
            return
            
        results = self.shared_data['analysis_results'][dataset_name]
        display_type = self.display_var.get()
        
        # Clear previous display
        self.ax.clear()
        
        if display_type == "summary":
            self._display_summary(results, dataset_name)
        elif display_type == "plots":
            self._display_plots(results, dataset_name)
        elif display_type == "detailed":
            self._display_detailed(results, dataset_name)
            
        self.canvas.draw()
        
    def _display_summary(self, results: Dict[str, Any], dataset_name: str):
        """Display analysis summary."""
        summary_text = f"Analysis Summary: {dataset_name}\n\n"
        
        for method, result in results.items():
            summary_text += f"{method}:\n"
            if 'error' in result:
                summary_text += f"  Error: {result['error']}\n"
            else:
                if 'hurst' in result:
                    summary_text += f"  Hurst: {result['hurst']:.4f}\n"
                if 'alpha' in result:
                    summary_text += f"  Alpha: {result['alpha']:.4f}\n"
            summary_text += "\n"
            
        self.ax.text(0.1, 0.9, summary_text, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title(f"Summary: {dataset_name}")
        self.ax.axis('off')
        
    def _display_plots(self, results: Dict[str, Any], dataset_name: str):
        """Display analysis plots."""
        n_methods = len(results)
        if n_methods == 0:
            return
            
        # Create subplots
        self.fig.clear()
        
        if n_methods <= 2:
            cols = n_methods
            rows = 1
        elif n_methods <= 4:
            cols = 2
            rows = 2
        else:
            cols = 3
            rows = (n_methods + 2) // 3
            
        axes = []
        for i, (method, result) in enumerate(results.items()):
            if 'error' in result:
                continue
                
            ax = self.fig.add_subplot(rows, cols, i + 1)
            axes.append(ax)
            
            if method == "DFA" and 'scales' in result and 'fluctuations' in result:
                ax.loglog(result['scales'], result['fluctuations'], 'o-')
                ax.set_title(f"DFA: H={result.get('hurst', 'N/A'):.3f}")
                ax.set_xlabel("Scale")
                ax.set_ylabel("Fluctuation")
                ax.grid(True, alpha=0.3)
                
            elif method == "RS" and 'lags' in result and 'rs_values' in result:
                ax.loglog(result['lags'], result['rs_values'], 'o-')
                ax.set_title(f"R/S: H={result.get('hurst', 'N/A'):.3f}")
                ax.set_xlabel("Lag")
                ax.set_ylabel("R/S")
                ax.grid(True, alpha=0.3)
                
            elif method == "Spectral" and 'frequencies' in result and 'power_spectrum' in result:
                ax.loglog(result['frequencies'], result['power_spectrum'])
                ax.set_title("Spectral Analysis")
                ax.set_xlabel("Frequency")
                ax.set_ylabel("Power Spectrum")
                ax.grid(True, alpha=0.3)
                
        self.fig.suptitle(f"Analysis Results: {dataset_name}")
        self.fig.tight_layout()
        
    def _display_detailed(self, results: Dict[str, Any], dataset_name: str):
        """Display detailed results."""
        detailed_text = f"Detailed Results: {dataset_name}\n\n"
        
        for method, result in results.items():
            detailed_text += f"{'='*50}\n"
            detailed_text += f"Method: {method}\n"
            detailed_text += f"{'='*50}\n"
            
            if 'error' in result:
                detailed_text += f"Error: {result['error']}\n\n"
            else:
                for key, value in result.items():
                    if key != 'method':
                        if isinstance(value, dict):
                            detailed_text += f"{key}:\n"
                            for k, v in value.items():
                                detailed_text += f"  {k}: {v}\n"
                        elif isinstance(value, list):
                            detailed_text += f"{key}: [{len(value)} values]\n"
                        else:
                            detailed_text += f"{key}: {value}\n"
                detailed_text += "\n"
                
        self.ax.text(0.02, 0.98, detailed_text, transform=self.ax.transAxes, 
                    fontsize=8, verticalalignment='top', fontfamily='monospace')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title(f"Detailed: {dataset_name}")
        self.ax.axis('off')
        
    def _update_display(self, event=None):
        """Update the display when display type changes."""
        self._update_results_display()
        
    def _refresh_display(self):
        """Refresh the current display."""
        self._update_results_display()
        
    def _clear_results(self):
        """Clear analysis results."""
        if messagebox.askyesno("Confirm", "Clear all analysis results?"):
            self.shared_data['analysis_results'].clear()
            self._create_empty_results()
            
    def _save_results(self):
        """Save analysis results to file."""
        if not self.shared_data['analysis_results']:
            messagebox.showwarning("Warning", "No results to save.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.shared_data['analysis_results'], f, indent=2)
                self.message_queue.put({'type': 'info', 'text': f'Results saved to {file_path}'})
            except Exception as e:
                self.message_queue.put({'type': 'error', 'text': f'Error saving results: {e}'})

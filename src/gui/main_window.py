"""
Main Window for Long-Range Dependence Project GUI

This module provides the main application window with a modern tabbed interface
for accessing all project features.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
from pathlib import Path
import threading
import queue
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from .data_manager_gui import DataManagerFrame
from .analysis_gui import AnalysisFrame
from .synthetic_data_gui import SyntheticDataFrame
from .submission_gui import SubmissionFrame
from .results_gui import ResultsFrame
from .config_gui import ConfigFrame


class MainWindow:
    """Main application window with tabbed interface."""
    
    def __init__(self, root: Optional[tk.Tk] = None):
        """Initialize the main window."""
        if root is None:
            self.root = tk.Tk()
        else:
            self.root = root
            
        self.root.title("Long-Range Dependence Analysis Benchmark")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set theme
        try:
            from ttkthemes import ThemedTk
            style = ttk.Style()
            style.theme_use('clam')
        except ImportError:
            pass
        
        # Initialize shared data
        self.shared_data = {
            'datasets': {},
            'current_dataset': None,
            'analysis_results': {},
            'synthetic_data': {},
            'config': {}
        }
        
        # Message queue for thread-safe GUI updates
        self.message_queue = queue.Queue()
        
        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._setup_tabs()
        
        # Start message processing
        self._process_messages()
        
    def _setup_ui(self):
        """Setup the main UI components."""
        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Dataset", command=self._load_dataset)
        file_menu.add_command(label="Save Results", command=self._save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Full Analysis", command=self._run_full_analysis)
        analysis_menu.add_command(label="Generate Synthetic Data", command=self._generate_synthetic)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="View Results", command=self._view_results)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Submit Model", command=self._submit_model)
        tools_menu.add_command(label="Submit Dataset", command=self._submit_dataset)
        tools_menu.add_separator()
        tools_menu.add_command(label="Configuration", command=self._open_config)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._show_documentation)
        help_menu.add_command(label="About", command=self._show_about)
        
    def _setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.grid(row=2, column=0, sticky="ew", padx=5, pady=2)
        
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side="left")
        
        self.progress_bar = ttk.Progressbar(self.status_bar, mode='indeterminate')
        self.progress_bar.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
    def _setup_tabs(self):
        """Setup the tabbed interface."""
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        
        # Create tab frames
        self.data_frame = DataManagerFrame(self.notebook, self.shared_data, self.message_queue)
        self.analysis_frame = AnalysisFrame(self.notebook, self.shared_data, self.message_queue)
        self.synthetic_frame = SyntheticDataFrame(self.notebook, self.shared_data, self.message_queue)
        self.submission_frame = SubmissionFrame(self.notebook, self.shared_data, self.message_queue)
        self.results_frame = ResultsFrame(self.notebook, self.shared_data, self.message_queue)
        self.config_frame = ConfigFrame(self.notebook, self.shared_data, self.message_queue)
        
        # Add tabs to notebook
        self.notebook.add(self.data_frame, text="Data Manager")
        self.notebook.add(self.analysis_frame, text="Analysis")
        self.notebook.add(self.synthetic_frame, text="Synthetic Data")
        self.notebook.add(self.submission_frame, text="Submission")
        self.notebook.add(self.results_frame, text="Results")
        self.notebook.add(self.config_frame, text="Configuration")
        
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        
    def _on_tab_changed(self, event):
        """Handle tab change events."""
        current_tab = self.notebook.select()
        tab_id = self.notebook.index(current_tab)
        tab_names = ["Data Manager", "Analysis", "Synthetic Data", "Submission", "Results", "Configuration"]
        
        if 0 <= tab_id < len(tab_names):
            self.status_label.config(text=f"Current tab: {tab_names[tab_id]}")
            
    def _process_messages(self):
        """Process messages from background threads."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self._handle_message(message)
        except queue.Empty:
            pass
        finally:
            # Schedule next check
            self.root.after(100, self._process_messages)
            
    def _handle_message(self, message: Dict[str, Any]):
        """Handle messages from background threads."""
        msg_type = message.get('type')
        
        if msg_type == 'status':
            self.status_label.config(text=message.get('text', ''))
        elif msg_type == 'progress_start':
            self.progress_bar.start()
        elif msg_type == 'progress_stop':
            self.progress_bar.stop()
        elif msg_type == 'error':
            messagebox.showerror("Error", message.get('text', 'An error occurred'))
        elif msg_type == 'info':
            messagebox.showinfo("Information", message.get('text', ''))
        elif msg_type == 'warning':
            messagebox.showwarning("Warning", message.get('text', ''))
            
    def _load_dataset(self):
        """Load a dataset from file."""
        file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx;*.xls"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            # Switch to data manager tab
            self.notebook.select(0)
            # Trigger dataset loading in data manager
            self.data_frame.load_dataset_from_file(file_path)
            
    def _save_results(self):
        """Save analysis results."""
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            # Trigger results saving in results frame
            self.results_frame.save_results_to_file(file_path)
            
    def _run_full_analysis(self):
        """Run full analysis pipeline."""
        # Switch to analysis tab
        self.notebook.select(1)
        # Trigger full analysis
        self.analysis_frame.run_full_analysis()
        
    def _generate_synthetic(self):
        """Generate synthetic data."""
        # Switch to synthetic data tab
        self.notebook.select(2)
        # Trigger synthetic data generation
        self.synthetic_frame.generate_data()
        
    def _view_results(self):
        """View analysis results."""
        # Switch to results tab
        self.notebook.select(4)
        # Refresh results display
        self.results_frame.refresh_results()
        
    def _submit_model(self):
        """Submit a new model."""
        # Switch to submission tab
        self.notebook.select(3)
        # Trigger model submission
        self.submission_frame.submit_model()
        
    def _submit_dataset(self):
        """Submit a new dataset."""
        # Switch to submission tab
        self.notebook.select(3)
        # Trigger dataset submission
        self.submission_frame.submit_dataset()
        
    def _open_config(self):
        """Open configuration panel."""
        # Switch to config tab
        self.notebook.select(5)
        
    def _show_documentation(self):
        """Show documentation."""
        import webbrowser
        try:
            # Try to open local documentation
            docs_path = Path(__file__).parent.parent.parent / "docs" / "index.html"
            if docs_path.exists():
                webbrowser.open(f"file://{docs_path.absolute()}")
            else:
                # Fallback to README
                readme_path = Path(__file__).parent.parent.parent / "README.md"
                if readme_path.exists():
                    webbrowser.open(f"file://{readme_path.absolute()}")
                else:
                    messagebox.showinfo("Documentation", 
                                      "Documentation not found locally.\n"
                                      "Please check the project README.md file.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open documentation: {e}")
            
    def _show_about(self):
        """Show about dialog."""
        about_text = """Long-Range Dependence Analysis Benchmark

A comprehensive Python framework for analyzing long-range dependence 
in time series data, featuring advanced fractal analysis methods 
and synthetic data generation capabilities.

Version: 1.0.0
Author: Research Team

Features:
• Multiple Analysis Methods (DFA, R/S, MFDFA, Wavelet, Spectral)
• Synthetic Data Generation
• Statistical Validation
• Model and Dataset Submission System
• Comprehensive Visualization
• Modern GUI Interface

For more information, see the documentation."""
        
        messagebox.showinfo("About", about_text)
        
    def _on_closing(self):
        """Handle application closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()
            
    def run(self):
        """Run the GUI application."""
        # Set up closing handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Start the main loop
        self.root.mainloop()
        
    def update_status(self, text: str):
        """Update status bar text."""
        self.message_queue.put({'type': 'status', 'text': text})
        
    def show_error(self, text: str):
        """Show error message."""
        self.message_queue.put({'type': 'error', 'text': text})
        
    def show_info(self, text: str):
        """Show info message."""
        self.message_queue.put({'type': 'info', 'text': text})
        
    def start_progress(self):
        """Start progress bar."""
        self.message_queue.put({'type': 'progress_start'})
        
    def stop_progress(self):
        """Stop progress bar."""
        self.message_queue.put({'type': 'progress_stop'})


def main():
    """Main entry point for the GUI application."""
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()

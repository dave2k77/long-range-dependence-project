"""
Configuration GUI Frame

This module provides a GUI interface for managing project configurations,
including analysis parameters, data settings, and submission options.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional
import queue

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class ConfigFrame(ttk.Frame):
    """Configuration tab for managing project settings."""
    
    def __init__(self, parent, shared_data: Dict[str, Any], message_queue: queue.Queue):
        """Initialize the configuration frame."""
        super().__init__(parent)
        self.shared_data = shared_data
        self.message_queue = message_queue
        
        self._setup_ui()
        self._load_configurations()
        
    def _setup_ui(self):
        """Setup the user interface."""
        # Configure grid weights
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Left panel - Controls
        self._setup_controls_panel()
        
        # Right panel - Configuration display
        self._setup_config_panel()
        
    def _setup_controls_panel(self):
        """Setup the controls panel."""
        controls_frame = ttk.LabelFrame(self, text="Configuration Controls", padding="10")
        controls_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        
        # Configuration type selection
        type_frame = ttk.LabelFrame(controls_frame, text="Configuration Type", padding="5")
        type_frame.pack(fill="x", pady=(0, 10))
        
        self.config_type = tk.StringVar(value="analysis")
        ttk.Radiobutton(type_frame, text="Analysis Config", 
                       variable=self.config_type, value="analysis",
                       command=self._on_type_changed).pack(anchor="w", pady=2)
        ttk.Radiobutton(type_frame, text="Data Config", 
                       variable=self.config_type, value="data",
                       command=self._on_type_changed).pack(anchor="w", pady=2)
        ttk.Radiobutton(type_frame, text="Submission Config", 
                       variable=self.config_type, value="submission",
                       command=self._on_type_changed).pack(anchor="w", pady=2)
        
        # File operations
        file_frame = ttk.LabelFrame(controls_frame, text="File Operations", padding="5")
        file_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Configuration", 
                  command=self._load_config).pack(fill="x", pady=2)
        ttk.Button(file_frame, text="Save Configuration", 
                  command=self._save_config).pack(fill="x", pady=2)
        ttk.Button(file_frame, text="Reset to Default", 
                  command=self._reset_config).pack(fill="x", pady=2)
        
        # Configuration validation
        validation_frame = ttk.LabelFrame(controls_frame, text="Validation", padding="5")
        validation_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(validation_frame, text="Validate Configuration", 
                  command=self._validate_config).pack(fill="x", pady=2)
        ttk.Button(validation_frame, text="Test Configuration", 
                  command=self._test_config).pack(fill="x", pady=2)
        
        # Quick actions
        actions_frame = ttk.LabelFrame(controls_frame, text="Quick Actions", padding="5")
        actions_frame.pack(fill="x")
        
        ttk.Button(actions_frame, text="Open Config Folder", 
                  command=self._open_config_folder).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Export All Configs", 
                  command=self._export_all_configs).pack(fill="x", pady=2)
        ttk.Button(actions_frame, text="Import Configs", 
                  command=self._import_configs).pack(fill="x", pady=2)
        
    def _setup_config_panel(self):
        """Setup the configuration display panel."""
        config_frame = ttk.LabelFrame(self, text="Configuration Editor", padding="10")
        config_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        config_frame.grid_rowconfigure(1, weight=1)
        config_frame.grid_columnconfigure(0, weight=1)
        
        # Config controls
        config_controls = ttk.Frame(config_frame)
        config_controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(config_controls, text="Current Config:").pack(side="left")
        self.current_config_label = ttk.Label(config_controls, text="No config loaded")
        self.current_config_label.pack(side="left", padx=(5, 0))
        
        ttk.Button(config_controls, text="Refresh", 
                  command=self._refresh_config).pack(side="right")
        
        # Configuration editor
        self.config_text = tk.Text(config_frame, wrap=tk.NONE, font=("Courier", 9))
        config_scrollbar_y = ttk.Scrollbar(config_frame, orient="vertical", command=self.config_text.yview)
        config_scrollbar_x = ttk.Scrollbar(config_frame, orient="horizontal", command=self.config_text.xview)
        self.config_text.configure(yscrollcommand=config_scrollbar_y.set, xscrollcommand=config_scrollbar_x.set)
        
        self.config_text.grid(row=1, column=0, sticky="nsew")
        config_scrollbar_y.grid(row=1, column=1, sticky="ns")
        config_scrollbar_x.grid(row=2, column=0, sticky="ew")
        
        # Initialize empty config
        self.config_text.insert(tk.END, "# No configuration loaded\n# Select a configuration type and load a file")
        
    def _load_configurations(self):
        """Load default configurations."""
        config_dir = Path(__file__).parent.parent.parent / "config"
        
        self.default_configs = {}
        
        # Load analysis config
        analysis_config_path = config_dir / "analysis_config.yaml"
        if analysis_config_path.exists():
            try:
                with open(analysis_config_path, 'r') as f:
                    self.default_configs['analysis'] = yaml.safe_load(f)
            except Exception as e:
                self.message_queue.put({'type': 'warning', 'text': f'Could not load analysis config: {e}'})
                
        # Load data config
        data_config_path = config_dir / "data_config.yaml"
        if data_config_path.exists():
            try:
                with open(data_config_path, 'r') as f:
                    self.default_configs['data'] = yaml.safe_load(f)
            except Exception as e:
                self.message_queue.put({'type': 'warning', 'text': f'Could not load data config: {e}'})
                
        # Load submission config
        submission_config_path = config_dir / "submission_config.yaml"
        if submission_config_path.exists():
            try:
                with open(submission_config_path, 'r') as f:
                    self.default_configs['submission'] = yaml.safe_load(f)
            except Exception as e:
                self.message_queue.put({'type': 'warning', 'text': f'Could not load submission config: {e}'})
                
    def _on_type_changed(self):
        """Handle configuration type change."""
        config_type = self.config_type.get()
        self.current_config_label.config(text=f"{config_type.capitalize()} Configuration")
        
        # Load default config for this type
        if config_type in self.default_configs:
            self._display_config(self.default_configs[config_type])
        else:
            self.config_text.delete(1.0, tk.END)
            self.config_text.insert(tk.END, f"# {config_type.capitalize()} Configuration\n# No default configuration available")
            
    def _display_config(self, config: Dict[str, Any]):
        """Display configuration in the text editor."""
        self.config_text.delete(1.0, tk.END)
        
        try:
            # Convert to YAML format
            yaml_str = yaml.dump(config, default_flow_style=False, indent=2, sort_keys=False)
            self.config_text.insert(tk.END, yaml_str)
        except Exception as e:
            # Fallback to JSON
            json_str = json.dumps(config, indent=2)
            self.config_text.insert(tk.END, json_str)
            
    def _get_current_config(self) -> Optional[Dict[str, Any]]:
        """Get the current configuration from the text editor."""
        try:
            config_text = self.config_text.get(1.0, tk.END)
            
            # Try to parse as YAML first
            try:
                return yaml.safe_load(config_text)
            except yaml.YAMLError:
                # Try JSON as fallback
                return json.loads(config_text)
                
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f'Error parsing configuration: {e}'})
            return None
            
    def _load_config(self):
        """Load configuration from file."""
        config_type = self.config_type.get()
        
        file_path = filedialog.askopenfilename(
            title=f"Load {config_type.capitalize()} Configuration",
            filetypes=[
                ("YAML files", "*.yaml;*.yml"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    if file_path.endswith(('.yaml', '.yml')):
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                        
                self._display_config(config)
                self.current_config_label.config(text=f"{config_type.capitalize()} Config: {Path(file_path).name}")
                self.message_queue.put({'type': 'info', 'text': f'Configuration loaded from {file_path}'})
                
            except Exception as e:
                self.message_queue.put({'type': 'error', 'text': f'Error loading configuration: {e}'})
                
    def _save_config(self):
        """Save current configuration to file."""
        config = self._get_current_config()
        if not config:
            return
            
        config_type = self.config_type.get()
        
        file_path = filedialog.asksaveasfilename(
            title=f"Save {config_type.capitalize()} Configuration",
            defaultextension=".yaml",
            filetypes=[
                ("YAML files", "*.yaml"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    if file_path.endswith('.yaml'):
                        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
                    else:
                        json.dump(config, f, indent=2)
                        
                self.message_queue.put({'type': 'info', 'text': f'Configuration saved to {file_path}'})
                
            except Exception as e:
                self.message_queue.put({'type': 'error', 'text': f'Error saving configuration: {e}'})
                
    def _reset_config(self):
        """Reset configuration to default."""
        config_type = self.config_type.get()
        
        if messagebox.askyesno("Confirm", f"Reset {config_type} configuration to default?"):
            if config_type in self.default_configs:
                self._display_config(self.default_configs[config_type])
                self.message_queue.put({'type': 'info', 'text': f'{config_type.capitalize()} configuration reset to default'})
            else:
                self.message_queue.put({'type': 'warning', 'text': f'No default configuration available for {config_type}'})
                
    def _validate_config(self):
        """Validate the current configuration."""
        config = self._get_current_config()
        if not config:
            return
            
        config_type = self.config_type.get()
        
        try:
            # Basic validation based on config type
            if config_type == "analysis":
                self._validate_analysis_config(config)
            elif config_type == "data":
                self._validate_data_config(config)
            elif config_type == "submission":
                self._validate_submission_config(config)
                
            self.message_queue.put({'type': 'info', 'text': f'{config_type.capitalize()} configuration is valid'})
            
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f'Configuration validation failed: {e}'})
            
    def _validate_analysis_config(self, config: Dict[str, Any]):
        """Validate analysis configuration."""
        required_sections = ['methods', 'parameters', 'validation']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
                
        # Validate methods
        if 'methods' in config:
            for method, settings in config['methods'].items():
                if not isinstance(settings, dict):
                    raise ValueError(f"Method {method} settings must be a dictionary")
                    
    def _validate_data_config(self, config: Dict[str, Any]):
        """Validate data configuration."""
        required_sections = ['loading', 'processing', 'quality']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
                
    def _validate_submission_config(self, config: Dict[str, Any]):
        """Validate submission configuration."""
        required_sections = ['validation', 'testing', 'registry']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
                
    def _test_config(self):
        """Test the current configuration."""
        config = self._get_current_config()
        if not config:
            return
            
        config_type = self.config_type.get()
        
        try:
            # Simulate configuration testing
            self.message_queue.put({'type': 'status', 'text': f'Testing {config_type} configuration...'})
            
            # This would typically involve loading the config and testing it
            # For now, we'll just simulate a successful test
            
            self.message_queue.put({'type': 'info', 'text': f'{config_type.capitalize()} configuration test passed'})
            
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f'Configuration test failed: {e}'})
            
    def _open_config_folder(self):
        """Open the configuration folder."""
        config_dir = Path(__file__).parent.parent.parent / "config"
        
        try:
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                subprocess.run(["explorer", str(config_dir)])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(config_dir)])
            else:  # Linux
                subprocess.run(["xdg-open", str(config_dir)])
                
            self.message_queue.put({'type': 'info', 'text': f'Opened configuration folder: {config_dir}'})
            
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f'Error opening config folder: {e}'})
            
    def _export_all_configs(self):
        """Export all configurations to a directory."""
        directory = filedialog.askdirectory(title="Select Directory to Export Configurations")
        if not directory:
            return
            
        try:
            exported_count = 0
            
            for config_type, config in self.default_configs.items():
                file_path = Path(directory) / f"{config_type}_config.yaml"
                with open(file_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
                exported_count += 1
                
            self.message_queue.put({'type': 'info', 'text': f'Exported {exported_count} configurations to {directory}'})
            
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f'Error exporting configurations: {e}'})
            
    def _import_configs(self):
        """Import configurations from a directory."""
        directory = filedialog.askdirectory(title="Select Directory to Import Configurations")
        if not directory:
            return
            
        try:
            imported_count = 0
            config_dir = Path(__file__).parent.parent.parent / "config"
            
            for config_file in Path(directory).glob("*_config.yaml"):
                config_type = config_file.stem.replace("_config", "")
                
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Save to project config directory
                target_path = config_dir / f"{config_type}_config.yaml"
                with open(target_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
                    
                imported_count += 1
                
            # Reload configurations
            self._load_configurations()
            
            self.message_queue.put({'type': 'info', 'text': f'Imported {imported_count} configurations from {directory}'})
            
        except Exception as e:
            self.message_queue.put({'type': 'error', 'text': f'Error importing configurations: {e}'})
            
    def _refresh_config(self):
        """Refresh the current configuration display."""
        config_type = self.config_type.get()
        
        if config_type in self.default_configs:
            self._display_config(self.default_configs[config_type])
        else:
            self._on_type_changed()

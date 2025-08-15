# Import Issues Fix Guide

## Problem
When running scripts or notebooks from this project, you may encounter import errors like:
```
ModuleNotFoundError: No module named 'data_processing'
ImportError: cannot import name 'SyntheticDataGenerator' from 'data_processing.synthetic_generator'
```

## Root Cause
The issue occurs because Python can't find the `src` directory, which contains all the project modules. This happens when:
- Running scripts from different directories
- Running notebooks from different locations
- The `src` directory is not in the Python path

## Solutions

### Solution 1: Use the Import Helper (Recommended)

The easiest way is to use the built-in import helper:

```python
# At the top of your script
from src.import_helper import setup_project_imports
setup_project_imports()

# Now you can import normally
from data_processing.synthetic_generator import SyntheticDataGenerator
from analysis.higuchi_analysis import higuchi_fractal_dimension
from analysis.dfa_analysis import dfa
```

### Solution 2: Manual Path Setup

If you prefer to handle it manually:

```python
import sys
from pathlib import Path

# Find project root and add src to path
current_dir = Path.cwd()
project_root = None

for parent in [current_dir] + list(current_dir.parents):
    if (parent / "src").exists() and (parent / "src" / "data_processing").exists():
        project_root = parent
        break

if project_root:
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    print(f"✓ Added {src_path} to Python path")

# Now you can import
from data_processing.synthetic_generator import SyntheticDataGenerator
```

### Solution 3: Run from Project Root

Always run your scripts from the project root directory:
```bash
cd long-range-dependence-project
python your_script.py
```

## Working Examples

### Example 1: Basic Data Generation
```python
#!/usr/bin/env python3
from src.import_helper import setup_project_imports
setup_project_imports()

from data_processing.synthetic_generator import SyntheticDataGenerator

# Create generator and generate data
generator = SyntheticDataGenerator(random_state=42)
data = generator.generate_arfima(n=1000, d=0.3)
print(f"Generated {len(data)} data points")
```

### Example 2: Fractal Analysis
```python
#!/usr/bin/env python3
from src.import_helper import setup_project_imports
setup_project_imports()

from data_processing.synthetic_generator import SyntheticDataGenerator
from analysis.higuchi_analysis import higuchi_fractal_dimension
from analysis.dfa_analysis import dfa

# Generate data
generator = SyntheticDataGenerator(random_state=42)
data = generator.generate_arfima(n=1000, d=0.3)

# Perform analysis
k_values, l_values, higuchi_result = higuchi_fractal_dimension(data)
dfa_scales, dfa_flucts, dfa_result = dfa(data)

print(f"Higuchi fractal dimension: {higuchi_result.fractal_dimension:.3f}")
print(f"DFA alpha: {dfa_result.alpha:.3f}")
```

## Available Scripts

The project includes several working examples:

1. **`working_example.py`** - Basic functionality test
2. **`my_analysis_template.py`** - Template for your own analysis
3. **`fix_imports.py`** - Standalone import fixer

## Module Structure

The main modules available are:

- **`data_processing.synthetic_generator`** - Generate synthetic time series
- **`analysis.higuchi_analysis`** - Higuchi fractal dimension analysis
- **`analysis.dfa_analysis`** - Detrended Fluctuation Analysis
- **`analysis.rs_analysis`** - R/S analysis
- **`visualisation.*`** - Plotting and visualization tools

## Troubleshooting

### Still getting import errors?
1. Make sure you're running from the project root directory
2. Check that the `src` directory exists and contains the modules
3. Verify all dependencies are installed: `pip install -r requirements.txt`
4. Try the import helper first: `from src.import_helper import setup_project_imports`

### Module not found?
1. Check the exact module name in the `src` directory
2. Use the functional API (e.g., `higuchi_fractal_dimension`) not class-based API
3. Check the module's `__init__.py` file for available exports

### Permission errors?
1. Make sure you have read access to the project directory
2. Try running with elevated permissions if needed
3. Check file ownership and permissions

## Quick Test

To verify everything is working, run:
```bash
python working_example.py
```

This should generate synthetic data and perform basic analysis without any import errors.

## Need Help?

If you're still having issues:
1. Check the error message carefully
2. Verify your current working directory
3. Ensure all dependencies are installed
4. Try the import helper approach first

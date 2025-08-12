# Software Requirements and Installation Guide

This document provides comprehensive information about software requirements, dependencies, and installation procedures for the Long-Range Dependence Analysis project.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Python Requirements](#python-requirements)
3. [Core Dependencies](#core-dependencies)
4. [Optional Dependencies](#optional-dependencies)
5. [Installation Procedures](#installation-procedures)
6. [Environment Setup](#environment-setup)
7. [Testing Installation](#testing-installation)
8. [Troubleshooting](#troubleshooting)
9. [Performance Considerations](#performance-considerations)
10. [Version Compatibility](#version-compatibility)

## System Requirements

### 1.1 Operating System

#### Supported Operating Systems

- **Windows**: Windows 10/11 (64-bit)
- **macOS**: macOS 10.15 (Catalina) or later
- **Linux**: Ubuntu 18.04+, CentOS 7+, RHEL 7+

#### System Architecture

- **CPU**: 64-bit processor (x86_64 or ARM64)
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: Minimum 2GB free space
- **Network**: Internet connection for package installation

### 1.2 Hardware Requirements

#### Minimum Requirements

- **CPU**: Intel i3/AMD Ryzen 3 or equivalent
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Graphics**: Basic graphics support (for plotting)

#### Recommended Requirements

- **CPU**: Intel i5/AMD Ryzen 5 or equivalent
- **RAM**: 8GB or more
- **Storage**: 5GB free space
- **Graphics**: Dedicated graphics card (for large datasets)

#### High-Performance Requirements

- **CPU**: Intel i7/AMD Ryzen 7 or equivalent
- **RAM**: 16GB or more
- **Storage**: 10GB free space (SSD recommended)
- **Graphics**: Dedicated graphics card with 4GB+ VRAM

### 1.3 Development Environment

#### Code Editor/IDE

- **VS Code**: Recommended with Python extension
- **PyCharm**: Professional Python IDE
- **Jupyter Notebook**: For interactive analysis
- **Spyder**: Scientific Python IDE

#### Version Control

- **Git**: 2.20+ for source code management
- **GitHub/GitLab**: For remote repository hosting

## Python Requirements

### 2.1 Python Version

#### Required Version

- **Python**: 3.8 or later
- **Recommended**: Python 3.9 or 3.10

#### Version Compatibility

| Python Version | Status | Notes |
|----------------|--------|-------|
| 3.7 | ❌ Not Supported | End of life |
| 3.8 | ✅ Supported | Minimum version |
| 3.9 | ✅ Supported | Recommended |
| 3.10 | ✅ Supported | Recommended |
| 3.11 | ✅ Supported | Latest stable |
| 3.12 | ⚠️ Testing | May have compatibility issues |

### 2.2 Python Distribution

#### Recommended Distributions

1. **Anaconda**: Complete scientific Python distribution
2. **Miniconda**: Minimal conda distribution
3. **Python.org**: Official Python distribution
4. **Pyenv**: Python version manager

#### Anaconda vs. Miniconda

| Feature | Anaconda | Miniconda |
|---------|----------|-----------|
| Size | ~3GB | ~400MB |
| Packages | 1500+ pre-installed | Minimal |
| GUI | Anaconda Navigator | Command line only |
| Use Case | Data science beginners | Experienced users |

### 2.3 Virtual Environment

#### Environment Management

- **Conda**: Recommended for scientific computing
- **venv**: Python built-in virtual environment
- **pipenv**: Alternative dependency management
- **poetry**: Modern Python packaging tool

#### Conda Environment Setup

```bash
# Create new environment
conda create -n lrd_analysis python=3.9

# Activate environment
conda activate lrd_analysis

# Install packages
conda install numpy pandas scipy matplotlib
```

## Core Dependencies

### 3.1 Numerical Computing

#### NumPy

```bash
# Installation
pip install numpy>=1.21.0
conda install numpy>=1.21.0

# Purpose
# - Array operations and linear algebra
# - Random number generation
# - Mathematical functions
```

#### Pandas

```bash
# Installation
pip install pandas>=1.3.0
conda install pandas>=1.3.0

# Purpose
# - Data manipulation and analysis
# - Time series functionality
# - Data I/O operations
```

#### SciPy

```bash
# Installation
pip install scipy>=1.7.0
conda install scipy>=1.7.0

# Purpose
# - Scientific computing functions
# - Statistical functions
# - Optimization algorithms
```

### 3.2 Statistical Analysis

#### Statsmodels

```bash
# Installation
pip install statsmodels>=0.13.0
conda install statsmodels>=0.13.0

# Purpose
# - Time series analysis
# - ARFIMA modeling
# - Statistical tests
```

#### Scikit-learn

```bash
# Installation
pip install scikit-learn>=1.0.0
conda install scikit-learn>=1.0.0

# Purpose
# - Machine learning utilities
# - Data preprocessing
# - Model validation
```

### 3.3 Visualization

#### Matplotlib

```bash
# Installation
pip install matplotlib>=3.5.0
conda install matplotlib>=3.5.0

# Purpose
# - Basic plotting and visualization
# - Publication-quality figures
# - Custom plot customization
```

#### Seaborn

```bash
# Installation
pip install seaborn>=0.11.0
conda install seaborn>=0.11.0

# Purpose
# - Statistical data visualization
# - Enhanced plot aesthetics
# - Quick exploratory plots
```

### 3.4 Wavelet Analysis

#### PyWavelets

```bash
# Installation
pip install PyWavelets>=1.1.0
conda install pywavelets>=1.1.0

# Purpose
# - Wavelet transforms
# - Wavelet coefficient analysis
# - Signal processing
```

### 3.5 Data Management

#### PyYAML

```bash
# Installation
pip install PyYAML>=6.0
conda install pyyaml>=6.0

# Purpose
# - Configuration file parsing
# - YAML format support
# - Settings management
```

#### Requests

```bash
# Installation
pip install requests>=2.25.0
conda install requests>=2.25.0

# Purpose
# - HTTP requests for data download
# - API communication
# - Web scraping utilities
```

#### YFinance

```bash
# Installation
pip install yfinance>=0.1.70
conda install -c conda-forge yfinance>=0.1.70

# Purpose
# - Financial data download
# - Stock price data
# - Market data access
```

## Optional Dependencies

### 4.1 Performance Enhancement

#### Numba

```bash
# Installation
pip install numba>=0.56.0
conda install numba>=0.56.0

# Purpose
# - JIT compilation for performance
# - Parallel processing
# - GPU acceleration (CUDA)
```

#### Cython

```bash
# Installation
pip install cython>=0.29.0
conda install cython>=0.29.0

# Purpose
# - C extensions for Python
# - Performance optimization
# - Custom algorithms
```

### 4.2 Advanced Visualization

#### Plotly

```bash
# Installation
pip install plotly>=5.0.0
conda install -c conda-forge plotly>=5.0.0

# Purpose
# - Interactive plots
# - Web-based visualization
# - Dashboard creation
```

#### Bokeh

```bash
# Installation
pip install bokeh>=2.4.0
conda install -c conda-forge bokeh>=2.4.0

# Purpose
# - Interactive web plots
# - Real-time data visualization
# - Custom web applications
```

### 4.3 Data Formats

#### HDF5

```bash
# Installation
pip install h5py>=3.1.0
conda install h5py>=3.1.0

# Purpose
# - Large dataset storage
# - Hierarchical data format
# - Fast I/O operations
```

#### Parquet

```bash
# Installation
pip install pyarrow>=7.0.0
conda install pyarrow>=7.0.0

# Purpose
# - Columnar data storage
# - Efficient data compression
# - Big data analytics
```

### 4.4 Development Tools

#### Jupyter

```bash
# Installation
pip install jupyter>=1.0.0
conda install jupyter>=1.0.0

# Purpose
# - Interactive notebooks
# - Code documentation
# - Data exploration
```

#### IPython

```bash
# Installation
pip install ipython>=7.0.0
conda install ipython>=7.0.0

# Purpose
# - Enhanced Python shell
# - Magic commands
# - Interactive debugging
```

## Installation Procedures

### 5.1 Fresh Installation

#### Step 1: Install Python

```bash
# Download Python from python.org
# Or use package manager

# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-pip

# CentOS/RHEL
sudo yum install python39 python39-pip

# macOS (using Homebrew)
brew install python@3.9
```

#### Step 2: Create Virtual Environment

```bash
# Create project directory
mkdir lrd_analysis
cd lrd_analysis

# Create virtual environment
python3.9 -m venv venv

# Activate environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, pandas, scipy; print('Core packages installed')"
```

### 5.2 Conda Installation

#### Step 1: Install Miniconda

```bash
# Download Miniconda installer
# Run installer and follow prompts

# Verify installation
conda --version
```

#### Step 2: Create Environment

```bash
# Create conda environment
conda create -n lrd_analysis python=3.9

# Activate environment
conda activate lrd_analysis
```

#### Step 3: Install Packages

```bash
# Install packages from conda
conda install numpy pandas scipy matplotlib seaborn statsmodels scikit-learn

# Install packages not available in conda
pip install pywavelets yfinance

# Verify installation
python -c "import numpy, pandas, scipy; print('Packages installed')"
```

### 5.3 Docker Installation

#### Step 1: Install Docker

```bash
# Install Docker Desktop or Docker Engine
# Follow platform-specific instructions
```

#### Step 2: Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/src

# Default command
CMD ["python", "scripts/run_full_analysis.py"]
```

#### Step 3: Build and Run

```bash
# Build image
docker build -t lrd_analysis .

# Run container
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results lrd_analysis
```

## Environment Setup

### 6.1 Configuration Files

#### Environment Variables

```bash
# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/path/to/project/src"
export LRD_DATA_DIR="/path/to/data"
export LRD_RESULTS_DIR="/path/to/results"

# Windows (PowerShell)
$env:PYTHONPATH = "$env:PYTHONPATH;C:\path\to\project\src"
$env:LRD_DATA_DIR = "C:\path\to\data"
$env:LRD_RESULTS_DIR = "C:\path\to\results"
```

#### Configuration Files

```yaml
# config/environment.yaml
environment:
  data_dir: ${LRD_DATA_DIR:-./data}
  results_dir: ${LRD_RESULTS_DIR:-./results}
  log_level: INFO
  parallel_workers: 4
```

### 6.2 IDE Configuration

#### VS Code Setup

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black"
}
```

#### PyCharm Setup

1. Open project in PyCharm
2. Set project interpreter to virtual environment
3. Configure run configurations
4. Set up code style and inspections

### 6.3 Jupyter Setup

```bash
# Install Jupyter extensions
pip install jupyter_contrib_nbextensions

# Enable extensions
jupyter contrib nbextension install --user
jupyter nbextension enable toc2/main
jupyter nbextension enable codefolding/main
```

## Testing Installation

### 7.1 Basic Tests

#### Import Test

```python
# test_imports.py
def test_imports():
    """Test that all required packages can be imported."""
    try:
        import numpy as np
        import pandas as pd
        import scipy as sp
        import matplotlib.pyplot as plt
        import seaborn as sns
        import statsmodels.api as sm
        import sklearn
        import pywt
        import yaml
        import requests
        import yfinance as yf
        print("✓ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()
```

#### Functionality Test

```python
# test_functionality.py
import numpy as np
import pandas as pd

def test_basic_functionality():
    """Test basic functionality of core packages."""
    # Test NumPy
    arr = np.random.randn(100)
    assert arr.shape == (100,)
    print("✓ NumPy functionality OK")
    
    # Test Pandas
    df = pd.DataFrame({'A': arr})
    assert len(df) == 100
    print("✓ Pandas functionality OK")
    
    # Test SciPy
    from scipy import stats
    mean = stats.tmean(arr)
    assert isinstance(mean, float)
    print("✓ SciPy functionality OK")

if __name__ == "__main__":
    test_functionality()
```

### 7.2 Project Tests

#### Run Project Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_fractal_analysis.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

#### Test Configuration

```python
# tests/conftest.py
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    import numpy as np
    np.random.seed(42)
    return np.random.randn(1000)
```

### 7.3 Performance Tests

#### Benchmark Tests

```python
# test_performance.py
import time
import numpy as np
from src.analysis.fractal_analysis import dfa

def test_dfa_performance():
    """Test DFA performance with different data sizes."""
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        data = np.random.randn(size)
        
        start_time = time.time()
        results = dfa(data)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"DFA {size} points: {duration:.3f}s")
        
        assert duration < 10.0  # Should complete within 10 seconds

if __name__ == "__main__":
    test_dfa_performance()
```

## Troubleshooting

### 8.1 Common Issues

#### Package Installation Issues

**Problem**: `pip install` fails with compilation errors
**Solutions**:
```bash
# Install build tools
conda install -c conda-forge compilers

# Use conda instead of pip
conda install package_name

# Install pre-compiled wheels
pip install --only-binary=all package_name
```

**Problem**: Version conflicts between packages
**Solutions**:
```bash
# Create fresh environment
conda create -n lrd_analysis python=3.9

# Install packages in specific order
conda install numpy pandas scipy
conda install matplotlib seaborn
pip install remaining_packages
```

#### Import Errors

**Problem**: `ModuleNotFoundError` for installed packages
**Solutions**:
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall package
pip uninstall package_name
pip install package_name

# Check virtual environment
which python
pip list
```

**Problem**: Package version mismatch
**Solutions**:
```bash
# Check package versions
pip list | grep package_name

# Install specific version
pip install package_name==version

# Update all packages
pip install --upgrade -r requirements.txt
```

### 8.2 Performance Issues

#### Memory Problems

**Problem**: Out of memory errors
**Solutions**:
```python
# Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Use chunked processing
def process_in_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        # Process chunk
        yield process_chunk(chunk)
```

#### Slow Performance

**Problem**: Analysis takes too long
**Solutions**:
```python
# Enable parallel processing
from src.config_loader import get_performance_config

config = get_performance_config()
if config['parallel']['enabled']:
    # Use parallel processing
    pass

# Profile code
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Run analysis
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### 8.3 Platform-Specific Issues

#### Windows Issues

**Problem**: Path length limitations
**Solutions**:
```bash
# Use short paths
# Enable long path support in Windows 10
# Use subst command for long paths
subst X: "C:\very\long\path\to\project"
```

**Problem**: Visual C++ compiler missing
**Solutions**:
```bash
# Install Visual Studio Build Tools
# Or use conda for pre-compiled packages
conda install package_name
```

#### macOS Issues

**Problem**: OpenMP not found
**Solutions**:
```bash
# Install OpenMP
brew install libomp

# Set environment variables
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
```

#### Linux Issues

**Problem**: Missing system libraries
**Solutions**:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel

# Install specific libraries
sudo apt-get install libblas-dev liblapack-dev
```

## Performance Considerations

### 9.1 Optimization Strategies

#### Algorithm Optimization

```python
# Use vectorized operations
# Instead of loops
for i in range(len(data)):
    result[i] = data[i] * 2

# Use NumPy operations
result = data * 2

# Use efficient data structures
import numpy as np
data = np.array(data)  # More efficient than list
```

#### Memory Management

```python
# Use appropriate data types
data = np.array(data, dtype=np.float32)  # Instead of float64

# Clear unused variables
del large_data
import gc
gc.collect()

# Use memory mapping for large files
import numpy as np
data = np.memmap('large_file.dat', dtype=np.float64, mode='r')
```

### 9.2 Parallel Processing

#### Multiprocessing

```python
from multiprocessing import Pool
from functools import partial

def parallel_analysis(data_list, n_workers=4):
    """Run analysis in parallel."""
    with Pool(n_workers) as pool:
        results = pool.map(analyze_data, data_list)
    return results

# Use in analysis
if __name__ == "__main__":
    results = parallel_analysis(data_list)
```

#### Joblib

```python
from joblib import Parallel, delayed

def parallel_analysis(data_list, n_jobs=-1):
    """Run analysis using joblib."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(analyze_data)(data) for data in data_list
    )
    return results
```

### 9.3 Caching

#### Function Caching

```python
from functools import lru_cache
import joblib

# Simple caching
@lru_cache(maxsize=128)
def expensive_function(x):
    # Expensive computation
    return result

# Persistent caching
@joblib.Memory('cache').cache
def expensive_function(x):
    # Expensive computation
    return result
```

#### Data Caching

```python
import pickle
import os

def cache_results(results, filename):
    """Cache results to file."""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_cached_results(filename):
    """Load cached results from file."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None
```

## Version Compatibility

### 10.1 Dependency Matrix

#### Core Package Compatibility

| Package | Min Version | Max Version | Notes |
|---------|-------------|-------------|-------|
| Python | 3.8 | 3.12 | 3.9-3.11 recommended |
| NumPy | 1.21.0 | 1.26.x | Latest stable |
| Pandas | 1.3.0 | 2.1.x | Latest stable |
| SciPy | 1.7.0 | 1.11.x | Latest stable |
| Matplotlib | 3.5.0 | 3.8.x | Latest stable |

#### Optional Package Compatibility

| Package | Min Version | Max Version | Notes |
|---------|-------------|-------------|-------|
| PyWavelets | 1.1.0 | 1.4.x | Latest stable |
| Statsmodels | 0.13.0 | 0.14.x | Latest stable |
| Scikit-learn | 1.0.0 | 1.3.x | Latest stable |
| YFinance | 0.1.70 | 0.2.x | Latest stable |

### 10.2 Testing Matrix

#### Tested Combinations

| Python | NumPy | Pandas | SciPy | Status |
|--------|-------|--------|-------|--------|
| 3.8 | 1.21.0 | 1.3.0 | 1.7.0 | ✅ Tested |
| 3.9 | 1.24.0 | 1.5.0 | 1.10.0 | ✅ Tested |
| 3.10 | 1.24.0 | 1.5.0 | 1.10.0 | ✅ Tested |
| 3.11 | 1.25.0 | 2.0.0 | 1.11.0 | ✅ Tested |
| 3.12 | 1.26.0 | 2.1.0 | 1.11.0 | ⚠️ Testing |

### 10.3 Migration Guide

#### Upgrading from Previous Versions

```bash
# Backup current environment
conda env export > environment_backup.yml

# Create new environment
conda create -n lrd_analysis_new python=3.11

# Install packages
conda install numpy=1.26 pandas=2.1 scipy=1.11

# Test compatibility
python -m pytest tests/ -v
```

#### Breaking Changes

- **Python 3.7**: No longer supported
- **NumPy 1.20**: Deprecated functions removed
- **Pandas 1.5**: Some API changes
- **SciPy 1.8**: Deprecated functions removed

---

*This document provides comprehensive information about software requirements and installation procedures. For specific issues, refer to the troubleshooting section or create an issue in the project repository.*

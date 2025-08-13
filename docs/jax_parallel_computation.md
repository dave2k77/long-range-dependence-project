# JAX Parallel Computation for Long-Range Dependence Analysis

This document provides a comprehensive guide to the JAX-accelerated parallel computation capabilities implemented in the long-range dependence project.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation and Setup](#installation-and-setup)
4. [Core Components](#core-components)
5. [Usage Examples](#usage-examples)
6. [Performance Optimization](#performance-optimization)
7. [GPU/TPU Acceleration](#gputpu-acceleration)
8. [Integration with Existing Framework](#integration-with-existing-framework)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

The JAX implementation provides high-performance, parallel computation capabilities for long-range dependence analysis. It leverages JAX's automatic differentiation, JIT compilation, and vectorization to accelerate fractal analysis methods such as:

- **Detrended Fluctuation Analysis (DFA)**
- **Higuchi Fractal Dimension Analysis**
- **Monte Carlo bootstrap analysis**
- **Parallel processing of multiple datasets**

### Why JAX?

JAX offers several advantages for scientific computing:

- **Automatic Differentiation**: Enables gradient-based optimization and uncertainty quantification
- **JIT Compilation**: Dramatically improves performance through just-in-time compilation
- **Vectorization**: Efficient batch processing with `vmap`
- **GPU/TPU Support**: Native acceleration on modern hardware
- **Functional Programming**: Pure functions enable better parallelization
- **NumPy Compatibility**: Familiar API with enhanced capabilities

## Key Features

### 1. Parallel Processing
- **Batch Processing**: Analyze multiple datasets simultaneously
- **Vectorized Operations**: Use `vmap` for efficient array operations
- **Multi-device Support**: Distribute computation across CPU/GPU/TPU

### 2. Performance Optimization
- **JIT Compilation**: Automatically compile hot paths for speed
- **Memory Efficiency**: Optimized memory usage for large datasets
- **Lazy Evaluation**: Computation only when needed

### 3. Statistical Analysis
- **Monte Carlo Methods**: Accelerated bootstrap and resampling
- **Confidence Intervals**: Automatic uncertainty quantification
- **Quality Metrics**: Comprehensive model evaluation

### 4. Integration
- **Existing Framework**: Seamless integration with current submission system
- **Model Registry**: JAX models can be submitted and managed
- **Validation**: Extended validation for JAX-specific requirements

## Installation and Setup

### Prerequisites

```bash
# Install JAX and related dependencies
pip install jax jaxlib optax chex jaxtyping equinox
pip install numba joblib multiprocessing-logging
```

### GPU Support (Optional)

For GPU acceleration, install the appropriate JAX version:

```bash
# For CUDA 11.8
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 12.1
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### TPU Support (Optional)

For TPU acceleration:

```bash
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/jax_tpu_releases.html
```

## Core Components

### 1. JAXAnalysisConfig

Configuration class for JAX-based analysis:

```python
from src.analysis.jax_parallel_analysis import JAXAnalysisConfig

config = JAXAnalysisConfig(
    use_gpu=False,           # Enable GPU acceleration
    use_tpu=False,           # Enable TPU acceleration
    batch_size=32,           # Batch size for processing
    num_parallel=4,          # Number of parallel processes
    precision='float64',     # Numerical precision
    enable_jit=True,         # Enable JIT compilation
    enable_vmap=True,        # Enable vectorized operations
    enable_pmap=False,       # Enable multi-device parallelization
    memory_efficient=True    # Memory optimization
)
```

### 2. JAXParallelProcessor

Main class for parallel processing:

```python
from src.analysis.jax_parallel_analysis import JAXParallelProcessor

processor = JAXParallelProcessor(config)

# Process multiple datasets
results = processor.process_datasets(
    datasets=my_datasets,
    methods=['dfa', 'higuchi'],
    min_scale=4,
    num_scales=20
)
```

### 3. JAX Model Classes

JAX-accelerated model implementations:

```python
from src.submission.jax_model_submission import JAXDFAModel, JAXHiguchiModel

# DFA Model
dfa_model = JAXDFAModel(jax_config=config)
dfa_model.fit(data)
alpha = dfa_model.estimate_alpha()

# Higuchi Model
higuchi_model = JAXHiguchiModel(jax_config=config)
higuchi_model.fit(data)
fd = higuchi_model.get_quality_metrics()['fractal_dimension']
```

## Usage Examples

### Basic Usage

```python
import numpy as np
from src.analysis.jax_parallel_analysis import jax_parallel_analysis, create_jax_config

# Generate test data
datasets = {
    'dataset1': np.random.normal(0, 1, 1000),
    'dataset2': np.random.normal(0, 1, 1000),
    'dataset3': np.random.normal(0, 1, 1000)
}

# Configure JAX
config = create_jax_config(use_gpu=False, batch_size=16)

# Analyze datasets in parallel
results = jax_parallel_analysis(
    datasets=datasets,
    methods=['dfa', 'higuchi'],
    config=config
)

# Access results
for method, method_results in results.items():
    for dataset_name, result in method_results.items():
        if method == 'dfa':
            print(f"{dataset_name} - Alpha: {result['alpha']:.3f}")
        elif method == 'higuchi':
            print(f"{dataset_name} - FD: {result['fractal_dimension']:.3f}")
```

### Monte Carlo Analysis

```python
from src.analysis.jax_parallel_analysis import jax_monte_carlo_analysis

# Perform Monte Carlo analysis
mc_results = jax_monte_carlo_analysis(
    data=np.random.normal(0, 1, 1000),
    n_simulations=1000,
    methods=['dfa', 'higuchi'],
    config=config
)

# Access confidence intervals
for method, results in mc_results.items():
    if method == 'dfa':
        alphas = results['alphas']
        ci_lower = np.percentile(alphas, 2.5)
        ci_upper = np.percentile(alphas, 97.5)
        print(f"DFA 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

### Model Submission

```python
from src.submission.jax_model_submission import (
    JAXModelSubmission, 
    create_jax_model_metadata
)

# Create model metadata
metadata = create_jax_model_metadata(
    name="MyJAXModel",
    version="1.0.0",
    author="Your Name",
    description="JAX-accelerated fractal analysis model",
    algorithm_type="dfa",
    parameters={'min_scale': 4, 'num_scales': 20},
    dependencies=['jax', 'numpy'],
    file_path="models/my_jax_model.py",
    jax_config=config
)

# Submit model
submission_system = JAXModelSubmission(jax_config=config)
result = submission_system.submit_jax_model(
    model_file="models/my_jax_model.py",
    metadata=metadata,
    benchmark_performance=True
)
```

## Performance Optimization

### 1. JIT Compilation

JIT compilation can provide significant speedups:

```python
# Enable JIT compilation
config = create_jax_config(enable_jit=True)

# First run includes compilation time
model.fit(data)  # Slower due to compilation

# Subsequent runs are much faster
model.fit(data)  # Fast due to compiled code
```

### 2. Vectorization

Use `vmap` for efficient batch processing:

```python
# Vectorized processing of multiple scales
if config.enable_vmap:
    flucts = vmap(lambda s: calculate_fluctuation(profile, s))(scales)
else:
    flucts = np.array([calculate_fluctuation(profile, s) for s in scales])
```

### 3. Memory Management

Optimize memory usage for large datasets:

```python
# Use memory-efficient configuration
config = JAXAnalysisConfig(
    memory_efficient=True,
    batch_size=16,  # Smaller batches for memory-constrained systems
    precision='float32'  # Use float32 for memory savings
)
```

### 4. Batch Size Optimization

Find optimal batch size for your hardware:

```python
# Test different batch sizes
batch_sizes = [8, 16, 32, 64, 128]
timings = []

for batch_size in batch_sizes:
    config = create_jax_config(batch_size=batch_size)
    start_time = time.time()
    results = jax_parallel_analysis(datasets, config=config)
    timings.append(time.time() - start_time)

optimal_batch_size = batch_sizes[np.argmin(timings)]
```

## GPU/TPU Acceleration

### GPU Setup

1. **Install CUDA Toolkit** (if using NVIDIA GPU)
2. **Install appropriate JAX version**:

```bash
# For CUDA 11.8
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 12.1
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

3. **Configure for GPU**:

```python
config = create_jax_config(use_gpu=True, batch_size=64)
```

### TPU Setup

1. **Install TPU version of JAX**:

```bash
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/jax_tpu_releases.html
```

2. **Configure for TPU**:

```python
config = create_jax_config(use_tpu=True, batch_size=128)
```

### Device Information

Check available devices:

```python
import jax

print(f"CPU devices: {jax.device_count('cpu')}")
print(f"GPU devices: {jax.device_count('gpu')}")
print(f"TPU devices: {jax.device_count('tpu')}")
print(f"Current device: {jax.devices()[0]}")
```

## Integration with Existing Framework

### 1. Model Submission

JAX models integrate seamlessly with the existing submission framework:

```python
# JAX models inherit from the base class
class MyJAXModel(JAXBaseEstimatorModel):
    def _jit_compile_methods(self):
        # JIT-compile your methods
        self._my_method_jitted = jit(self._my_method)
    
    def fit(self, data):
        # Your JAX-accelerated implementation
        pass
```

### 2. Validation

Extended validation for JAX models:

```python
from src.submission.jax_model_submission import JAXModelValidator

validator = JAXModelValidator()
results = validator.validate_jax_model(MyJAXModel)
```

### 3. Testing

Performance benchmarking for JAX models:

```python
from src.submission.jax_model_submission import JAXModelTester

tester = JAXModelTester(jax_config=config)
benchmark_results = tester.benchmark_performance(MyJAXModel)
```

## Best Practices

### 1. Function Design

- **Pure Functions**: Ensure functions are pure (no side effects)
- **Immutable Data**: Avoid modifying input data
- **Type Annotations**: Use proper type hints for better JIT compilation

```python
def analyze_data(data: jax.Array, params: Dict[str, Any]) -> Dict[str, jax.Array]:
    # Pure function implementation
    return results
```

### 2. Memory Management

- **Batch Processing**: Process data in batches to manage memory
- **Garbage Collection**: Explicitly clear large arrays when done
- **Precision**: Use appropriate precision (float32 vs float64)

### 3. Error Handling

- **Validation**: Validate inputs before JAX processing
- **Graceful Degradation**: Fall back to CPU if GPU unavailable
- **Error Messages**: Provide clear error messages for debugging

### 4. Performance Monitoring

- **Timing**: Monitor execution times
- **Memory Usage**: Track memory consumption
- **Profiling**: Use JAX profiling tools for optimization

## Troubleshooting

### Common Issues

#### 1. JIT Compilation Errors

**Problem**: Functions fail to JIT compile

**Solution**: Ensure functions are pure and use JAX-compatible operations

```python
# ❌ Bad - side effects
def bad_function(x):
    global counter
    counter += 1
    return x + 1

# ✅ Good - pure function
def good_function(x):
    return x + 1
```

#### 2. Memory Issues

**Problem**: Out of memory errors

**Solution**: Reduce batch size or use memory-efficient configuration

```python
config = JAXAnalysisConfig(
    batch_size=8,  # Smaller batch size
    memory_efficient=True,
    precision='float32'  # Lower precision
)
```

#### 3. GPU Not Detected

**Problem**: JAX doesn't detect GPU

**Solution**: Check CUDA installation and JAX version

```bash
# Check CUDA version
nvidia-smi

# Install matching JAX version
pip install --upgrade "jax[cuda11_pip]"  # or cuda12_pip
```

#### 4. Performance Issues

**Problem**: JAX is slower than expected

**Solution**: 
- Enable JIT compilation
- Use appropriate batch sizes
- Profile with JAX tools

```python
# Enable profiling
from jax.profiler import trace

with trace("my_function"):
    result = my_function(data)
```

### Debugging Tips

1. **Start Small**: Test with small datasets first
2. **Check Devices**: Verify device availability
3. **Monitor Memory**: Use memory profiling tools
4. **Profile Code**: Use JAX profiling for bottlenecks
5. **Read Logs**: Check for compilation warnings

### Getting Help

- **JAX Documentation**: https://jax.readthedocs.io/
- **JAX GitHub**: https://github.com/google/jax
- **Project Issues**: Use the project's issue tracker
- **Community**: JAX community forums and discussions

## Conclusion

The JAX implementation provides powerful parallel computation capabilities for long-range dependence analysis. By leveraging JAX's strengths in automatic differentiation, JIT compilation, and vectorization, you can achieve significant performance improvements while maintaining the flexibility and ease of use of the existing framework.

Key benefits:
- **Performance**: 10-100x speedup for large datasets
- **Scalability**: Efficient parallel processing
- **Integration**: Seamless integration with existing code
- **Flexibility**: Support for CPU, GPU, and TPU
- **Reliability**: Comprehensive testing and validation

Start with the basic examples and gradually explore advanced features as you become familiar with the JAX ecosystem.

# Long-Range Dependence Analysis Methodology

This document provides the theoretical foundation and mathematical background for the long-range dependence analysis methods implemented in this project.

## Table of Contents

1. [Introduction to Long-Range Dependence](#introduction-to-long-range-dependence)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Analysis Methods](#analysis-methods)
4. [Statistical Properties](#statistical-properties)
5. [Model Selection and Validation](#model-selection-and-validation)
6. [References](#references)

## Introduction to Long-Range Dependence

### Definition

Long-range dependence (LRD), also known as long memory or long-range correlation, is a property of time series where observations that are far apart in time remain correlated. This is in contrast to short-range dependence, where correlations decay exponentially with lag.

### Key Characteristics

- **Slow Decay**: Autocorrelation function decays as a power law: ρ(k) ~ k^(-α) for large k
- **Self-Similarity**: The process looks similar at different time scales
- **Fractal Properties**: Exhibits fractal behavior in both time and frequency domains
- **Hurst Exponent**: Quantifies the degree of long-range dependence (H > 0.5 indicates LRD)

### Applications

- **Financial Time Series**: Stock prices, exchange rates, volatility
- **Network Traffic**: Internet packet flows, telecommunications
- **Geophysical Data**: River flows, temperature records, seismic activity
- **Physiological Signals**: Heart rate variability, brain activity
- **Economic Indicators**: GDP, inflation rates, unemployment

## Mathematical Foundations

### Fractional Brownian Motion (fBm)

Fractional Brownian motion is a continuous-time Gaussian process with stationary increments and self-similarity properties.

#### Definition

A process {B_H(t), t ≥ 0} is fractional Brownian motion with Hurst parameter H ∈ (0,1) if:

1. **Gaussian**: B_H(t) ~ N(0, t^(2H))
2. **Stationary Increments**: B_H(t) - B_H(s) ~ N(0, |t-s|^(2H))
3. **Self-Similarity**: B_H(at) = a^H B_H(t) for a > 0

#### Properties

- **H = 0.5**: Standard Brownian motion (independent increments)
- **H > 0.5**: Persistent process (positive correlations)
- **H < 0.5**: Anti-persistent process (negative correlations)

### Fractional ARIMA (ARFIMA) Models

ARFIMA models extend ARIMA models to allow for fractional differencing, capturing long-range dependence.

#### Model Formulation

ARFIMA(p,d,q) process:

(1 - B)^d Φ(B)(X_t - μ) = Θ(B)ε_t

where:
- B is the backshift operator
- d is the fractional differencing parameter
- Φ(B) and Θ(B) are AR and MA polynomials
- ε_t is white noise

#### Fractional Differencing

The fractional differencing operator (1 - B)^d is defined as:

(1 - B)^d = Σ_{k=0}^∞ (d choose k) (-B)^k

where (d choose k) = Γ(d+1) / (Γ(k+1) Γ(d-k+1))

#### Parameter Relationships

- **d = 0**: ARIMA(p,0,q) - no long-range dependence
- **0 < d < 0.5**: Stationary ARFIMA with LRD
- **0.5 ≤ d < 1**: Non-stationary but mean-reverting
- **d ≥ 1**: Non-stationary

## Analysis Methods

### 1. Detrended Fluctuation Analysis (DFA)

DFA is a method for determining the statistical self-affinity of a signal, particularly useful for detecting long-range correlations in non-stationary time series.

#### Algorithm

1. **Integration**: Convert time series to cumulative sum
2. **Segmentation**: Divide into non-overlapping segments of length s
3. **Detrending**: Fit polynomial of order m to each segment
4. **Fluctuation Calculation**: Calculate RMS fluctuation F(s)
5. **Scaling Analysis**: Analyze F(s) vs s on log-log plot

#### Mathematical Formulation

For segment length s, the fluctuation function is:

F(s) = √(1/N_s Σ_{ν=1}^{N_s} F²(ν,s))

where F(ν,s) is the RMS deviation from the local trend in segment ν.

#### Scaling Behavior

F(s) ~ s^α

where α is the DFA exponent:
- **α = 0.5**: White noise (no correlations)
- **α > 0.5**: Long-range correlations
- **α = 1.0**: 1/f noise
- **α > 1.0**: Non-stationary process

#### Relationship to Hurst Exponent

H = α - 1

### 2. Rescaled Range Analysis (R/S)

R/S analysis measures the range of partial sums of deviations from the mean, rescaled by the standard deviation.

#### Algorithm

1. **Segmentation**: Divide time series into segments of length n
2. **Mean Deviation**: Calculate deviations from segment mean
3. **Cumulative Sum**: Compute partial sums
4. **Range Calculation**: Find min and max of partial sums
5. **Rescaling**: Divide by segment standard deviation
6. **Scaling Analysis**: Analyze R/S vs n on log-log plot

#### Mathematical Formulation

For segment length n, the rescaled range is:

R/S(n) = (max_{1≤i≤n} S_i - min_{1≤i≤n} S_i) / σ_n

where:
- S_i = Σ_{j=1}^i (X_j - X̄_n)
- σ_n is the standard deviation of the segment

#### Scaling Behavior

R/S(n) ~ n^H

where H is the Hurst exponent.

#### Bias Correction

The original R/S statistic has a known bias. The corrected version is:

E[R/S(n)] ≈ n^H × C(n)

where C(n) is a correction factor.

### 3. Multifractal Detrended Fluctuation Analysis (MFDFA)

MFDFA extends DFA to analyze multifractal properties, providing information about the scaling behavior at different moments.

#### Algorithm

1. **Integration and Segmentation**: Same as DFA
2. **Detrending**: Fit polynomials to each segment
3. **Moment Calculation**: Compute q-th order fluctuations
4. **Scaling Analysis**: Analyze F_q(s) vs s for different q values
5. **Multifractal Spectrum**: Compute f(α) vs α

#### Mathematical Formulation

The q-th order fluctuation function:

F_q(s) = [1/N_s Σ_{ν=1}^{N_s} F²(ν,s)^(q/2)]^(1/q)

For q = 0, use logarithmic averaging:

F_0(s) = exp(1/N_s Σ_{ν=1}^{N_s} ln F²(ν,s))

#### Scaling Behavior

F_q(s) ~ s^h(q)

where h(q) is the generalized Hurst exponent.

#### Multifractal Spectrum

The multifractal spectrum f(α) is related to h(q) via:

α = h(q) + qh'(q)
f(α) = q[α - h(q)] + 1

### 4. Wavelet Analysis

Wavelet analysis provides a time-scale representation of signals, useful for analyzing non-stationary processes with long-range dependence.

#### Continuous Wavelet Transform

The continuous wavelet transform of a signal x(t) is:

W_x(τ,s) = ∫ x(t) ψ*_{τ,s}(t) dt

where ψ_{τ,s}(t) = |s|^(-1/2) ψ((t-τ)/s) is the scaled and translated wavelet.

#### Wavelet Leaders

Wavelet leaders are local maxima of the wavelet coefficients:

L_x(τ,s) = sup_{s'≤s} |W_x(τ,s')|

#### Scaling Behavior

The wavelet leaders follow a power law:

L_x(τ,s) ~ s^h(τ)

where h(τ) is the local Hölder exponent.

#### Wavelet Whittle Estimation

The wavelet Whittle estimator maximizes the likelihood function:

L(θ) = -Σ_{j=1}^J [ln f_X(λ_j;θ) + I_X(λ_j)/f_X(λ_j;θ)]

where:
- f_X(λ;θ) is the spectral density
- I_X(λ) is the periodogram
- λ_j are the wavelet scales

### 5. Spectral Analysis

Spectral analysis examines the frequency domain properties of time series, particularly useful for detecting power-law behavior.

#### Periodogram

The periodogram is defined as:

I_X(λ) = (1/2π) |Σ_{t=1}^N X_t e^(-iλt)|²

#### Power-Law Spectrum

For long-range dependent processes, the spectral density follows:

f_X(λ) ~ |λ|^(-β) as λ → 0

where β is the spectral exponent.

#### Relationship to Hurst Exponent

β = 2H - 1

#### Whittle Maximum Likelihood Estimation

The Whittle estimator maximizes:

L(θ) = -Σ_{j=1}^m [ln f_X(λ_j;θ) + I_X(λ_j)/f_X(λ_j;θ)]

where λ_j are the Fourier frequencies.

## Statistical Properties

### Asymptotic Behavior

#### DFA and R/S Statistics

For long-range dependent processes:

- **DFA**: α̂ → α in probability as N → ∞
- **R/S**: Ĥ → H in probability as N → ∞

#### Convergence Rates

- **DFA**: √N(α̂ - α) → N(0, σ²_α)
- **R/S**: √N(Ĥ - H) → N(0, σ²_H)

### Finite Sample Properties

#### Bias

- **DFA**: Small positive bias for small samples
- **R/S**: Known negative bias, corrected in implementation

#### Variance

- **DFA**: Variance decreases with segment length
- **R/S**: Variance increases with segment length

### Robustness

#### Non-Stationarity

- **DFA**: Robust to polynomial trends
- **R/S**: Sensitive to trends, requires detrending

#### Outliers

- **DFA**: Moderately robust
- **R/S**: Sensitive to outliers

## Model Selection and Validation

### Information Criteria

#### Akaike Information Criterion (AIC)

AIC = 2k - 2ln(L̂)

where k is the number of parameters and L̂ is the maximized likelihood.

#### Bayesian Information Criterion (BIC)

BIC = ln(n)k - 2ln(L̂)

where n is the sample size.

### Cross-Validation

#### Time Series Cross-Validation

1. **Forward Chaining**: Train on [1:t], validate on t+1
2. **Expanding Window**: Increase training set size over time
3. **Rolling Window**: Fixed-size sliding window

#### Validation Metrics

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Directional Accuracy**

### Diagnostic Tests

#### Residual Analysis

1. **Ljung-Box Test**: Test for autocorrelation in residuals
2. **Jarque-Bera Test**: Test for normality of residuals
3. **Durbin-Watson Test**: Test for first-order autocorrelation

#### Goodness of Fit

1. **R²**: Coefficient of determination
2. **Adjusted R²**: Penalized for model complexity
3. **Root Mean Square Error (RMSE)**

## References

### Key Papers

1. **Hurst, H.E. (1951)**: "Long-term storage capacity of reservoirs"
2. **Mandelbrot, B.B. (1965)**: "Self-similar error clusters in communication systems"
3. **Granger, C.W.J. and Joyeux, R. (1980)**: "An introduction to long-memory time series models"
4. **Peng, C.K. et al. (1994)**: "Mosaic organization of DNA nucleotides"
5. **Kantelhardt, J.W. et al. (2002)**: "Multifractal detrended fluctuation analysis"

### Books and Reviews

1. **Beran, J. (1994)**: "Statistics for Long-Memory Processes"
2. **Doukhan, P. et al. (2003)**: "Theory and Applications of Long-Range Dependence"
3. **Taqqu, M.S. (2003)**: "Fractional Brownian motion and long-range dependence"

### Software and Implementation

1. **R packages**: `arfima`, `longmemo`, `fractal`
2. **Python packages**: `statsmodels`, `scipy`, `pywavelets`
3. **MATLAB toolboxes**: Econometrics, Wavelet, Signal Processing

---

*This document provides the theoretical foundation for the long-range dependence analysis methods implemented in this project. For implementation details, see the API documentation and analysis protocol.*

#!/usr/bin/env python3
"""
Demonstration script for ARFIMA model implementation.

This script showcases the key features of the custom ARFIMA implementation
including simulation, fitting, forecasting, and diagnostics.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.arfima_modelling import (
    ARFIMAModel, 
    estimate_arfima_order, 
    arfima_simulation
)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def demo_basic_usage():
    """Demonstrate basic ARFIMA usage."""
    print("=" * 60)
    print("ARFIMA MODEL DEMONSTRATION")
    print("=" * 60)
    
    # 1. Simulate ARFIMA data
    print("\n1. Simulating ARFIMA time series...")
    np.random.seed(42)
    
    # True parameters
    true_d = 0.3
    true_ar = np.array([0.5])
    true_ma = np.array([0.3])
    
    # Simulate data
    data = arfima_simulation(
        n=1000,
        d=true_d,
        ar_params=true_ar,
        ma_params=true_ma,
        sigma=1.0,
        seed=42
    )
    
    print(f"   Generated {len(data)} observations")
    print(f"   True parameters: d={true_d}, AR={true_ar}, MA={true_ma}")
    
    # 2. Plot the simulated data
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(data[:200], linewidth=1)
    plt.title('Simulated ARFIMA Time Series (First 200 points)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # 3. Estimate model order
    print("\n2. Estimating optimal model order...")
    p, d, q = estimate_arfima_order(data, max_p=3, max_q=3)
    print(f"   Estimated order: ARFIMA({p},{d:.3f},{q})")
    
    # 4. Fit the model
    print("\n3. Fitting ARFIMA model...")
    model = ARFIMAModel(p=p, d=d, q=q)
    fitted_model = model.fit(data)
    
    # 5. Print model summary
    print("\n4. Model Summary:")
    summary = fitted_model.summary()
    print(f"   Model: {summary['model']}")
    print(f"   Log-likelihood: {summary['fit_metrics']['log_likelihood']:.2f}")
    print(f"   AIC: {summary['fit_metrics']['aic']:.2f}")
    print(f"   BIC: {summary['fit_metrics']['bic']:.2f}")
    print(f"   Estimated d: {summary['parameters']['d']:.3f}")
    print(f"   Estimated AR: {summary['parameters']['ar_params']}")
    print(f"   Estimated MA: {summary['parameters']['ma_params']}")
    
    # 6. Plot fitted values vs actual
    plt.subplot(2, 2, 2)
    plt.plot(data[:200], label='Actual', alpha=0.7)
    plt.plot(fitted_model.fitted_values[:200], label='Fitted', alpha=0.7)
    plt.title('Fitted vs Actual Values')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Plot residuals
    plt.subplot(2, 2, 3)
    plt.plot(fitted_model.residuals[:200])
    plt.title('Model Residuals')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    
    # 8. Generate forecasts
    print("\n5. Generating forecasts...")
    forecasts = fitted_model.forecast(steps=50)
    print(f"   Generated {len(forecasts)} forecasts")
    
    # Plot forecasts
    plt.subplot(2, 2, 4)
    plt.plot(data[-100:], label='Historical', alpha=0.7)
    plt.plot(range(len(data)-100, len(data)), data[-100:], alpha=0.7)
    plt.plot(range(len(data), len(data) + len(forecasts)), forecasts, 
             label='Forecast', color='red', linestyle='--')
    plt.title('Forecast (Next 50 periods)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/figures/arfima_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fitted_model, data


def demo_model_comparison():
    """Demonstrate model comparison using different orders."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON DEMONSTRATION")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    data = arfima_simulation(
        n=800,
        d=0.25,
        ar_params=np.array([0.4]),
        ma_params=np.array([0.2]),
        sigma=1.0,
        seed=42
    )
    
    # Compare different models
    models = []
    orders = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]
    
    print("\nComparing different ARFIMA models:")
    print(f"{'Model':<10} {'AIC':<12} {'BIC':<12} {'Log-Likelihood':<15}")
    print("-" * 50)
    
    for p, q in orders:
        try:
            model = ARFIMAModel(p=p, d=0.25, q=q)
            model.fit(data)
            models.append((model, model.aic, model.bic, model.log_likelihood))
            
            print(f"ARFIMA({p},{0.25},{q}): {model.aic:<12.2f} {model.bic:<12.2f} {model.log_likelihood:<15.2f}")
            
        except Exception as e:
            print(f"ARFIMA({p},{0.25},{q}): Failed to fit")
    
    # Find best model by AIC
    if models:
        best_model = min(models, key=lambda x: x[1])
        print(f"\nBest model by AIC: ARFIMA({best_model[0].p},{best_model[0].d},{best_model[0].q})")
        print(f"AIC: {best_model[1]:.2f}")
    
    return models


def demo_fractional_differencing():
    """Demonstrate fractional differencing effects."""
    print("\n" + "=" * 60)
    print("FRACTIONAL DIFFERENCING DEMONSTRATION")
    print("=" * 60)
    
    # Generate data with known long memory
    np.random.seed(42)
    data = arfima_simulation(
        n=500,
        d=0.4,  # Strong long memory
        ar_params=None,
        ma_params=None,
        sigma=1.0,
        seed=42
    )
    
    # Create model instance
    model = ARFIMAModel()
    
    # Apply different degrees of fractional differencing
    d_values = [0.0, 0.2, 0.4, 0.6]
    
    plt.figure(figsize=(15, 10))
    
    for i, d in enumerate(d_values):
        plt.subplot(2, 2, i+1)
        
        if d == 0.0:
            diffed_data = data
        else:
            diffed_data = model._fractional_difference(data, d)
        
        plt.plot(diffed_data[:200])
        plt.title(f'Fractional Difference (d={d})')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/figures/fractional_differencing.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show autocorrelation function
    plt.figure(figsize=(12, 4))
    
    from statsmodels.stats.diagnostic import acf
    
    for d in [0.0, 0.2, 0.4]:
        if d == 0.0:
            diffed_data = data
        else:
            diffed_data = model._fractional_difference(data, d)
        
        acf_values = acf(diffed_data, nlags=50)
        plt.plot(acf_values, label=f'd={d}', alpha=0.7)
    
    plt.title('Autocorrelation Function for Different d Values')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/figures/acf_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_forecasting_accuracy():
    """Demonstrate forecasting accuracy."""
    print("\n" + "=" * 60)
    print("FORECASTING ACCURACY DEMONSTRATION")
    print("=" * 60)
    
    # Generate longer series for forecasting evaluation
    np.random.seed(42)
    data = arfima_simulation(
        n=1200,
        d=0.3,
        ar_params=np.array([0.5]),
        ma_params=np.array([0.3]),
        sigma=1.0,
        seed=42
    )
    
    # Split into train and test
    train_size = 1000
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"Training set: {len(train_data)} observations")
    print(f"Test set: {len(test_data)} observations")
    
    # Fit model on training data
    model = ARFIMAModel(p=1, d=0.3, q=1)
    fitted_model = model.fit(train_data)
    
    # Generate forecasts
    forecasts = fitted_model.forecast(steps=len(test_data))
    
    # Calculate accuracy metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    mse = mean_squared_error(test_data, forecasts)
    mae = mean_absolute_error(test_data, forecasts)
    rmse = np.sqrt(mse)
    
    print(f"\nForecasting Accuracy:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Plot forecasts vs actual
    plt.figure(figsize=(12, 6))
    
    plt.plot(range(len(train_data)), train_data, label='Training Data', alpha=0.7)
    plt.plot(range(len(train_data), len(data)), test_data, label='Actual Test Data', alpha=0.7)
    plt.plot(range(len(train_data), len(data)), forecasts, 
             label='Forecasts', color='red', linestyle='--')
    
    plt.title('ARFIMA Forecasting Performance')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/figures/forecasting_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mse, mae, rmse


def main():
    """Run all demonstrations."""
    print("ARFIMA Model Implementation Demonstration")
    print("This script demonstrates the key features of the custom ARFIMA implementation.")
    
    # Create results directory if it doesn't exist
    os.makedirs('../results/figures', exist_ok=True)
    
    try:
        # Run demonstrations
        fitted_model, data = demo_basic_usage()
        models = demo_model_comparison()
        demo_fractional_differencing()
        mse, mae, rmse = demo_forecasting_accuracy()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey features demonstrated:")
        print("✓ ARFIMA model simulation")
        print("✓ Automatic order estimation")
        print("✓ Model fitting and parameter estimation")
        print("✓ Forecasting capabilities")
        print("✓ Model diagnostics and comparison")
        print("✓ Fractional differencing effects")
        print("✓ Forecasting accuracy evaluation")
        
        print(f"\nForecasting performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        print(f"\nFigures saved to: ../results/figures/")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

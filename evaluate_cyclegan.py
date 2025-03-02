import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from cycle_gan import CycleGAN, preprocess_data

def load_model_and_data(model_dir='models', 
                       model_name='cyclegan_final',
                       data_path='continuous_factory_process.csv',
                       column_info_path='column_info.json'):
    """
    Load a trained CycleGAN model and prepare test data
    
    Parameters:
    -----------
    model_dir : str
        Directory containing the model files
    model_name : str
        Prefix of the model files to load
    data_path : str
        Path to the original data CSV
    column_info_path : str
        Path to the JSON file with column information
    
    Returns:
    --------
    tuple : (model, test_data_dict)
    """
    # Load column information
    with open(column_info_path, 'r') as f:
        column_info = json.load(f)
    
    # Load original data
    df = pd.read_csv(data_path)
    
    # Get column lists
    input_cols = column_info['input_cols']
    stage1_cols = column_info['stage1_cols']
    stage2_cols = column_info['stage2_cols']
    
    # Preprocess data
    X_scaled, stage1_scaled, stage2_scaled, scalers = preprocess_data(
        df, stage1_cols, stage2_cols, input_cols
    )
    
    # Create a test set (using the last 15% of data)
    test_size = int(0.15 * X_scaled.shape[0])
    X_test = X_scaled[-test_size:]
    stage1_test = stage1_scaled[-test_size:]
    stage2_test = stage2_scaled[-test_size:]
    
    # Also get the original (unscaled) data for better visualization
    X_test_orig = df[input_cols].values[-test_size:]
    stage1_test_orig = df[stage1_cols].values[-test_size:]
    stage2_test_orig = df[stage2_cols].values[-test_size:]
    
    # Create a dummy initialization to reconstruct the model
    input_dims = X_scaled.shape[1]
    stage1_dims = stage1_scaled.shape[1]
    stage2_dims = stage2_scaled.shape[1]
    
    dummy_model = CycleGAN(
        input_dims=input_dims,
        stage1_dims=stage1_dims,
        stage2_dims=stage2_dims
    )
    
    # Load the trained weights
    model_path = os.path.join(model_dir, model_name)
    dummy_model.load_models(model_path)
    
    # Prepare test data dictionary
    test_data = {
        'X_test': X_test,
        'stage1_test': stage1_test,
        'stage2_test': stage2_test,
        'X_test_orig': X_test_orig,
        'stage1_test_orig': stage1_test_orig,
        'stage2_test_orig': stage2_test_orig,
        'scalers': scalers,
        'column_info': column_info
    }
    
    return dummy_model, test_data

def generate_predictions(model, test_data):
    """
    Generate predictions using the trained model
    
    Parameters:
    -----------
    model : CycleGAN
        The trained CycleGAN model
    test_data : dict
        Dictionary with test data
    
    Returns:
    --------
    dict : Dictionary with predictions
    """
    # Generate predictions
    pred_stage2 = model.predict_stage2(test_data['X_test'], test_data['stage1_test'])
    pred_stage1 = model.predict_stage1(test_data['X_test'], test_data['stage2_test'])
    
    # Generate cycle reconstructions
    cycle_stage1 = model.predict_stage1(test_data['X_test'], pred_stage2)
    cycle_stage2 = model.predict_stage2(test_data['X_test'], pred_stage1)
    
    # Inverse transform predictions to original scale
    stage1_scaler = test_data['scalers']['stage1']
    stage2_scaler = test_data['scalers']['stage2']
    
    pred_stage1_orig = stage1_scaler.inverse_transform(pred_stage1)
    pred_stage2_orig = stage2_scaler.inverse_transform(pred_stage2)
    cycle_stage1_orig = stage1_scaler.inverse_transform(cycle_stage1)
    cycle_stage2_orig = stage2_scaler.inverse_transform(cycle_stage2)
    
    # Assemble results
    predictions = {
        'pred_stage1': pred_stage1,
        'pred_stage2': pred_stage2,
        'cycle_stage1': cycle_stage1,
        'cycle_stage2': cycle_stage2,
        'pred_stage1_orig': pred_stage1_orig,
        'pred_stage2_orig': pred_stage2_orig,
        'cycle_stage1_orig': cycle_stage1_orig,
        'cycle_stage2_orig': cycle_stage2_orig
    }
    
    return predictions

def calculate_metrics(test_data, predictions):
    """
    Calculate evaluation metrics for the predictions
    
    Parameters:
    -----------
    test_data : dict
        Dictionary with test data
    predictions : dict
        Dictionary with model predictions
    
    Returns:
    --------
    dict : Dictionary with evaluation metrics
    """
    # Calculate metrics in scaled space
    stage1_mae = mean_absolute_error(test_data['stage1_test'], predictions['pred_stage1'])
    stage2_mae = mean_absolute_error(test_data['stage2_test'], predictions['pred_stage2'])
    
    stage1_mse = mean_squared_error(test_data['stage1_test'], predictions['pred_stage1'])
    stage2_mse = mean_squared_error(test_data['stage2_test'], predictions['pred_stage2'])
    
    cycle_s1_mae = mean_absolute_error(test_data['stage1_test'], predictions['cycle_stage1'])
    cycle_s2_mae = mean_absolute_error(test_data['stage2_test'], predictions['cycle_stage2'])
    
    # Calculate metrics in original space
    stage1_mae_orig = mean_absolute_error(test_data['stage1_test_orig'], predictions['pred_stage1_orig'])
    stage2_mae_orig = mean_absolute_error(test_data['stage2_test_orig'], predictions['pred_stage2_orig'])
    
    stage1_mse_orig = mean_squared_error(test_data['stage1_test_orig'], predictions['pred_stage1_orig'])
    stage2_mse_orig = mean_squared_error(test_data['stage2_test_orig'], predictions['pred_stage2_orig'])
    
    # Calculate R² scores for each measurement
    r2_stage1 = []
    r2_stage2 = []
    
    for i in range(test_data['stage1_test_orig'].shape[1]):
        r2_stage1.append(r2_score(test_data['stage1_test_orig'][:, i], predictions['pred_stage1_orig'][:, i]))
    
    for i in range(test_data['stage2_test_orig'].shape[1]):
        r2_stage2.append(r2_score(test_data['stage2_test_orig'][:, i], predictions['pred_stage2_orig'][:, i]))
    
    # Compile metrics
    metrics = {
        'stage1_mae': stage1_mae,
        'stage2_mae': stage2_mae,
        'stage1_mse': stage1_mse,
        'stage2_mse': stage2_mse,
        'cycle_s1_mae': cycle_s1_mae,
        'cycle_s2_mae': cycle_s2_mae,
        'stage1_mae_orig': stage1_mae_orig,
        'stage2_mae_orig': stage2_mae_orig,
        'stage1_mse_orig': stage1_mse_orig,
        'stage2_mse_orig': stage2_mse_orig,
        'r2_stage1': r2_stage1,
        'r2_stage2': r2_stage2,
        'avg_r2_stage1': np.mean(r2_stage1),
        'avg_r2_stage2': np.mean(r2_stage2)
    }
    
    return metrics

def plot_predictions(test_data, predictions, sample_indices=None, num_samples=3):
    """
    Plot sample predictions against ground truth
    
    Parameters:
    -----------
    test_data : dict
        Dictionary with test data
    predictions : dict
        Dictionary with model predictions
    sample_indices : list, optional
        Specific indices to plot. If None, random samples are chosen.
    num_samples : int
        Number of samples to plot (if sample_indices is None)
    """
    if sample_indices is None:
        # Choose random samples
        sample_indices = np.random.choice(len(test_data['stage1_test']), num_samples, replace=False)
    
    # Create a directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Get column names
    stage1_cols = test_data['column_info']['stage1_cols']
    stage2_cols = test_data['column_info']['stage2_cols']
    
    # Create nicer display names
    stage1_names = [col.replace('Stage1.Output.Measurement', 'S1.Meas').replace('.U.Actual', '') 
                   for col in stage1_cols]
    stage2_names = [col.replace('Stage2.Output.Measurement', 'S2.Meas').replace('.U.Actual', '') 
                   for col in stage2_cols]
    
    # Plot predictions for each sample
    for i, idx in enumerate(sample_indices):
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Stage 1 measurements
        ax = axes[0]
        ax.set_title(f'Sample {i+1}: Stage 1 Measurements (Primary Output)')
        
        # Ground truth
        ax.plot(stage1_names, test_data['stage1_test_orig'][idx], 'o-', label='Ground Truth')
        
        # Prediction
        ax.plot(stage1_names, predictions['pred_stage1_orig'][idx], 's--', label='Predicted')
        
        # Cycle reconstruction
        ax.plot(stage1_names, predictions['cycle_stage1_orig'][idx], '^:', label='Cycle Reconstruction')
        
        ax.set_xticks(range(len(stage1_names)))
        ax.set_xticklabels(stage1_names, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Stage 2 measurements
        ax = axes[1]
        ax.set_title(f'Sample {i+1}: Stage 2 Measurements (Secondary Output)')
        
        # Ground truth
        ax.plot(stage2_names, test_data['stage2_test_orig'][idx], 'o-', label='Ground Truth')
        
        # Prediction
        ax.plot(stage2_names, predictions['pred_stage2_orig'][idx], 's--', label='Predicted')
        
        # Cycle reconstruction
        ax.plot(stage2_names, predictions['cycle_stage2_orig'][idx], '^:', label='Cycle Reconstruction')
        
        ax.set_xticks(range(len(stage2_names)))
        ax.set_xticklabels(stage2_names, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/sample_{i+1}_predictions.png')
        plt.close()
    
    # Plot R² values for each measurement
    metrics = calculate_metrics(test_data, predictions)
    
    plt.figure(figsize=(14, 6))
    
    # Stage 1 R²
    plt.subplot(1, 2, 1)
    plt.bar(stage1_names, metrics['r2_stage1'])
    plt.title('R² Scores for Stage 1 Measurements')
    plt.xticks(rotation=45)
    plt.ylim([-0.5, 1.0])  # R² can be negative for poor fits
    plt.grid(True, alpha=0.3)
    
    # Stage 2 R²
    plt.subplot(1, 2, 2)
    plt.bar(stage2_names, metrics['r2_stage2'])
    plt.title('R² Scores for Stage 2 Measurements')
    plt.xticks(rotation=45)
    plt.ylim([-0.5, 1.0])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/r2_scores.png')
    plt.close()
    
    # Plot error distributions
    plt.figure(figsize=(14, 6))
    
    # Stage 1 errors
    errors_stage1 = test_data['stage1_test_orig'] - predictions['pred_stage1_orig']
    plt.subplot(1, 2, 1)
    plt.boxplot(errors_stage1, labels=stage1_names)
    plt.title('Error Distribution for Stage 1 Measurements')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Stage 2 errors
    errors_stage2 = test_data['stage2_test_orig'] - predictions['pred_stage2_orig']
    plt.subplot(1, 2, 2)
    plt.boxplot(errors_stage2, labels=stage2_names)
    plt.title('Error Distribution for Stage 2 Measurements')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/error_distributions.png')
    plt.close()

def plot_time_series(test_data, predictions, num_samples=200, measurement_indices=None):
    """
    Plot time series of predictions vs. ground truth
    
    Parameters:
    -----------
    test_data : dict
        Dictionary with test data
    predictions : dict
        Dictionary with model predictions
    num_samples : int
        Number of consecutive time steps to plot
    measurement_indices : list, optional
        Indices of specific measurements to plot. If None, a representative set is chosen.
    """
    if measurement_indices is None:
        # Use a set of 3 measurements for each stage as representative examples
        stage1_indices = [0, 7, 14]  # First, middle, last measurement
        stage2_indices = [0, 7, 14]  # First, middle, last measurement
    else:
        stage1_indices = measurement_indices
        stage2_indices = measurement_indices
    
    # Create a directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Get column names
    stage1_cols = test_data['column_info']['stage1_cols']
    stage2_cols = test_data['column_info']['stage2_cols']
    
    # Select a continuous segment of the test data
    start_idx = 0
    end_idx = min(start_idx + num_samples, len(test_data['stage1_test']))
    
    # Create time indices
    time_indices = np.arange(end_idx - start_idx)
    
    # Plot Stage 1 time series
    plt.figure(figsize=(15, 12))
    for i, idx in enumerate(stage1_indices):
        plt.subplot(3, 1, i+1)
        
        # Ground truth
        plt.plot(time_indices, test_data['stage1_test_orig'][start_idx:end_idx, idx], 
                'b-', label='Ground Truth')
        
        # Prediction
        plt.plot(time_indices, predictions['pred_stage1_orig'][start_idx:end_idx, idx], 
                'r--', label='Predicted')
        
        # Get measurement name
        meas_name = stage1_cols[idx].replace('Stage1.Output.Measurement', 'Meas').replace('.U.Actual', '')
        plt.title(f'Stage 1 - {meas_name} Time Series')
        plt.xlabel('Time Step')
        plt.ylabel('Measurement Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/stage1_time_series.png')
    plt.close()
    
    # Plot Stage 2 time series
    plt.figure(figsize=(15, 12))
    for i, idx in enumerate(stage2_indices):
        plt.subplot(3, 1, i+1)
        
        # Ground truth
        plt.plot(time_indices, test_data['stage2_test_orig'][start_idx:end_idx, idx], 
                'b-', label='Ground Truth')
        
        # Prediction
        plt.plot(time_indices, predictions['pred_stage2_orig'][start_idx:end_idx, idx], 
                'r--', label='Predicted')
        
        # Get measurement name
        meas_name = stage2_cols[idx].replace('Stage2.Output.Measurement', 'Meas').replace('.U.Actual', '')
        plt.title(f'Stage 2 - {meas_name} Time Series')
        plt.xlabel('Time Step')
        plt.ylabel('Measurement Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/stage2_time_series.png')
    plt.close()

def evaluate_and_visualize(model_dir='models', model_name='cyclegan_final'):
    """
    Main function to evaluate a trained CycleGAN model and visualize results
    
    Parameters:
    -----------
    model_dir : str
        Directory containing the model files
    model_name : str
        Prefix of the model files to load
    """
    print(f"Loading model from {model_dir}/{model_name}...")
    model, test_data = load_model_and_data(model_dir, model_name)
    
    print("Generating predictions...")
    predictions = generate_predictions(model, test_data)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(test_data, predictions)
    
    # Save metrics
    with open(f'{model_dir}/evaluation_metrics.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}
        json.dump(metrics_json, f, indent=2)
    
    # Print summary metrics
    print("\nEvaluation Metrics:")
    print(f"Stage 1 MAE: {metrics['stage1_mae_orig']:.4f}")
    print(f"Stage 2 MAE: {metrics['stage2_mae_orig']:.4f}")
    print(f"Stage 1 Average R²: {metrics['avg_r2_stage1']:.4f}")
    print(f"Stage 2 Average R²: {metrics['avg_r2_stage2']:.4f}")
    print(f"Cycle Consistency Stage 1 MAE: {metrics['cycle_s1_mae']:.4f}")
    print(f"Cycle Consistency Stage 2 MAE: {metrics['cycle_s2_mae']:.4f}")
    
    print("\nCreating visualization plots...")
    
    # Plot sample predictions
    print("Plotting sample predictions...")
    plot_predictions(test_data, predictions, num_samples=5)
    
    # Plot time series
    print("Plotting time series...")
    plot_time_series(test_data, predictions, num_samples=200)
    
    print(f"Evaluation complete. Visualizations saved to the 'plots' directory.")

if __name__ == "__main__":
    # You can change these parameters to evaluate different models
    evaluate_and_visualize(model_dir='models', model_name='cyclegan_final') 
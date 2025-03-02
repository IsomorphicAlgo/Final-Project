# LSTM Model for Manufacturing Process Data

This repository contains code for implementing a Long Short-Term Memory (LSTM) neural network to predict manufacturing process outputs based on time series data.

## Files

- `lstm_model.py`: Contains the core LSTM model implementation and helper functions
- `lstm_example.py`: Example script showing how to use the LSTM model with the manufacturing data

## Dataset

The dataset (`continuous_factory_process.csv`) contains time series data from a multi-stage manufacturing process with:
- Time stamps at 1Hz sampling rate
- Ambient conditions
- Raw material properties
- Machine process variables
- Output measurements from multiple stages

## How to Use

### Option 1: Run the Example Script

The simplest way to get started is to run the example script:

```bash
python lstm_example.py
```

This will:
1. Load the manufacturing data
2. Split it into training and testing sets
3. Prepare sequences for the LSTM model
4. Train the LSTM model
5. Evaluate the model on test data
6. Plot predictions vs actual values
7. Calculate RMSE for each output feature

### Option 2: Use in Jupyter Notebook

To use the LSTM model in your Jupyter notebook:

```python
# Import necessary functions
from lstm_model import (
    prepare_data, 
    train_lstm_model, 
    plot_training_history, 
    evaluate_model, 
    plot_predictions, 
    calculate_rmse
)

# Assuming train_df and test_df are already defined
X_train, y_train, X_test, y_test, scaler, output_cols, output_indices = prepare_data(train_df, test_df)

# Get input and output shapes
input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
output_shape = y_train.shape[1]  # Number of output features

# Train the model
lstm_model, history = train_lstm_model(X_train, y_train, input_shape, output_shape)

# Plot training history
plot_training_history(history)

# Evaluate the model
y_test_inv, pred_inv = evaluate_model(lstm_model, X_test, y_test, scaler, output_indices)

# Plot predictions
plot_predictions(y_test_inv, pred_inv, output_cols)

# Calculate RMSE
calculate_rmse(y_test_inv, pred_inv, output_cols)
```

## Model Architecture

The LSTM model architecture consists of:
1. First LSTM layer with 64 units and return sequences
2. Second LSTM layer with 32 units
3. Dropout layer (20% dropout rate)
4. Dense hidden layer with 32 units
5. Output layer with units matching the number of output features

The model is compiled with:
- Adam optimizer (learning rate = 0.001)
- Mean Squared Error (MSE) loss function
- Mean Absolute Error (MAE) metric

## Customization

You can customize various aspects of the model:

- Sequence length: Change the `seq_length` parameter in `prepare_data()`
- Model architecture: Modify the `create_lstm_model()` function
- Training parameters: Adjust epochs, batch size in `train_lstm_model()`
- Output features: By default, the model predicts Stage 2 output measurements

## Requirements

- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn 
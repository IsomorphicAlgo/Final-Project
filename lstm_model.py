import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import math

def prepare_data(data, target_column, sequence_length, train_size=0.8, val_size=0.1):
    """Prepare data for LSTM model"""
    # Extract target series
    series = data[target_column].values.reshape(-1, 1)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series)
    
    # Create sequences and labels
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    X, y = np.array(X), np.array(y)
    
    # Split into train, validation, and test sets
    train_size = int(len(X) * train_size)
    val_size = int(len(X) * val_size)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'sequence_length': sequence_length
    }

def train_lstm_model(data_dict, units=50, dropout=0.2, epochs=100, batch_size=32):
    """Train LSTM model"""
    # Build model
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, 
                  input_shape=(data_dict['X_train'].shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        data_dict['X_train'], data_dict['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(data_dict['X_val'], data_dict['y_val']),
        callbacks=[early_stop],
        verbose=1
    )
    
    return model, history

def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def evaluate_model(model, data_dict):
    """Evaluate model on test data"""
    y_pred = model.predict(data_dict['X_test'])
    
    # Inverse transform
    y_pred_inv = data_dict['scaler'].inverse_transform(y_pred)
    y_test_inv = data_dict['scaler'].inverse_transform(data_dict['y_test'])
    
    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print(f'Test RMSE: {rmse:.4f}')
    
    return rmse

def plot_predictions(model, data_dict, data, target_column):
    """Plot test predictions vs actual values"""
    # Generate predictions
    y_pred = model.predict(data_dict['X_test'])
    
    # Inverse transform
    y_pred_inv = data_dict['scaler'].inverse_transform(y_pred)
    y_test_inv = data_dict['scaler'].inverse_transform(data_dict['y_test'])
    
    # Create a DataFrame for plotting
    train_size = len(data_dict['X_train']) + data_dict['sequence_length']
    val_size = len(data_dict['X_val'])
    
    # Adjust index for test predictions
    test_index = range(train_size + val_size, train_size + val_size + len(y_pred_inv))
    
    plt.figure(figsize=(16, 8))
    plt.plot(data[target_column], label='Actual Data')
    plt.plot(test_index, y_pred_inv, label='LSTM Predictions', color='red')
    plt.title(f'LSTM Predictions vs Actual {target_column}')
    plt.xlabel('Time')
    plt.ylabel(target_column)
    plt.legend()
    plt.show()

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error"""
    return math.sqrt(mean_squared_error(y_true, y_pred)) 
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class CycleGAN:
    def __init__(self, 
                 input_dims,
                 stage1_dims=15,  # 15 primary measurements 
                 stage2_dims=15,  # 15 secondary measurements
                 gen_hidden_layers=[256, 128, 64],
                 disc_hidden_layers=[64, 32, 16],
                 gen_dropout_rate=0.2,
                 disc_dropout_rate=0.2,
                 gen_learning_rate=0.0002,
                 disc_learning_rate=0.0002,
                 lambda_cycle=10.0,
                 lambda_identity=1.0):
        """
        Initialize the CycleGAN model for manufacturing process prediction.
        
        Parameters:
        -----------
        input_dims : int
            Dimension of the input features (e.g., machine parameters, conditions)
        stage1_dims : int
            Dimension of Stage 1 measurements (primary outputs)
        stage2_dims : int
            Dimension of Stage 2 measurements (secondary outputs)
        gen_hidden_layers : list
            List of hidden layer sizes for the generators
        disc_hidden_layers : list
            List of hidden layer sizes for the discriminators
        gen_dropout_rate : float
            Dropout rate for the generator
        disc_dropout_rate : float
            Dropout rate for the discriminator
        gen_learning_rate : float
            Learning rate for the generator
        disc_learning_rate : float
            Learning rate for the discriminator
        lambda_cycle : float
            Weight for cycle consistency loss
        lambda_identity : float
            Weight for identity mapping loss
        """
        self.input_dims = input_dims
        self.stage1_dims = stage1_dims
        self.stage2_dims = stage2_dims
        
        self.gen_hidden_layers = gen_hidden_layers
        self.disc_hidden_layers = disc_hidden_layers
        
        self.gen_dropout_rate = gen_dropout_rate
        self.disc_dropout_rate = disc_dropout_rate
        
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        # Initialize optimizers
        self.gen_optimizer = Adam(learning_rate=gen_learning_rate, beta_1=0.5)
        self.disc_optimizer = Adam(learning_rate=disc_learning_rate, beta_1=0.5)
        
        # Build the models
        self._build_generators()
        self._build_discriminators()
        self._build_combined_models()
        
        # Initialize history tracking
        self.history = {
            'gen_loss': [],
            'disc_loss': [],
            'cycle_loss': [],
            'identity_loss': [],
            'g_s1_to_s2_loss': [],
            'g_s2_to_s1_loss': [],
            'd_s1_loss': [],
            'd_s2_loss': []
        }

    def _build_generators(self):
        """Build the generator models:
        1. G_S1_to_S2: Maps input + Stage 1 measurements to Stage 2 measurements
        2. G_S2_to_S1: Maps input + Stage 2 measurements to Stage 1 measurements
        """
        # Generator: Input + Stage 1 -> Stage 2
        input_features = Input(shape=(self.input_dims,), name='input_features_s1_to_s2')
        stage1_measurements = Input(shape=(self.stage1_dims,), name='stage1_measurements')
        
        # Concatenate inputs
        x = Concatenate()([input_features, stage1_measurements])
        
        # Build hidden layers
        for i, units in enumerate(self.gen_hidden_layers):
            x = Dense(units, name=f'g_s1_to_s2_dense_{i}')(x)
            x = LeakyReLU(0.2)(x)
            if self.gen_dropout_rate > 0:
                x = Dropout(self.gen_dropout_rate)(x)
        
        # Output layer
        stage2_pred = Dense(self.stage2_dims, activation='linear', name='stage2_predictions')(x)
        
        # Define the model
        self.G_S1_to_S2 = Model([input_features, stage1_measurements], stage2_pred, name='generator_s1_to_s2')
        
        # Generator: Input + Stage 2 -> Stage 1
        input_features = Input(shape=(self.input_dims,), name='input_features_s2_to_s1')
        stage2_measurements = Input(shape=(self.stage2_dims,), name='stage2_measurements')
        
        # Concatenate inputs
        x = Concatenate()([input_features, stage2_measurements])
        
        # Build hidden layers
        for i, units in enumerate(self.gen_hidden_layers):
            x = Dense(units, name=f'g_s2_to_s1_dense_{i}')(x)
            x = LeakyReLU(0.2)(x)
            if self.gen_dropout_rate > 0:
                x = Dropout(self.gen_dropout_rate)(x)
        
        # Output layer
        stage1_pred = Dense(self.stage1_dims, activation='linear', name='stage1_predictions')(x)
        
        # Define the model
        self.G_S2_to_S1 = Model([input_features, stage2_measurements], stage1_pred, name='generator_s2_to_s1')
        
        print("Generator models summary:")
        self.G_S1_to_S2.summary()
        self.G_S2_to_S1.summary()

    def _build_discriminators(self):
        """Build the discriminator models:
        1. D_S1: Determines if Stage 1 measurements are real or generated
        2. D_S2: Determines if Stage 2 measurements are real or generated
        """
        # Discriminator for Stage 1 measurements
        input_features = Input(shape=(self.input_dims,), name='input_features_d_s1')
        stage1_measurements = Input(shape=(self.stage1_dims,), name='stage1_measurements_disc')
        
        # Concatenate inputs
        x = Concatenate()([input_features, stage1_measurements])
        
        # Build hidden layers
        for i, units in enumerate(self.disc_hidden_layers):
            x = Dense(units, name=f'd_s1_dense_{i}')(x)
            x = LeakyReLU(0.2)(x)
            if self.disc_dropout_rate > 0:
                x = Dropout(self.disc_dropout_rate)(x)
        
        # Output layer - single neuron with sigmoid for real/fake classification
        validity = Dense(1, activation='sigmoid', name='d_s1_validity')(x)
        
        # Define the model
        self.D_S1 = Model([input_features, stage1_measurements], validity, name='discriminator_s1')
        self.D_S1.compile(loss='binary_crossentropy', optimizer=self.disc_optimizer, metrics=['accuracy'])
        
        # Discriminator for Stage 2 measurements
        input_features = Input(shape=(self.input_dims,), name='input_features_d_s2')
        stage2_measurements = Input(shape=(self.stage2_dims,), name='stage2_measurements_disc')
        
        # Concatenate inputs
        x = Concatenate()([input_features, stage2_measurements])
        
        # Build hidden layers
        for i, units in enumerate(self.disc_hidden_layers):
            x = Dense(units, name=f'd_s2_dense_{i}')(x)
            x = LeakyReLU(0.2)(x)
            if self.disc_dropout_rate > 0:
                x = Dropout(self.disc_dropout_rate)(x)
        
        # Output layer - single neuron with sigmoid for real/fake classification
        validity = Dense(1, activation='sigmoid', name='d_s2_validity')(x)
        
        # Define the model
        self.D_S2 = Model([input_features, stage2_measurements], validity, name='discriminator_s2')
        self.D_S2.compile(loss='binary_crossentropy', optimizer=self.disc_optimizer, metrics=['accuracy'])
        
        print("Discriminator models summary:")
        self.D_S1.summary()
        self.D_S2.summary()

    def _build_combined_models(self):
        """
        Build the combined models for cycle consistency training
        """
        # Input shapes
        input_features = Input(shape=(self.input_dims,), name='input_features_combined')
        stage1_measurements = Input(shape=(self.stage1_dims,), name='stage1_measurements_combined')
        stage2_measurements = Input(shape=(self.stage2_dims,), name='stage2_measurements_combined')
        
        # Freeze the discriminators during generator training
        self.D_S1.trainable = False
        self.D_S2.trainable = False
        
        # Cycle S1 -> S2 -> S1
        fake_stage2 = self.G_S1_to_S2([input_features, stage1_measurements])
        valid_stage2 = self.D_S2([input_features, fake_stage2])
        reconstructed_stage1 = self.G_S2_to_S1([input_features, fake_stage2])
        
        # Cycle S2 -> S1 -> S2
        fake_stage1 = self.G_S2_to_S1([input_features, stage2_measurements])
        valid_stage1 = self.D_S1([input_features, fake_stage1])
        reconstructed_stage2 = self.G_S1_to_S2([input_features, fake_stage1])
        
        # Identity mapping
        identity_stage1 = self.G_S2_to_S1([input_features, stage1_measurements])
        identity_stage2 = self.G_S1_to_S2([input_features, stage2_measurements])
        
        # Combined model
        self.combined = Model(
            inputs=[input_features, stage1_measurements, stage2_measurements],
            outputs=[
                valid_stage1, valid_stage2,                     # Adversarial loss
                reconstructed_stage1, reconstructed_stage2,     # Cycle consistency loss
                identity_stage1, identity_stage2                # Identity loss
            ],
            name='combined_cyclegan'
        )
        
        # Compile the combined model
        self.combined.compile(
            loss=['binary_crossentropy', 'binary_crossentropy',  # Adversarial loss
                 'mae', 'mae',                                   # Cycle loss
                 'mae', 'mae'],                                  # Identity loss
            loss_weights=[1, 1,                                  # Adversarial loss
                         self.lambda_cycle, self.lambda_cycle,   # Cycle loss
                         self.lambda_identity, self.lambda_identity],  # Identity loss
            optimizer=self.gen_optimizer
        )
        
        print("Combined model created")

    def train(self, 
              input_features, 
              stage1_measurements, 
              stage2_measurements,
              epochs=100, 
              batch_size=32, 
              validate_every=10,
              sample_interval=10, 
              save_model_interval=50,
              validation_data=None):
        """
        Train the CycleGAN model
        
        Parameters:
        -----------
        input_features : array
            Input features (process parameters)
        stage1_measurements : array
            Stage 1 measurements (primary outputs)
        stage2_measurements : array
            Stage 2 measurements (secondary outputs)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validate_every : int
            How often to validate
        sample_interval : int
            How often to show sample predictions
        save_model_interval : int
            How often to save the model
        validation_data : tuple
            (val_input_features, val_stage1, val_stage2) tuple for validation
        """
        # Labels for adversarial training
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Start training
        for epoch in range(epochs):
            # Get a random batch of samples
            idx = np.random.randint(0, input_features.shape[0], batch_size)
            batch_input = input_features[idx]
            batch_stage1 = stage1_measurements[idx]
            batch_stage2 = stage2_measurements[idx]
            
            # --------------------
            # Train Discriminators
            # --------------------
            
            # Generate fake Stage 2 measurements
            fake_stage2 = self.G_S1_to_S2.predict([batch_input, batch_stage1])
            
            # Train the Stage 2 discriminator
            d_s2_loss_real = self.D_S2.train_on_batch([batch_input, batch_stage2], valid)
            d_s2_loss_fake = self.D_S2.train_on_batch([batch_input, fake_stage2], fake)
            d_s2_loss = 0.5 * np.add(d_s2_loss_real, d_s2_loss_fake)
            
            # Generate fake Stage 1 measurements
            fake_stage1 = self.G_S2_to_S1.predict([batch_input, batch_stage2])
            
            # Train the Stage 1 discriminator
            d_s1_loss_real = self.D_S1.train_on_batch([batch_input, batch_stage1], valid)
            d_s1_loss_fake = self.D_S1.train_on_batch([batch_input, fake_stage1], fake)
            d_s1_loss = 0.5 * np.add(d_s1_loss_real, d_s1_loss_fake)
            
            # --------------------
            # Train Generators
            # --------------------
            
            # Train the generators and combined model
            g_loss = self.combined.train_on_batch(
                [batch_input, batch_stage1, batch_stage2],
                [valid, valid,                                      # Adversarial loss
                 batch_stage1, batch_stage2,                        # Cycle loss
                 batch_stage1, batch_stage2]                        # Identity loss
            )
            
            # Store losses
            self.history['disc_loss'].append(0.5 * (d_s1_loss[0] + d_s2_loss[0]))
            self.history['gen_loss'].append(g_loss[0])
            self.history['cycle_loss'].append(g_loss[3] + g_loss[4])
            self.history['identity_loss'].append(g_loss[5] + g_loss[6])
            self.history['g_s1_to_s2_loss'].append(g_loss[1])
            self.history['g_s2_to_s1_loss'].append(g_loss[2])
            self.history['d_s1_loss'].append(d_s1_loss[0])
            self.history['d_s2_loss'].append(d_s2_loss[0])
            
            # Print the progress
            if epoch % sample_interval == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"D_S1 Loss: {d_s1_loss[0]:.4f}, Acc: {d_s1_loss[1]:.4f}")
                print(f"D_S2 Loss: {d_s2_loss[0]:.4f}, Acc: {d_s2_loss[1]:.4f}")
                print(f"G Loss: {g_loss[0]:.4f}")
                print(f"Cycle Loss: {g_loss[3] + g_loss[4]:.4f}")
                print(f"Identity Loss: {g_loss[5] + g_loss[6]:.4f}")
                print("-" * 50)
            
            # Validate if requested
            if validation_data is not None and epoch % validate_every == 0:
                self._validate_model(*validation_data)
            
            # Save the model periodically
            if epoch % save_model_interval == 0 and epoch > 0:
                self.save_models(f"cyclegan_epoch_{epoch}")
    
    def _validate_model(self, val_input, val_stage1, val_stage2):
        """
        Validate the model on validation data
        
        Parameters:
        -----------
        val_input : array
            Validation input features
        val_stage1 : array
            Validation Stage 1 measurements
        val_stage2 : array
            Validation Stage 2 measurements
        """
        # Generate predictions
        pred_stage2 = self.G_S1_to_S2.predict([val_input, val_stage1])
        pred_stage1 = self.G_S2_to_S1.predict([val_input, val_stage2])
        
        # Calculate MAE
        stage1_mae = np.mean(np.abs(val_stage1 - pred_stage1))
        stage2_mae = np.mean(np.abs(val_stage2 - pred_stage2))
        
        print("Validation Results:")
        print(f"Stage 1 -> Stage 2 MAE: {stage2_mae:.4f}")
        print(f"Stage 2 -> Stage 1 MAE: {stage1_mae:.4f}")
        print("-" * 50)
    
    def predict_stage2(self, input_features, stage1_measurements):
        """
        Predict Stage 2 measurements from input features and Stage 1 measurements
        
        Parameters:
        -----------
        input_features : array
            Input features (process parameters)
        stage1_measurements : array
            Stage 1 measurements (primary outputs)
        
        Returns:
        --------
        array : Predicted Stage 2 measurements
        """
        return self.G_S1_to_S2.predict([input_features, stage1_measurements])
    
    def predict_stage1(self, input_features, stage2_measurements):
        """
        Predict Stage 1 measurements from input features and Stage 2 measurements
        
        Parameters:
        -----------
        input_features : array
            Input features (process parameters)
        stage2_measurements : array
            Stage 2 measurements (secondary outputs)
        
        Returns:
        --------
        array : Predicted Stage 1 measurements
        """
        return self.G_S2_to_S1.predict([input_features, stage2_measurements])
    
    def save_models(self, model_name_prefix="cyclegan"):
        """
        Save all models (generators, discriminators, combined)
        
        Parameters:
        -----------
        model_name_prefix : str
            Prefix for the model filenames
        """
        self.G_S1_to_S2.save(f"{model_name_prefix}_g_s1_to_s2.keras")
        self.G_S2_to_S1.save(f"{model_name_prefix}_g_s2_to_s1.keras")
        self.D_S1.save(f"{model_name_prefix}_d_s1.keras")
        self.D_S2.save(f"{model_name_prefix}_d_s2.keras")
        
        # Save weights separately for the combined model (can't save the entire model easily)
        # TensorFlow requires the .weights.h5 format
        self.combined.save_weights(f"{model_name_prefix}_combined.weights.h5")
        
        # Save history
        np.save(f"{model_name_prefix}_history.npy", self.history)
        
    def load_models(self, model_name_prefix="cyclegan"):
        """
        Load saved models
        
        Parameters:
        -----------
        model_name_prefix : str
            Prefix for the model filenames
        """
        self.G_S1_to_S2 = tf.keras.models.load_model(f"{model_name_prefix}_g_s1_to_s2.keras")
        self.G_S2_to_S1 = tf.keras.models.load_model(f"{model_name_prefix}_g_s2_to_s1.keras")
        self.D_S1 = tf.keras.models.load_model(f"{model_name_prefix}_d_s1.keras")
        self.D_S2 = tf.keras.models.load_model(f"{model_name_prefix}_d_s2.keras")
        
        # Rebuild and load weights for the combined model
        self._build_combined_models()
        # Use the same .weights.h5 format for consistency
        self.combined.load_weights(f"{model_name_prefix}_combined.weights.h5")
        
        # Load history
        try:
            self.history = np.load(f"{model_name_prefix}_history.npy", allow_pickle=True).item()
        except:
            print("Could not load training history")
    
    def plot_history(self):
        """
        Plot the training history
        """
        plt.figure(figsize=(15, 10))
        
        # Plot generator and discriminator losses
        plt.subplot(2, 2, 1)
        plt.plot(self.history['gen_loss'], label='Generator Loss')
        plt.plot(self.history['disc_loss'], label='Discriminator Loss')
        plt.title('Generator and Discriminator Loss')
        plt.legend()
        
        # Plot cycle and identity losses
        plt.subplot(2, 2, 2)
        plt.plot(self.history['cycle_loss'], label='Cycle Loss')
        plt.plot(self.history['identity_loss'], label='Identity Loss')
        plt.title('Cycle and Identity Losses')
        plt.legend()
        
        # Plot individual generator losses
        plt.subplot(2, 2, 3)
        plt.plot(self.history['g_s1_to_s2_loss'], label='G: S1 -> S2')
        plt.plot(self.history['g_s2_to_s1_loss'], label='G: S2 -> S1')
        plt.title('Generator Losses by Direction')
        plt.legend()
        
        # Plot individual discriminator losses
        plt.subplot(2, 2, 4)
        plt.plot(self.history['d_s1_loss'], label='D: S1')
        plt.plot(self.history['d_s2_loss'], label='D: S2')
        plt.title('Discriminator Losses')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('cyclegan_training_history.png')
        plt.show()
        
    def evaluate_model(self, input_features, stage1_measurements, stage2_measurements):
        """
        Evaluate the model performance
        
        Parameters:
        -----------
        input_features : array
            Input features
        stage1_measurements : array
            Stage 1 measurements (ground truth)
        stage2_measurements : array
            Stage 2 measurements (ground truth)
            
        Returns:
        --------
        dict : Dictionary with evaluation metrics
        """
        # Generate predictions
        pred_stage2 = self.predict_stage2(input_features, stage1_measurements)
        pred_stage1 = self.predict_stage1(input_features, stage2_measurements)
        
        # Calculate MAE and MSE
        stage1_mae = np.mean(np.abs(stage1_measurements - pred_stage1))
        stage2_mae = np.mean(np.abs(stage2_measurements - pred_stage2))
        
        stage1_mse = np.mean(np.square(stage1_measurements - pred_stage1))
        stage2_mse = np.mean(np.square(stage2_measurements - pred_stage2))
        
        # Calculate cycle consistency
        reconstructed_stage1 = self.predict_stage1(input_features, pred_stage2)
        reconstructed_stage2 = self.predict_stage2(input_features, pred_stage1)
        
        cycle_s1_mae = np.mean(np.abs(stage1_measurements - reconstructed_stage1))
        cycle_s2_mae = np.mean(np.abs(stage2_measurements - reconstructed_stage2))
        
        # Return metrics
        return {
            'stage1_mae': stage1_mae,
            'stage2_mae': stage2_mae,
            'stage1_mse': stage1_mse,
            'stage2_mse': stage2_mse,
            'cycle_s1_mae': cycle_s1_mae,
            'cycle_s2_mae': cycle_s2_mae
        }

# Utility functions for data preprocessing
def preprocess_data(df, stage1_cols, stage2_cols, input_cols=None):
    """
    Preprocess the manufacturing data for CycleGAN training
    
    Parameters:
    -----------
    df : DataFrame
        The original manufacturing data
    stage1_cols : list
        Column names for Stage 1 outputs
    stage2_cols : list
        Column names for Stage 2 outputs
    input_cols : list
        Column names for input features (if None, all non-stage columns are used)
    
    Returns:
    --------
    tuple : (input_features, stage1_data, stage2_data, scalers)
    """
    # If input columns not specified, use all except stage columns
    if input_cols is None:
        all_stage_cols = stage1_cols + stage2_cols
        input_cols = [col for col in df.columns if col not in all_stage_cols]
    
    # Extract data
    X = df[input_cols].values
    stage1_data = df[stage1_cols].values
    stage2_data = df[stage2_cols].values
    
    # Create scalers
    x_scaler = StandardScaler()
    stage1_scaler = StandardScaler()
    stage2_scaler = StandardScaler()
    
    # Fit and transform
    X_scaled = x_scaler.fit_transform(X)
    stage1_scaled = stage1_scaler.fit_transform(stage1_data)
    stage2_scaled = stage2_scaler.fit_transform(stage2_data)
    
    scalers = {
        'input': x_scaler,
        'stage1': stage1_scaler,
        'stage2': stage2_scaler
    }
    
    return X_scaled, stage1_scaled, stage2_scaled, scalers

def train_test_validation_split(X, stage1, stage2, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train, test, and validation sets
    
    Parameters:
    -----------
    X : array
        Input features
    stage1 : array
        Stage 1 measurements
    stage2 : array
        Stage 2 measurements
    test_size : float
        Fraction of data to use for testing
    val_size : float
        Fraction of data to use for validation
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X_train, stage1_train, stage2_train, X_val, stage1_val, stage2_val, X_test, stage1_test, stage2_test)
    """
    # First split: separate test set
    X_temp, X_test, stage1_temp, stage1_test, stage2_temp, stage2_test = train_test_split(
        X, stage1, stage2, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation set from temporary training set
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, stage1_train, stage1_val, stage2_train, stage2_val = train_test_split(
        X_temp, stage1_temp, stage2_temp, test_size=val_ratio, random_state=random_state
    )
    
    return (X_train, stage1_train, stage2_train, 
            X_val, stage1_val, stage2_val, 
            X_test, stage1_test, stage2_test) 
# CycleGAN for Multi-Stage Manufacturing Process Prediction

This implementation uses a CycleGAN architecture to predict measurements in a multi-stage manufacturing process. The model is designed to translate between two domains: Stage 1 (primary) measurements and Stage 2 (secondary) measurements.

## Problem Description

In this manufacturing process:
- The first stage consists of Machines 1, 2, and 3 operating in parallel, feeding into a combiner
- The output from the combiner is measured in 15 locations (Stage 1 measurements)
- The flow then enters a second stage, where Machines 4 and 5 process in series
- Measurements are made again in the same 15 locations (Stage 2 measurements)

The goal is to predict:
1. Stage 1 measurements from Stage 2 measurements and process parameters
2. Stage 2 measurements from Stage 1 measurements and process parameters

This bidirectional prediction capability is what makes CycleGAN an ideal architecture for this problem.

## CycleGAN Architecture

The implemented CycleGAN consists of:

1. **Two Generator Networks**:
   - G_S1_to_S2: Transforms input features + Stage 1 measurements → Stage 2 measurements
   - G_S2_to_S1: Transforms input features + Stage 2 measurements → Stage 1 measurements

2. **Two Discriminator Networks**:
   - D_S1: Determines if Stage 1 measurements are real or generated
   - D_S2: Determines if Stage 2 measurements are real or generated

3. **Loss Functions**:
   - Adversarial loss: Makes generated measurements indistinguishable from real ones
   - Cycle consistency loss: Ensures that translating measurements from one stage to another and back results in the original measurements
   - Identity loss: Helps preserve characteristics of the input domain

## Files Structure

- `cycle_gan.py`: Main CycleGAN model implementation with all necessary components
- `train_cyclegan.py`: Script for training and tuning the CycleGAN on manufacturing data
- `evaluate_cyclegan.py`: Utilities for evaluating the trained model and visualizing results
- `continuous_factory_process.csv`: Original dataset (provided separately)

## How to Use

### Training a Model

To train a CycleGAN model on the manufacturing data:

```bash
python train_cyclegan.py
```

This will:
1. Load and preprocess the manufacturing data
2. Split the data into training, validation, and test sets
3. Train the CycleGAN model
4. Save the trained model to the `models` directory
5. Evaluate performance on the test set

### Hyperparameter Tuning

The training script includes a commented-out section for hyperparameter tuning. To enable it, uncomment the relevant code in `train_cyclegan.py`.

Key hyperparameters for tuning:
- Network architecture (layer sizes)
- Learning rates
- Dropout rates
- Cycle consistency and identity loss weights

### Evaluating and Visualizing

To evaluate a trained model and generate visualizations:

```bash
python evaluate_cyclegan.py
```

This will:
1. Load the trained model
2. Generate predictions on the test set
3. Calculate evaluation metrics
4. Create visualizations in the `plots` directory, including:
   - Sample predictions vs. ground truth
   - Time series plots
   - R² scores for each measurement
   - Error distributions

## Model Performance

The performance of the model can be evaluated using several metrics:

- Mean Absolute Error (MAE): Average absolute difference between predictions and ground truth
- Mean Squared Error (MSE): Average squared difference between predictions and ground truth
- R² Score: Proportion of variance in the output that is predictable from the input
- Cycle Consistency Error: Error after a full cycle transformation (S1→S2→S1 or S2→S1→S2)

## Requirements

- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Acknowledgments

- Data source: https://www.kaggle.com/datasets/supergus/multistage-continuousflow-manufacturing-process/data
- Implementation based on the CycleGAN paper: https://arxiv.org/abs/1703.10593 
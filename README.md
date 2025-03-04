# Multi-Stage Manufacturing Deep Learning Analysis

## Overview
This project explores the application of deep learning techniques for anomaly detection and attribute prediction in multi-stage manufacturing processes. The project utilizes two main approaches:
1. Long Short-Term Memory (LSTM) networks for time-series prediction
2. Cycle-Consistent Adversarial Networks (CycleGAN) for bidirectional mapping between manufacturing stages

## Problem Statement
As manufacturing resources become finite and global product demand grows, there is an increasing need to leverage new technologies and improve manufacturing processes. This project aims to explore deep learning methods for predicting manufacturing process outcomes and detecting anomalies in real-time.

## Data Description
The dataset comes from a continuous flow manufacturing process with the following characteristics:
- Sample rate: 1 Hz
- First stage: Three parallel machines (1, 2, 3) feeding into a combiner
- Second stage: Two machines (4, 5) in series
- Measurements: 15 locations in both stages
- Data includes setpoints and actual values for controlled and uncontrolled variables

### Data Structure
- Time stamp
- Factory ambient conditions (2 variables)
- Machine 1-3 raw material properties and process variables
- Combiner stage parameters
- Primary output measurements (15 features)
- Machine 4-5 process variables
- Secondary output measurements (15 features)

## Methodology

### LSTM Model
- Designed for sequence-based prediction
- Architecture:
  - Multiple LSTM layers
  - Dropout for regularization
  - Dense layers for output
- Optimized with Adam optimizer
- Early stopping to prevent overfitting

### CycleGAN Model
- Focuses on bidirectional mapping between manufacturing stages
- Configuration:
  - Generator and Discriminator networks
  - Cycle consistency loss
  - Identity mapping loss
- Hyperparameters:
  - Learning rates: 0.0002
  - Batch size: 64
  - Hidden layer configurations
  - Dropout rates

## Results

### LSTM Performance
- Test Loss (MSE): 0.0375
- Test MAE: 0.0881
- Feature-specific RMSE:
  - Best: Stage 1 Measurement 12 (RMSE = 0.0855)
  - Worst: Stage 2 Measurement 9 (RMSE = 14.7440)

### CycleGAN Performance
- Stage 1 MAE: 0.4054
- Stage 2 MAE: 0.3771
- Stage 1 MSE: 0.7778
- Stage 2 MSE: 0.7909
- Cycle Consistency MAE:
  - Stage 1: 0.3852
  - Stage 2: 0.3598

## Conclusions
The project demonstrates the effectiveness of deep learning in manufacturing process prediction:
- LSTM shows strong performance for specific measurements but varies across features
- CycleGAN provides more consistent performance across all measurements
- Combined approach recommended for practical applications:
  - LSTM for general prediction tasks
  - CycleGAN for understanding stage relationships and consistent predictions

## Dependencies
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Other requirements listed in requirements.txt

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the Jupyter notebook: `jupyter notebook`
3. Execute cells in sequence for full analysis

## License

## Contact
Michael Hansen
University of Colorado

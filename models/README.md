# SpeedTransformer Models - Clean Directory

This directory contains the essential files from the SpeedTransformer experiments, optimized for GitHub repository sharing.

## Structure

- **Source Code**: All `.py` and `.sh` files
- **Experiment Logs**: Training and evaluation logs (`.log` files)
- **Visualizations**: Confusion matrices and analysis plots (`.png` files)
- **Notebooks**: Data analysis and modeling notebooks (`.ipynb` files)

## Excluded Files

To reduce repository size, the following files have been excluded:
- Model checkpoints (`.pth` files)
- Preprocessed data (`.joblib`, `.pkl` files)
- Python cache directories (`__pycache__`)

## Size Comparison

- **Original directory**: 3.2 GB
- **Clean directory**: 6.7 MB
- **Space saved**: 99.8%

## Models Included

### Transformer
- Hyperparameter sweeps for Geolife and Mobis datasets
- Fine-tuning experiments with various strategies
- Window size optimization sweeps
- Miniprogram fine-tuning experiments

### LSTM
- Baseline implementations for comparison
- Fine-tuning experiments across different data percentages
- Cross-dataset transfer learning experiments

### Random Forest
- Traditional ML baseline for comparison
- Feature engineering and evaluation results

## Usage

To reproduce experiments, refer to the shell scripts (`.sh` files) and training logs for hyperparameter configurations and training procedures.

## Data Analysis

The `data_analysis.ipynb` notebook contains comprehensive visualizations and analysis of experimental results with consistent theming and styling.

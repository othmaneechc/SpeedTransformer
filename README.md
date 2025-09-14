# SpeedTransformer

This repository contains the code used in the paper **"[Predicting Human Mobility Using Dense Smartphone GPS Trajectories and Transformer Models](#)"**. 

## Table of Contents

1. [Requirements](#requirements)
2. [Preparing the Data](#preparing-the-data)  
   - [Geolife Dataset](#geolife-dataset)  
   - [MOBIS Dataset](#mobis-dataset)  
3. [Running the Models](#running-the-models)  
   - [LSTM Model](#lstm-model)   
   - [Transformer Model](#transformer-model)  
4. [Replicating Results](#replicating-results)

---

## Requirements

## Preparing the Data

### Geolife Dataset

The Geolife dataset provides GPS trajectories collected from users. To preprocess this dataset:

1. **Download the Dataset**

   - Obtain the Geolife GPS trajectory dataset from [Microsoft Research](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/).
   - Unzip the dataset to a directory on your machine.

2. **Run the Preprocessing Script**

   Use the `data/geolife.py` script to process the data. This script utilizes multiprocessing for efficient processing and typically completes in under 20 minutes:

   ```bash
   python process_geolife.py --data-folder "Geolife Trajectories 1.3/Data" --output-file "geolife.csv"
   ```
3. **Post-Processing** 

After preprocessing, run `extract_speed_geolife.py` to compute additional features like speed and distance:

```bash
python extract_speed_geolife.py geolife.csv --output_file geolife_processed.csv
``` 

### MOBIS Dataset

_The MOBIS dataset can be processed using a similar method. The processed MOBIS data can be found here: https://zenodo.org/records/15530797_

## Running the Models
This repository provides two primary model architectures:

- LSTM-based trip classification (`models/lstm/`).
- Transformer-based trip classification (`models/transformer/`).

Each architecture includes dedicated scripts for training and fine-tuning. The following shell scripts are available:

### Shell Scripts Overview

#### Transformer Scripts (`models/transformer/`)

- **`run_sweep.sh`** - Comprehensive hyperparameter sweep across learning rates, batch sizes, model dimensions, attention heads, and dropout rates
- **`ws_sweep.sh`** - Window size optimization sweep to find optimal trajectory sequence lengths (tests 20, 50, 100, 200, 300, 400, 500)
- **`finetune.sh`** - Transfer learning from MOBIS pretrained model to Geolife with multiple fine-tuning strategies (full, layer freezing, gradual unfreezing)
- **`finetune_miniprogram.sh`** - Specialized fine-tuning for miniprogram (WeChat) mobility data with various data subset sizes (15%, 20%, 30%, 40%, 50%)

#### LSTM Scripts (`models/lstm/`)

- **`run_sweep.sh`** - Comprehensive hyperparameter sweep across learning rates, batch sizes, hidden dimensions, layer counts, and dropout rates
- **`finetune.sh`** - Enhanced transfer learning from MOBIS to Geolife with comprehensive hyperparameter sweeps and smart dependency waiting
- **`finetune_miniprogram.sh`** - Specialized fine-tuning for miniprogram (WeChat) mobility data with LSTM architecture and various data subset sizes (15%, 20%, 30%, 40%, 50%)

**Note**: LSTM models use a fixed sequence length of 200 frames (hardcoded in the model architecture), so window size optimization is not applicable.

### Script Usage Guide

1. **Hyperparameter Optimization**: Use `run_sweep.sh` for automated parameter search across multiple dimensions
2. **Sequence Length Tuning**: Use `ws_sweep.sh` (transformer only) to determine optimal window sizes for trajectory segmentation
3. **Transfer Learning**: Use `finetune.sh` for cross-dataset adaptation or `finetune_miniprogram.sh` for WeChat data
4. **Baseline Comparison**: Run LSTM scripts to establish traditional sequence model benchmarks

---

#### Experiment Types

**Transformer Models** (`models/transformer/`):
1. **Basic Training**: Use `train.sh` scripts with specified random seeds
2. **Hyperparameter Optimization**: Run `run_sweep.sh` for automated parameter search across attention heads, model dimensions, and learning rates
3. **Window Size Analysis**: Execute `ws_sweep.sh` for sequence length optimization (20-500 windows)
4. **Transfer Learning**: Use `finetune.sh` for MOBIS→Geolife cross-dataset experiments
5. **Data Efficiency**: Run `finetune_miniprogram.sh` for miniprogram subset size analysis (15%-50% data)

**LSTM Models** (`models/lstm/`):
1. **Comprehensive Sweeps**: `run_sweep.sh` performs exhaustive hyperparameter search across hidden dimensions (64-256), layer counts (2-3), and learning rates (1e-3 to 2e-3)
2. **Window Optimization**: `ws_sweep.sh` finds optimal sequence lengths for LSTM memory efficiency
3. **Enhanced Transfer Learning**: `finetune.sh` includes learning rate and hidden dimension grid search for MOBIS→Geolife transfer
4. **Advanced Miniprogram Experiments**: `finetune_miniprogram.sh` combines hyperparameter tuning with data subset analysis, plus automated summary generation

#### Model Comparison Framework

- **Transformer Scripts**: Focus on attention mechanisms, multi-head configurations, and model depth
- **LSTM Scripts**: Emphasize memory cell optimization, hidden state dimensions, and recurrent layer stacking
- **Shared Features**: Both model types use identical random seeds, data preprocessing, and evaluation metrics for fair comparison

The provided shell scripts ensure reproducible experiments with consistent configurations. All experiment logs, model checkpoints, and performance metrics are preserved in organized subdirectories under `models/`.

---

#### Quick Start Guide

**Run All Transformer Experiments:**
```bash
cd models/transformer
tmux new-session -d -s transformer_experiments
tmux send-keys "cd /data/A-SpeedTransformer/models/transformer" C-m
tmux send-keys "./run_sweep.sh" C-m
# Add more windows for parallel execution
tmux new-window -t transformer_experiments
tmux send-keys "cd /data/A-SpeedTransformer/models/transformer && ./ws_sweep.sh" C-m
```

**Run All LSTM Experiments:**
```bash
cd models/lstm  
tmux new-session -d -s lstm_experiments
tmux send-keys "cd /data/A-SpeedTransformer/models/lstm" C-m
tmux send-keys "./run_sweep.sh" C-m
# Parallel window execution
tmux new-window -t lstm_experiments
tmux send-keys "cd /data/A-SpeedTransformer/models/lstm && ./ws_sweep.sh" C-m
```

**Monitor Progress:**
```bash
tmux list-sessions
tmux attach-session -t lstm_experiments
tmux attach-session -t transformer_experiments
```

**Note**: Make sure to use the correct model checkpoints and data paths when running the scripts!

## License & Contact

This project is licensed under the MIT License. Feel free to open issues or pull requests on GitHub.
For questions or contributions, please reach out to [Othmane Echchabi](mailto:othmane.echchabi@mail.mcgill.ca).


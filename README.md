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

- **`train.sh`** - Basic LSTM model training for baseline comparisons on Geolife and MOBIS datasets
- **`finetune.sh`** - LSTM transfer learning experiments from MOBIS to Geolife for baseline comparison

### Script Usage Guide

1. **New Dataset Training**: Use `train.sh` for initial model training with standard configurations
2. **Hyperparameter Optimization**: Use `run_sweep.sh` for automated parameter search across multiple dimensions
3. **Sequence Length Tuning**: Use `ws_sweep.sh` to determine optimal window sizes for trajectory segmentation
4. **Transfer Learning**: Use `finetune.sh` for cross-dataset adaptation or `finetune_miniprogram.sh` for WeChat data
5. **Baseline Comparison**: Run LSTM scripts to establish traditional sequence model benchmarks

---

#### Experiment Types

1. **Basic Training**: Use `train.sh` scripts with the specified random seeds
2. **Hyperparameter Optimization**: Run `run_sweep.sh` for automated parameter search  
3. **Window Size Analysis**: Execute `ws_sweep.sh` for sequence length optimization
4. **Transfer Learning**: Use `finetune.sh` for cross-dataset experiments
5. **Data Efficiency**: Run `finetune_miniprogram.sh` for subset size analysis

The provided shell scripts ensure the same random seeds and configurations are used to replicate the reported accuracy and performance metrics. All experiment logs and configurations are preserved in the `models/` directory structure.

**Note**: Make sure to use the correct model checkpoints and data paths when running the scripts!

## License & Contact

This project is licensed under the MIT License. Feel free to open issues or pull requests on GitHub.
For questions or contributions, please reach out to [Othmane Echchabi](mailto:othmane.echchabi@mail.mcgill.ca).


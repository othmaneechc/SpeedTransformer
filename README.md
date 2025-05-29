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

_The MOBIS dataset can be processed using a similar method. More detailed instructions will be added soon._
The MOBIS Processed data can be found here: https://zenodo.org/records/15530797

## Running the Models
This repository provides two primary model architectures:

- LSTM-based trip classification (`models/lstm/`).
- Transformer-based trip classification (`models/transformer/`).

Each architecture includes dedicated scripts for training and fine-tuning. Predefined shell scripts (`train.sh` and `finetune.sh`) are available for streamlined execution with the exact random states used for the reported results.

### LSTM Model

**Training**

Use the `train.sh` script for quick training with the proper random states.

```bash
# lstm/train.sh
# MOBIS
python lstm.py --data_path /data/SpeedTransformer/data/mobis_processed.csv --random_state 316

# Geolife
python lstm.py --data_path /data/SpeedTransformer/data/geolife_processed.csv --random_state 1
```

This saves the best model and also saves `scaler.joblib` / `label_encoder.joblib` for fine-tuning and/or inference purposes.

**Fine-tuning**

Use the `finetune.sh` script to fine-tune the pre-trained LSTM model with a default random state of `42`

```bash
# lstm/finetune.sh
python finetune.py \
  --pretrained_model_path /data/SpeedTransformer/models/lstm/mobis/best_model.pth \
  --data_path /data/SpeedTransformer/data/geolife_processed.csv \
  --scaler_path /data/SpeedTransformer/models/lstm/mobis/scaler.joblib \
  --label_encoder_path /data/SpeedTransformer/models/lstm/mobis/label_encoder.joblib \
  --test_size 0.79 \
  --val_size 0.2 \
  --random_state 42
```

### Transformer Model

**Training**

Use the `train.sh` script to train a Transformer model with the specified random states:

```bash
# transformer/train.sh
# MOBIS
python train.py --data_path /data/SpeedTransformer/data/mobis_processed.csv --random_state 316

# Geolife
python train.py --data_path /data/SpeedTransformer/data/geolife_processed.csv --random_state 1
```

**Fine-Tuning**

```bash
# transformer/finetune.sh
python finetune.py \
  --pretrained_model_path /data/SpeedTransformer/models/transformer/mobis/best_model.pth \
  --data_path /data/SpeedTransformer/data/geolife_processed.csv \
  --label_encoder_path /data/SpeedTransformer/models/transformer/mobis/label_encoder.joblib \
  --test_size 0.79 \
  --val_size 0.2 \
  --random_state 42
```

---

### Replicating Results

To reproduce the results from the paper, the following random states were used:

LSTM Geolife: `1`

LSTM MOBIS: `316`

Transformer Geolife: `316`

Transformer MOBIS: `1`

Fine-Tuning Tasks: `42`

The provided `.sh` scripts ensure the same random seeds are used to replicate the reported accuracy and performance metrics. Make sure to use the right checkpoints!

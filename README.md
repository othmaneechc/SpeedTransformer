# TrajectoryTransformer

This repository contains the code used in the paper **"[Predicting Human Mobility Using Smartphone GPS Trajectories and Transformer Models](#)"**. 

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
   
   Replace "/path/to/data", "/path/to/output.csv", and "/path/to/temp_folder" with your dataset path, desired output file, and a temporary folder for intermediate files. 

3. **Post-Processing** 

After preprocessing, run `extract_speed_geolife.py` to compute additional features like speed and distance:

```bash
python extract_speed_geolife.py geolife.csv --output_file geolife_processed.csv
``` 

### MOBIS Dataset

(Details to be added.)

## Running the Models
This repository provides two primary model architectures:

- LSTM-based trip classification (`models/lstm/`).
- Transformer-based trip classification (`models/transformer/`).

Both have scripts to handle training and fine-tuning and (for Transformer) a dedicated test script.

### LSTM Model

**Training**

Use `models/lstm/lstm.py` to train an LSTM on your dataset. For example:

```bash
python lstm.py --data_path geolife_processed.csv
```

This saves the best model and also saves `scaler.joblib` / `label_encoder.joblib` for future use.

**Fine-tuning**

Use `models/lstm/finetune.py` to load a pre-trained LSTM model, optionally freeze or unfreeze specific layers, and continue training on new data. Now it accepts command-line arguments, for example:

```bash
python finetune.py \
  --pretrained_model_path /path/to/best_model.pth \
  --data_path /path/to/new_data.csv \
  --scaler_path /path/to/scaler.joblib \
  --label_encoder_path /path/to/label_encoder.joblib \
  --batch_size 64 \
  --num_epochs 15 \
  --patience 5
```

It then tests the fine-tuned model on a new test set and saves the updated checkpoint.


### Transformer Model

**Training**

Use `models/transformer/train.py` to train a Transformer-based classifier. It saves the best model (best_model.pth), the fitted label encoder (label_encoder.joblib), and tests automatically at the end. For instance:

```bash
python train.py --data_path geolife_processed.csv
```

**Fine-Tuning**
Use `models/transformer/fine_une.py` to load a pre-trained Transformer, overwrite the newly fitted label encoder with your old one, and continue training on new data:

```bash
python finetune.py
--data_path /path/to/fine_tune_data.csv
--pretrained_model_path best_model.pth
--label_encoder_path label_encoder.joblib
--num_epochs 10
```

It will save a fine-tuned model and evaluate on the new test set.

---

## Replicating results

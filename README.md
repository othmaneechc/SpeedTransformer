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
   python geolife.py --data-folder "/path/to/data" --output-file "/path/to/output.csv" --temp-folder "/path/to/temp_folder"
   ```
   
   Replace "/path/to/data", "/path/to/output.csv", and "/path/to/temp_folder" with your dataset path, desired output file, and a temporary folder for intermediate files. 

3. **Post-Processing** 

After preprocessing, run `extract_speed_geolife.py` to compute additional features like speed and distance:

```bash
python extract_speed_geolife.py /path/to/output.csv --output_file /path/to/final_
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
python lstm.py --data_path /path/to/data.csv --feature_columns speed --target_column label
```

This saves the best model and also saves `scaler.joblib` / `label_encoder.joblib` for future use.

**Fine-tuning**

Use `models/lstm/fine_tune.py` to load a pre-trained LSTM model, optionally freeze or unfreeze specific layers, and continue training on new data. Now it accepts command-line arguments, for example:

```bash
python fine_tune.py \
  --pre_trained_model_path /path/to/best_model.pth \
  --fine_tune_data_path /path/to/new_data.csv \
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
python train.py --data_path /path/to/data.csv --feature_columns speed --target_column
```

**Fine-Tuning**
Use `models/transformer/fine_tune.py` to load a pre-trained Transformer, overwrite the newly fitted label encoder with your old one, and continue training on new data:

```bash
python fine_tune.py
--data_path /path/to/fine_tune_data.csv
--pretrained_model_path best_model.pth
--label_encoder_path label_encoder.joblib
--num_epochs 10
```

It will save a fine-tuned model and evaluate on the new test set.

**Testing**  
Use `models/transformer/test.py` for a dedicated test script, pointing to your saved model and label encoder:

```bash
python test.py
--data_path /path/to/test_data.csv
--model_path best_model.pth
--label_encoder_path label_encoder.joblib
```

---

## Replicating results
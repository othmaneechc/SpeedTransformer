# TrajectoryTransformer

This repository contains the code used in the paper **"[Predicting Human Mobility Using Smartphone GPS Trajectories and Transformer Models](#)"**. 

## Table of Contents

1. [Preparing the Data](#preparing-the-data)
   - [Geolife Dataset](#geolife-dataset)
   - [MOBIS Dataset](#mobis-dataset)
2. [Running the Models](#running-the-models)
3. [Requirements](#requirements)

---

## Preparing the Data

### Geolife Dataset

The Geolife dataset provides GPS trajectories collected from users. To preprocess this dataset:

1. **Download the Dataset**

   - Obtain the Geolife GPS trajectory dataset from [Microsoft Research](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/).
   - Unzip the dataset to a directory on your machine.

2. **Run the Preprocessing Script**

   Use the `data/geolife.py` script under to process the data. This script utilizes multiprocessing for efficient processing and typically completes in under 20 minutes.

   ```bash
   python geolife.py --data-folder "/path/to/data" --output-file "/path/to/output.csv" --temp-folder "/path/to/temp_folder"
   ```
   
   Replace "/path/to/data", "/path/to/output.csv", and "/path/to/temp_folder" with your dataset path, desired output file, and a temporary folder for intermediate files. 

### MOBIS Dataset

## Running the Models
Details on running models will be added soon.

## Requirements

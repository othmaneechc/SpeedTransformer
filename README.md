# TrajectoryTransformer

This repository contains the code used in the paper **"[Predicting Human Mobility Using Smartphone GPS Trajectories and Transformer Models](#)"**. The code includes steps for data preparation and running models to analyze and predict human mobility patterns.

## Table of Contents

1. [Preparing the Data](#preparing-the-data)
   - [Geolife Dataset](#geolife-dataset)
2. [Running the Models](#running-the-models)
3. [Requirements](#requirements)

---

## Preparing the Data

### Geolife Dataset

The first dataset used in this project is the Geolife dataset, which provides GPS trajectories collected from users. To preprocess this dataset and prepare it for training, follow these steps:

#### 1. **Download the Dataset**

   - Download the Geolife GPS trajectory dataset from [Microsoft Research](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/).
   - Unzip the dataset to a folder on your machine.

#### 2. **Run the Preprocessing Script**

   Use the provided `geolife.py` script to preprocess the Geolife data. This script extracts relevant information, processes trajectories, and prepares the dataset for training.

   **Command**:

   ```bash
   python geolife.py --data-folder "/path/to/data" --output-file "/path/to/output.csv" --temp-folder "/path/to/temp_folder"

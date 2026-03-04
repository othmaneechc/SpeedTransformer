# SpeedTransformer

This repository contains the code for the paper:

**SpeedTransformer: Detecting Transportation Mode Using Dense Smartphone GPS Trajectories**

📄 **Paper (arXiv):** https://arxiv.org/abs/2603.00340  
🌐 **Project Page:** https://othmaneechc.github.io/pages/speedtransformer.html

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

_The MOBIS dataset can be processed using a similar method. The processed MOBIS data can be found here: https://zenodo.org/records/17429944_

## Running the Models
This repository provides two primary model architectures:

- LSTM-based trip classification (`models/lstm/`).
- Transformer-based trip classification (`models/transformer/`).

Each architecture includes dedicated scripts for training and fine-tuning. The following shell scripts are available:

### Shell Scripts Overview

#### Replication helpers (`models/replication/`)

- `run_training_experiments.sh` – replays the best transformer and LSTM Geolife/Mobis training jobs.
- `run_gl_finetune_experiments.sh` – reproduces the Geolife finetuning winners for both model families.
- `run_gl_lowshot_finetune_experiments.sh` – fine-tunes the MOBIS transformer on 100/200 Geolife trajectories (low-shot).
- `run_miniprogram_finetune_experiments.sh` – regenerates the CarbonClever finetuning leaderboard models.
- `run_window_sweep_experiments.sh` – reruns the top Geolife window sweep configuration.
- `metrics_gen.py` – converts experiment logs into the replication figures and summary table (`experiment_summary.csv`).

All scripts assume the datasets under `data/` and write results back into their respective `models/**/experiments/` folders so checkpoints, logs, and metrics line up with the paper tables.

### Quick Start Snippet

```bash
# Example: reproduce the key training checkpoints
cd /data/A-SpeedTransformer/models/replication
./run_training_experiments.sh

# Then regenerate plots / tables
python metrics_gen.py
```

### Colab Notebook

For an end-to-end, notebook-based replication you can open `SpeedTransformer.ipynb` directly in Google Colab. Appendix I in the paper lists the expected runtimes and resource notes for that workflow.

## License & Contact

This project is licensed under the MIT License. Feel free to open issues or pull requests on GitHub.
For questions or contributions, please reach out to [Othmane Echchabi](mailto:othmane.echchabi@mail.mcgill.ca).

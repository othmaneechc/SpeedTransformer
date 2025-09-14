# transformer/train.sh

# Geolife
python train.py --data_path /data/A-SpeedTransformer/data/geolife_processed.csv --random_state 1 --run_name geolife_transformer1

# Geolife
python train.py --data_path /data/A-SpeedTransformer/data/geolife_processed.csv --random_state 316 --run_name geolife_transformer316

# MOBIS
python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 1 --run_name mobis_transformer1

# MOBIS
python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --run_name mobis_transformer316
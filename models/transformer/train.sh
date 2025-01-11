# transformer/train.sh

# MOBIS
python train.py --data_path /data/A-TrajectoryTransformer/data/mobis_processed.csv --random_state 316
# Geolife
python train.py --data_path /data/A-TrajectoryTransformer/data/geolife_processed.csv --random_state 1
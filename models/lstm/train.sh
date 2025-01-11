# lstm/train.sh

# MOBIS
python lstm.py --data_path /data/A-TrajectoryTransformer/data/mobis_processed.csv --random_state 316
# Geolife
python lstm.py --data_path /data/A-TrajectoryTransformer/data/geolife_processed.csv --random_state 1

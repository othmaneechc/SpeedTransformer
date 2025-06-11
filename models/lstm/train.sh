# lstm/train.sh

### MOBIS ###
# python lstm.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv \
#             --random_state 316 \
#             --test_size 0.15 \
#             --val_size 0.15 \
#             --checkpoint_dir ./mobis \
#             --scaler_path ./mobis/scaler.joblib \
#             --label_encoder_path ./mobis/label_encoder.joblib \

### Geolife ###
python lstm.py --data_path /data/A-SpeedTransformer/data/geolife_processed.csv \
            --random_state 1 \
            --test_size 0.15 \
            --val_size 0.15 \
            --checkpoint_dir ./geolife \
            --scaler_path ./geolife/scaler.joblib \
            --label_encoder_path ./geolife/label_encoder.joblib 
# python lstm.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.1 --val_size 0.1
# python lstm.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.15 --val_size 0.15
# python lstm.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.2 --val_size 0.2
# python lstm.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.25 --val_size 0.25
# python lstm.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.3 --val_size 0.3
# python lstm.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.35 --val_size 0.35
# python lstm.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.4 --val_size 0.4
# python lstm.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.425 --val_size 0.425
# python lstm.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.45 --val_size 0.45
# python lstm.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.475 --val_size 0.475
# python lstm.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.4875 --val_size 0.4875



# 80%, 70%, 60%, 50%, 40%, 30%, 20%, 15%, 10%, 5%, 2.5%
# 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 500, 250

# python lstm.py --data_path /data/A-SpeedTransformer/data/miniprogram_merged.csv --random_state 316 --test_size 0.15 --val_size 0.15

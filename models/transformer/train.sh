# transformer/train.sh

### MOBIS ###
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv \
#             --random_state 1 \
#             --test_size 0.15 \
#             --val_size 0.15 \
#             --checkpoint_dir ./mobis \
#             --save_model_path ./mobis/model.pth \
#             --scaler_path ./mobis/scaler.joblib \
#             --label_encoder_path ./mobis/label_encoder.joblib \

### Geolife ###
python train.py --data_path /data/A-SpeedTransformer/data/geolife_processed.csv \
            --random_state 316 \
            --test_size 0.15 \
            --val_size 0.15 \
            --checkpoint_dir ./geolife \
            --save_model_path ./geolife/model.pth \
            --save_scaler_path ./geolife/scaler.joblib \
            --save_label_encoder_path ./geolife/label_encoder.joblib \

python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 1 --test_size 0.15 --val_size 0.15
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.1 --val_size 0.1
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.15 --val_size 0.15
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.2 --val_size 0.2
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.25 --val_size 0.25
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.3 --val_size 0.3
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.35 --val_size 0.35
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.4 --val_size 0.4
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.425 --val_size 0.425
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.45 --val_size 0.45
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.475 --val_size 0.475
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.4875 --val_size 0.4875

### Geolife ###
# python train.py --data_path /data/A-SpeedTransformer/data/mobis_processed.csv --random_state 316 --test_size 0.15 --val_size 0.15
# 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 500, 250

# python train.py --data_path /data/A-SpeedTransformer/data/miniprogram_merged.csv --random_state 316 --test_size 0.15 --val_size 0.15

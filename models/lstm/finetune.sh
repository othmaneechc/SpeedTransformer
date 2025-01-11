python finetune.py \
  --pretrained_model_path /data/A-TrajectoryTransformer/models/lstm/mobis/best_model.pth \
  --data_path /data/A-TrajectoryTransformer/data/geolife_processed.csv \
  --scaler_path /data/A-TrajectoryTransformer/models/lstm/mobis/scaler.joblib \
  --label_encoder_path /data/A-TrajectoryTransformer/models/lstm/mobis/label_encoder.joblib \
  --test_size 0.79 \
  --val_size 0.2 \
  --random_state 42
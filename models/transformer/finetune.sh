# transformer/finetune.sh

python finetune.py \
  --pretrained_model_path /data/A-TrajectoryTransformer/models/transformer/mobis/best_model.pth \
  --data_path /data/A-TrajectoryTransformer/data/geolife_processed.csv \
  --label_encoder_path /data/A-TrajectoryTransformer/models/transformer/mobis/label_encoder.joblib \
  --test_size 0.7786 \
  --val_size 0.2 \
  --random_state 42
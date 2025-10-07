#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRANSFORMER_DIR="$ROOT/models/transformer"
DATA_DIR="$ROOT/data"
mkdir -p "$TRANSFORMER_DIR/experiments/geolife_window_sweeps"

RUN_NAME="geolife_ws200_lr2e-4_bs512_h8_d128_kv4_do0.1"
OUT_DIR="$TRANSFORMER_DIR/experiments/geolife_window_sweeps"

echo "[Window Sweep] Geolife → $RUN_NAME"
python "$TRANSFORMER_DIR/train.py" \
  --data_path "$DATA_DIR/geolife_processed.csv" \
  --feature_columns speed \
  --target_column label \
  --traj_id_column traj_id \
  --test_size 0.15 \
  --val_size 0.15 \
  --random_state 316 \
  --chunksize 1000000 \
  --window_size 200 \
  --stride 50 \
  --batch_size 512 \
  --num_workers 4 \
  --num_epochs 50 \
  --patience 7 \
  --d_model 128 \
  --nhead 8 \
  --num_layers 4 \
  --dropout 0.1 \
  --learning_rate 2e-4 \
  --weight_decay 1e-4 \
  --gradient_clip 1.0 \
  --kv_heads 4 \
  --use_amp \
  --run_name "$RUN_NAME" \
  --out_dir "$OUT_DIR"

echo "\nWindow sweep replication run completed."

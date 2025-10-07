#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRANSFORMER_DIR="$ROOT/models/transformer"
LSTM_DIR="$ROOT/models/lstm"
DATA_DIR="$ROOT/data"

TRANSFORMER_PRETRAIN_DIR="$TRANSFORMER_DIR/experiments/mobis_transformer_sweeps/mobis_lr1e-4_bs512_h8_d128_kv4_do0.1"
LSTM_PRETRAIN_DIR="$LSTM_DIR/experiments/mobis_lstm_sweeps/mobis_lr5e-4_bs128_h256_l3_do0.1"

mkdir -p "$TRANSFORMER_DIR/experiments/finetune_sweeps"
mkdir -p "$LSTM_DIR/experiments/finetune_sweeps"

if [[ ! -f "$TRANSFORMER_PRETRAIN_DIR/best_model.pth" ]]; then
  echo "Missing transformer pretrained weights at $TRANSFORMER_PRETRAIN_DIR. Run the training replication script first." >&2
  exit 1
fi

if [[ ! -f "$LSTM_PRETRAIN_DIR/best_model.pth" ]]; then
  echo "Missing LSTM pretrained weights at $LSTM_PRETRAIN_DIR. Run the training replication script first." >&2
  exit 1
fi

echo "\n[Transformer Finetune] Geolife → lr2e-4_warmup0_freeze_attention"
TF_OUT_DIR="$TRANSFORMER_DIR/experiments/finetune_sweeps/lr2e-4_warmup0_freeze_attention"
mkdir -p "$TF_OUT_DIR"
python "$TRANSFORMER_DIR/finetune.py" \
  --pretrained_model_path "$TRANSFORMER_PRETRAIN_DIR/best_model.pth" \
  --label_encoder_path "$TRANSFORMER_PRETRAIN_DIR/label_encoder.joblib" \
  --data_path "$DATA_DIR/geolife_processed.csv" \
  --test_size 0.79 \
  --val_size 0.2 \
  --random_state 42 \
  --batch_size 512 \
  --num_epochs 60 \
  --patience 7 \
  --learning_rate 2e-4 \
  --warmup_steps 0 \
  --d_model 128 \
  --nhead 8 \
  --kv_heads 4 \
  --dropout 0.1 \
  --freeze_attention \
  --use_amp \
  --save_model_path "$TF_OUT_DIR/best_model.pth"

echo "\n[LSTM Finetune] Geolife → lstm_lr1e-3_h256"
LSTM_OUT_DIR="$LSTM_DIR/experiments/finetune_sweeps/lstm_lr1e-3_h256"
mkdir -p "$LSTM_OUT_DIR"
python "$LSTM_DIR/finetune.py" \
  --pretrained_model_path "$LSTM_PRETRAIN_DIR/best_model.pth" \
  --scaler_path "$LSTM_PRETRAIN_DIR/scaler.joblib" \
  --label_encoder_path "$LSTM_PRETRAIN_DIR/label_encoder.joblib" \
  --data_path "$DATA_DIR/geolife_processed.csv" \
  --test_size 0.79 \
  --val_size 0.2 \
  --random_state 42 \
  --batch_size 128 \
  --num_epochs 60 \
  --patience 7 \
  --learning_rate 1e-3 \
  --hidden_size 256 \
  --dropout 0.1 \
  --checkpoint_dir "$LSTM_OUT_DIR"

echo "\nGeolife finetuning replication runs completed."

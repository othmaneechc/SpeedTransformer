#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRANSFORMER_DIR="$ROOT/models/transformer"
LSTM_DIR="$ROOT/models/lstm"
DATA_DIR="$ROOT/data"

TRANSFORMER_PRETRAIN_DIR="$TRANSFORMER_DIR/experiments/mobis_transformer_sweeps/mobis_lr1e-4_bs512_h8_d128_kv4_do0.1"
LSTM_PRETRAIN_DIR="$LSTM_DIR/experiments/mobis_lstm_sweeps/mobis_lr1e-3_bs128_h128_l2_do0.1"

mkdir -p "$TRANSFORMER_DIR/experiments/miniprogram_finetune"
mkdir -p "$LSTM_DIR/experiments/miniprogram_finetune"

if [[ ! -f "$TRANSFORMER_PRETRAIN_DIR/best_model.pth" ]]; then
  echo "Missing transformer pretrained weights at $TRANSFORMER_PRETRAIN_DIR. Run the training replication script first." >&2
  exit 1
fi

if [[ ! -f "$LSTM_PRETRAIN_DIR/best_model.pth" ]]; then
  echo "Missing LSTM pretrained weights at $LSTM_PRETRAIN_DIR. Run the original MOBIS LSTM sweep to generate them." >&2
  exit 1
fi

echo "\n[Transformer Miniprogram] final_50pct_313trips_lr5e-4_warmup0_none"
TF_OUT_DIR="$TRANSFORMER_DIR/experiments/miniprogram_finetune/final_50pct_313trips_lr5e-4_warmup0_none"
mkdir -p "$TF_OUT_DIR"
python "$TRANSFORMER_DIR/finetune.py" \
  --pretrained_model_path "$TRANSFORMER_PRETRAIN_DIR/best_model.pth" \
  --label_encoder_path "$TRANSFORMER_PRETRAIN_DIR/label_encoder.joblib" \
  --data_path "$DATA_DIR/miniprogram_balanced.csv" \
  --test_size 0.3217 \
  --val_size 0.2 \
  --random_state 42 \
  --batch_size 512 \
  --num_epochs 50 \
  --patience 10 \
  --learning_rate 5e-4 \
  --warmup_steps 0 \
  --d_model 128 \
  --nhead 8 \
  --kv_heads 4 \
  --dropout 0.1 \
  --use_amp \
  --save_model_path "$TF_OUT_DIR/best_model.pth"

echo "\n[LSTM Miniprogram] final_30pct_189trips_lr5e-4_h128"
LSTM_OUT_DIR="$LSTM_DIR/experiments/miniprogram_finetune/final_30pct_189trips_lr5e-4_h128"
mkdir -p "$LSTM_OUT_DIR"
python "$LSTM_DIR/finetune.py" \
  --pretrained_model_path "$LSTM_PRETRAIN_DIR/best_model.pth" \
  --scaler_path "$LSTM_PRETRAIN_DIR/scaler.joblib" \
  --label_encoder_path "$LSTM_PRETRAIN_DIR/label_encoder.joblib" \
  --data_path "$DATA_DIR/miniprogram_balanced.csv" \
  --test_size 0.5097 \
  --val_size 0.2 \
  --random_state 42 \
  --batch_size 128 \
  --num_epochs 50 \
  --patience 10 \
  --learning_rate 5e-4 \
  --hidden_size 128 \
  --dropout 0.1 \
  --checkpoint_dir "$LSTM_OUT_DIR"

echo "\nMiniprogram finetuning replication runs completed."

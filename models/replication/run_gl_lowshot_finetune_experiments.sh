#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRANSFORMER_DIR="$ROOT/models/transformer"
PRETRAIN_DIR="$TRANSFORMER_DIR/experiments/mobis_transformer_sweeps/mobis_lr1e-4_bs512_h8_d128_kv4_do0.1"
OUT_DIR="$TRANSFORMER_DIR/experiments/finetune_lowshot"
LSTM_DIR="$ROOT/models/lstm"
LSTM_PRETRAIN_DIR="$LSTM_DIR/experiments/mobis_lstm_sweeps/mobis_lr5e-4_bs128_h256_l3_do0.1"
LSTM_OUT_DIR="$LSTM_DIR/experiments/finetune_lowshot"
DATA_DIR="$ROOT/data"

if [[ ! -f "$PRETRAIN_DIR/best_model.pth" ]]; then
  echo "Missing MOBIS pretrained weights at $PRETRAIN_DIR" >&2
  exit 1
fi

if [[ ! -f "$LSTM_PRETRAIN_DIR/best_model.pth" ]]; then
  echo "Missing MOBIS pretrained LSTM weights at $LSTM_PRETRAIN_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
mkdir -p "$LSTM_OUT_DIR"

# Fractions chosen to yield ~100 and ~200 train trajectories (total Geolife traj_ids = 9347)
declare -A VAL_SIZE=(
  [train100]="0.494597196961592"
  [train200]="0.489247887022574"
)

declare -A TEST_SIZE=(
  [train100]="0.494704183160372"
  [train200]="0.489354873221354"
)

declare -A TRAIN_TARGET=(
  [train100]=100
  [train200]=200
)

for key in train100 train200; do
  val_frac="${VAL_SIZE[$key]}"
  test_frac="${TEST_SIZE[$key]}"
  target_count="${TRAIN_TARGET[$key]}"

  run_name="${key}_lr2e-4_warmup0_freeze_attention"
  run_dir="$OUT_DIR/$run_name"
  mkdir -p "$run_dir"

  echo "\n[lowshot] Target train trajs: $target_count"
  echo "  val_size=$val_frac, test_size=$test_frac"
  echo "  transformer run=$run_name"

  python "$TRANSFORMER_DIR/finetune.py" \
    --pretrained_model_path "$PRETRAIN_DIR/best_model.pth" \
    --label_encoder_path "$PRETRAIN_DIR/label_encoder.joblib" \
    --data_path "$DATA_DIR/geolife_processed.csv" \
    --test_size "$test_frac" \
    --val_size "$val_frac" \
    --random_state 42 \
    --batch_size 512 \
    --num_epochs 20 \
    --patience 7 \
    --learning_rate 2e-4 \
    --warmup_steps 0 \
    --d_model 128 \
    --nhead 8 \
    --kv_heads 4 \
    --dropout 0.1 \
    --freeze_attention \
    --use_amp \
    --save_model_path "$run_dir/best_model.pth" \
    > "$run_dir/finetune.log" 2>&1

  echo "  transformer saved → $run_dir"

  lstm_run_name="${key}_lr5e-5_bs128_do0.3"
  lstm_run_dir="$LSTM_OUT_DIR/$lstm_run_name"
  mkdir -p "$lstm_run_dir"

  echo "  lstm run=$lstm_run_name"

  python "$LSTM_DIR/finetune.py" \
    --pretrained_model_path "$LSTM_PRETRAIN_DIR/best_model.pth" \
    --scaler_path "$LSTM_PRETRAIN_DIR/scaler.joblib" \
    --label_encoder_path "$LSTM_PRETRAIN_DIR/label_encoder.joblib" \
    --data_path "$DATA_DIR/geolife_processed.csv" \
    --feature_columns speed \
    --test_size "$test_frac" \
    --val_size "$val_frac" \
    --random_state 42 \
    --batch_size 128 \
    --num_workers 4 \
    --num_epochs 20 \
    --patience 3 \
    --learning_rate 5e-5 \
    --weight_decay 5e-3 \
    --dropout 0.3 \
    --max_grad_norm 0.25 \
    --checkpoint_dir "$lstm_run_dir" \
    > "$lstm_run_dir/finetune.log" 2>&1

  echo "  lstm saved → $lstm_run_dir"
done

echo "\nLow-shot Geolife finetuning runs completed."

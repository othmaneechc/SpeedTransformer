#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRANSFORMER_DIR="$ROOT/models/transformer"
LSTM_DIR="$ROOT/models/lstm"
DATA_DIR="$ROOT/data"

# Ensure experiment output directories exist
mkdir -p "$TRANSFORMER_DIR/experiments/geolife_transformer_sweeps"
mkdir -p "$TRANSFORMER_DIR/experiments/mobis_transformer_sweeps"
mkdir -p "$LSTM_DIR/experiments/geolife_lstm_sweeps"
mkdir -p "$LSTM_DIR/experiments/mobis_lstm_sweeps"

# Transformer best runs
declare -a TRANSFORMER_RUNS=(
  "dataset=geolife_processed.csv;run_name=lr2e-4_bs512_h8_d128_kv4_do0.1;lr=2e-4;batch=512;nhead=8;dmodel=128;kv_heads=4;dropout=0.1;window=200;stride=50"
  "dataset=mobis_processed.csv;run_name=mobis_lr1e-4_bs512_h8_d128_kv4_do0.1;lr=1e-4;batch=512;nhead=8;dmodel=128;kv_heads=4;dropout=0.1;window=200;stride=50"
)

for entry in "${TRANSFORMER_RUNS[@]}"; do
  IFS=';' read -ra parts <<< "$entry"
  declare -A cfg=()
  for part in "${parts[@]}"; do
    key="${part%%=*}"
    value="${part#*=}"
    cfg["$key"]="$value"
  done

  dataset="${cfg[dataset]}"
  run_name="${cfg[run_name]}"
  dataset_name="${dataset%_processed.csv}"
  out_dir="$TRANSFORMER_DIR/experiments/${dataset_name}_transformer_sweeps"

  echo "\n[Transformer] ${dataset_name^} → $run_name"
  python "$TRANSFORMER_DIR/train.py" \
    --data_path "$DATA_DIR/$dataset" \
    --feature_columns speed \
    --target_column label \
    --traj_id_column traj_id \
    --test_size 0.15 \
    --val_size 0.15 \
    --random_state 316 \
    --chunksize 1000000 \
    --window_size "${cfg[window]}" \
    --stride "${cfg[stride]}" \
    --batch_size "${cfg[batch]}" \
    --num_workers 4 \
    --num_epochs 50 \
    --patience 7 \
    --d_model "${cfg[dmodel]}" \
    --nhead "${cfg[nhead]}" \
    --num_layers 4 \
    --dropout "${cfg[dropout]}" \
    --learning_rate "${cfg[lr]}" \
    --weight_decay 1e-4 \
    --gradient_clip 1.0 \
    --kv_heads "${cfg[kv_heads]}" \
    --use_amp \
    --run_name "$run_name" \
    --out_dir "$out_dir"

  unset cfg
  declare -A cfg=()
done

# LSTM best runs
declare -a LSTM_RUNS=(
  "dataset=geolife_processed.csv;run_name=geolife_lr5e-4_bs128_h256_l3_do0.1;lr=5e-4;batch=128;hidden=256;layers=3;dropout=0.1"
  "dataset=mobis_processed.csv;run_name=mobis_lr5e-4_bs128_h256_l3_do0.1;lr=5e-4;batch=128;hidden=256;layers=3;dropout=0.1"
)

for entry in "${LSTM_RUNS[@]}"; do
  IFS=';' read -ra parts <<< "$entry"
  declare -A cfg=()
  for part in "${parts[@]}"; do
    key="${part%%=*}"
    value="${part#*=}"
    cfg["$key"]="$value"
  done

  dataset="${cfg[dataset]}"
  run_name="${cfg[run_name]}"
  dataset_name="${dataset%_processed.csv}"
  checkpoint_dir="$LSTM_DIR/experiments/${dataset_name}_lstm_sweeps/$run_name"
  mkdir -p "$checkpoint_dir"

  echo "\n[LSTM] ${dataset_name^} → $run_name"
  python "$LSTM_DIR/lstm.py" \
    --data_path "$DATA_DIR/$dataset" \
    --feature_columns speed \
    --target_column label \
    --traj_id_column traj_id \
    --test_size 0.15 \
    --val_size 0.15 \
    --random_state 42 \
    --batch_size "${cfg[batch]}" \
    --learning_rate "${cfg[lr]}" \
    --hidden_size "${cfg[hidden]}" \
    --num_layers "${cfg[layers]}" \
    --dropout "${cfg[dropout]}" \
    --num_epochs 50 \
    --patience 7 \
    --checkpoint_dir "$checkpoint_dir" \
    --scaler_path "$checkpoint_dir/scaler.joblib" \
    --label_encoder_path "$checkpoint_dir/label_encoder.joblib" \
    --num_workers 4 
  unset cfg
  declare -A cfg=()
done

echo "\nAll training replication runs completed."

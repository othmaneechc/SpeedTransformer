#!/usr/bin/env bash
set -euo pipefail

# Base paths and configuration
ROOT="/data/A-SpeedTransformer"
MOBIS_MODEL_DIR="$ROOT/models/transformer/experiments/mobis_transformer_sweeps/mobis_lr1e-4_bs512_h8_d128_kv4_do0.1/mobis_lr1e-4_bs512_h8_d128_kv4_do0.1"
EXPERIMENT_DIR="$ROOT/models/transformer/experiments/finetune_sweeps"

# Fixed parameters
BASE_ARGS=(
  --pretrained_model_path "$MOBIS_MODEL_DIR/best_model.pth"
  --label_encoder_path "$MOBIS_MODEL_DIR/label_encoder.joblib"
  --data_path "/data/A-SpeedTransformer/data/geolife_processed.csv"
  --test_size 0.79
  --val_size 0.2
  --random_state 42
  --batch_size 512
  --num_epochs 60
  --patience 7
  --use_amp
)

# Sweep configurations
declare -a LEARNING_RATES=("5e-5" "1e-4" "2e-4")
declare -a WARMUP_STEPS=(0 100 500)
declare -a LAYER_LR_DECAYS=("1.0" "0.95" "0.9")

# Fine-tuning strategies
declare -A STRATEGIES=(
  ["full"]=""
  ["freeze_first_half"]="--freeze_layers 0,1"
  ["freeze_embeddings"]="--freeze_embeddings"
  ["freeze_attention"]="--freeze_attention"
  ["freeze_feedforward"]="--freeze_feedforward"
  ["reinit_last"]="--reinit_layers 3"
  ["grad_unfreeze"]="--freeze_layers 0,1,2,3 --layer_lr_decay 0.9"
)

# Run sweep
for lr in "${LEARNING_RATES[@]}"; do
  for warmup in "${WARMUP_STEPS[@]}"; do
    for strategy_name in "${!STRATEGIES[@]}"; do
      strategy_args="${STRATEGIES[$strategy_name]}"
      
      # Set run name and output directory
      RUN_NAME="lr${lr}_warmup${warmup}_${strategy_name}"
      OUT_DIR="$EXPERIMENT_DIR/$RUN_NAME"
      mkdir -p "$OUT_DIR"
      
      # Skip if already completed
      if [[ -f "$OUT_DIR/best_model.pth" ]]; then
        echo "[skip] $RUN_NAME already exists"
        continue
      fi
      
      echo "[run] Starting $RUN_NAME..."
      
      # Run fine-tuning with current configuration
      python finetune.py \
        "${BASE_ARGS[@]}" \
        --learning_rate "$lr" \
        --warmup_steps "$warmup" \
        --save_model_path "$OUT_DIR/best_model.pth" \
        $strategy_args \
        2>&1 | tee "$OUT_DIR/finetune.log"
        
      echo "[done] Completed $RUN_NAME"
    done
  done
done

echo "All fine-tuning sweeps complete."
#!/usr/bin/env bash
set -euo pipefail

# Activate torchgpu environment
source /data/oe23/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu

# LSTM Fine-tuning Experiments
# Transfer learning from MOBIS pretrained model to Geolife dataset
# Testing different learning rates and configurations

ROOT="/data/A-SpeedTransformer"
MOBIS_MODEL_DIR="$ROOT/models/lstm/mobis"
EXPERIMENT_DIR="$ROOT/models/lstm/experiments/finetune_sweeps"

# Check if pretrained model exists
if [[ ! -f "$MOBIS_MODEL_DIR/best_model.pth" ]]; then
  echo "Error: MOBIS pretrained LSTM model not found at: $MOBIS_MODEL_DIR/best_model.pth"
  echo "Checking alternative locations..."
  
  # Check the sweep experiments directory
  MOBIS_SWEEP_MODEL="/data/A-SpeedTransformer/models/lstm/experiments/mobis_lstm_sweeps/mobis_lr1e-3_bs128_h128_l2_do0.1/best_model.pth"
  if [[ -f "$MOBIS_SWEEP_MODEL" ]]; then
    echo "Found MOBIS model in sweep experiments: $MOBIS_SWEEP_MODEL"
    MOBIS_MODEL_DIR="/data/A-SpeedTransformer/models/lstm/experiments/mobis_lstm_sweeps/mobis_lr1e-3_bs128_h128_l2_do0.1"
  else
    echo "No pretrained MOBIS LSTM model found. Available MOBIS models:"
    find /data/A-SpeedTransformer/models/lstm -name "best_model.pth" -type f | head -5
    echo ""
    echo "Please run the hyperparameter sweep first: ./run_sweep.sh"
    echo "This script will wait for the MOBIS model to be available..."
    
    # Wait for model to be available
    while [[ ! -f "$MOBIS_SWEEP_MODEL" ]]; do
      echo "Waiting for MOBIS model training to complete..."
      sleep 60
    done
    echo "MOBIS model found! Continuing with fine-tuning..."
    MOBIS_MODEL_DIR="/data/A-SpeedTransformer/models/lstm/experiments/mobis_lstm_sweeps/mobis_lr1e-3_bs128_h128_l2_do0.1"
  fi
fi

# Base paths and configuration
BASE_ARGS=(
  --pretrained_model_path "$MOBIS_MODEL_DIR/best_model.pth"
  --scaler_path "$MOBIS_MODEL_DIR/scaler.joblib"
  --label_encoder_path "$MOBIS_MODEL_DIR/label_encoder.joblib"
  --data_path "$ROOT/data/geolife_processed.csv"
  --test_size 0.79
  --val_size 0.2
  --random_state 42
  --batch_size 128
  --num_epochs 60
  --patience 7
)

# Sweep configurations
declare -a LEARNING_RATES=("1e-4" "5e-4" "1e-3")
declare -a HIDDEN_DIMS=("64" "128" "256")

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR"

echo "Starting LSTM fine-tuning experiments..."
echo "Base model: $MOBIS_MODEL_DIR"
echo "Target dataset: Geolife"

# Fine-tuning sweep
for lr in "${LEARNING_RATES[@]}"; do
  for hidden_dim in "${HIDDEN_DIMS[@]}"; do
    RUN_NAME="lstm_lr${lr}_h${hidden_dim}"
    RUN_DIR="$EXPERIMENT_DIR/$RUN_NAME"
    mkdir -p "$RUN_DIR"
    
    echo "Running: $RUN_NAME"
    echo "  Learning rate: $lr"
    echo "  Hidden dim: $hidden_dim"
    echo "  Output dir: $RUN_DIR"
    
    python finetune.py \
      "${BASE_ARGS[@]}" \
      --learning_rate "$lr" \
      --hidden_size "$hidden_dim" \
      --checkpoint_dir "$RUN_DIR" \
      > "$RUN_DIR/finetune.log" 2>&1
    
    echo "Completed: $RUN_NAME"
    echo "---"
  done
done

echo "All fine-tuning experiments completed!"
echo "Results can be found in: $EXPERIMENT_DIR"
#!/usr/bin/env bash
set -euo pipefail

# Activate torchgpu environment
source /data/oe23/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu

# Miniprogram Finetune Experiments for LSTM
# Hyperparameter tuning for MOBIS pretrained LSTM model on miniprogram data
# Testing different combinations of learning rates and data subset sizes

ROOT="/data/A-SpeedTransformer"
MOBIS_MODEL_DIR="$ROOT/models/lstm/experiments/mobis_lstm_sweeps/mobis_lr1e-3_bs128_h128_l2_do0.1"
EXPERIMENT_DIR="$ROOT/models/lstm/experiments/miniprogram_finetune"

# Check if pretrained model exists
if [[ ! -f "$MOBIS_MODEL_DIR/best_model.pth" ]]; then
  echo "Error: MOBIS pretrained LSTM model not found at: $MOBIS_MODEL_DIR/best_model.pth"
  echo "Checking alternative locations..."
  
  # Try the standard mobis directory
  ALT_MODEL_DIR="$ROOT/models/lstm/mobis"
  if [[ -f "$ALT_MODEL_DIR/best_model.pth" ]]; then
    echo "Found model in: $ALT_MODEL_DIR"
    MOBIS_MODEL_DIR="$ALT_MODEL_DIR"
  else
    echo "No pretrained MOBIS LSTM model found. Available MOBIS models:"
    find "$ROOT/models/lstm" -name "best_model.pth" -path "*mobis*" | head -5
    exit 1
  fi
fi

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR"

# Base parameters (fixed across all experiments)
BASE_ARGS=(
  --pretrained_model_path "$MOBIS_MODEL_DIR/best_model.pth"
  --scaler_path "$MOBIS_MODEL_DIR/scaler.joblib"
  --label_encoder_path "$MOBIS_MODEL_DIR/label_encoder.joblib"
  --data_path "/data/A-SpeedTransformer/data/miniprogram_balanced.csv"
  --random_state 42
  --batch_size 128
  --num_epochs 50
  --patience 10
)

# Hyperparameter combinations to test
declare -a LEARNING_RATES=("1e-4" "5e-4" "1e-3")
declare -a HIDDEN_DIMS=("64" "128" "256")

# Data subset for hyperparameter tuning (use 15% = ~94 trips as it's middle-sized)
TUNING_TEST_SIZE="0.6506"

echo "Starting LSTM miniprogram hyperparameter tuning..."
echo "Using data subset: ~15% (test_size=$TUNING_TEST_SIZE)"
echo "Base model: $MOBIS_MODEL_DIR"

# Hyperparameter tuning runs
for lr in "${LEARNING_RATES[@]}"; do
  for hidden_dim in "${HIDDEN_DIMS[@]}"; do
    RUN_NAME="tune_lr${lr}_h${hidden_dim}"
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
      --test_size "$TUNING_TEST_SIZE" \
      --val_size 0.2 \
      --checkpoint_dir "$RUN_DIR" \
      > "$RUN_DIR/finetune.log" 2>&1
    
    echo "Completed: $RUN_NAME"
    echo "---"
  done
done

echo "Hyperparameter tuning completed!"

# Best configuration from tuning (assuming lr=5e-4, hidden_dim=128 works best)
BEST_LR="5e-4"
BEST_HIDDEN_DIM="128"

echo "Starting data subset experiments with best config: lr=$BEST_LR, hidden_dim=$BEST_HIDDEN_DIM"

# Data subset experiments with different percentages
declare -A SUBSET_CONFIGS=(
  ["15pct_94trips"]="0.6506"    # 15% of data ≈ 94 trips
  ["20pct_125trips"]="0.6037"   # 20% of data ≈ 125 trips  
  ["30pct_189trips"]="0.5097"   # 30% of data ≈ 189 trips
  ["40pct_251trips"]="0.4157"   # 40% of data ≈ 251 trips
  ["50pct_313trips"]="0.3217"   # 50% of data ≈ 313 trips
)

for subset_name in "${!SUBSET_CONFIGS[@]}"; do
  test_size="${SUBSET_CONFIGS[$subset_name]}"
  
  RUN_NAME="final_${subset_name}_lr${BEST_LR}_h${BEST_HIDDEN_DIM}"
  RUN_DIR="$EXPERIMENT_DIR/$RUN_NAME"
  mkdir -p "$RUN_DIR"
  
  echo "Running data subset experiment: $RUN_NAME"
  echo "  Data subset: $subset_name"
  echo "  Test size: $test_size"
  echo "  Learning rate: $BEST_LR"
  echo "  Hidden dim: $BEST_HIDDEN_DIM"
  echo "  Output dir: $RUN_DIR"
  
  python finetune.py \
    "${BASE_ARGS[@]}" \
    --learning_rate "$BEST_LR" \
    --hidden_size "$BEST_HIDDEN_DIM" \
    --test_size "$test_size" \
    --val_size 0.2 \
    --checkpoint_dir "$RUN_DIR" \
    > "$RUN_DIR/finetune.log" 2>&1
  
  echo "Completed: $RUN_NAME"
  echo "---"
done

echo "All miniprogram fine-tuning experiments completed!"
echo "Results can be found in: $EXPERIMENT_DIR"

# Generate summary
echo "Generating experiment summary..."
echo "LSTM Miniprogram Fine-tuning Results Summary" > "$EXPERIMENT_DIR/summary.txt"
echo "=============================================" >> "$EXPERIMENT_DIR/summary.txt"
echo "" >> "$EXPERIMENT_DIR/summary.txt"
echo "Hyperparameter Tuning Runs:" >> "$EXPERIMENT_DIR/summary.txt"
for lr in "${LEARNING_RATES[@]}"; do
  for hidden_dim in "${HIDDEN_DIMS[@]}"; do
    run_name="tune_lr${lr}_h${hidden_dim}"
    if [[ -f "$EXPERIMENT_DIR/$run_name/finetune.log" ]]; then
      echo "  $run_name" >> "$EXPERIMENT_DIR/summary.txt"
    fi
  done
done

echo "" >> "$EXPERIMENT_DIR/summary.txt"
echo "Data Subset Experiments:" >> "$EXPERIMENT_DIR/summary.txt"
for subset_name in "${!SUBSET_CONFIGS[@]}"; do
  run_name="final_${subset_name}_lr${BEST_LR}_h${BEST_HIDDEN_DIM}"
  if [[ -f "$EXPERIMENT_DIR/$run_name/finetune.log" ]]; then
    echo "  $run_name (${subset_name})" >> "$EXPERIMENT_DIR/summary.txt"
  fi
done

echo "Summary saved to: $EXPERIMENT_DIR/summary.txt"

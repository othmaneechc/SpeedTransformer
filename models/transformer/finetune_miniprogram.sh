#!/usr/bin/env bash
set -euo pipefail

# Miniprogram Finetune Experiments
# Hyperparameter tuning for MOBIS pretrained model on miniprogram data
# Testing different combinations of learning rates, warmup steps, and freezing strategies

ROOT="/data/A-SpeedTransformer"
MOBIS_MODEL_DIR="$ROOT/models/transformer/experiments/mobis_transformer_sweeps/mobis_lr1e-4_bs512_h8_d128_kv4_do0.1"
EXPERIMENT_DIR="$ROOT/models/transformer/experiments/miniprogram_finetune"

# Check if pretrained model exists
if [[ ! -f "$MOBIS_MODEL_DIR/best_model.pth" ]]; then
  echo "Error: MOBIS pretrained model not found at: $MOBIS_MODEL_DIR/best_model.pth"
  echo "Checking alternative locations..."
  
  # Try the transformer_vanilla directory
  ALT_MODEL_DIR="$ROOT/models/transformer_vanilla/mobis"
  if [[ -f "$ALT_MODEL_DIR/best_model.pth" ]]; then
    echo "Found model in: $ALT_MODEL_DIR"
    MOBIS_MODEL_DIR="$ALT_MODEL_DIR"
  else
    echo "No pretrained MOBIS model found. Available MOBIS models:"
    find "$ROOT/models" -name "best_model.pth" -path "*/mobis*" | head -5
    exit 1
  fi
fi

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR"

# Base parameters (fixed across all experiments)
BASE_ARGS=(
  --pretrained_model_path "$MOBIS_MODEL_DIR/best_model.pth"
  --label_encoder_path "$MOBIS_MODEL_DIR/label_encoder.joblib"
  --data_path "/data/A-SpeedTransformer/data/miniprogram_balanced.csv"
  --random_state 42
  --batch_size 512
  --num_epochs 50
  --patience 10
  --use_amp
)

# Hyperparameter combinations to test (matching LSTM experiments)
declare -a LEARNING_RATES=("1e-4" "5e-4" "1e-3")
declare -a WARMUP_STEPS=(0 50 100)
declare -a FREEZE_STRATEGIES=("none" "freeze_attention" "freeze_feedforward")

# Data subset for hyperparameter tuning (use 15% = ~94 trips as it's middle-sized)
TUNING_TEST_SIZE="0.6506"
TUNING_VAL_SIZE="0.2"

# Data subset configurations matching the LSTM experiments exactly
# Total trips in miniprogram: 629
declare -A SUBSET_CONFIGS=(
  ["15pct_94trips"]="0.6506"    # 15% of data ≈ 94 trips (val_size handled separately)
  ["20pct_125trips"]="0.6037"   # 20% of data ≈ 125 trips  
  ["30pct_189trips"]="0.5097"   # 30% of data ≈ 189 trips
  ["40pct_251trips"]="0.4157"   # 40% of data ≈ 251 trips
  ["50pct_313trips"]="0.3217"   # 50% of data ≈ 313 trips
)

echo "=== Transformer Miniprogram Fine-tuning Experiments ==="
echo "Pretrained model: $MOBIS_MODEL_DIR/best_model.pth"
echo "Target dataset: miniprogram_balanced.csv (629 total trips)"
echo "Using data subset: ~15% (test_size=$TUNING_TEST_SIZE) for hyperparameter tuning"
echo "Base model: $MOBIS_MODEL_DIR"
echo ""

# Switch to the transformer directory for execution
cd "$ROOT/models/transformer"

# Hyperparameter tuning runs (simplified approach matching LSTM)
echo "Starting Transformer miniprogram hyperparameter tuning..."
for lr in "${LEARNING_RATES[@]}"; do
  for warmup in "${WARMUP_STEPS[@]}"; do
    for strategy in "${FREEZE_STRATEGIES[@]}"; do
      
      # Set freeze arguments based on strategy
      freeze_args=""
      case "$strategy" in
        "freeze_attention")
          freeze_args="--freeze_attention"
          ;;
        "freeze_feedforward")
          freeze_args="--freeze_feedforward"
          ;;
        "none")
          freeze_args=""
          ;;
      esac
      
      RUN_NAME="tune_lr${lr}_warmup${warmup}_${strategy}"
      RUN_DIR="$EXPERIMENT_DIR/$RUN_NAME"
      mkdir -p "$RUN_DIR"
      
      echo "Running: $RUN_NAME"
      echo "  Learning rate: $lr"
      echo "  Warmup steps: $warmup"
      echo "  Strategy: $strategy"
      echo "  Output dir: $RUN_DIR"
      
      python finetune.py \
        "${BASE_ARGS[@]}" \
        --learning_rate "$lr" \
        --warmup_steps "$warmup" \
        $freeze_args \
        --test_size "$TUNING_TEST_SIZE" \
        --val_size "$TUNING_VAL_SIZE" \
        --save_model_path "$RUN_DIR/best_model.pth" \
        > "$RUN_DIR/finetune.log" 2>&1
      
      echo "Completed: $RUN_NAME"
      echo "---"
    done
  done
done

echo "Hyperparameter tuning completed!"

# Best configuration from tuning (assuming lr=1e-4, warmup=50, none works best based on MOBIS results)
BEST_LR="1e-4"
BEST_WARMUP="50"
BEST_STRATEGY="none"

echo "Starting data subset experiments with best config: lr=$BEST_LR, warmup=$BEST_WARMUP, strategy=$BEST_STRATEGY"
# Data subset experiments with different percentages (matching LSTM experiments)
for subset_name in "${!SUBSET_CONFIGS[@]}"; do
  test_size="${SUBSET_CONFIGS[$subset_name]}"
  
  RUN_NAME="final_${subset_name}_lr${BEST_LR}_warmup${BEST_WARMUP}_${BEST_STRATEGY}"
  RUN_DIR="$EXPERIMENT_DIR/$RUN_NAME"
  mkdir -p "$RUN_DIR"
  
  echo "Running data subset experiment: $RUN_NAME"
  echo "  Data subset: $subset_name"
  echo "  Test size: $test_size"
  echo "  Learning rate: $BEST_LR"
  echo "  Warmup steps: $BEST_WARMUP"
  echo "  Strategy: $BEST_STRATEGY"
  echo "  Output dir: $RUN_DIR"
  
  python finetune.py \
    "${BASE_ARGS[@]}" \
    --learning_rate "$BEST_LR" \
    --warmup_steps "$BEST_WARMUP" \
    --test_size "$test_size" \
    --val_size 0.2 \
    --save_model_path "$RUN_DIR/best_model.pth" \
    > "$RUN_DIR/finetune.log" 2>&1
  
  echo "Completed: $RUN_NAME"
  echo "---"
done

echo "All transformer miniprogram fine-tuning experiments completed!"
echo "Results can be found in: $EXPERIMENT_DIR"

# Generate summary
echo "Generating experiment summary..."
echo "Transformer Miniprogram Fine-tuning Results Summary" > "$EXPERIMENT_DIR/summary.txt"
echo "====================================================" >> "$EXPERIMENT_DIR/summary.txt"
echo "" >> "$EXPERIMENT_DIR/summary.txt"
echo "Hyperparameter Tuning Runs:" >> "$EXPERIMENT_DIR/summary.txt"
for lr in "${LEARNING_RATES[@]}"; do
  for warmup in "${WARMUP_STEPS[@]}"; do
    for strategy in "${FREEZE_STRATEGIES[@]}"; do
      run_name="tune_lr${lr}_warmup${warmup}_${strategy}"
      if [[ -f "$EXPERIMENT_DIR/$run_name/finetune.log" ]]; then
        echo "  $run_name" >> "$EXPERIMENT_DIR/summary.txt"
      fi
    done
  done
done

echo "" >> "$EXPERIMENT_DIR/summary.txt"
echo "Data Subset Experiments:" >> "$EXPERIMENT_DIR/summary.txt"
for subset_name in "${!SUBSET_CONFIGS[@]}"; do
  run_name="final_${subset_name}_lr${BEST_LR}_warmup${BEST_WARMUP}_${BEST_STRATEGY}"
  if [[ -f "$EXPERIMENT_DIR/$run_name/finetune.log" ]]; then
    echo "  $run_name (${subset_name})" >> "$EXPERIMENT_DIR/summary.txt"
  fi
done

echo "Summary saved to: $EXPERIMENT_DIR/summary.txt"

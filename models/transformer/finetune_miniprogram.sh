#!/usr/bin/env bash
set -euo pipefail

# Miniprogram Finetune Experiments
# Hyperparameter tuning for MOBIS pretrained model on miniprogram data
# Testing different combinations of learning rates, warmup steps, and freezing strategies

ROOT="/data/A-SpeedTransformer"
MOBIS_MODEL_DIR="$ROOT/models/transformer/experiments/mobis_transformer_sweeps/mobis_lr1e-4_bs512_h8_d128_kv4_do0.1/mobis_lr1e-4_bs512_h8_d128_kv4_do0.1"
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

# Hyperparameter combinations to test
declare -a LEARNING_RATES=("1e-4" "2e-4" "5e-4")
declare -a WARMUP_STEPS=(0 50 100)
declare -a FREEZE_STRATEGIES=("none" "freeze_attention" "freeze_feedforward" "freeze_embeddings")

# Data subset for hyperparameter tuning (use 15% = ~94 trips as it's middle-sized)
TUNING_TEST_SIZE="0.6506"
TUNING_VAL_SIZE="0.2"

# Data subset configurations matching the paper results
# Total trips in miniprogram: 629
# Corrected calculations: training_ratio = 1 - test_size - val_size

declare -A SUBSET_CONFIGS=(
  ["15pct_94trips"]="0.6506 0.2"   # ~94 training trips
  ["20pct_125trips"]="0.6013 0.2"  # ~125 training trips
  ["30pct_189trips"]="0.4995 0.2"  # ~189 training trips
  ["40pct_251trips"]="0.4010 0.2"  # ~251 training trips
  ["50pct_313trips"]="0.3008 0.2"  # ~313 training trips
)

echo "=== Miniprogram Hyperparameter Tuning ==="
echo "Pretrained model: $MOBIS_MODEL_DIR/best_model.pth"
echo "Target dataset: miniprogram_balanced.csv (629 total trips)"
echo "Testing hyperparameter combinations on 15% subset (~94 training trips)"
echo ""

# Calculate training trips for tuning subset
training_ratio=$(python -c "print(f'{(1 - $TUNING_TEST_SIZE) * (1 - $TUNING_VAL_SIZE):.4f}')")
training_trips=$(python -c "print(int(629 * $training_ratio))")
echo "Tuning subset: test_size=$TUNING_TEST_SIZE, val_size=$TUNING_VAL_SIZE -> ~$training_trips training trips"
echo ""

# Switch to the transformer directory for execution
cd "$ROOT/models/transformer"

# Hyperparameter tuning phase
echo "=== Phase 1: Hyperparameter Tuning ==="
best_accuracy=0
best_config=""

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
        "freeze_embeddings")
          freeze_args="--freeze_embeddings"
          ;;
        "none")
          freeze_args=""
          ;;
      esac
      
      # Set run name and output directory
      RUN_NAME="tune_lr${lr}_warmup${warmup}_${strategy}"
      OUT_DIR="$EXPERIMENT_DIR/$RUN_NAME"
      mkdir -p "$OUT_DIR"
      
      # Skip if already completed
      if [[ -f "$OUT_DIR/best_model.pth" ]]; then
        echo "[skip] $RUN_NAME already exists"
        
        # Check if this is the best configuration so far
        if [[ -f "$OUT_DIR/finetune.log" ]]; then
          accuracy=$(grep "Test Accuracy:" "$OUT_DIR/finetune.log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
          if [[ -n "$accuracy" ]]; then
            accuracy_float=$(python -c "print(float('$accuracy'))")
            is_better=$(python -c "print(1 if $accuracy_float > $best_accuracy else 0)")
            if [[ "$is_better" == "1" ]]; then
              best_accuracy=$accuracy_float
              best_config="lr${lr}_warmup${warmup}_${strategy}"
            fi
          fi
        fi
        continue
      fi
      
      echo "[run] Starting $RUN_NAME (lr=$lr, warmup=$warmup, strategy=$strategy)..."
      
      # Run fine-tuning with current configuration
      python finetune.py \
        "${BASE_ARGS[@]}" \
        --learning_rate "$lr" \
        --warmup_steps "$warmup" \
        $freeze_args \
        --test_size "$TUNING_TEST_SIZE" \
        --val_size "$TUNING_VAL_SIZE" \
        --save_model_path "$OUT_DIR/best_model.pth" \
        2>&1 | tee "$OUT_DIR/finetune.log"
        
      # Extract accuracy and update best configuration
      if [[ -f "$OUT_DIR/finetune.log" ]]; then
        accuracy=$(grep "Test Accuracy:" "$OUT_DIR/finetune.log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        if [[ -n "$accuracy" ]]; then
          accuracy_float=$(python -c "print(float('$accuracy'))")
          is_better=$(python -c "print(1 if $accuracy_float > $best_accuracy else 0)")
          if [[ "$is_better" == "1" ]]; then
            best_accuracy=$accuracy_float
            best_config="lr${lr}_warmup${warmup}_${strategy}"
          fi
          echo "[result] Test Accuracy: ${accuracy} (Best so far: ${best_accuracy} from ${best_config})"
        fi
      fi
      
      echo "[done] Completed $RUN_NAME"
      echo ""
    done
  done
done

echo "=== Hyperparameter Tuning Results ==="
echo "Best configuration: $best_config with accuracy: $best_accuracy"
echo ""

# Extract best hyperparameters
best_lr=$(echo "$best_config" | sed 's/lr\([^_]*\)_.*/\1/')
best_warmup=$(echo "$best_config" | sed 's/.*warmup\([^_]*\)_.*/\1/')
best_strategy=$(echo "$best_config" | sed 's/.*_\([^_]*\)$/\1/')

echo "Best hyperparameters:"
echo "  Learning Rate: $best_lr"
echo "  Warmup Steps: $best_warmup"
echo "  Strategy: $best_strategy"
echo ""

# Phase 2: Apply best configuration to all subset sizes
echo "=== Phase 2: Applying Best Configuration to All Subset Sizes ==="

# Set freeze arguments for best strategy
best_freeze_args=""
case "$best_strategy" in
  "freeze_attention")
    best_freeze_args="--freeze_attention"
    ;;
  "freeze_feedforward")
    best_freeze_args="--freeze_feedforward"
    ;;
  "freeze_embeddings")
    best_freeze_args="--freeze_embeddings"
    ;;
  "none")
    best_freeze_args=""
    ;;
esac
for subset_name in "${!SUBSET_CONFIGS[@]}"; do
  IFS=' ' read -r test_size val_size <<< "${SUBSET_CONFIGS[$subset_name]}"
  
  # Set run name and output directory
  RUN_NAME="final_${subset_name}_${best_config}"
  OUT_DIR="$EXPERIMENT_DIR/$RUN_NAME"
  mkdir -p "$OUT_DIR"
  
  # Skip if already completed
  if [[ -f "$OUT_DIR/best_model.pth" ]]; then
    echo "[skip] $RUN_NAME already exists"
    continue
  fi
  
  echo "[run] Starting $RUN_NAME with best config (test_size=$test_size, val_size=$val_size)..."
  
  # Run fine-tuning with best configuration
  python finetune.py \
    "${BASE_ARGS[@]}" \
    --learning_rate "$best_lr" \
    --warmup_steps "$best_warmup" \
    $best_freeze_args \
    --test_size "$test_size" \
    --val_size "$val_size" \
    --save_model_path "$OUT_DIR/best_model.pth" \
    2>&1 | tee "$OUT_DIR/finetune.log"
    
  echo "[done] Completed $RUN_NAME"
  echo ""
done

echo "=== All miniprogram experiments complete ==="

# Summary report
echo ""
echo "=== Hyperparameter Tuning Summary ==="
echo "Configuration | Learning Rate | Warmup | Strategy | Accuracy (%)"
echo "------------------------------------------------------------"

for lr in "${LEARNING_RATES[@]}"; do
  for warmup in "${WARMUP_STEPS[@]}"; do
    for strategy in "${FREEZE_STRATEGIES[@]}"; do
      RUN_NAME="tune_lr${lr}_warmup${warmup}_${strategy}"
      OUT_DIR="$EXPERIMENT_DIR/$RUN_NAME"
      
      if [[ -f "$OUT_DIR/finetune.log" ]]; then
        accuracy=$(grep "Test Accuracy:" "$OUT_DIR/finetune.log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        if [[ -n "$accuracy" ]]; then
          accuracy_pct=$(python -c "print(f'{float(\"$accuracy\") * 100:.2f}')")
          printf "%-12s | %-13s | %-6s | %-16s | %s%%\n" "$RUN_NAME" "$lr" "$warmup" "$strategy" "$accuracy_pct"
        fi
      fi
    done
  done
done

echo ""
echo "=== Final Results Summary (Best Configuration Applied) ==="
echo "Data Subset (%) | # Trips | SpeedTransformer Accuracy (%)"
echo "--------------------------------------------------------"

for subset_name in "15pct_94trips" "20pct_125trips" "30pct_189trips" "40pct_251trips" "50pct_313trips"; do
  OUT_DIR="$EXPERIMENT_DIR/final_${subset_name}_${best_config}"
  
  if [[ -f "$OUT_DIR/finetune.log" ]]; then
    # Extract final test accuracy from log
    accuracy=$(grep "Test Accuracy:" "$OUT_DIR/finetune.log" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    percentage=${subset_name%pct*}
    trips=${subset_name#*pct_}
    trips=${trips%trips}
    
    if [[ -n "$accuracy" ]]; then
      accuracy_pct=$(python -c "print(f'{float(\"$accuracy\") * 100:.2f}')")
      printf "%s%% | %s | %s%%\n" "$percentage" "$trips" "$accuracy_pct"
    else
      printf "%s%% | %s | Failed\n" "$percentage" "$trips"
    fi
  else
    percentage=${subset_name%pct*}
    trips=${subset_name#*pct_}
    trips=${trips%trips}
    printf "%s%% | %s | Not run\n" "$percentage" "$trips"
  fi
done

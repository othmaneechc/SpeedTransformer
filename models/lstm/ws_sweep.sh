#!/usr/bin/env bash
set -euo pipefail

# ===== Environment =====
# If 'ml' exists, purge; otherwise ignore.
if command -v ml &>/dev/null; then ml purge; fi

# Activate torchgpu environment
source /data/oe23/miniconda3/etc/profile.d/conda.sh
conda activate torchgpu

# Pin to the single GPU
export CUDA_VISIBLE_DEVICES=0

# Threads & misc
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

# ===== Base configuration (using best LSTM params) =====
ROOT="/data/A-SpeedTransformer"
cd "$ROOT/models/lstm"

# Fixed hyperparameters (best from previous sweep)
LR="1e-3"
BS="128"
HIDDEN_DIM="128"
NUM_LAYERS="2"
DROPOUT="0.1"

# Datasets to process
declare -a DATASETS=(
  "geolife_processed.csv"
  # "mobis_processed.csv"
)

# Window sizes to test (sequence lengths for LSTM)
declare -a SIZES=(
  20 50 100 200 300 400 500
)

# ===== Runner (sequential on one GPU) =====
for dataset in "${DATASETS[@]}"; do
  DATASET_NAME="${dataset%_processed.csv}"  # Remove _processed.csv suffix
  DATA="$ROOT/data/$dataset"
  
  # Skip if dataset doesn't exist
  if [[ ! -f "$DATA" ]]; then
    echo "Warning: Dataset $DATA not found, skipping..."
    continue
  fi
  
  echo "Processing dataset: $DATASET_NAME for window size optimization"
  
  # ===== Window size sweep =====
  for ws in "${SIZES[@]}"; do
    # Construct run name
    RUN_NAME="${DATASET_NAME}_ws${ws}_lr${LR}_bs${BS}_h${HIDDEN_DIM}_l${NUM_LAYERS}_do${DROPOUT}"
    
    # Create experiment directory
    EXP_DIR="$ROOT/models/lstm/experiments/${DATASET_NAME}_window_sweeps/$RUN_NAME"
    mkdir -p "$EXP_DIR"
    
    echo "Starting window size experiment: $RUN_NAME"
    echo "  Window size: $ws"
    echo "  Learning rate: $LR"
    echo "  Batch size: $BS"
    echo "  Hidden dim: $HIDDEN_DIM"
    echo "  Num layers: $NUM_LAYERS"
    echo "  Dropout: $DROPOUT"
    echo "  Output dir: $EXP_DIR"
    
    # Run LSTM training with specific window size
    python lstm.py \
      --data_path "$DATA" \
      --window_size "$ws" \
      --learning_rate "$LR" \
      --batch_size "$BS" \
      --hidden_size "$HIDDEN_DIM" \
      --num_layers "$NUM_LAYERS" \
      --dropout "$DROPOUT" \
      --random_state 42 \
      --test_size 0.15 \
      --val_size 0.15 \
      --checkpoint_dir "$EXP_DIR" \
      --scaler_path "$EXP_DIR/scaler.joblib" \
      --label_encoder_path "$EXP_DIR/label_encoder.joblib" \
      > "$EXP_DIR/train.log" 2>&1
    
    echo "Completed window size: $ws"
    echo "Log saved to: $EXP_DIR/train.log"
    echo "---"
  done
  
  echo "Completed all window size experiments for dataset: $DATASET_NAME"
  echo "======================================="
done

echo "All window size sweeps completed!"
echo "Results can be found in: $ROOT/models/lstm/experiments/"

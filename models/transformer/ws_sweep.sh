#!/usr/bin/env bash
set -euo pipefail

# ===== Environment (mirrors your sbatch) =====
# If 'ml' exists, purge; otherwise ignore.
if command -v ml &>/dev/null; then ml purge; fi

# Pin to the single GPU
export CUDA_VISIBLE_DEVICES=0

# Threads & misc
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

# ===== Base configuration (using best params) =====
ROOT="/data/A-SpeedTransformer"
cd "$ROOT/models/transformer"

# Fixed hyperparameters (best from previous sweep)
LR="2e-4"
BS="256"
NHEAD="8"
DMODEL="128"
KVH="4"
DROPOUT="0.1"

# Datasets to process
declare -a DATASETS=(
  "geolife_processed.csv"
  # "mobis_processed.csv"
)

# Window sizes to test
declare -a SIZES=(
  # 20 50 100 200 
  # 300 
  400 500
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

  echo "Processing dataset: $DATASET_NAME"
  
  for WS in "${SIZES[@]}"; do
    RUN_NAME="${DATASET_NAME}_ws${WS}_lr${LR}_bs${BS}_h${NHEAD}_d${DMODEL}_kv${KVH}_do${DROPOUT}"
    OUT_DIR="$ROOT/models/transformer/experiments/${DATASET_NAME}_window_sweeps/${RUN_NAME}"
    mkdir -p "$OUT_DIR"

    # Skip if already finished
    if [[ -f "$OUT_DIR/best_model.pth" ]]; then
        echo "[skip] $RUN_NAME already exists"
        continue
    fi

    echo "[run] Window Size: $WS - $RUN_NAME"
    python -u "$ROOT/models/transformer/train.py" \
        --data_path "$DATA" \
        --feature_columns speed \
        --target_column label \
        --traj_id_column traj_id \
        --test_size 0.15 \
        --val_size 0.15 \
        --random_state 316 \
        --chunksize 1000000 \
        --window_size "$WS" \
        --stride 50 \
        --batch_size "$BS" \
        --num_workers 4 \
        --num_epochs 50 \
        --patience 7 \
        --d_model "$DMODEL" \
        --nhead "$NHEAD" \
        --num_layers 4 \
        --dropout "$DROPOUT" \
        --learning_rate "$LR" \
        --weight_decay 1e-4 \
        --gradient_clip 1.0 \
        --kv_heads "$KVH" \
        --use_amp \
        --run_name "$RUN_NAME" \
        --out_dir "$OUT_DIR" \
        --save_model_path "$OUT_DIR/best_model.pth" \
        --save_label_encoder_path "$OUT_DIR/label_encoder.joblib" \
        --save_scaler_path "$OUT_DIR/scaler.joblib" \
        > "$OUT_DIR/train.log" 2>&1

    echo "[done] $RUN_NAME"
  done
done

echo "All window size sweeps complete."
# ===== End ====
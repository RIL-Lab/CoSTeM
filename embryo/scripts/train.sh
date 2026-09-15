#!/bin/bash
# Launch a single EmbryoNet (EQENet) training run.
#
# Usage:
#   ROOT_PATH=/path/to/embryo_videos bash train.sh
#   ROOT_PATH=/path/to/embryo_videos SEED=3407 EPOCHS=50 NPROC=1 bash train.sh
#   CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash train.sh
#
# All settings are passed through environment variables, see the table below.
set -euo pipefail

cd "$(dirname "$0")"

ROOT_PATH="${ROOT_PATH:-../data/embryo_videos}"  # dataset root (videos + train.xlsx / val.xlsx)
OUTPUT_DIR="${OUTPUT_DIR:-../experiments}"       # where logs, configs and checkpoints are written
SEED="${SEED:-3407}"
EXP_NAME="${EXP_NAME:-EmbryoNet.seed_${SEED}}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-2.5e-5}"
TASK="${TASK:-Grading}"                        # Grading (3 classes) or Evaluation (2 classes)
NUM_CLASSES="${NUM_CLASSES:-3}"
MASTER_PORT="${MASTER_PORT:-29571}"
NPROC="${NPROC:-1}"                            # number of GPUs / processes per node
EXTRA_ARGS="${EXTRA_ARGS:-}"                   # extra flags forwarded to train_new_version.py

mkdir -p "$OUTPUT_DIR"

echo "[train.sh] exp_name=$EXP_NAME seed=$SEED root_path=$ROOT_PATH output_dir=$OUTPUT_DIR"

torchrun --nproc-per-node "$NPROC" --master_port "$MASTER_PORT" train_new_version.py \
    --exp_name "$EXP_NAME" \
    --seed "$SEED" \
    --task "$TASK" \
    --num_classes "$NUM_CLASSES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --root_path "$ROOT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

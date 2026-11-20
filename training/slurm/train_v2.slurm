#!/bin/bash
# =============================================================================
# Cluster example: train ONE variant from scratch on a single-GPU node (the released models were
# trained this way, one 24 h job each on an NVIDIA GH200). Adapt the #SBATCH lines and the three
# folders to your cluster. Submit from the repository root, for example:
#   sbatch --export=ALL,NAME=v2_s8c512_seed0,S=8,CROP=512 -J v2_s8c512_seed0 training/slurm/train_v2.slurm
# Resubmitting the same command resumes from runs/<NAME>/ckpt/latest.pth; a DONE marker ends it.
# =============================================================================
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
# #SBATCH --partition=<your GPU partition>
# #SBATCH --account=<your allocation>

set -euo pipefail
: "${NAME:?set NAME}" "${S:?set S (1/2/4/8)}" "${CROP:?set CROP (128/256/512/512 for s1/s2/s4/s8)}"
FIELD_DIR="${FIELD_DIR:-data/DIV2K_train_HR}"     # DIV2K training images (see training/README.md)
RUNS_ROOT="${RUNS_ROOT:-training/runs}"           # checkpoints, log.csv, args.json
CACHE_ROOT="${CACHE_ROOT:-training/cache}"        # the fixed validation set
BATCH="${BATCH:-8}"; ITERS="${ITERS:-60000}"; LR="${LR:-2e-4}"; SEED="${SEED:-0}"
MIN_EDGE="${MIN_EDGE:-0.125}"; P_DETAIL="${P_DETAIL:-1.0}"; OTF="${OTF:-auto}"; OTF_IMPL="${OTF_IMPL:-loop}"

echo "=== $NAME (s$S crop $CROP batch $BATCH iters $ITERS lr $LR seed $SEED min_edge $MIN_EDGE p_detail $P_DETAIL otf $OTF) | job ${SLURM_JOB_ID:-?} @ $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ -f "$RUNS_ROOT/$NAME/DONE" ]; then echo "[done] DONE marker present"; exit 0; fi
[ "$(ls "$FIELD_DIR"/*.png 2>/dev/null | wc -l)" -ge 100 ] || { echo "ERROR: no DIV2K images in $FIELD_DIR"; exit 1; }

srun python -u training/train_v2.py --name "$NAME" --s "$S" --crop "$CROP" \
    --batch "$BATCH" --total-iters "$ITERS" --lr "$LR" --seed "$SEED" \
    --min-edge "$MIN_EDGE" --p-detail "$P_DETAIL" --otf "$OTF" --otf-impl "$OTF_IMPL" \
    --field-dir "$FIELD_DIR" --runs-root "$RUNS_ROOT" --cache-root "$CACHE_ROOT" \
    --eval-every 1000 --val-n 64 --ckpt-every 5000
echo "=== $NAME finished @ $(date) ==="

#!/bin/bash
# DINOv2 ViT-S/14, 50 epochs on SSv2 frames, 1x L40S (48 GB).
#
# Submit from the repo root:
#   export WANDB_API_KEY=...        # once per shell; sbatch passes it to the job
#   sbatch scripts/train_ssv2_l40s.sbatch
#
# If the job dies (time limit, preemption), submit the same command again:
# training resumes from the last checkpoint in OUT_DIR.

#SBATCH --job-name=dinov2-ssv2
# L40S partitions; Slurm starts the job on whichever has a free GPU first.
# Override at submit time if needed: sbatch -p gl40s_long scripts/train_ssv2_l40s.sbatch
#SBATCH --partition=gl40s_short,gl40s_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

CONDA_ENV=/gpfs/scratch/sd6701/conda-envs/dinov2
SSV2=/gpfs/scratch/sd6701/personal/salt/datasets/ssv2
OUT_DIR=./out/ssv2_50ep_bs128
BATCH_SIZE=128          # per GPU; ViT-S fits easily in 48 GB. Drop to 96 if you hit OOM.
NUM_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))   # video decoding is the bottleneck

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONPATH=.
export OMP_NUM_THREADS=1

echo "Job $SLURM_JOB_ID on $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
[ -n "${WANDB_API_KEY:-}" ] || echo "WARNING: WANDB_API_KEY is not set; wandb logging will fail"

torchrun --standalone --nproc_per_node=1 dinov2/train/train.py \
  --config-file dinov2/configs/train/vits14_video.yaml \
  --output-dir "$OUT_DIR" \
  --enable-wandb --wandb-name "ssv2-50ep-bs${BATCH_SIZE}" \
  train.dataset_path=VideoFiles:root=$SSV2/labels/train.json:extra=$SSV2/20bn-something-something-v2 \
  train.batch_size_per_gpu=$BATCH_SIZE \
  train.num_workers=$NUM_WORKERS \
  optim.epochs=50 \
  optim.warmup_epochs=5 \
  teacher.warmup_teacher_temp_epochs=10

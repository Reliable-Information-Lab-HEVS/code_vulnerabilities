#!/bin/bash
#SBATCH --job-name=cweval-embeddings
#SBATCH --output=run_logs/%x-%j.out
#SBATCH --error=run_logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nodes
#SBATCH --chdir=/cluster/raid/home/stea/CWEval
#SBATCH --qos=normal

source ~/.conda/envs/llm/bin/activate llm 2>/dev/null \
    || { source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate llm; }

mkdir -p run_logs

echo "Starting embedding computation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Time:   $(date)"
echo "=============================================="

python compute_embeddings.py "$@"

echo "=============================================="
echo "Finished at $(date)"

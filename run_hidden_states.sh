#!/bin/bash
#SBATCH --job-name=cweval-hidden
#SBATCH --output=run_logs/%x-%j.out
#SBATCH --error=run_logs/%x-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:5
#SBATCH --partition=nodes
#SBATCH --chdir=/cluster/raid/home/stea/CWEval
#SBATCH --qos=normal

source ~/.conda/envs/llm/bin/activate llm 2>/dev/null \
    || { source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate llm; }

mkdir -p run_logs

echo "Starting hidden-state computation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Time:   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv | head -5
echo "=============================================="

python compute_hidden_states.py "$@"

echo "=============================================="
echo "Finished at $(date)"

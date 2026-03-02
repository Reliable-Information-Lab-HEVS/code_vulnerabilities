#!/bin/bash
#SBATCH --job-name=cweval_evaluate
#SBATCH --output=run_logs/%x-%j.out
#SBATCH --error=run_logs/%x-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --partition=nodes
#SBATCH --chdir=/cluster/raid/home/stea/CWEval
#SBATCH --qos=normal

# ============================================================
# SETTINGS
# ============================================================

COMMON_ENV="source /home/ubuntu/miniforge3/etc/profile.d/conda.sh &&
            source .env"

# ============================================================
# 1. Launch evaluation
# ============================================================

echo "Running CWEval evaluation..."

apptainer exec --nv cweval.sif \
bash -c "
    $COMMON_ENV ;
    python cweval/evaluate.py pipeline \
        --eval_path evals/c_mutated_prompts_character_Qwen3-Coder-30B-A3B-Instruct \
        --num_proc 20 \
        --docker False \

"

echo "Job complete."

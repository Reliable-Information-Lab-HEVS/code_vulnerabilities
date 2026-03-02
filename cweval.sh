#!/bin/bash
#SBATCH --job-name=cweval_run
#SBATCH --output=run_logs/%x-%j.out
#SBATCH --error=run_logs/%x-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=11
#SBATCH --mem=50G
#SBATCH --partition=nodes
#SBATCH --chdir=/cluster/raid/home/stea/CWEval
#SBATCH --qos=normal
#SBATCH --nodelist=gpu-node1

# ============================================================
# ARGUMENTS
# ============================================================

MODEL_NAME="$1"
PORT="$2"
LANGUAGE="$3"

#Run name is combination model_name_language
RUN_NAME="${MODEL_NAME}_${LANGUAGE}"

# CHECK IF model name is provided
if [ -z "$MODEL_NAME" ] || [ -z "$PORT" ] || [ -z "$LANGUAGE" ]; then
    echo "Usage: sbatch cweval.sh <model_name> <port> <language>"
    exit 1
fi

# ============================================================
# SETTINGS
# ============================================================

PREFACE="${LANGUAGE}_mutated3_prompts_character_"
LANGUAGES="$LANGUAGE"

COMMON_ENV="source /home/ubuntu/miniforge3/etc/profile.d/conda.sh &&
            source .env"

# LOCAL USE: API_BASE="http://localhost:${PORT}/v1"
API_BASE="https://api.swissai.cscs.ch/v1"

# Derived names
MODEL="hosted_vllm/${MODEL_NAME}"
SHORT_MODEL_NAME="${MODEL_NAME##*/}"
NAME="${PREFACE}${SHORT_MODEL_NAME}"
DATASETS="new_datasets/$NAME"
EVAL_PATH="evals/$NAME"

echo "=============================================="
echo "Model        : $MODEL_NAME"
echo "API base     : $API_BASE"
echo "Dataset name : $NAME"
echo "=============================================="

# ============================================================
# 1. Handle new data
# ============================================================

echo "Checking for new data to handle..."

apptainer exec --nv cweval.sif \
bash -c "
    $COMMON_ENV ;
    python cweval/handle_new_data.py \
        --datasets $DATASETS
"

# ============================================================
# 2. Run CWEval generator
# ============================================================

echo "Running CWEval generator..."

apptainer exec --nv cweval.sif \
bash -c "
    $COMMON_ENV ;
    python cweval/generate.py gen \
        --n 10 \
        --temperature 0.8 \
        --num_proc 5 \
        --eval_path $EVAL_PATH \
        --model $MODEL \
        --api_base $API_BASE \
        --langs $LANGUAGES
"

# ============================================================
# 3. Launch evaluation
# ============================================================
echo "Running CWEval evaluation..."
apptainer exec --nv cweval.sif \
bash -c "
    $COMMON_ENV ;
    python cweval/evaluate.py pipeline \
        --eval_path $EVAL_PATH \
        --num_proc 20 \
        --docker False \

"

# ============================================================
# 4. Cleanup
# ============================================================

echo "Cleaning up new data..."
python cweval/cleanup_new_data.py --datasets $DATASETS

echo "Job complete."

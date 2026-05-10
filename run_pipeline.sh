#!/bin/bash
#SBATCH --job-name=cweval-pipeline
#SBATCH --output=run_logs/%x-%j.out
#SBATCH --error=run_logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=nodes
#SBATCH --chdir=/cluster/raid/home/stea/CWEval
#SBATCH --qos=normal
#
# End-to-end probing pipeline (steps 1+2 run inside this slurm job;
# step 3 submits one GPU slurm job per model).
#
# Submit with:
#   sbatch run_pipeline.sh --evals-dir <dir> --name <name> --categories <...> [...]
#
# Goes from a folder of res_all.json eval files to probe-result CSVs in a
# single command. Designed to leave previous results untouched (everything
# new is written under results/<NAME>/).
#
# Usage:
#   bash run_pipeline.sh \
#       --evals-dir evals_temp0 \
#       --name      temp0 \
#       --categories mutated_prompts_character mutated3_prompts_character mutated_token_replacement \
#       [--src-emb-dir embeddings] \
#       [--models Qwen3-Coder-30B-A3B-Instruct deepseek-coder-33b-instruct CodeLlama-70b-Instruct-hf] \
#       [--languages py c cpp js go]                    # default: all five
#       [--features layer_48_last_pos]                  # optional: probe only these features
#       [--pca-dims 0 50 100]                           # default: 0
#       [--dropouts 0.1 0.3]                            # default: 0.3
#       [--hidden-sizes default 128,32]                 # default: default
#       [--probes mlp2 logreg]                          # default: mlp2 logreg
#       [--logreg-c 0.001 0.01 0.1 1.0]                 # default: 1.0 (logreg only)
#       [--min-sig-frac 0.20]                           # default: 0.20
#       [--skip-update | --skip-labels | --skip-probe]
#
# Outputs:
#   results/<NAME>/
#     embeddings/<model>_hidden_states_multilayer.npz   # updated metadata
#     stat_change_labels.json
#     joint_tests.parquet
#     probe_results/<model>.csv (one slurm job per model)
#     pipeline.log
#     config.json
#
# Steps 1-2 run inline on the login node (CPU, fast); step 3 (probing) is
# submitted as one slurm job per model, all writing into the same
# results/<NAME>/probe_results/ folder.

cd /cluster/raid/home/stea/CWEval

# ── Defaults ────────────────────────────────────────────────────────────
EVALS_DIR=""
NAME=""
CATEGORIES=()
SRC_EMB_DIR="embeddings"
MODELS=("Qwen3-Coder-30B-A3B-Instruct" "deepseek-coder-33b-instruct" "CodeLlama-70b-Instruct-hf")
LANGUAGES=(py c cpp js go)
FEATURES=()
PCA_DIMS=(0)
DROPOUTS=(0.3)
HIDDEN_SIZES=(default)
PROBES=(mlp2 logreg)
LOGREG_CS=(1.0)
MIN_SIG_FRAC=0.20
SKIP_UPDATE=0
SKIP_LABELS=0
SKIP_PROBE=0

# ── Parse args ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --evals-dir)       EVALS_DIR="$2"; shift 2 ;;
        --name)            NAME="$2"; shift 2 ;;
        --categories)      shift; CATEGORIES=();
                           while [[ $# -gt 0 && "$1" != --* ]]; do CATEGORIES+=("$1"); shift; done ;;
        --src-emb-dir)     SRC_EMB_DIR="$2"; shift 2 ;;
        --models)          shift; MODELS=();
                           while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
        --languages)       shift; LANGUAGES=();
                           while [[ $# -gt 0 && "$1" != --* ]]; do LANGUAGES+=("$1"); shift; done ;;
        --features)        shift; FEATURES=();
                           while [[ $# -gt 0 && "$1" != --* ]]; do FEATURES+=("$1"); shift; done ;;
        --pca-dims)        shift; PCA_DIMS=();
                           while [[ $# -gt 0 && "$1" != --* ]]; do PCA_DIMS+=("$1"); shift; done ;;
        --dropouts)        shift; DROPOUTS=();
                           while [[ $# -gt 0 && "$1" != --* ]]; do DROPOUTS+=("$1"); shift; done ;;
        --hidden-sizes)    shift; HIDDEN_SIZES=();
                           while [[ $# -gt 0 && "$1" != --* ]]; do HIDDEN_SIZES+=("$1"); shift; done ;;
        --probes)          shift; PROBES=();
                           while [[ $# -gt 0 && "$1" != --* ]]; do PROBES+=("$1"); shift; done ;;
        --logreg-c)        shift; LOGREG_CS=();
                           while [[ $# -gt 0 && "$1" != --* ]]; do LOGREG_CS+=("$1"); shift; done ;;
        --min-sig-frac)    MIN_SIG_FRAC="$2"; shift 2 ;;
        --skip-update)     SKIP_UPDATE=1; shift ;;
        --skip-labels)     SKIP_LABELS=1; shift ;;
        --skip-probe)      SKIP_PROBE=1; shift ;;
        *)                 echo "Unknown arg: $1"; exit 2 ;;
    esac
done

if [[ -z "$EVALS_DIR" || -z "$NAME" || ${#CATEGORIES[@]} -eq 0 ]]; then
    echo "Required: --evals-dir, --name, --categories"
    echo "Run with no args to see full usage."
    head -40 "$0"
    exit 2
fi

RESULT_DIR="results/$NAME"
OVERLAY="$RESULT_DIR/metadata_overlay.parquet"
LABELS_JSON="$RESULT_DIR/stat_change_labels.json"
PARQUET="$RESULT_DIR/joint_tests.parquet"
PROBE_DIR="$RESULT_DIR/probe_results"
LOG="$RESULT_DIR/pipeline.log"
CFG="$RESULT_DIR/config.json"

mkdir -p "$RESULT_DIR" "$PROBE_DIR"

# Activate the conda env that has the analysis dependencies.
# (Done before strict-mode kicks in: conda's init hooks can have non-zero
# returns that 'set -e' would treat as a fatal error.)
source ~/.conda/envs/llm/bin/activate llm 2>/dev/null \
    || { source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate llm; }
set -euo pipefail

# Save what was run.
NAME="$NAME" EVALS_DIR="$EVALS_DIR" SRC_EMB_DIR="$SRC_EMB_DIR" \
    CATEGORIES_S="$(IFS=,; echo "${CATEGORIES[*]}")" \
    MODELS_S="$(IFS=,; echo "${MODELS[*]}")" \
    LANGUAGES_S="$(IFS=,; echo "${LANGUAGES[*]}")" \
    FEATURES_S="$(IFS=,; echo "${FEATURES[*]:-}")" \
    PCA_DIMS_S="$(IFS=,; echo "${PCA_DIMS[*]}")" \
    DROPOUTS_S="$(IFS=,; echo "${DROPOUTS[*]}")" \
    HIDDEN_SIZES_S="$(IFS=,; echo "${HIDDEN_SIZES[*]}")" \
    PROBES_S="$(IFS=,; echo "${PROBES[*]}")" \
    LOGREG_CS_S="$(IFS=,; echo "${LOGREG_CS[*]}")" \
    MIN_SIG_FRAC="$MIN_SIG_FRAC" \
    CFG="$CFG" \
python -c "
import json, os
cfg = dict(
    name=os.environ['NAME'],
    evals_dir=os.environ['EVALS_DIR'],
    src_emb_dir=os.environ['SRC_EMB_DIR'],
    categories=os.environ.get('CATEGORIES_S', '').split(','),
    models=os.environ.get('MODELS_S', '').split(','),
    languages=os.environ.get('LANGUAGES_S', '').split(','),
    features=[s for s in os.environ.get('FEATURES_S', '').split(',') if s],
    pca_dims=[int(x) for x in os.environ.get('PCA_DIMS_S', '').split(',') if x],
    dropouts=[float(x) for x in os.environ.get('DROPOUTS_S', '').split(',') if x],
    hidden_sizes=os.environ.get('HIDDEN_SIZES_S', '').split(','),
    probes=os.environ.get('PROBES_S', '').split(','),
    logreg_cs=[float(x) for x in os.environ.get('LOGREG_CS_S', '').split(',') if x],
    min_sig_frac=float(os.environ['MIN_SIG_FRAC']),
)
json.dump(cfg, open(os.environ['CFG'], 'w'), indent=2, default=str)
"

echo "===== Pipeline started: $(date) =====" | tee -a "$LOG"
echo "  name=$NAME" | tee -a "$LOG"
echo "  evals_dir=$EVALS_DIR" | tee -a "$LOG"
echo "  categories=${CATEGORIES[*]}" | tee -a "$LOG"
echo "  models=${MODELS[*]}" | tee -a "$LOG"
echo "  languages=${LANGUAGES[*]}" | tee -a "$LOG"
echo "  result_dir=$RESULT_DIR" | tee -a "$LOG"

# ── Step 1: write metadata overlay (small parquet, no npz copy) ────────
if [[ "$SKIP_UPDATE" == "1" ]]; then
    echo "[1/3] write_metadata_overlay.py — SKIPPED (--skip-update)" | tee -a "$LOG"
else
    echo "[1/3] write_metadata_overlay.py" | tee -a "$LOG"
    python write_metadata_overlay.py \
        --evals-dir "$EVALS_DIR" \
        --categories "${CATEGORIES[@]}" \
        --models "${MODELS[@]}" \
        --languages "${LANGUAGES[@]}" \
        --out "$OVERLAY" 2>&1 | tee -a "$LOG"
fi

# ── Step 2: build stat_change_labels.json + joint_tests.parquet ────────
if [[ "$SKIP_LABELS" == "1" ]]; then
    echo "[2/3] build_stat_change_labels.py — SKIPPED (--skip-labels)" | tee -a "$LOG"
else
    echo "[2/3] build_stat_change_labels.py" | tee -a "$LOG"
    python build_stat_change_labels.py \
        --evals-dir "$EVALS_DIR" \
        --categories "${CATEGORIES[@]}" \
        --models "${MODELS[@]}" \
        --languages "${LANGUAGES[@]}" \
        --out-json "$LABELS_JSON" \
        --out-parquet "$PARQUET" 2>&1 | tee -a "$LOG"
fi

# ── Step 3: probing — submit one slurm job per model ───────────────────
if [[ "$SKIP_PROBE" == "1" ]]; then
    echo "[3/3] probe_embeddings.py — SKIPPED (--skip-probe)" | tee -a "$LOG"
else
    echo "[3/3] probe_embeddings.py — submitting one slurm job per model" | tee -a "$LOG"
    EXTRA_ARGS=()
    if [[ ${#FEATURES[@]} -gt 0 ]]; then
        EXTRA_ARGS+=(--features "${FEATURES[@]}")
    fi
    PROBE_JOBS=()
    for MODEL in "${MODELS[@]}"; do
        OUT_CSV="$PROBE_DIR/${MODEL}.csv"
        JOB=$(sbatch --parsable run_probing.sh \
            --emb-dir "$SRC_EMB_DIR" \
            --metadata-overlay "$OVERLAY" \
            --labels-path "$LABELS_JSON" \
            --models "$MODEL" \
            --probes "${PROBES[@]}" \
            --pca-dims "${PCA_DIMS[@]}" \
            --dropouts "${DROPOUTS[@]}" \
            --hidden-sizes "${HIDDEN_SIZES[@]}" \
            --logreg-c "${LOGREG_CS[@]}" \
            --min-sig-frac "$MIN_SIG_FRAC" \
            --out "$OUT_CSV" \
            "${EXTRA_ARGS[@]}")
        PROBE_JOBS+=("$JOB")
        echo "  $MODEL  →  jobid $JOB  →  $OUT_CSV" | tee -a "$LOG"
    done

    # Final summary — depends on *all* probe jobs (afterany so it still runs
    # even if one model fails; the summary script just skips missing CSVs).
    SUMMARY_OUT="$RESULT_DIR/summary.csv"
    DEP="afterany:$(IFS=:; echo "${PROBE_JOBS[*]}")"
    SUM_JOB=$(sbatch --parsable --dependency="$DEP" run_summary.sh \
        --probe-dir "$PROBE_DIR" \
        --out "$SUMMARY_OUT")
    echo "  summary  →  jobid $SUM_JOB  →  $SUMMARY_OUT  (depends on ${PROBE_JOBS[*]})" | tee -a "$LOG"
fi

echo "===== Pipeline submitted: $(date) =====" | tee -a "$LOG"
echo
echo "Watch progress:"
echo "  tail -f $LOG"
echo "  squeue -u \$USER"
echo "Results will land in $PROBE_DIR/<model>.csv"

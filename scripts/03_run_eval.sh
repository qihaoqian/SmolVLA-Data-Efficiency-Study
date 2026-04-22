#!/usr/bin/env bash
# Step 3: Evaluate all fine-tuned checkpoints and copy the zero-shot baseline.
#
# Run from the lerobot root directory:
#   bash project1_vla_data_efficiency/scripts/03_run_eval.sh
#   bash project1_vla_data_efficiency/scripts/03_run_eval.sh 0.50
#   bash project1_vla_data_efficiency/scripts/03_run_eval.sh 0.10 0.50
#   bash project1_vla_data_efficiency/scripts/03_run_eval.sh --finetune-output-base-dir /tmp/ft --results-dir /tmp/eval 0.25
#   CKPT_PATH=/path/to/pretrained_model bash .../03_run_eval.sh
#   bash .../03_run_eval.sh --ckpt-path /path/to/pretrained_model
#
# If CKPT_PATH is not set (no env, no --ckpt-path), checkpoints are taken from
# FINETUNE_OUTPUT_BASE_DIR/<frac>/checkpoints/last/pretrained_model (non-interactive).
#
# Output: project1_vla_data_efficiency/results/<fraction>/eval_info.json
#
# Each eval run takes ~20–40 min (500 episodes, 10 parallel envs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FINETUNE_OUTPUT_BASE_DIR="${PROJECT_ROOT}/outputs/finetuning-action-expert"
RESULTS_DIR="${PROJECT_ROOT}/results"
TARGET_FRACTIONS=()
CKPT_PATH_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt-path)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --ckpt-path requires a value" >&2
                exit 1
            fi
            CKPT_PATH_OVERRIDE="$2"
            shift 2
            ;;
        --finetune-output-base-dir)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --finetune-output-base-dir requires a value" >&2
                exit 1
            fi
            FINETUNE_OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --results-dir)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --results-dir requires a value" >&2
                exit 1
            fi
            RESULTS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [--ckpt-path DIR] [--finetune-output-base-dir DIR] [--results-dir DIR] [FRACTION ...]

Examples:
  $(basename "$0")
  $(basename "$0") 0.25
  $(basename "$0") --ckpt-path /path/to/pretrained_model
  $(basename "$0") --finetune-output-base-dir /tmp/ft --results-dir /tmp/eval 0.10 0.25
EOF
            exit 0
            ;;
        *)
            TARGET_FRACTIONS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$CKPT_PATH_OVERRIDE" ]]; then
    CKPT_PATH_OVERRIDE="${CKPT_PATH:-}"
fi

mkdir -p "$RESULTS_DIR"

# ── 0. Copy zero-shot baseline ────────────────────────────────────────────────
ZERO_SHOT_SRC="${PROJECT_ROOT}/outputs/eval/2026-04-06/22-05-32_libero_smolvla/eval_info.json"
mkdir -p "$RESULTS_DIR/zero_shot"
if [ -f "$ZERO_SHOT_SRC" ]; then
    cp "$ZERO_SHOT_SRC" "$RESULTS_DIR/zero_shot/eval_info.json"
    echo "[zero_shot] Copied baseline from $ZERO_SHOT_SRC"
else
    echo "[zero_shot] WARNING: baseline not found at $ZERO_SHOT_SRC — skipping"
fi

# ── Eval helper ──────────────────────────────────────────────────────────────
fraction_to_label() {
    local fraction="$1"
    case "$fraction" in
        1|1.0|1.00|frac100) echo "frac100" ;;
        0.5|0.50|frac050) echo "frac050" ;;
        0.25|frac025) echo "frac025" ;;
        0.1|0.10|frac010) echo "frac010" ;;
        0.05|frac005) echo "frac005" ;;
        *)
            echo "ERROR: unsupported fraction '$fraction'. Use one of: 1.00, 0.50, 0.25, 0.10, 0.05" >&2
            exit 1
            ;;
    esac
}

run_lerobot_eval() {
    local CKPT_PATH="$1"
    local OUT_DIR="$2"
    local LOG_TAG="$3"

    mkdir -p "$OUT_DIR"
    echo ""
    echo "[$LOG_TAG] Evaluating checkpoint: $CKPT_PATH"

    MUJOCO_GL=egl python3 -m lerobot.scripts.lerobot_eval \
        --policy.type=smolvla \
        --policy.pretrained_path="$CKPT_PATH" \
        --env.type=libero \
        --env.task=libero_spatial \
        --eval.n_episodes=200 \
        --eval.batch_size=4 \
        --output_dir="$OUT_DIR"

    echo "[$LOG_TAG] Done. Results at $OUT_DIR/eval_info.json"
}

eval_checkpoint() {
    local fraction="$1"
    local LABEL
    LABEL="$(fraction_to_label "$fraction")"
    local CKPT_PATH="${FINETUNE_OUTPUT_BASE_DIR}/${LABEL}/checkpoints/last/pretrained_model"
    local OUT_DIR="$RESULTS_DIR/${LABEL}"

    if [ ! -d "$CKPT_PATH" ]; then
        echo "[$LABEL] WARNING: checkpoint not found at $CKPT_PATH — skipping"
        return
    fi

    run_lerobot_eval "$CKPT_PATH" "$OUT_DIR" "$LABEL"
}

eval_checkpoint_explicit() {
    local CKPT_PATH="$1"
    local OUT_DIR="$RESULTS_DIR/explicit_ckpt"

    if [ ! -d "$CKPT_PATH" ]; then
        echo "ERROR: checkpoint not found at $CKPT_PATH" >&2
        exit 1
    fi

    run_lerobot_eval "$CKPT_PATH" "$OUT_DIR" "explicit_ckpt"
}

# ── 1. Evaluate target fractions (from args or defaults) ─────────────────────
DEFAULT_FRACTIONS=("0.05" "0.10" "0.25" "0.50" "1.00")
if [[ -z "$CKPT_PATH_OVERRIDE" ]] && [[ ${#TARGET_FRACTIONS[@]} -eq 0 ]]; then
    TARGET_FRACTIONS=("${DEFAULT_FRACTIONS[@]}")
fi

if [[ -n "$CKPT_PATH_OVERRIDE" ]]; then
    if [[ ${#TARGET_FRACTIONS[@]} -gt 0 ]]; then
        echo "NOTE: explicit checkpoint set (--ckpt-path or CKPT_PATH); ignoring fraction arguments." >&2
    fi
    echo "Explicit checkpoint: ${CKPT_PATH_OVERRIDE}"
    echo "Results dir: ${RESULTS_DIR}"
    eval_checkpoint_explicit "$CKPT_PATH_OVERRIDE"
else
    echo "Target fractions: ${TARGET_FRACTIONS[*]}"
    echo "Checkpoint base dir: ${FINETUNE_OUTPUT_BASE_DIR}"
    echo "Results dir: ${RESULTS_DIR}"
    for fraction in "${TARGET_FRACTIONS[@]}"; do
        eval_checkpoint "$fraction"
    done
fi

echo ""
echo "All evaluations complete."
echo "Run analysis: python3 project1_vla_data_efficiency/scripts/04_analyze_results.py"

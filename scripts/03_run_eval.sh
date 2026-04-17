#!/usr/bin/env bash
# Step 3: Evaluate all fine-tuned checkpoints and copy the zero-shot baseline.
#
# Run from the lerobot root directory:
#   bash project1_vla_data_efficiency/scripts/03_run_eval.sh
#   bash project1_vla_data_efficiency/scripts/03_run_eval.sh frac050
#   bash project1_vla_data_efficiency/scripts/03_run_eval.sh frac010 frac050
#
# Output: project1_vla_data_efficiency/results/<fraction>/eval_info.json
#
# Each eval run takes ~20–40 min (500 episodes, 10 parallel envs).

set -euo pipefail

RESULTS_DIR="project1_vla_data_efficiency/results"
mkdir -p "$RESULTS_DIR"

# ── 0. Copy zero-shot baseline ────────────────────────────────────────────────
ZERO_SHOT_SRC="outputs/eval/2026-04-06/22-05-32_libero_smolvla/eval_info.json"
mkdir -p "$RESULTS_DIR/zero_shot"
if [ -f "$ZERO_SHOT_SRC" ]; then
    cp "$ZERO_SHOT_SRC" "$RESULTS_DIR/zero_shot/eval_info.json"
    echo "[zero_shot] Copied baseline from $ZERO_SHOT_SRC"
else
    echo "[zero_shot] WARNING: baseline not found at $ZERO_SHOT_SRC — skipping"
fi

# ── Eval helper ──────────────────────────────────────────────────────────────
eval_checkpoint() {
    local LABEL="$1"
    local CKPT_PATH="outputs/project1/${LABEL}/checkpoints/last/pretrained_model"
    local OUT_DIR="$RESULTS_DIR/${LABEL}"

    if [ ! -d "$CKPT_PATH" ]; then
        echo "[$LABEL] WARNING: checkpoint not found at $CKPT_PATH — skipping"
        return
    fi

    mkdir -p "$OUT_DIR"
    echo ""
    echo "[$LABEL] Evaluating checkpoint: $CKPT_PATH"

    MUJOCO_GL=egl pipenv run python -m lerobot.scripts.lerobot_eval \
        --policy.type=smolvla \
        --policy.pretrained_path="$CKPT_PATH" \
        --env.type=libero \
        --env.task=libero_spatial \
        --eval.n_episodes=500 \
        --eval.batch_size=2 \
        --output_dir="$OUT_DIR"

    echo "[$LABEL] Done. Results at $OUT_DIR/eval_info.json"
}

# ── 1. Evaluate target fractions (from args or defaults) ─────────────────────
DEFAULT_LABELS=("frac005" "frac010" "frac025" "frac050" "frac100")
TARGET_LABELS=("${@:-${DEFAULT_LABELS[@]}}")

echo "Target labels: ${TARGET_LABELS[*]}"
for label in "${TARGET_LABELS[@]}"; do
    eval_checkpoint "$label"
done

echo ""
echo "All evaluations complete."
echo "Run analysis: pipenv run python project1_vla_data_efficiency/scripts/04_analyze_results.py"

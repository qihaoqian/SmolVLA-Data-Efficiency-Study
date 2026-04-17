#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
RUNNER_LOG="${LOG_DIR}/finetune_serial_runner.log"
OUTPUT_BASE_DIR="outputs/finetuning-action-expert"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-base-dir)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --output-base-dir requires a value" >&2
        exit 1
      fi
      OUTPUT_BASE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [--output-base-dir DIR]

Launch all fine-tuning jobs in a detached serial worker.
Each run writes to <DIR>/fracXXX. Default: ${OUTPUT_BASE_DIR}
EOF
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

run_one() {
  local fraction="$1"
  local tag="${fraction/./}"
  local log_file="${LOG_DIR}/finetune_frac${tag}.log"

  echo "Starting fraction=${fraction}, log=${log_file}"
  pipenv run python scripts/02_run_finetuning.py \
    --fraction "${fraction}" \
    --output-base-dir "${OUTPUT_BASE_DIR}" \
    > "${log_file}" 2>&1
  echo "Finished fraction=${fraction}"
}

# Launch a detached worker once; worker then runs jobs serially.
if [[ "${RUN_FINETUNE_WORKER:-0}" != "1" ]]; then
  echo "Starting serial worker in background..."
  nohup env RUN_FINETUNE_WORKER=1 bash "$0" --output-base-dir "${OUTPUT_BASE_DIR}" > "${RUNNER_LOG}" 2>&1 &
  echo "Worker PID: $!"
  disown -a
  echo "Submitted. Follow runner log: tail -f ${RUNNER_LOG}"
  exit 0
fi

# Worker mode: run all four jobs one by one.
run_one "1.00"
run_one "0.50"
run_one "0.25"
run_one "0.10"
run_one "0.05"

echo "All serial jobs finished."

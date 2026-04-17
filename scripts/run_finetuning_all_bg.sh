#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
RUNNER_LOG="${LOG_DIR}/finetune_serial_runner.log"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

run_one() {
  local fraction="$1"
  local tag="${fraction/./}"
  local log_file="${LOG_DIR}/finetune_frac${tag}.log"

  echo "Starting fraction=${fraction}, log=${log_file}"
  pipenv run python scripts/02_run_finetuning.py --fraction "${fraction}" > "${log_file}" 2>&1
  echo "Finished fraction=${fraction}"
}

# Launch a detached worker once; worker then runs jobs serially.
if [[ "${RUN_FINETUNE_WORKER:-0}" != "1" ]]; then
  echo "Starting serial worker in background..."
  nohup env RUN_FINETUNE_WORKER=1 bash "$0" > "${RUNNER_LOG}" 2>&1 &
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

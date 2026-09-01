#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
profile=${1:-"$repo_root/experiments/slo_optimization_testing/proxqp_policy.conf"}
experiment_dir="$repo_root/experiments/slo_optimization_testing"
build_dir=${SLO_BUILD_DIR:-"$repo_root/build-slo-proxqp"}
experiment="$build_dir/slo_optimization_experiment"
cache=${SLO_CACHE:-"$experiment_dir/backend_measurements.csv"}
model=${QUICKSERVE_TEST_MODEL:-"/Users/vijaygoyal/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q4_K_M.gguf"}
prox_prefix=${PROXSUITE_PREFIX:-"$(brew --prefix proxsuite)"}

[[ -f "$model" ]] || { echo "error: model does not exist: $model" >&2; exit 2; }
mkdir -p "$(dirname "$profile")"

echo "[1/3] Building the calibration experiment"
env CC="${SLO_CC:-/usr/bin/clang}" CXX="${SLO_CXX:-/usr/bin/clang++}" \
cmake -S "$experiment_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$prox_prefix"
cmake -E env CARGO_NET_OFFLINE=true \
  cmake --build "$build_dir" --target slo_optimization_experiment \
  -j "${SLO_BUILD_JOBS:-4}"

os_identity="$(uname -sr)"
if command -v sw_vers >/dev/null 2>&1; then
  os_identity="$os_identity|$(sw_vers -productVersion)"
fi
environment_id=${SLO_ENVIRONMENT_ID:-"$(uname -m)|$os_identity|default-power-mode"}

echo "[2/3] Filling the cached backend measurement grid"
"$experiment" collect \
  --model "$model" \
  --cache "$cache" \
  --prefill-list "${SLO_PREFILL_LIST:-32,64,128,256}" \
  --decode-list "${SLO_DECODE_LIST:-1,4,8,16}" \
  --context-list "${SLO_CONTEXT_LIST:-512,2048}" \
  --warmups "${SLO_WARMUPS:-3}" \
  --repetitions "${SLO_REPETITIONS:-30}" \
  --threads "${SLO_THREADS:-1}" \
  --input-seed "${SLO_INPUT_SEED:-0}" \
  --seed "${SLO_SEED:-1}" \
  --environment-id "$environment_id"

echo "[3/3] Fitting and exporting the fixed ProxQP profile"
"$experiment" export-profile \
  --cache "$cache" \
  --output "$profile" \
  --upper-bound-quantile "${SLO_UPPER_BOUND_QUANTILE:-0.98}" \
  --context-capacity "${QUICKSERVE_CONTEXT_SIZE:-16384}" \
  --token-capacity "${QUICKSERVE_BATCH_CAPACITY:-512}" \
  --sequence-capacity "${QUICKSERVE_MAX_SEQUENCES:-16}" \
  --ttft-target-ns "${PROXQP_TTFT_TARGET_NS:-2000000000}" \
  --tpot-target-ns "${PROXQP_TPOT_TARGET_NS:-200000000}" \
  --window-ns "${PROXQP_WINDOW_NS:-60000000000}" \
  --runtime-weight "${PROXQP_RUNTIME_WEIGHT:-0.1}" \
  --rho-prefill "${PROXQP_RHO_PREFILL:-0.2}" \
  --rho-decode "${PROXQP_RHO_DECODE:-0.2}" \
  --boundary-buffer-z "${PROXQP_BOUNDARY_BUFFER_Z:-1}" \
  --boundary-weight "${PROXQP_BOUNDARY_WEIGHT:-0.25}"

echo "Calibration profile: $profile"

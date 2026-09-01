#!/bin/bash

set -euo pipefail

usage() {
  echo "usage: $0 POLICY.cpp [OUTPUT_DIR]" >&2
  echo "POLICY.cpp must have a sibling .hpp or .h declaring quickserve_benchmark_policy::create." >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
policy_input="$1"

[[ -f "$policy_input" ]] || {
  echo "error: policy source does not exist: $policy_input" >&2
  exit 2
}

policy_dir="$(cd "$(dirname "$policy_input")" && pwd -P)"
policy_source="$policy_dir/$(basename "$policy_input")"
case "$policy_source" in
*.cpp) policy_stem="${policy_source%.cpp}" ;;
*.cc) policy_stem="${policy_source%.cc}" ;;
*.cxx) policy_stem="${policy_source%.cxx}" ;;
*)
  echo "error: policy source must end in .cpp, .cc, or .cxx" >&2
  exit 2
  ;;
esac

if [[ -f "${policy_stem}.hpp" ]]; then
  policy_header="${policy_stem}.hpp"
elif [[ -f "${policy_stem}.h" ]]; then
  policy_header="${policy_stem}.h"
else
  echo "error: expected sibling policy header ${policy_stem}.hpp or ${policy_stem}.h" >&2
  exit 2
fi

policy_name="$(basename "$policy_stem")"
policy_name="${policy_name%_policy}"
safe_policy_name="$(printf '%s' "$policy_name" | tr -c 'A-Za-z0-9._-' '-')"
build_dir="/private/tmp/quickserve-benchmark-${safe_policy_name}"
output_dir="${2:-/private/tmp/quickserve-${safe_policy_name}-calibration}"
trace="$repo_root/data/AzureLLMInferenceTrace_code_1week.qst"
model=${QUICKSERVE_TEST_MODEL:-"/Users/vijaygoyal/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q4_K_M.gguf"}

[[ -f "$trace" ]] || {
  echo "error: trace does not exist: $trace" >&2
  exit 2
}
[[ -f "$model" ]] || {
  echo "error: model does not exist: $model" >&2
  exit 2
}
[[ ! -e "$output_dir" ]] || {
  echo "error: output directory already exists: $output_dir" >&2
  exit 2
}

cmake_options=()
policy_options=()
if [[ "$(basename "$policy_source")" == "proxqp_scheduler.cpp" ]]; then
  policy_config=${QUICKSERVE_POLICY_CONFIG:-"$build_dir/proxqp_policy.conf"}
  if [[ -z "${QUICKSERVE_POLICY_CONFIG:-}" && ! -f "$policy_config" ]]; then
    "$repo_root/scripts/calibrate_proxqp_policy.sh" "$policy_config"
  fi
  [[ -f "$policy_config" ]] || {
    echo "error: ProxQP policy profile does not exist: $policy_config" >&2
    exit 2
  }
  cmake_options+=(
    -DQUICKSERVE_BUILD_PROXQP_POLICY=ON
    "-DCMAKE_PREFIX_PATH=${PROXSUITE_PREFIX:-$(brew --prefix proxsuite)}"
  )
  policy_options+=(--policy-config "$policy_config")
fi

env CC="${QUICKSERVE_CC:-/usr/bin/clang}" \
  CXX="${QUICKSERVE_CXX:-/usr/bin/clang++}" \
  cmake -S "$repo_root" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQUICKSERVE_BENCHMARK_POLICY_SOURCE="$policy_source" \
  -DQUICKSERVE_BENCHMARK_POLICY_HEADER="$policy_header" \
  ${cmake_options[@]+"${cmake_options[@]}"}

cmake --build "$build_dir" --target quickserve_benchmark \
  -j "$(sysctl -n hw.logicalcpu)"

caffeinate -i "$build_dir/quickserve_benchmark" \
  --trace "$trace" \
  --model "$model" \
  --target-qps 1.1 \
  --max-requests 128 \
  --output-mode trace-exact \
  --output-dir "$output_dir" \
  --context-size 16384 \
  --batch-capacity 512 \
  --max-sequences 16 \
  --token-budget 512 \
  ${policy_options[@]+"${policy_options[@]}"}

echo "Calibration result: $output_dir/summary.json"

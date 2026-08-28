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
  *.cc)  policy_stem="${policy_source%.cc}" ;;
  *.cxx) policy_stem="${policy_source%.cxx}" ;;
  *) echo "error: policy source must end in .cpp, .cc, or .cxx" >&2; exit 2 ;;
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
model="/Users/vijaygoyal/.cache/huggingface/hub/models--unsloth--Qwen3.5-2B-GGUF/snapshots/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_K_M.gguf"

[[ -f "$trace" ]] || { echo "error: trace does not exist: $trace" >&2; exit 2; }
[[ -f "$model" ]] || { echo "error: model does not exist: $model" >&2; exit 2; }
[[ ! -e "$output_dir" ]] || {
  echo "error: output directory already exists: $output_dir" >&2
  exit 2
}

cmake -S "$repo_root" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQUICKSERVE_BENCHMARK_POLICY_SOURCE="$policy_source" \
  -DQUICKSERVE_BENCHMARK_POLICY_HEADER="$policy_header"

cmake --build "$build_dir" --target quickserve_benchmark \
  -j "$(sysctl -n hw.logicalcpu)"

caffeinate -i "$build_dir/quickserve_benchmark" \
  --trace "$trace" \
  --model "$model" \
  --target-qps 1000 \
  --max-requests 128 \
  --output-mode trace-exact \
  --output-dir "$output_dir" \
  --context-size 16384 \
  --batch-capacity 512 \
  --max-sequences 4 \
  --token-budget 512

echo "Calibration result: $output_dir/summary.json"

#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
main_build_dir=${QUICKSERVE_BUILD_DIR:-"$repo_root/build"}
slo_build_dir=${SLO_BUILD_DIR:-"$repo_root/build-slo"}
experiment=${SLO_EXPERIMENT_BINARY:-"$slo_build_dir/slo_optimization_experiment"}
cache=${SLO_CACHE:-"$script_dir/backend_measurements.csv"}
output=${SLO_OUTPUT:-"$script_dir/offline_results.csv"}
repetitions=${SLO_REPETITIONS:-30}
snapshots=${SLO_SNAPSHOTS:-100}
seed=${SLO_SEED:-1}

needs_build=false
if [[ ! -x "$experiment" ]]; then
  needs_build=true
elif find "$script_dir" -maxdepth 1 -type f \
    \( -name '*.cpp' -o -name '*.hpp' -o -name 'CMakeLists.txt' \) \
    -newer "$experiment" -print -quit | grep -q .; then
  needs_build=true
fi
if [[ "$needs_build" == true ]]; then
  echo "Experiment binary missing or stale; configuring and building it in $slo_build_dir"
  cmake -S "$script_dir" -B "$slo_build_dir" -DCMAKE_BUILD_TYPE=Release
  cmake -E env CARGO_NET_OFFLINE=true \
    cmake --build "$slo_build_dir" -j "${SLO_BUILD_JOBS:-4}"
fi
if [[ ! -x "$experiment" ]]; then
  echo "Build completed without producing the experiment binary: $experiment" >&2
  exit 1
fi

model=${QUICKSERVE_TEST_MODEL:-}
if [[ -z "$model" ]]; then
  cmake_cache="$main_build_dir/CMakeCache.txt"
  if [[ ! -f "$cmake_cache" ]]; then
    echo "Main CMake cache not found: $cmake_cache" >&2
    echo "Set QUICKSERVE_TEST_MODEL or QUICKSERVE_BUILD_DIR." >&2
    exit 1
  fi
  model=$(sed -n 's/^QUICKSERVE_TEST_MODEL:FILEPATH=//p' "$cmake_cache" | tail -n 1)
fi
if [[ -z "$model" || ! -f "$model" ]]; then
  echo "Configured QUICKSERVE_TEST_MODEL is missing: ${model:-<empty>}" >&2
  exit 1
fi

mkdir -p "$(dirname "$cache")" "$(dirname "$output")"

os_identity=$(uname -sr)
if command -v sw_vers >/dev/null 2>&1; then
  os_identity="$os_identity|$(sw_vers -productVersion)"
fi
environment_id=${SLO_ENVIRONMENT_ID:-"$(uname -m)|$os_identity|default-power-mode"}
echo "[1/2] Checking and filling the backend measurement grid in $cache"
"$experiment" collect \
  --model "$model" \
  --cache "$cache" \
  --prefill-list "${SLO_PREFILL_LIST:-32,64,128,256}" \
  --decode-list "${SLO_DECODE_LIST:-1,4,8,16}" \
  --context-list "${SLO_CONTEXT_LIST:-512,2048}" \
  --warmups "${SLO_WARMUPS:-3}" \
  --repetitions "$repetitions" \
  --threads "${SLO_THREADS:-1}" \
  --input-seed "${SLO_INPUT_SEED:-0}" \
  --seed "$seed" \
  --environment-id "$environment_id"

echo "[2/2] Running offline evaluation"
"$experiment" evaluate \
  --cache "$cache" \
  --output "$output" \
  --snapshots "$snapshots" \
  --seed "$seed"
echo "Offline results written to $output"

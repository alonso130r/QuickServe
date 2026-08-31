#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

mkdir -p "$tmp_dir/main-build" "$tmp_dir/slo-build"
touch "$tmp_dir/model.gguf"
printf 'QUICKSERVE_TEST_MODEL:FILEPATH=%s\n' "$tmp_dir/model.gguf" > "$tmp_dir/main-build/CMakeCache.txt"

cat > "$tmp_dir/slo-build/slo_optimization_experiment" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$CALL_LOG"
case "$1" in
  collect) printf 'cached\n' > "$CACHE_PATH" ;;
  evaluate) printf 'evaluated\n' > "$OUTPUT_PATH" ;;
esac
EOF
chmod +x "$tmp_dir/slo-build/slo_optimization_experiment"

export CALL_LOG="$tmp_dir/calls.log"
export CACHE_PATH="$tmp_dir/cache.csv"
export OUTPUT_PATH="$tmp_dir/results.csv"
QUICKSERVE_BUILD_DIR="$tmp_dir/main-build" SLO_BUILD_DIR="$tmp_dir/slo-build" \
  SLO_CACHE="$CACHE_PATH" SLO_OUTPUT="$OUTPUT_PATH" SLO_REPETITIONS=2 \
  "$script_dir/run_offline_experiment.sh"

grep -q '^collect ' "$CALL_LOG"
grep -q '^evaluate ' "$CALL_LOG"
grep -q '^collect ' "$CALL_LOG"

rm "$tmp_dir/slo-build/slo_optimization_experiment"
mkdir -p "$tmp_dir/bin"
cat > "$tmp_dir/bin/cmake" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$CMAKE_LOG"
for argument in "$@"; do
  if [[ "$argument" == "--build" ]]; then
    cp "$FAKE_EXPERIMENT" "$SLO_BUILD_DIR/slo_optimization_experiment"
    chmod +x "$SLO_BUILD_DIR/slo_optimization_experiment"
  fi
done
EOF
chmod +x "$tmp_dir/bin/cmake"
cp "$tmp_dir/slo-build/slo_optimization_experiment" "$tmp_dir/fake-experiment" 2>/dev/null || true
cat > "$tmp_dir/fake-experiment" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$CALL_LOG"
case "$1" in
  collect) printf 'cached\n' > "$CACHE_PATH" ;;
  evaluate) printf 'evaluated\n' > "$OUTPUT_PATH" ;;
esac
EOF
chmod +x "$tmp_dir/fake-experiment"
export CMAKE_LOG="$tmp_dir/cmake.log" FAKE_EXPERIMENT="$tmp_dir/fake-experiment"
: > "$CALL_LOG"
PATH="$tmp_dir/bin:$PATH" QUICKSERVE_BUILD_DIR="$tmp_dir/main-build" \
  SLO_BUILD_DIR="$tmp_dir/slo-build" SLO_CACHE="$CACHE_PATH" \
  SLO_OUTPUT="$OUTPUT_PATH" "$script_dir/run_offline_experiment.sh"
grep -q '^-S .*slo_optimization_testing -B .*slo-build -DCMAKE_BUILD_TYPE=Release$' "$CMAKE_LOG"
grep -q -- '--build .*slo-build' "$CMAKE_LOG"
grep -q '^evaluate ' "$CALL_LOG"
: > "$CALL_LOG"
QUICKSERVE_BUILD_DIR="$tmp_dir/main-build" SLO_BUILD_DIR="$tmp_dir/slo-build" \
  SLO_CACHE="$CACHE_PATH" SLO_OUTPUT="$OUTPUT_PATH" SLO_REPETITIONS=2 \
  "$script_dir/run_offline_experiment.sh"

grep -q '^collect ' "$CALL_LOG"
grep -q -- '--prefill-list 32,64,128,256' "$CALL_LOG"
grep -q -- '--decode-list 1,4,8,16' "$CALL_LOG"
grep -q '^evaluate ' "$CALL_LOG"

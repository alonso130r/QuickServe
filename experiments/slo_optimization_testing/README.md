# SLO Optimization Experiment

This standalone experiment compares the linearly constrained QP solved by ProxQP with the uncertainty-aware SOCP solved by Clarabel. It does not instantiate `Scheduler`, `Environment`, or `Handoff`.

## Build

```sh
cmake -S experiments/slo_optimization_testing -B build-slo -DCMAKE_BUILD_TYPE=Release
cmake -E env CARGO_NET_OFFLINE=true cmake --build build-slo -j
ctest --test-dir build-slo --output-on-failure
```

ProxSuite must be discoverable by CMake. Clarabel.cpp and llama.cpp are consumed from `external/`.

## Tier 1 smoke benchmark

```sh
./build-slo/slo_optimization_experiment synthetic \
  --output /tmp/slo-tier1.csv --instances 12 --repetitions 20 --seed 1
```

Use 729 instances and at least 1,000 repetitions for the full run.

## Cached backend collection

```sh
./build-slo/slo_optimization_experiment collect \
  --model /absolute/model.gguf --cache /tmp/slo-cache.csv \
  --prefill-list 32,64,128,256 \
  --decode-list 1,4,8,16 \
  --context-list 512,2048 --warmups 3 --repetitions 30 \
  --threads 1 --input-seed 0 \
  --environment-id "machine|OS|backend-driver|power-mode"
```

The collector randomizes composition order, reuses existing observations, and appends only missing cells or repetitions. The cache key includes a fingerprint of the backend measurement implementation, the llama.cpp revision, build type, model fingerprint, environment identity, backend settings, context, and batch composition. Unrelated QuickServe commits do not invalidate backend measurements.

## Offline evaluation

```sh
./build-slo/slo_optimization_experiment evaluate \
  --cache /tmp/slo-cache.csv --output /tmp/slo-evaluation.csv \
  --snapshots 100 --seed 1 \
  --ttft-target-multiplier 3 --tpot-target-multiplier 2 \
  --risk-buffer-z 1 --risk-weight 0.25 \
  --upper-bound-quantile 0.98
```

Evaluation uses a deterministic 60/20/20 repetition split inside every measured allocation cell. This preserves the full measured allocation grid for nearest-neighbor projection while keeping evaluation timings unseen. A separate leave-cell-out coverage metric measures generalization to unseen batch compositions. Snapshots span reference pressure levels \(-2,-1,-0.5,0,0.5\), and both formulations combine hard standardized constraints with a squared-hinge boundary-risk objective.

The predictive runtime envelope defaults to the one-sided 98th percentile of calibration residuals. This is distinct from the hard sliding-window p99 SLO, which remains unchanged. Override the envelope tradeoff with `--upper-bound-quantile`.

To collect missing measurements with the model configured for the main test suite and then run evaluation:

```sh
experiments/slo_optimization_testing/run_offline_experiment.sh
```

To collect or reuse the same measurements and export a fixed profile consumed
by `ProxQPScheduler`:

```sh
scripts/calibrate_proxqp_policy.sh /tmp/proxqp-policy.conf
```

The exporter fits nonnegative runtime coefficients, calibrates the one-sided
p98 residual margin, and writes the file atomically. It rejects caches that mix
model, backend, build, hardware, or environment cohorts.

The script builds the experiment when its binary is missing or stale, reads `QUICKSERVE_TEST_MODEL` from `build/CMakeCache.txt`, fills any missing cells in the 32-cell measurement grid, and writes `offline_results.csv` in this directory. Override locations with `QUICKSERVE_BUILD_DIR`, `SLO_BUILD_DIR`, `SLO_CACHE`, or `SLO_OUTPUT`.

See [experiment_design.md](experiment_design.md) for the full protocol and [formulations.md](formulations.md) for the mathematics.

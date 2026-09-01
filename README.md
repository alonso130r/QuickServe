# QuickServe

QuickServe is an experimental C++ inference scheduler built on
[`llama.cpp`](https://github.com/ggerganov/llama.cpp). It includes FIFO,
continuous-batching, and heuristic AIMD policies, along with a trace-driven
benchmark for comparing scheduling behavior on Apple Silicon.

## Requirements

- macOS on Apple Silicon
- CMake 3.18 or newer
- A C++17 compiler
- Git submodules
- Optional: Homebrew `libomp` for OpenMP support
- ProxSuite when building the optional ProxQP scheduler policy

## Build and test

```sh
git clone --recurse-submodules <repository-url>
cd QuickServe
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Enable the separate ProxQP policy and its focused tests with:

```sh
cmake -S . -B build -DQUICKSERVE_BUILD_PROXQP_POLICY=ON \
  -DCMAKE_PREFIX_PATH="$(brew --prefix proxsuite)"
cmake --build build
```

`ProxQPScheduler` requires a calibrated fixed profile and emits decision and
completed-batch measurements through callbacks. It does not update the profile
online. Generate the profile, then run the QPS calibration with:

```sh
scripts/calibrate_proxqp_policy.sh /tmp/proxqp-policy.conf
QUICKSERVE_POLICY_CONFIG=/tmp/proxqp-policy.conf \
  scripts/run_policy_qps_calibration.sh src/policies/proxqp_scheduler.cpp
```

Run the full 10,000-request benchmark with the same profile:

```sh
QUICKSERVE_POLICY_CONFIG=/tmp/proxqp-policy.conf \
  scripts/run_policy_benchmark.sh src/policies/proxqp_scheduler.cpp
```

If `QUICKSERVE_POLICY_CONFIG` is omitted, either benchmark script calibrates a
profile once under its temporary policy build directory and reuses it. Override
the calibration model with `QUICKSERVE_TEST_MODEL`; the grid and fit controls
use the `SLO_*` variables documented by the optimization experiment.

To include the real-backend test, configure with a local GGUF model:

```sh
cmake -S . -B build -DQUICKSERVE_TEST_MODEL=/path/to/model.gguf
```

## Benchmarking

`scripts/run_policy_benchmark.sh` builds and runs a scheduling policy against
the prepared Azure LLM inference trace. The script currently expects the trace
and model paths used by this development environment, so update those paths
before running it on another machine.

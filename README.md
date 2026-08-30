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

## Build and test

```sh
git clone --recurse-submodules <repository-url>
cd QuickServe
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

To include the real-backend test, configure with a local GGUF model:

```sh
cmake -S . -B build -DQUICKSERVE_TEST_MODEL=/path/to/model.gguf
```

## Benchmarking

`scripts/run_policy_benchmark.sh` builds and runs a scheduling policy against
the prepared Azure LLM inference trace. The script currently expects the trace
and model paths used by this development environment, so update those paths
before running it on another machine.

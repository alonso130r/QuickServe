#include "benchmark/trace.hpp"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

int main(int argc, char **argv) {
  std::filesystem::path input, output;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "--input" || arg == "--output") && i + 1 < argc) {
      (arg == "--input" ? input : output) = argv[++i];
    } else {
      std::fprintf(stderr, "usage: quickserve_prepare_trace --input CSV --output QST\n");
      return 2;
    }
  }
  if (input.empty() || output.empty()) {
    std::fprintf(stderr, "usage: quickserve_prepare_trace --input CSV --output QST\n");
    return 2;
  }
  try {
    quickserve::benchmark::prepare_trace(input, output);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "quickserve_prepare_trace: %s\n", e.what());
    return 1;
  }
  return 0;
}

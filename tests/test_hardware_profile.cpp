#include "runtime/hardware_profile.hpp"

#include <cstdio>

namespace {

int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);       \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

void test_valid_profile_preserves_values() {
  const auto profile = make_hardware_profile(
      {"Mac15,7", 32ULL * 1024 * 1024 * 1024, 10, 10, 16'384});
  CHECK(profile.has_value());
  if (!profile) return;
  CHECK(profile->model_identifier == "Mac15,7");
  CHECK(profile->total_memory_bytes == 32ULL * 1024 * 1024 * 1024);
  CHECK(profile->physical_cpu_count == 10);
  CHECK(profile->logical_cpu_count == 10);
  CHECK(profile->page_size_bytes == 16'384);
}

void test_invalid_profile_is_rejected() {
  CHECK(!make_hardware_profile({"", 1, 1, 1, 1}));
  CHECK(!make_hardware_profile({"Mac", 0, 1, 1, 1}));
  CHECK(!make_hardware_profile({"Mac", 1, 0, 1, 1}));
  CHECK(!make_hardware_profile({"Mac", 1, 2, 1, 1}));
  CHECK(!make_hardware_profile({"Mac", 1, 1, 1, 0}));
}

void test_macos_collector_returns_sane_values() {
  const auto profile = collect_macos_hardware_profile();
  CHECK(profile.has_value());
  if (!profile) return;
  CHECK(!profile->model_identifier.empty());
  CHECK(profile->total_memory_bytes > 0);
  CHECK(profile->physical_cpu_count > 0);
  CHECK(profile->logical_cpu_count >= profile->physical_cpu_count);
  CHECK(profile->page_size_bytes > 0);
}

} // namespace

int main() {
  test_valid_profile_preserves_values();
  test_invalid_profile_is_rejected();
  test_macos_collector_returns_sane_values();
  if (failures != 0) return 1;
  std::printf("all hardware profile checks passed\n");
  return 0;
}

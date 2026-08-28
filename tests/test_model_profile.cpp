#include "runtime/model_profile.hpp"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>

namespace {

int g_failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                         \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);       \
      ++g_failures;                                                             \
    }                                                                           \
  } while (false)

RawModelProfile valid_raw_profile() {
  return RawModelProfile{
      7'000'000'000,
      4'200'000'000,
      32,
      4096,
      32,
      8,
      2.0,
      1.5,
      8192,
      2048,
      64,
  };
}

void test_valid_profile_preserves_and_derives_values() {
  std::printf("test_valid_profile_preserves_and_derives_values\n");
  const std::optional<ModelProfile> profile =
      make_model_profile(valid_raw_profile());

  CHECK(profile.has_value());
  if (!profile.has_value()) {
    return;
  }
  CHECK(profile->parameter_count == 7'000'000'000ULL);
  CHECK(profile->model_bytes == 4'200'000'000ULL);
  CHECK(profile->layer_count == 32);
  CHECK(profile->embedding_dimension == 4096);
  CHECK(profile->attention_head_count == 32);
  CHECK(profile->kv_head_count == 8);
  CHECK(profile->head_dimension == 128);
  CHECK(profile->key_effective_bytes_per_scalar == 2.0);
  CHECK(profile->value_effective_bytes_per_scalar == 1.5);
  CHECK(profile->mean_bytes_per_parameter == 0.6);
  CHECK(profile->context_capacity == 8192);
  CHECK(profile->batch_capacity == 2048);
  CHECK(profile->max_sequences == 64);
}

void test_zero_sizes_and_capacities_are_rejected() {
  std::printf("test_zero_sizes_and_capacities_are_rejected\n");
  RawModelProfile raw = valid_raw_profile();
  raw.parameter_count = 0;
  CHECK(!make_model_profile(raw).has_value());
  raw = valid_raw_profile();
  raw.model_bytes = 0;
  CHECK(!make_model_profile(raw).has_value());
  raw = valid_raw_profile();
  raw.context_capacity = 0;
  CHECK(!make_model_profile(raw).has_value());
  raw = valid_raw_profile();
  raw.batch_capacity = 0;
  CHECK(!make_model_profile(raw).has_value());
  raw = valid_raw_profile();
  raw.max_sequences = 0;
  CHECK(!make_model_profile(raw).has_value());
}

void test_all_signed_inputs_reject_zero_and_negative_values() {
  std::printf("test_all_signed_inputs_reject_zero_and_negative_values\n");
  for (const std::int64_t invalid : {std::int64_t{0}, std::int64_t{-1}}) {
    RawModelProfile raw = valid_raw_profile();
    raw.layer_count = invalid;
    CHECK(!make_model_profile(raw).has_value());
    raw = valid_raw_profile();
    raw.embedding_dimension = invalid;
    CHECK(!make_model_profile(raw).has_value());
    raw = valid_raw_profile();
    raw.attention_head_count = invalid;
    CHECK(!make_model_profile(raw).has_value());
    raw = valid_raw_profile();
    raw.kv_head_count = invalid;
    CHECK(!make_model_profile(raw).has_value());
    raw = valid_raw_profile();
    raw.context_capacity = invalid;
    CHECK(!make_model_profile(raw).has_value());
    raw = valid_raw_profile();
    raw.batch_capacity = invalid;
    CHECK(!make_model_profile(raw).has_value());
    raw = valid_raw_profile();
    raw.max_sequences = invalid;
    CHECK(!make_model_profile(raw).has_value());
  }
}

void test_zero_parameter_count_is_rejected_before_ratio_derivation() {
  std::printf("test_zero_parameter_count_is_rejected_before_ratio_derivation\n");
  RawModelProfile raw = valid_raw_profile();
  raw.parameter_count = 0;

  std::feclearexcept(FE_ALL_EXCEPT);
  CHECK(!make_model_profile(raw).has_value());
  CHECK(std::fetestexcept(FE_DIVBYZERO) == 0);
}

void test_uint64_model_size_boundaries_are_accepted() {
  std::printf("test_uint64_model_size_boundaries_are_accepted\n");
  RawModelProfile raw = valid_raw_profile();
  raw.parameter_count = std::numeric_limits<std::uint64_t>::max();
  raw.model_bytes = std::numeric_limits<std::uint64_t>::max();

  const std::optional<ModelProfile> profile = make_model_profile(raw);
  CHECK(profile.has_value());
  if (profile.has_value()) {
    CHECK(profile->parameter_count == std::numeric_limits<std::uint64_t>::max());
    CHECK(profile->model_bytes == std::numeric_limits<std::uint64_t>::max());
    CHECK(profile->mean_bytes_per_parameter == 1.0);
  }
}

void test_invalid_head_geometry_is_rejected() {
  std::printf("test_invalid_head_geometry_is_rejected\n");
  RawModelProfile raw = valid_raw_profile();
  raw.embedding_dimension = 4097;
  CHECK(!make_model_profile(raw).has_value());
  raw = valid_raw_profile();
  raw.kv_head_count = 33;
  CHECK(!make_model_profile(raw).has_value());
}

void test_invalid_effective_scalar_widths_are_rejected() {
  std::printf("test_invalid_effective_scalar_widths_are_rejected\n");
  const double invalid_values[] = {
      0.0,
      -1.0,
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
  };
  for (const double invalid : invalid_values) {
    RawModelProfile raw = valid_raw_profile();
    raw.key_effective_bytes_per_scalar = invalid;
    CHECK(!make_model_profile(raw).has_value());
    raw = valid_raw_profile();
    raw.value_effective_bytes_per_scalar = invalid;
    CHECK(!make_model_profile(raw).has_value());
  }
}

} // namespace

int main() {
  test_valid_profile_preserves_and_derives_values();
  test_zero_sizes_and_capacities_are_rejected();
  test_all_signed_inputs_reject_zero_and_negative_values();
  test_zero_parameter_count_is_rejected_before_ratio_derivation();
  test_uint64_model_size_boundaries_are_accepted();
  test_invalid_head_geometry_is_rejected();
  test_invalid_effective_scalar_widths_are_rejected();

  if (g_failures != 0) {
    std::printf("\n%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("\nall model profile checks passed\n");
  return 0;
}

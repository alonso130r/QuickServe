#pragma once

#include <cstdint>
#include <optional>

struct RawModelProfile {
  std::uint64_t parameter_count{};
  std::uint64_t model_bytes{};
  std::int64_t layer_count{};
  std::int64_t embedding_dimension{};
  std::int64_t attention_head_count{};
  std::int64_t kv_head_count{};
  double key_effective_bytes_per_scalar{};
  double value_effective_bytes_per_scalar{};
  std::int64_t context_capacity{};
  std::int64_t batch_capacity{};
  std::int64_t max_sequences{};
};

struct ModelProfile {
  std::uint64_t parameter_count{};
  std::uint64_t model_bytes{};
  std::uint64_t layer_count{};
  std::uint64_t embedding_dimension{};
  std::uint64_t attention_head_count{};
  std::uint64_t kv_head_count{};
  std::uint64_t head_dimension{};
  double key_effective_bytes_per_scalar{};
  double value_effective_bytes_per_scalar{};
  double mean_bytes_per_parameter{};
  std::uint64_t context_capacity{};
  std::uint64_t batch_capacity{};
  std::uint64_t max_sequences{};
};

[[nodiscard]] std::optional<ModelProfile>
make_model_profile(const RawModelProfile &raw);

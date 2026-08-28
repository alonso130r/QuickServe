#include "runtime/model_profile.hpp"

#include <cmath>

namespace {

bool is_positive_finite(double value) {
  return std::isfinite(value) && value > 0.0;
}

} // namespace

std::optional<ModelProfile>
make_model_profile(const RawModelProfile &raw) {
  if (raw.parameter_count == 0 || raw.model_bytes == 0 ||
      raw.layer_count <= 0 || raw.embedding_dimension <= 0 ||
      raw.attention_head_count <= 0 || raw.kv_head_count <= 0 ||
      raw.context_capacity <= 0 || raw.batch_capacity <= 0 ||
      raw.max_sequences <= 0 ||
      raw.embedding_dimension % raw.attention_head_count != 0 ||
      raw.kv_head_count > raw.attention_head_count ||
      !is_positive_finite(raw.key_effective_bytes_per_scalar) ||
      !is_positive_finite(raw.value_effective_bytes_per_scalar)) {
    return std::nullopt;
  }

  const double mean_bytes_per_parameter =
      static_cast<double>(raw.model_bytes) /
      static_cast<double>(raw.parameter_count);
  if (!is_positive_finite(mean_bytes_per_parameter)) {
    return std::nullopt;
  }

  return ModelProfile{
      raw.parameter_count,
      raw.model_bytes,
      static_cast<std::uint64_t>(raw.layer_count),
      static_cast<std::uint64_t>(raw.embedding_dimension),
      static_cast<std::uint64_t>(raw.attention_head_count),
      static_cast<std::uint64_t>(raw.kv_head_count),
      static_cast<std::uint64_t>(raw.embedding_dimension /
                                 raw.attention_head_count),
      raw.key_effective_bytes_per_scalar,
      raw.value_effective_bytes_per_scalar,
      mean_bytes_per_parameter,
      static_cast<std::uint64_t>(raw.context_capacity),
      static_cast<std::uint64_t>(raw.batch_capacity),
      static_cast<std::uint64_t>(raw.max_sequences),
  };
}

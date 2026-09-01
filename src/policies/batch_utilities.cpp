#include "batch_utilities.hpp"

#include <cmath>

namespace quickserve::policy {

BatchResourceUsage evaluate_batch_resources(
    const std::list<RequestState> &requests, const std::vector<WorkItem> &work,
    std::uint32_t token_budget, const ModelProfile &model,
    const HardwareProfile &hardware) {
  BatchResourceUsage result;
  const long double kv_dimension =
      static_cast<long double>(model.kv_head_count) * model.head_dimension;
  const long double kv_scalar_bytes = model.key_effective_bytes_per_scalar +
                                      model.value_effective_bytes_per_scalar;
  const long double kv_bytes_per_token =
      static_cast<long double>(model.layer_count) * kv_dimension *
      kv_scalar_bytes;
  std::uint64_t resident_tokens = 0;
  for (const RequestState &request : requests) {
    if (request.stage != RequestState::Stage::Prefill &&
        request.stage != RequestState::Stage::Decode)
      continue;
    resident_tokens += request.prefill_position;
    if (request.decoded_count > 0)
      resident_tokens += request.decoded_count - 1;
  }
  for (const WorkItem &item : work) {
    result.total_tokens += item.token_count();
    resident_tokens += item.kind == WorkKind::Prefill ? item.token_count() : 1;
  }
  result.work_items = work.size();
  result.resident_kv_bytes = kv_bytes_per_token * resident_tokens;
  result.required_memory_bytes =
      static_cast<long double>(model.model_bytes) + result.resident_kv_bytes;
  result.valid = result.total_tokens <= token_budget &&
                 result.total_tokens <= model.batch_capacity &&
                 result.work_items <= model.max_sequences &&
                 std::isfinite(result.resident_kv_bytes) &&
                 std::isfinite(result.required_memory_bytes) &&
                 result.required_memory_bytes <= hardware.total_memory_bytes;
  return result;
}

} // namespace quickserve::policy

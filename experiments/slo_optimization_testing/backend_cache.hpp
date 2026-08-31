#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace sloexp {
struct Measurement {
  std::string key, observation_id, run_id;
  std::uint64_t timestamp_ns{};
  std::uint32_t prefill_tokens{}, decode_items{}, context_tokens{};
  std::uint64_t duration_ns{};
  std::uint64_t monotonic_ns{};
  std::uint32_t execution_position{};
  std::uint32_t sequence_count{};
};

class MeasurementCache {
public:
  explicit MeasurementCache(std::filesystem::path path) : path_(std::move(path)) {}
  std::vector<Measurement> load() const;
  bool append(const Measurement &measurement);
private:
  std::filesystem::path path_;
};

std::vector<std::uint64_t> measure_backend(const std::string &model_path,
                                           std::uint32_t prefill_tokens,
                                           std::uint32_t decode_items,
                                           std::uint32_t context_tokens,
                                           std::uint32_t threads,
                                           std::size_t warmups,
                                           std::size_t repetitions);
} // namespace sloexp

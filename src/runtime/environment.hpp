#pragma once

#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>

#include "handoff.hpp"
#include "model_profile.hpp"

struct EnvironmentConfig {
  std::string model_path;
  std::uint32_t context_size = 0;
  std::uint32_t batch_capacity = 0;
  std::uint32_t max_sequences = 0;
};

struct EnvironmentStartupResult {
  bool success = false;
  std::string error;
  std::optional<ModelProfile> model_profile;
};

[[nodiscard]] constexpr ErrorCode validate_synthetic_workload(
    std::uint32_t prompt_tokens, std::uint32_t max_output_tokens,
    std::uint32_t context_size) noexcept {
  return static_cast<std::uint64_t>(prompt_tokens) + max_output_tokens >
                 context_size
             ? ErrorCode::ContextCapacityExceeded
             : ErrorCode::None;
}

class Environment {
public:
  Environment(Handoff &handoff, EnvironmentConfig config);
  Environment(const Environment &) = delete;
  Environment &operator=(const Environment &) = delete;
  Environment(Environment &&) = delete;
  Environment &operator=(Environment &&) = delete;

  void run();
  [[nodiscard]] EnvironmentStartupResult wait_for_startup();

private:
  void publish_startup(EnvironmentStartupResult result);

  Handoff &handoff_;
  EnvironmentConfig config_;
  std::mutex startup_mutex_;
  std::condition_variable startup_cv_;
  EnvironmentStartupResult startup_result_;
  bool startup_ready_ = false;
};

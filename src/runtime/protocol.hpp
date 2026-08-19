#pragma once

#include <cstdint>
#include <string>
#include <type_traits>

using RequestId = std::uint32_t;
using Token = std::int32_t;

enum class WorkKind : std::uint8_t { Prefill, Decode };

enum class ErrorCode : std::uint8_t {
  None,
  TokenizationFailed,
  ContextCapacityExceeded,
  DecodeFailed,
  SamplingFailed,
  ProtocolViolation,
  EnvironmentStopped,
};

struct Admission {
  RequestId id = 0;
  std::string prompt;
  std::uint32_t max_output_tokens = 0;
};

struct AdmissionResult {
  RequestId id = 0;
  std::uint32_t prompt_tokens = 0;
  ErrorCode error = ErrorCode::None;
};

struct WorkItem {
  RequestId id = 0;
  std::uint32_t token_begin = 0;
  std::uint32_t token_end = 0;
  WorkKind kind = WorkKind::Prefill;

  [[nodiscard]] std::uint32_t token_count() const {
    return token_end - token_begin;
  }
};

struct Completion {
  RequestId id = 0;
  std::uint32_t prefill_position = 0;
  std::uint32_t decoded_tokens = 0;
  Token token = 0;
  WorkKind kind = WorkKind::Prefill;
  ErrorCode error = ErrorCode::None;
  bool generated_token = false;
  bool eos = false;
};

struct OutputPiece {
  RequestId id = 0;
  Token token = 0;
  std::string text;
};

struct Release {
  RequestId id = 0;
};

struct ReleaseAck {
  RequestId id = 0;
};

struct RunFatal {
  ErrorCode error = ErrorCode::None;
};

static_assert(std::is_trivially_copyable_v<WorkItem>,
              "WorkItem must be trivially copyable across threads");
static_assert(std::is_trivially_copyable_v<Completion>,
              "Completion must be trivially copyable across threads");

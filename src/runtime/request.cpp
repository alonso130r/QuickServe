#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <llama.h>
#include <vector>

struct Request {
  std::size_t num_tokens;
  std::vector<llama_token> tokenized_prompt;
  std::chrono::time_point<std::chrono::high_resolution_clock> time_of_arrival;
};

enum class SplitKind : std::uint8_t { Prefill, Decode };

// A unit of work handed from the scheduler to the runtime. Kept trivially
// copyable and <= 16 bytes so it can travel through std::atomic and the SPSC
// queue by value; `id` indexes a request table rather than pointing at one,
// since an 8-byte pointer would push this past the lock-free width on arm64.
struct SplitRequest {
  std::uint32_t id; // key for hashes; index into the request table
  std::uint32_t tok_begin;
  std::uint32_t tok_end; // decode is the degenerate range [pos, pos + 1)
  SplitKind kind;
};

static_assert(std::is_trivially_copyable_v<SplitRequest>,
              "SplitRequest must be trivially copyable to cross threads by "
              "value");
static_assert(std::atomic<SplitRequest>::is_always_lock_free,
              "SplitRequest must fit the platform's lock-free atomic width");

// Reported by the runtime once a unit of work has executed. Positions are
// absolute rather than incremental so that applying a completion is idempotent
// and order-insensitive: the scheduler folds them in with max().
struct Completion {
  std::uint32_t id;
  std::uint32_t prefill_pos; // total prompt tokens prefilled so far
  std::uint32_t decoded;     // total tokens generated so far
  llama_token last_token;    // meaningful only when kind == Decode
  SplitKind kind;
  bool eos;
};

static_assert(std::is_trivially_copyable_v<Completion>,
              "Completion must be trivially copyable to cross threads by "
              "value");

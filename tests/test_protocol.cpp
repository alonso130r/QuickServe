#include "runtime/protocol.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>

static_assert(std::is_same_v<RequestId, std::uint32_t>);
static_assert(std::is_same_v<Token, std::int32_t>);
static_assert(std::is_trivially_copyable_v<WorkItem>);
static_assert(std::is_trivially_copyable_v<Completion>);

namespace {

int g_failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);       \
      ++g_failures;                                                            \
    }                                                                          \
  } while (false)

void test_work_item_counts_tokens() {
  std::printf("test_work_item_counts_tokens\n");
  const WorkItem work{7, 3, 11, WorkKind::Prefill};
  CHECK(work.token_count() == work.token_end - work.token_begin);
  CHECK(work.token_count() == 8);

  const WorkItem decode{7, 11, 12, WorkKind::Decode};
  CHECK(decode.token_count() == 1);
}

void test_completion_stores_errors() {
  std::printf("test_completion_stores_errors\n");
  Completion failed{};
  failed.id = 42;
  failed.error = ErrorCode::DecodeFailed;
  CHECK(failed.id == 42);
  CHECK(failed.error == ErrorCode::DecodeFailed);
}

void test_messages_have_deterministic_defaults() {
  std::printf("test_messages_have_deterministic_defaults\n");

  const Admission admission{};
  CHECK(admission.id == 0);
  CHECK(admission.prompt.empty());
  CHECK(admission.max_output_tokens == 0);

  const AdmissionResult admission_result{};
  CHECK(admission_result.id == 0);
  CHECK(admission_result.prompt_tokens == 0);
  CHECK(admission_result.error == ErrorCode::None);

  const WorkItem work{};
  CHECK(work.id == 0);
  CHECK(work.token_begin == 0);
  CHECK(work.token_end == 0);
  CHECK(work.kind == WorkKind::Prefill);
  CHECK(work.token_count() == 0);

  const Completion completion{};
  CHECK(completion.id == 0);
  CHECK(completion.prefill_position == 0);
  CHECK(completion.decoded_tokens == 0);
  CHECK(completion.token == 0);
  CHECK(completion.kind == WorkKind::Prefill);
  CHECK(completion.error == ErrorCode::None);
  CHECK(!completion.generated_token);
  CHECK(!completion.eos);

  const OutputPiece output{};
  CHECK(output.id == 0);
  CHECK(output.token == 0);
  CHECK(output.text.empty());

  const Release release{};
  CHECK(release.id == 0);
  const ReleaseAck release_ack{};
  CHECK(release_ack.id == 0);
  const RunFatal fatal{};
  CHECK(fatal.error == ErrorCode::None);
}

void test_all_error_codes_are_distinct() {
  std::printf("test_all_error_codes_are_distinct\n");
  CHECK(ErrorCode::None != ErrorCode::TokenizationFailed);
  CHECK(ErrorCode::TokenizationFailed != ErrorCode::ContextCapacityExceeded);
  CHECK(ErrorCode::ContextCapacityExceeded != ErrorCode::DecodeFailed);
  CHECK(ErrorCode::DecodeFailed != ErrorCode::SamplingFailed);
  CHECK(ErrorCode::SamplingFailed != ErrorCode::ProtocolViolation);
  CHECK(ErrorCode::ProtocolViolation != ErrorCode::EnvironmentStopped);
}

} // namespace

int main() {
  test_work_item_counts_tokens();
  test_completion_stores_errors();
  test_messages_have_deterministic_defaults();
  test_all_error_codes_are_distinct();

  if (g_failures != 0) {
    std::printf("\n%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("\nall protocol checks passed\n");
  return 0;
}

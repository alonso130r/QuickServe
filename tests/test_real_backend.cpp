#include "policies/fifo.hpp"
#include "runtime/environment.hpp"

#include <atomic>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>
#include <thread>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: test_real_backend /absolute/model.gguf\n");
    return 2;
  }

  Handoff handoff(2);
  Environment environment(
      handoff, EnvironmentConfig{argv[1], 256, 64, 1});
  FifoScheduler scheduler(handoff, 64);
  const RequestId oversized = scheduler.submit_synthetic(
      std::numeric_limits<std::uint32_t>::max(),
      /*max_output_tokens=*/1, OutputMode::TraceExact);
  const RequestId first = scheduler.submit("The capital of France is", 8);
  const RequestId second = scheduler.submit("The opposite of hot is", 8);
  const RequestId synthetic = scheduler.submit_synthetic(
      /*prompt_tokens=*/11, /*max_output_tokens=*/3,
      OutputMode::TraceExact);

  std::exception_ptr environment_error;
  std::exception_ptr scheduler_error;
  std::atomic<bool> environment_exited{false};
  std::atomic<bool> scheduler_exited{false};

  std::thread environment_thread([&] {
    try {
      environment.run();
    } catch (...) {
      environment_error = std::current_exception();
    }
    environment_exited.store(true, std::memory_order_release);
  });

  const EnvironmentStartupResult startup = environment.wait_for_startup();
  if (!startup.success) {
    std::fprintf(stderr, "environment startup failed: %s\n",
                 startup.error.c_str());
    handoff.request_stop();
    environment_thread.join();
    return 1;
  }

  std::thread scheduler_thread([&] {
    try {
      scheduler.run();
    } catch (...) {
      scheduler_error = std::current_exception();
      handoff.request_stop();
    }
    scheduler_exited.store(true, std::memory_order_release);
  });

  scheduler_thread.join();
  environment_thread.join();

  CHECK(environment_error == nullptr);
  CHECK(scheduler_error == nullptr);
  CHECK(environment_exited.load(std::memory_order_acquire));
  CHECK(scheduler_exited.load(std::memory_order_acquire));
  CHECK(scheduler.all_terminal());
  CHECK(scheduler.last_error().valid);
  CHECK(scheduler.last_error().code == ErrorCode::None);

  const auto &requests = scheduler.requests();
  CHECK(requests.size() == 4);
  if (requests.size() == 4) {
    const RequestState &oversized_state = requests[oversized];
    const RequestState &first_state = requests[first];
    const RequestState &second_state = requests[second];
    const RequestState &synthetic_state = requests[synthetic];
    CHECK(oversized_state.terminal_error ==
          ErrorCode::ContextCapacityExceeded);
    CHECK(first_state.stage == RequestState::Stage::Terminal);
    CHECK(second_state.stage == RequestState::Stage::Terminal);
    CHECK(first_state.terminal_error == ErrorCode::None);
    CHECK(second_state.terminal_error == ErrorCode::None);
    CHECK(synthetic_state.terminal_error == ErrorCode::None);
    CHECK(first_state.decoded_count >= 1);
    CHECK(second_state.decoded_count >= 1);
    CHECK(!first_state.output_token_ids.empty());
    CHECK(!second_state.output_token_ids.empty());
    CHECK(!first_state.output_text.empty());
    CHECK(!second_state.output_text.empty());
    CHECK(synthetic_state.prompt_length == 11);
    CHECK(synthetic_state.decoded_count == 3);
    CHECK(synthetic_state.output_token_ids.size() == 3);
    CHECK(first_state.finish_recorded);
    CHECK(second_state.start_recorded);
    if (first_state.finish_recorded && second_state.start_recorded) {
      CHECK(second_state.start_time >= first_state.finish_time);
    }
  }

  if (g_failures != 0) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("real backend FIFO test passed\n");
  return 0;
}

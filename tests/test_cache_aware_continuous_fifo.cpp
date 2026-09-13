#include "policies/cache_aware_continuous_fifo.hpp"

#include <cstdio>
#include <vector>

namespace {
int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
  std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); ++failures; \
} } while (false)

void admit(CacheAwareContinuousFifoScheduler &scheduler, Handoff &handoff,
           const std::vector<std::pair<std::uint32_t, std::uint32_t>> &sizes) {
  scheduler.run_once();
  for (RequestId id = 0; id < sizes.size(); ++id) {
    Admission admission{};
    CHECK(handoff.try_take_admission(admission));
    CHECK(handoff.try_report_admission(
        AdmissionResult{id, sizes[id].first, ErrorCode::None,
                        sizes[id].second}));
  }
  scheduler.run_once();
}

void test_prefers_cache_savings_inside_fifo_window() {
  Handoff handoff(8);
  CacheAwareContinuousFifoScheduler scheduler(handoff, 4, 3);
  scheduler.submit("old cold", 2);
  scheduler.submit("middle hot", 2);
  scheduler.submit("new warm", 2);
  admit(scheduler, handoff, {{8, 0}, {8, 6}, {8, 3}});

  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  if (plan) {
    CHECK(plan->work.size() == 2);
    if (plan->work.size() == 2) {
      CHECK(plan->work[0].id == 1);
      CHECK(plan->work[0].token_begin == 6);
      CHECK(plan->work[0].token_end == 8);
      CHECK(plan->work[1].id == 2);
      CHECK(plan->work[1].token_begin == 3);
      CHECK(plan->work[1].token_end == 5);
    }
    handoff.retire_plan(plan);
  }
}

void test_does_not_reorder_beyond_fifo_window() {
  Handoff handoff(8);
  CacheAwareContinuousFifoScheduler scheduler(handoff, 2, 2);
  scheduler.submit("first", 2);
  scheduler.submit("second", 2);
  scheduler.submit("outside", 2);
  admit(scheduler, handoff, {{4, 0}, {4, 0}, {4, 3}});

  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  if (plan) {
    CHECK(plan->work.size() == 1);
    if (!plan->work.empty()) CHECK(plan->work[0].id == 0);
    handoff.retire_plan(plan);
  }
}
}

int main() {
  test_prefers_cache_savings_inside_fifo_window();
  test_does_not_reorder_beyond_fifo_window();
  if (failures) return 1;
  std::printf("all cache-aware continuous FIFO checks passed\n");
  return 0;
}

#include "backend_cache.hpp"
#include "experiment.hpp"
#include "solvers.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {
int failures = 0;
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                 \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

void test_nearest_rank_percentile() {
  CHECK(sloexp::nearest_rank({4.0, 1.0, 3.0, 2.0}, 0.50) == 2.0);
  CHECK(sloexp::nearest_rank({4.0, 1.0, 3.0, 2.0}, 0.99) == 4.0);
}

void test_scenarios_are_deterministic() {
  const auto a = sloexp::generate_instances(17, 12);
  const auto b = sloexp::generate_instances(17, 12);
  CHECK(a.size() == 12);
  CHECK(a == b);
}

void test_projection_uses_distance_then_ties() {
  sloexp::ContinuousSolution target{100.0, 8.0};
  std::vector<sloexp::Candidate> candidates{{96, 8, 5.0, 0}, {104, 8, 4.0, 0}};
  const auto chosen = sloexp::project_nearest(target, candidates, 512, 64);
  CHECK(chosen.has_value());
  CHECK(chosen && chosen->prefill_tokens == 104);
}

void test_one_step_score_adds_completed_and_censored_values() {
  sloexp::Snapshot snapshot;
  snapshot.now_ns = 1'000;
  snapshot.window_ns = 10'000;
  snapshot.ttft_target_ns = 1'000;
  snapshot.tpot_target_ns = 200;
  snapshot.historical_ttft = {{500, 900}};
  snapshot.historical_tpot = {{900, 100}};
  snapshot.prefills = {{1, 100, 8, 8}};
  snapshot.decodes = {{2, 900}};
  const sloexp::Candidate candidate{8, 0, 0.0, 0};
  const auto score = sloexp::score_one_step(snapshot, candidate, 100, 512, 64,
                                            1.0, 1.0, 0.0, 0.0, 0.0);
  CHECK(score.ttft_p99_ns == 1'000.0);
  CHECK(score.tpot_p99_ns == 200.0);
  CHECK(score.completed_ttft == 1);
  CHECK(score.censored_tpot == 1);
}

void test_sliding_window_headroom_uses_percentile_state() {
  sloexp::Snapshot snapshot;
  snapshot.now_ns = 1'000;
  snapshot.window_ns = 10'000;
  snapshot.ttft_target_ns = 1'000;
  snapshot.tpot_target_ns = 200;
  snapshot.historical_ttft = {{500, 400}};
  snapshot.historical_tpot = {{900, 100}};
  snapshot.prefills = {{1, 100, 8, 8}};
  snapshot.decodes = {{2, 900}};
  CHECK(std::abs(sloexp::sliding_window_headroom(snapshot, true) - 100.0) <
        1e-6);
  CHECK(std::abs(sloexp::sliding_window_headroom(snapshot, false) - 100.0) <
        1e-6);
}

void test_standardized_slack_is_unit_invariant() {
  const double ns = sloexp::standardized_runtime_slack(900.0, 1'000.0, 50.0);
  const double ms = sloexp::standardized_runtime_slack(0.9, 1.0, 0.05);
  CHECK(std::abs(ns - ms) < 1e-12);
  CHECK(std::abs(ns + 2.0) < 1e-12);
}

void test_cache_deduplicates_observations() {
  CHECK(std::string(SLO_COLLECTOR_ID).size() == 64);
  const auto path =
      std::filesystem::temp_directory_path() /
      ("slo-cache-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".csv");
  sloexp::MeasurementCache cache(path);
  sloexp::Measurement observation{"key", "obs-1", "run", 1, 2, 3, 4, 5};
  CHECK(cache.append(observation));
  CHECK(!cache.append(observation));
  CHECK(cache.load().size() == 1);
  std::filesystem::remove(path);
  std::filesystem::remove(path.string() + ".lock");
}

void test_solvers_agree_on_shared_qp() {
  const auto instance = sloexp::generate_instances(9, 1).front();
  const auto prox = sloexp::solve_proxqp(instance, 1);
  const auto clarabel = sloexp::solve_clarabel_qp(instance, 1);
  CHECK(prox.success);
  CHECK(clarabel.success);
  CHECK(std::abs(prox.p - clarabel.p) < 1e-4);
  CHECK(std::abs(prox.d - clarabel.d) < 1e-4);
  const auto socp = sloexp::solve_clarabel_socp(instance, 1);
  CHECK(socp.success);
  CHECK(socp.p >= -1e-7 && socp.d >= -1e-7);
  const auto covariance_socp =
      sloexp::solve_clarabel_covariance_socp(instance, 1);
  CHECK(covariance_socp.success);
  CHECK(covariance_socp.p >= -1e-7 && covariance_socp.d >= -1e-7);
}

void test_evaluate_mode_emits_paired_decisions(
    const std::filesystem::path &binary_dir) {
  const auto stamp = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto cache_path = std::filesystem::temp_directory_path() /
                          ("slo-eval-cache-" + stamp + ".csv");
  const auto output_path = std::filesystem::temp_directory_path() /
                           ("slo-eval-output-" + stamp + ".csv");
  sloexp::MeasurementCache cache(cache_path);
  std::uint32_t position = 0;
  for (unsigned mix = 0; mix < 3; ++mix)
    for (unsigned k = 0; k < 10; ++k) {
      unsigned p = 0, d = 0;
      if (mix == 0) {
        p = 16 * (k + 1);
        d = 1;
      } else if (mix == 1) {
        d = k + 1;
        p = 4 * d;
      } else {
        p = 0;
        d = k + 1;
      }
      const std::string key =
          "cell-" + std::to_string(mix) + "-" + std::to_string(k);
      for (unsigned rep = 0; rep < 10; ++rep) {
        const auto duration = 1'000'000 + 1'000 * p + 5'000 * d + rep * 100;
        cache.append({key, "obs-" + std::to_string(rep), "run", rep, p, d, 512,
                      duration, rep, position++, d + (p ? 1 : 0)});
      }
    }
  const auto executable = binary_dir / "slo_optimization_experiment";
  const std::string command = "\"" + executable.string() +
                              "\" evaluate --cache \"" + cache_path.string() +
                              "\" --output \"" + output_path.string() +
                              "\" --snapshots 3 --seed 1";
  CHECK(std::system(command.c_str()) == 0);
  std::ifstream in(output_path);
  std::string contents((std::istreambuf_iterator<char>(in)), {});
  CHECK(contents.find("#decision") != std::string::npos);
  CHECK(contents.find("regret") != std::string::npos);
  CHECK(contents.find("solver_status") != std::string::npos);
  CHECK(contents.find("reference_z") != std::string::npos);
  CHECK(contents.find("boundary_penalty") != std::string::npos);
  CHECK(contents.find("decision_candidate_cells,30") != std::string::npos);
  CHECK(contents.find("upper_bound_quantile,0.98") != std::string::npos);
  CHECK(contents.find("covariance_socp_kappa") != std::string::npos);
  CHECK(contents.find(",clarabel_qp,") != std::string::npos);
  CHECK(contents.find(",covariance_socp,") != std::string::npos);
  CHECK(contents.find("projection_bound_source") != std::string::npos);
  CHECK(contents.find("nan") == std::string::npos);
  std::filesystem::remove(cache_path);
  std::filesystem::remove(output_path);
}

void test_export_profile_emits_calibrated_context_tiers(
    const std::filesystem::path &binary_dir) {
  const auto stamp = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto cache_path = std::filesystem::temp_directory_path() /
                          ("slo-profile-cache-" + stamp + ".csv");
  const auto output_path = std::filesystem::temp_directory_path() /
                           ("slo-profile-" + stamp + ".conf");
  sloexp::MeasurementCache cache(cache_path);
  std::uint32_t position = 0;
  for (unsigned context : {512U, 2048U})
    for (const auto [p, d] : std::vector<std::pair<unsigned, unsigned>>{
             {32, 1}, {64, 1}, {32, 4}, {64, 4}})
      for (unsigned rep = 0; rep < 10; ++rep) {
        const auto key = "schema=3|collector=" SLO_COLLECTOR_ID
                         "|llama=" SLO_LLAMA_REV
                         "|build=" SLO_BUILD_TYPE "|cohort=test|p=" + std::to_string(p) +
                         "|d=" + std::to_string(d) + "|ctx=" +
                         std::to_string(context) + "|seq=" +
                         std::to_string(d + 1);
        cache.append({key, "obs-" + std::to_string(context) + "-" +
                               std::to_string(p) + "-" + std::to_string(d) +
                               "-" + std::to_string(rep),
                      "run", rep, p, d, context,
                      1'000'000 + 2'000 * p + 5'000 * d + rep * 100,
                      rep, position++, d + 1});
      }
  for (unsigned rep = 0; rep < 10; ++rep)
    cache.append({"schema=3|collector=stale|cohort=test|p=32|d=1|ctx=512|seq=2",
                  "stale-" + std::to_string(rep), "stale-run", rep, 32, 1,
                  512, 1'000'000, rep, position++, 2});
  const auto executable = binary_dir / "slo_optimization_experiment";
  const std::string command =
      "\"" + executable.string() + "\" export-profile --cache \"" +
      cache_path.string() + "\" --output \"" + output_path.string() +
      "\" --context-capacity 16384 --token-capacity 512 "
      "--sequence-capacity 4";
  CHECK(std::system(command.c_str()) == 0);
  std::ifstream in(output_path);
  std::string contents((std::istreambuf_iterator<char>(in)), {});
  CHECK(contents.find("schema_version=1") != std::string::npos);
  CHECK(contents.find("tier_count=2") != std::string::npos);
  CHECK(contents.find("tier.0.tau_prefill_ns_per_token=") != std::string::npos);
  CHECK(contents.find("tier.1.context_max=16384") != std::string::npos);
  std::filesystem::remove(cache_path);
  std::filesystem::remove(output_path);
}
} // namespace

int main(int argc, char **argv) {
  test_nearest_rank_percentile();
  test_scenarios_are_deterministic();
  test_projection_uses_distance_then_ties();
  test_one_step_score_adds_completed_and_censored_values();
  test_sliding_window_headroom_uses_percentile_state();
  test_standardized_slack_is_unit_invariant();
  test_cache_deduplicates_observations();
  test_solvers_agree_on_shared_qp();
  if (argc > 0)
    test_evaluate_mode_emits_paired_decisions(
        std::filesystem::path(argv[0]).parent_path());
  if (argc > 0)
    test_export_profile_emits_calibrated_context_tiers(
        std::filesystem::path(argv[0]).parent_path());
  if (failures)
    return 1;
  std::printf("all SLO optimization experiment checks passed\n");
  return 0;
}

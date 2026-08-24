#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace quickserve::benchmark {

struct RunMetadata {
  std::string policy_name;
  std::string policy_source_sha256;
  std::string policy_header_sha256;
  std::string compiler_id;
  std::string compiler_version;
  std::string build_type;
  std::string quickserve_revision;
  std::string llama_revision;
  std::string model_path;
  std::string trace_path;
  std::string trace_source_sha256;
  std::uint64_t trace_record_count{};
  std::int64_t trace_first_timestamp_ns{};
  std::int64_t trace_last_timestamp_ns{};
  std::uint64_t selected_requests{};
  std::uint64_t selected_source_duration_ns{};
  std::optional<std::uint64_t> max_requests;
  std::optional<std::uint64_t> trace_duration_limit_ns;
  std::uint64_t scheduled_duration_ns{};
  std::uint64_t last_scheduled_deadline_ns{};
  double target_qps{};
  std::optional<double> offered_request_qps;
  std::string output_mode;
  std::uint32_t context_size{};
  std::uint32_t batch_capacity{};
  std::uint32_t max_sequences{};
  std::uint32_t token_budget{};
};

struct RequestMetrics {
  std::uint32_t request_id{};
  std::uint64_t source_offset_ns{};
  std::uint64_t scheduled_arrival_ns{};
  std::uint64_t actual_arrival_ns{};
  std::int64_t arrival_lag_ns{};
  std::uint32_t input_tokens{};
  std::uint32_t executed_input_tokens{};
  std::uint32_t requested_output_tokens{};
  std::uint32_t generated_output_tokens{};
  std::optional<std::uint64_t> queue_delay_ns;
  std::optional<std::uint64_t> ttft_ns;
  std::optional<std::uint64_t> prefill_ns;
  std::optional<std::uint64_t> decode_span_ns;
  std::optional<std::uint64_t> tpot_ns;
  std::optional<std::uint64_t> e2e_latency_ns;
  std::optional<double> normalized_latency_ns_per_token;
  std::string terminal_error;
  std::string disposition;
  bool eog_observed{};
  std::string output_mode;
};

struct LogSketch {
  void add(std::uint64_t value);
  [[nodiscard]] std::optional<double> mean() const;
  [[nodiscard]] std::optional<std::uint64_t> percentile(double q) const;
  std::array<std::uint64_t, 4096> bins{};
  std::uint64_t count{};
  std::uint64_t minimum{};
  std::uint64_t maximum{};
  std::uint64_t underflow{};
  std::uint64_t overflow{};
  long double sum{};
};

[[nodiscard]] long double log_sketch_relative_error();

class AtomicResults {
public:
  explicit AtomicResults(std::filesystem::path destination);
  ~AtomicResults();
  void begin(const RunMetadata &metadata);
  bool observe(const RequestMetrics &metrics) noexcept;
  void sample_counts(std::uint64_t now_ns, std::uint64_t active,
                     std::uint64_t queued);
  void finish(std::uint64_t wall_duration_ns);

private:
  std::filesystem::path destination_;
  std::filesystem::path temporary_;
  std::ofstream requests_;
  RunMetadata metadata_;
  LogSketch arrival_lag_;
  LogSketch queue_delay_;
  LogSketch ttft_;
  LogSketch tpot_;
  LogSketch e2e_;
  std::uint64_t input_tokens_{};
  std::uint64_t output_tokens_{};
  std::uint64_t first_arrival_ns_{};
  std::uint64_t last_success_terminal_ns_{};
  bool has_arrival_{};
  std::array<std::uint64_t, 4> input_buckets_{};
  std::array<std::uint64_t, 4> output_buckets_{};
  struct BucketSummary {
    std::uint64_t count{}, success{}, failure{}, rejection{};
    long double e2e_sum{}, ttft_sum{};
    std::uint64_t e2e_count{}, ttft_count{};
  };
  std::array<BucketSummary, 4> input_bucket_summaries_{};
  std::array<BucketSummary, 4> output_bucket_summaries_{};
  std::uint64_t observed_{};
  std::uint64_t successful_{};
  std::uint64_t failed_{};
  std::uint64_t rejected_{};
  std::uint64_t admission_rejections_{};
  std::uint64_t eog_shortened_{};
  long double fairness_sum_{};
  long double fairness_square_sum_{};
  std::uint64_t fairness_count_{};
  std::uint64_t count_last_time_{};
  std::uint64_t last_active_{};
  std::uint64_t last_queued_{};
  long double active_integral_{};
  long double queued_integral_{};
  std::uint64_t peak_active_{};
  std::uint64_t peak_queued_{};
  bool begun_{};
  bool finished_{};
};

} // namespace quickserve::benchmark

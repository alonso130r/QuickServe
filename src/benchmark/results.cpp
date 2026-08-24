#include "benchmark/results.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#ifdef __APPLE__
#include <fcntl.h>
#include <stdio.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#endif

namespace quickserve::benchmark {
namespace {

std::string json_string(const std::string &text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    switch (c) {
    case '"': out << "\\\""; break;
    case '\\': out << "\\\\"; break;
    case '\n': out << "\\n"; break;
    case '\r': out << "\\r"; break;
    case '\t': out << "\\t"; break;
    default:
      if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
      else out << c;
    }
  }
  return out.str() + '"';
}

std::string csv_field(const std::string &text) {
  if (text.find_first_of(",\"\r\n") == std::string::npos) return text;
  std::string out = "\"";
  for (char c : text) out += c == '"' ? "\"\"" : std::string(1, c);
  return out + '"';
}

template <typename T> std::string optional_number(const std::optional<T> &value) {
  if (!value) return {};
  std::ostringstream out;
  out << *value;
  return out.str();
}

constexpr std::uint64_t kSketchMaximum = 1000000000000000000ULL;
const long double kLogGamma = std::log(static_cast<long double>(kSketchMaximum)) / 4093.0L;

std::size_t bin_for(std::uint64_t value) {
  if (value == 0) return 0;
  if (value > kSketchMaximum) return 4095;
  const auto index = static_cast<long double>(std::log(static_cast<long double>(value))) / kLogGamma + 1;
  if (index >= 4095) return 4094;
  return static_cast<std::size_t>(index);
}

std::uint64_t quantile(const std::array<std::uint64_t, 4096> &bins,
                       std::uint64_t count, std::uint64_t minimum,
                       std::uint64_t maximum, double q) {
  if (count == 1) return maximum;
  const auto rank = static_cast<std::uint64_t>(std::ceil(q * count));
  std::uint64_t seen = 0;
  for (std::size_t i = 0; i < bins.size(); ++i) {
    seen += bins[i];
    if (seen >= std::max<std::uint64_t>(1, rank)) {
      if (i == 0) return 0;
      if (i == bins.size() - 1) return maximum;
      const auto estimate = static_cast<std::uint64_t>(std::llround(std::exp((i - 1) * kLogGamma)));
      return std::clamp(estimate, minimum, maximum);
    }
  }
  return maximum;
}

void publish_directory_exclusive(const std::filesystem::path &source,
                                 const std::filesystem::path &destination) {
#ifdef __APPLE__
  if (::renameatx_np(AT_FDCWD, source.c_str(), AT_FDCWD, destination.c_str(),
                     RENAME_EXCL) != 0)
    throw std::system_error(errno, std::generic_category(), "cannot publish output directory");
#elif defined(__linux__)
  if (::syscall(SYS_renameat2, AT_FDCWD, source.c_str(), AT_FDCWD,
                destination.c_str(), RENAME_NOREPLACE) != 0)
    throw std::system_error(errno, std::generic_category(), "cannot publish output directory");
#else
  if (std::filesystem::exists(destination))
    throw std::invalid_argument("output directory already exists");
  std::filesystem::rename(source, destination);
#endif
}

} // namespace

void LogSketch::add(std::uint64_t value) {
  const auto bin = bin_for(value);
  ++bins[bin];
  ++count;
  sum += value;
  if (count == 1) minimum = maximum = value;
  else { minimum = std::min(minimum, value); maximum = std::max(maximum, value); }
  if (value == 0) ++underflow;
  if (value > kSketchMaximum) ++overflow;
}

std::optional<double> LogSketch::mean() const {
  if (count == 0) return std::nullopt;
  return static_cast<double>(sum / count);
}

std::optional<std::uint64_t> LogSketch::percentile(double q) const {
  if (count == 0) return std::nullopt;
  return quantile(bins, count, minimum, maximum, q);
}

long double log_sketch_relative_error() {
  return std::expm1(kLogGamma);
}

AtomicResults::AtomicResults(std::filesystem::path destination)
    : destination_(std::move(destination)) {}

AtomicResults::~AtomicResults() {
  if (!finished_ && !temporary_.empty()) {
    requests_.close();
    std::error_code ignored;
    std::filesystem::remove_all(temporary_, ignored);
  }
}

void AtomicResults::begin(const RunMetadata &metadata) {
  if (begun_) throw std::logic_error("results already begun");
  if (std::filesystem::exists(destination_)) throw std::invalid_argument("output directory already exists");
  const auto parent = destination_.parent_path().empty() ? std::filesystem::path(".") : destination_.parent_path();
  temporary_ = parent / ("." + destination_.filename().string() + ".tmp-" + std::to_string(::getpid()));
  if (!std::filesystem::create_directory(temporary_)) throw std::runtime_error("cannot create temporary output directory");
  requests_.open(temporary_ / "requests.csv", std::ios::binary | std::ios::out | std::ios::trunc);
  if (!requests_) throw std::runtime_error("cannot create requests.csv");
  requests_ << "request_id,source_offset_ns,scheduled_arrival_ns,actual_arrival_ns,arrival_lag_ns,input_tokens,executed_input_tokens,requested_output_tokens,generated_output_tokens,queue_delay_ns,ttft_ns,prefill_ns,decode_span_ns,tpot_ns,e2e_latency_ns,normalized_latency_ns_per_token,terminal_error,terminal_disposition,eog_observed,output_mode\r\n";
  metadata_ = metadata;
  begun_ = true;
}

bool AtomicResults::observe(const RequestMetrics &m) noexcept {
  try {
    requests_ << m.request_id << ',' << m.source_offset_ns << ',' << m.scheduled_arrival_ns << ','
              << m.actual_arrival_ns << ',' << m.arrival_lag_ns << ',' << m.input_tokens << ',' << m.executed_input_tokens << ','
              << m.requested_output_tokens << ',' << m.generated_output_tokens << ','
              << optional_number(m.queue_delay_ns) << ',' << optional_number(m.ttft_ns) << ','
              << optional_number(m.prefill_ns) << ',' << optional_number(m.decode_span_ns) << ','
              << optional_number(m.tpot_ns) << ',' << optional_number(m.e2e_latency_ns) << ','
              << optional_number(m.normalized_latency_ns_per_token) << ',' << csv_field(m.terminal_error) << ',' << csv_field(m.disposition) << ','
              << (m.eog_observed ? "true" : "false") << ',' << csv_field(m.output_mode) << "\r\n";
    if (!requests_) return false;
    ++observed_;
    arrival_lag_.add(static_cast<std::uint64_t>(std::max<std::int64_t>(0, m.arrival_lag_ns)));
    if (m.queue_delay_ns) queue_delay_.add(*m.queue_delay_ns);
    if (m.ttft_ns) ttft_.add(*m.ttft_ns);
    if (m.tpot_ns) tpot_.add(*m.tpot_ns);
    input_tokens_ += m.executed_input_tokens;
    output_tokens_ += m.generated_output_tokens;
    if (!has_arrival_) { first_arrival_ns_ = m.actual_arrival_ns; has_arrival_ = true; }
    else first_arrival_ns_ = std::min(first_arrival_ns_, m.actual_arrival_ns);
    const std::size_t ib = m.input_tokens <= 512 ? 0 : m.input_tokens <= 2048 ? 1 : m.input_tokens <= 8192 ? 2 : 3;
    const std::size_t ob = m.requested_output_tokens <= 32 ? 0 : m.requested_output_tokens <= 128 ? 1 : m.requested_output_tokens <= 512 ? 2 : 3;
    ++input_buckets_[ib]; ++output_buckets_[ob];
    for (BucketSummary *b : {&input_bucket_summaries_[ib], &output_bucket_summaries_[ob]}) {
      ++b->count;
      if (m.disposition == "success") ++b->success;
      else if (m.disposition == "admission_rejection") ++b->rejection;
      else ++b->failure;
      if (m.e2e_latency_ns) { b->e2e_sum += *m.e2e_latency_ns; ++b->e2e_count; }
      if (m.ttft_ns) { b->ttft_sum += *m.ttft_ns; ++b->ttft_count; }
    }
    if (m.disposition == "success") ++successful_; else {
      if (m.disposition == "admission_rejection") { ++rejected_; ++admission_rejections_; }
      else ++failed_;
    }
    if (m.output_mode == "natural" && m.eog_observed &&
        m.generated_output_tokens < m.requested_output_tokens) ++eog_shortened_;
    if (m.e2e_latency_ns) e2e_.add(*m.e2e_latency_ns);
    if (m.disposition == "success" && m.e2e_latency_ns)
      last_success_terminal_ns_ = std::max(last_success_terminal_ns_, m.actual_arrival_ns + *m.e2e_latency_ns);
    if (m.disposition == "success" && m.normalized_latency_ns_per_token &&
        *m.normalized_latency_ns_per_token > 0) {
      const long double score = 1.0L / *m.normalized_latency_ns_per_token;
      fairness_sum_ += score;
      fairness_square_sum_ += score * score;
      ++fairness_count_;
    }
    return true;
  } catch (...) { return false; }
}

void AtomicResults::sample_counts(std::uint64_t now_ns, std::uint64_t active,
                                  std::uint64_t queued) {
  if (now_ns < count_last_time_) throw std::invalid_argument("count sample time decreased");
  const auto elapsed = now_ns - count_last_time_;
  active_integral_ += static_cast<long double>(last_active_) * elapsed;
  queued_integral_ += static_cast<long double>(last_queued_) * elapsed;
  count_last_time_ = now_ns;
  last_active_ = active;
  last_queued_ = queued;
  peak_active_ = std::max(peak_active_, active);
  peak_queued_ = std::max(peak_queued_, queued);
}

void AtomicResults::finish(std::uint64_t wall_duration_ns) {
  if (!begun_ || finished_) throw std::logic_error("results are not open");
  sample_counts(wall_duration_ns, last_active_, last_queued_);
  requests_.flush();
  if (!requests_) throw std::runtime_error("failed writing requests.csv");
  requests_.close();
  if (requests_.fail()) throw std::runtime_error("failed closing requests.csv");
  std::ofstream summary(temporary_ / "summary.json", std::ios::binary | std::ios::trunc);
  if (!summary) throw std::runtime_error("cannot create summary.json");
  const auto nullable = [](long double value, bool valid) {
    if (!valid) return std::string("null");
    std::ostringstream out; out << std::setprecision(17) << static_cast<double>(value); return out.str();
  };
  const auto emit_sketch = [&](const char *name, const LogSketch &sketch) {
    const auto number = [](const auto &v) { return v ? std::to_string(*v) : std::string("null"); };
    summary << json_string(name) << ":{\"mean_ns\":" << number(sketch.mean())
      << ",\"p50_ns\":" << number(sketch.percentile(.50))
      << ",\"p90_ns\":" << number(sketch.percentile(.90))
      << ",\"p95_ns\":" << number(sketch.percentile(.95))
      << ",\"p99_ns\":" << number(sketch.percentile(.99))
      << ",\"max_ns\":" << (sketch.count ? std::to_string(sketch.maximum) : "null")
      << ",\"sketch_relative_error\":" << std::setprecision(20)
      << log_sketch_relative_error()
      << ",\"sketch_min_ns\":1,\"sketch_max_ns\":1000000000000000000"
      << ",\"underflow_count\":" << sketch.underflow << ",\"overflow_count\":" << sketch.overflow << '}';
  };
  const long double fairness = fairness_count_ == 0 || fairness_square_sum_ == 0 ? 0 :
      fairness_sum_ * fairness_sum_ / (fairness_count_ * fairness_square_sum_);
  summary << '{'
    << "\"policy_identity\":{"
    << "\"name\":" << json_string(metadata_.policy_name) << ','
    << "\"source_sha256\":" << json_string(metadata_.policy_source_sha256) << ','
    << "\"header_sha256\":" << json_string(metadata_.policy_header_sha256) << "},"
    << "\"build\":{" << "\"compiler_id\":" << json_string(metadata_.compiler_id) << ','
    << "\"compiler_version\":" << json_string(metadata_.compiler_version) << ','
    << "\"build_type\":" << json_string(metadata_.build_type) << ','
    << "\"quickserve_revision\":" << json_string(metadata_.quickserve_revision) << ','
    << "\"llama_revision\":" << json_string(metadata_.llama_revision) << "},"
    << "\"trace_path\":" << json_string(metadata_.trace_path) << ','
    << "\"model_path\":" << json_string(metadata_.model_path) << ','
    << "\"trace_source_sha256\":" << json_string(metadata_.trace_source_sha256) << ','
    << "\"trace_header\":{" << "\"record_count\":" << metadata_.trace_record_count << ','
    << "\"first_timestamp_ns\":" << metadata_.trace_first_timestamp_ns << ','
    << "\"last_timestamp_ns\":" << metadata_.trace_last_timestamp_ns << "},"
    << "\"selected_requests\":" << metadata_.selected_requests << ','
    << "\"selected_source_duration_ns\":" << metadata_.selected_source_duration_ns << ','
    << "\"max_requests\":" << (metadata_.max_requests ? std::to_string(*metadata_.max_requests) : "null") << ','
    << "\"trace_duration_limit_ns\":" << (metadata_.trace_duration_limit_ns ? std::to_string(*metadata_.trace_duration_limit_ns) : "null") << ','
    << "\"scheduled_duration_ns\":" << metadata_.scheduled_duration_ns << ','
    << "\"last_scheduled_deadline_ns\":" << metadata_.last_scheduled_deadline_ns << ','
    << "\"target_qps\":" << metadata_.target_qps << ','
    << "\"output_mode\":" << json_string(metadata_.output_mode) << ','
    << "\"environment\":{" << "\"context_size\":" << metadata_.context_size << ','
    << "\"batch_capacity\":" << metadata_.batch_capacity << ','
    << "\"max_sequences\":" << metadata_.max_sequences << ','
    << "\"token_budget\":" << metadata_.token_budget << "},"
    << "\"counts\":{" << "\"observed\":" << observed_ << ',' << "\"successful\":" << successful_ << ','
    << "\"failed\":" << failed_ << ',' << "\"rejected\":" << rejected_ << ','
    << "\"admission_rejections\":" << admission_rejections_ << ','
    << "\"eog_shortened\":" << eog_shortened_ << "},"
    << "\"wall_duration_ns\":" << wall_duration_ns << ','
    << "\"executed_input_tokens\":" << input_tokens_ << ','
    << "\"generated_output_tokens\":" << output_tokens_ << ','
    << "\"offered_request_qps\":" << (metadata_.offered_request_qps ? std::to_string(*metadata_.offered_request_qps) : "null") << ','
    << "\"achieved_request_qps\":" << (successful_ && last_success_terminal_ns_ > first_arrival_ns_ ? std::to_string(static_cast<double>(successful_) * 1e9 / (last_success_terminal_ns_ - first_arrival_ns_)) : "null") << ','
    << "\"failure_request_rate\":" << (wall_duration_ns ? std::to_string(static_cast<double>(failed_) * 1e9 / wall_duration_ns) : "null") << ','
    << "\"rejection_request_rate\":" << (wall_duration_ns ? std::to_string(static_cast<double>(rejected_) * 1e9 / wall_duration_ns) : "null") << ','
    << "\"input_token_throughput\":" << (wall_duration_ns ? std::to_string(static_cast<double>(input_tokens_) * 1e9 / wall_duration_ns) : "null") << ','
    << "\"output_token_throughput\":" << (wall_duration_ns ? std::to_string(static_cast<double>(output_tokens_) * 1e9 / wall_duration_ns) : "null") << ','
    << "\"total_token_throughput\":" << (wall_duration_ns ? std::to_string(static_cast<double>(input_tokens_ + output_tokens_) * 1e9 / wall_duration_ns) : "null") << ','
    << "\"mean_active_requests\":" << nullable(active_integral_ / std::max<std::uint64_t>(1, wall_duration_ns), wall_duration_ns != 0) << ','
    << "\"mean_queued_requests\":" << nullable(queued_integral_ / std::max<std::uint64_t>(1, wall_duration_ns), wall_duration_ns != 0) << ','
    << "\"peak_active_requests\":" << peak_active_ << ',' << "\"peak_queued_requests\":" << peak_queued_ << ','
    << "\"jain_fairness\":" << nullable(fairness, fairness_count_ != 0 && fairness_square_sum_ != 0) << ',';
  emit_sketch("arrival_lag", arrival_lag_); summary << ',';
  emit_sketch("queue_delay", queue_delay_); summary << ',';
  emit_sketch("ttft", ttft_); summary << ',';
  emit_sketch("tpot", tpot_); summary << ',';
  emit_sketch("e2e_latency", e2e_);
  const auto emit_buckets = [&](const char *name, const std::array<BucketSummary, 4> &buckets,
                                const std::array<const char *, 4> &names) {
    summary << ',' << json_string(name) << ":{";
    for (std::size_t i = 0; i < buckets.size(); ++i) {
      if (i) summary << ',';
      const auto &b = buckets[i];
      summary << json_string(names[i]) << ":{\"count\":" << b.count << ",\"success\":" << b.success
              << ",\"failure\":" << b.failure << ",\"rejection\":" << b.rejection
              << ",\"mean_e2e_ns\":" << (b.e2e_count ? std::to_string(static_cast<double>(b.e2e_sum / b.e2e_count)) : "null")
              << ",\"mean_ttft_ns\":" << (b.ttft_count ? std::to_string(static_cast<double>(b.ttft_sum / b.ttft_count)) : "null") << '}';
    }
    summary << '}';
  };
  emit_buckets("input_buckets", input_bucket_summaries_, {"1_512", "513_2048", "2049_8192", "8193_inf"});
  emit_buckets("output_buckets", output_bucket_summaries_, {"1_32", "33_128", "129_512", "513_inf"});
  summary << "}\n";
  summary.flush();
  if (!summary) throw std::runtime_error("failed writing summary.json");
  summary.close();
  if (summary.fail()) throw std::runtime_error("failed closing summary.json");
  publish_directory_exclusive(temporary_, destination_);
  finished_ = true;
}

} // namespace quickserve::benchmark

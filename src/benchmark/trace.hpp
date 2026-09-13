#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace quickserve::benchmark {

struct TraceHeader {
  std::uint64_t record_count{};
  std::int64_t first_timestamp_ns{};
  std::int64_t last_timestamp_ns{};
  std::string source_sha256_hex;
};

struct TraceRecord {
  TraceRecord() = default;
  TraceRecord(std::uint64_t offset, std::uint32_t context,
              std::uint32_t generated)
      : arrival_offset_ns(offset), context_tokens(context),
        generated_tokens(generated) {}
  std::uint64_t arrival_offset_ns{};
  std::uint32_t context_tokens{};
  std::uint32_t generated_tokens{};
  std::uint32_t turn_index{};
  std::string conversation_id;
  std::string prompt;
};

class TraceCursor {
public:
  TraceCursor(TraceCursor &&) noexcept = default;
  TraceCursor &operator=(TraceCursor &&) noexcept = default;
  TraceCursor(const TraceCursor &) = delete;
  TraceCursor &operator=(const TraceCursor &) = delete;
  bool next(TraceRecord &record);

private:
  friend class TraceReader;
  TraceCursor(const std::filesystem::path &path, const TraceHeader &header,
              bool conversation_trace);
  std::ifstream input_;
  TraceHeader header_;
  std::uint64_t index_{};
  std::uint64_t previous_offset_{};
  bool conversation_trace_{};
};

std::string sha256_hex(const std::string &bytes);
void prepare_trace(const std::filesystem::path &source_csv,
                   const std::filesystem::path &destination);

class TraceReader {
public:
  explicit TraceReader(std::filesystem::path path);
  const TraceHeader &header() const noexcept { return header_; }
  TraceRecord record(std::uint64_t index) const;
  TraceCursor cursor() const;

private:
  std::filesystem::path path_;
  TraceHeader header_;
  bool conversation_trace_{};
  std::vector<std::uint64_t> record_offsets_;
};

} // namespace quickserve::benchmark

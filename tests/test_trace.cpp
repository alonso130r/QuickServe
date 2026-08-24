#include "benchmark/trace.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using quickserve::benchmark::TraceReader;
using quickserve::benchmark::TraceRecord;
using quickserve::benchmark::prepare_trace;
using quickserve::benchmark::sha256_hex;

namespace {
int failures = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failures; } } while (false)

template <typename F> void check_throws(F &&f) {
  try { f(); ++failures; std::printf("FAIL expected exception\n"); }
  catch (const std::exception &) {}
}

fs::path temp_dir() {
  const auto name = "quickserve-trace-test-" + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = fs::temp_directory_path() / name;
  fs::create_directory(path);
  return path;
}

void write_bytes(const fs::path &path, const std::string &bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void test_sha256_vectors() {
  CHECK(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb924"
                           "27ae41e4649b934ca495991b7852b855");
  CHECK(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223"
                              "b00361a396177a9cb410ff61f20015ad");
}

void test_prepare_and_read_normative_layout() {
  const auto dir = temp_dir();
  const auto csv = dir / "input.csv";
  const auto qst = dir / "output.qst";
  const std::string bytes =
      "TIMESTAMP,ContextTokens,GeneratedTokens\r\n"
      "2024-05-10 01:00:00.000000001+01:00,7,3\r\n"
      "2024-05-10 00:00:00.000000001Z,8,4\r\n"
      "2024-05-10 00:00:01.25+00:00,9,5\r\n";
  write_bytes(csv, bytes);
  prepare_trace(csv, qst);

  CHECK(fs::file_size(qst) == 96 + 3 * 16);
  TraceReader reader(qst);
  CHECK(reader.header().record_count == 3);
  CHECK(reader.header().first_timestamp_ns == 1715299200000000001LL);
  CHECK(reader.header().last_timestamp_ns == 1715299201250000000LL);
  CHECK(reader.header().source_sha256_hex == sha256_hex(bytes));
  CHECK(reader.record(0).arrival_offset_ns == 0);
  CHECK(reader.record(0).context_tokens == 7);
  CHECK(reader.record(1).arrival_offset_ns == 0);
  CHECK(reader.record(2).arrival_offset_ns == 1249999999ULL);
  CHECK(reader.record(2).generated_tokens == 5);
  auto cursor = reader.cursor();
  TraceRecord streamed{};
  CHECK(cursor.next(streamed));
  CHECK(streamed.context_tokens == 7);
  CHECK(cursor.next(streamed));
  CHECK(streamed.arrival_offset_ns == 0);
  CHECK(cursor.next(streamed));
  CHECK(streamed.generated_tokens == 5);
  CHECK(!cursor.next(streamed));
  fs::remove_all(dir);
}

void test_rejects_bad_csv_and_preserves_destination() {
  const auto dir = temp_dir();
  const auto csv = dir / "input.csv";
  const auto qst = dir / "output.qst";
  write_bytes(csv, "TIMESTAMP,ContextTokens,Wrong\n");
  check_throws([&] { prepare_trace(csv, qst); });
  CHECK(!fs::exists(qst));

  write_bytes(csv, "TIMESTAMP,ContextTokens,GeneratedTokens\n"
                   "2024-05-10 00:00:01Z,1,1\n"
                   "2024-05-10 00:00:00Z,1,1\n");
  check_throws([&] { prepare_trace(csv, qst); });
  CHECK(!fs::exists(qst));

  write_bytes(csv, "TIMESTAMP,ContextTokens,GeneratedTokens\n"
                   "2024-05-10 00:00:00Z,0,1\n");
  check_throws([&] { prepare_trace(csv, qst); });

  write_bytes(qst, "keep");
  write_bytes(csv, "TIMESTAMP,ContextTokens,GeneratedTokens\n"
                   "2024-05-10 00:00:00Z,1,1\n");
  check_throws([&] { prepare_trace(csv, qst); });
  CHECK(fs::file_size(qst) == 4);
  fs::remove_all(dir);
}

void test_reader_rejects_corruption_and_truncation() {
  const auto dir = temp_dir();
  const auto csv = dir / "input.csv";
  const auto good = dir / "good.qst";
  write_bytes(csv, "TIMESTAMP,ContextTokens,GeneratedTokens\n"
                   "2024-05-10 00:00:00Z,1,1\n"
                   "2024-05-10 00:00:01Z,2,2\n");
  prepare_trace(csv, good);
  std::ifstream in(good, std::ios::binary);
  std::vector<char> data((std::istreambuf_iterator<char>(in)), {});

  for (const auto &entry : std::array<std::pair<const char *, std::size_t>, 5>{{
           {"magic", 0}, {"version", 8}, {"flags", 20}, {"reserved", 80}, {"offset", 112}}}) {
    const auto *name = entry.first;
    const auto index = entry.second;
    auto damaged = data;
    damaged[index] ^= 1;
    const auto path = dir / (std::string(name) + ".qst");
    std::ofstream out(path, std::ios::binary);
    out.write(damaged.data(), static_cast<std::streamsize>(damaged.size())); out.close();
    check_throws([&] {
      TraceReader reader(path);
      if (std::string(name) == "offset") {
        auto cursor = reader.cursor();
        TraceRecord record{};
        while (cursor.next(record)) {}
      }
    });
  }
  const auto truncated = dir / "truncated.qst";
  std::ofstream out(truncated, std::ios::binary);
  out.write(data.data(), static_cast<std::streamsize>(data.size() - 1)); out.close();
  check_throws([&] { TraceReader reader(truncated); });
  fs::remove_all(dir);
}

void test_cursor_rejects_nonzero_first_offset() {
  const auto dir = temp_dir();
  const auto csv = dir / "input.csv";
  const auto path = dir / "bad-first-offset.qst";
  write_bytes(csv, "TIMESTAMP,ContextTokens,GeneratedTokens\n"
                   "2024-05-10 00:00:00Z,1,1\n");
  prepare_trace(csv, path);
  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  file.seekp(96);
  const char one = 1;
  file.write(&one, 1);
  file.seekp(40);
  file.write(&one, 1); // Keep last_timestamp == first_timestamp + offset.
  file.close();
  TraceReader reader(path); // Structural validation remains O(1).
  auto cursor = reader.cursor();
  TraceRecord record{};
  check_throws([&] { cursor.next(record); });
  fs::remove_all(dir);
}
} // namespace

int main() {
  test_sha256_vectors();
  test_prepare_and_read_normative_layout();
  test_rejects_bad_csv_and_preserves_destination();
  test_reader_rejects_corruption_and_truncation();
  test_cursor_rejects_nonzero_first_offset();
  if (failures) return 1;
  std::puts("all trace checks passed");
  return 0;
}

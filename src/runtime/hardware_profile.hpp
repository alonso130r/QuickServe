#pragma once

#include <cstdint>
#include <optional>
#include <string>

struct RawHardwareProfile {
  std::string model_identifier;
  std::uint64_t total_memory_bytes{};
  std::uint32_t physical_cpu_count{};
  std::uint32_t logical_cpu_count{};
  std::uint64_t page_size_bytes{};
};

struct HardwareProfile {
  std::string model_identifier;
  std::uint64_t total_memory_bytes{};
  std::uint32_t physical_cpu_count{};
  std::uint32_t logical_cpu_count{};
  std::uint64_t page_size_bytes{};
};

[[nodiscard]] std::optional<HardwareProfile>
make_hardware_profile(RawHardwareProfile raw);

[[nodiscard]] std::optional<HardwareProfile>
collect_macos_hardware_profile();

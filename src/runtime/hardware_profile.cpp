#include "runtime/hardware_profile.hpp"

#include <utility>

#if defined(__APPLE__)
#include <sys/sysctl.h>

#include <vector>
#endif

std::optional<HardwareProfile>
make_hardware_profile(RawHardwareProfile raw) {
  if (raw.model_identifier.empty() || raw.total_memory_bytes == 0 ||
      raw.physical_cpu_count == 0 ||
      raw.logical_cpu_count < raw.physical_cpu_count ||
      raw.page_size_bytes == 0) {
    return std::nullopt;
  }

  return HardwareProfile{std::move(raw.model_identifier),
                         raw.total_memory_bytes,
                         raw.physical_cpu_count,
                         raw.logical_cpu_count,
                         raw.page_size_bytes};
}

std::optional<HardwareProfile> collect_macos_hardware_profile() {
#if defined(__APPLE__)
  auto read_value = [](const char *name, auto &value) {
    std::size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 &&
           size == sizeof(value);
  };
  auto read_string = [](const char *name) -> std::optional<std::string> {
    std::size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size <= 1) {
      return std::nullopt;
    }
    std::vector<char> buffer(size);
    if (sysctlbyname(name, buffer.data(), &size, nullptr, 0) != 0) {
      return std::nullopt;
    }
    return std::string(buffer.data());
  };

  const auto model = read_string("hw.model");
  std::uint64_t memory = 0;
  std::uint32_t physical_cpus = 0;
  std::uint32_t logical_cpus = 0;
  std::uint64_t page_size = 0;
  if (!model || !read_value("hw.memsize", memory) ||
      !read_value("hw.physicalcpu", physical_cpus) ||
      !read_value("hw.logicalcpu", logical_cpus) ||
      !read_value("hw.pagesize", page_size)) {
    return std::nullopt;
  }
  return make_hardware_profile(
      {*model, memory, physical_cpus, logical_cpus, page_size});
#else
  return std::nullopt;
#endif
}

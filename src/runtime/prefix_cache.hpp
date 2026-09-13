#pragma once

#include "protocol.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

struct PrefixCacheEntry {
  std::vector<Token> tokens;
  std::vector<std::uint8_t> state;
  std::uint64_t last_used = 0;
};

inline const PrefixCacheEntry *find_longest_full_prefix(
    const std::vector<PrefixCacheEntry> &entries,
    const std::vector<Token> &prompt) {
  const PrefixCacheEntry *best = nullptr;
  for (const PrefixCacheEntry &entry : entries) {
    if (entry.tokens.empty() || entry.tokens.size() > prompt.size() ||
        (best != nullptr && entry.tokens.size() <= best->tokens.size()))
      continue;
    if (std::equal(entry.tokens.begin(), entry.tokens.end(), prompt.begin()))
      best = &entry;
  }
  return best;
}

inline PrefixCacheEntry *find_longest_full_prefix(
    std::vector<PrefixCacheEntry> &entries, const std::vector<Token> &prompt) {
  return const_cast<PrefixCacheEntry *>(find_longest_full_prefix(
      static_cast<const std::vector<PrefixCacheEntry> &>(entries), prompt));
}

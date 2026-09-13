#pragma once

#include "protocol.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

struct PrefixCacheEntry {
  std::vector<Token> tokens;
  std::vector<std::uint8_t> state;
  std::uint64_t last_used = 0;
  std::string conversation_id;
};

inline void retain_conversation_prefix(
    std::vector<PrefixCacheEntry> &entries, PrefixCacheEntry entry,
    std::size_t capacity) {
  if (!entry.conversation_id.empty()) {
    const auto existing = std::find_if(
        entries.begin(), entries.end(), [&](const PrefixCacheEntry &candidate) {
          return candidate.conversation_id == entry.conversation_id;
        });
    if (existing != entries.end()) {
      *existing = std::move(entry);
      return;
    }
  }
  if (entries.size() == capacity) {
    const auto oldest = std::min_element(
        entries.begin(), entries.end(), [](const PrefixCacheEntry &left,
                                           const PrefixCacheEntry &right) {
          return left.last_used < right.last_used;
        });
    entries.erase(oldest);
  }
  entries.push_back(std::move(entry));
}

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

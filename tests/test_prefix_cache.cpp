#include "runtime/prefix_cache.hpp"

#include <cstdio>
#include <vector>

int main() {
  const std::vector<PrefixCacheEntry> entries{
      {{1, 2}, {10}}, {{1, 2, 3, 4}, {20}}, {{8, 9, 10}, {30}}};
  const std::vector<Token> prompt{1, 2, 3, 4, 5};
  const PrefixCacheEntry *match = find_longest_full_prefix(entries, prompt);
  if (match == nullptr || match->tokens.size() != 4 ||
      match->state != std::vector<std::uint8_t>{20}) {
    std::printf("longest reusable full prefix was not selected\n");
    return 1;
  }
  const std::vector<Token> unrelated{7, 8};
  if (find_longest_full_prefix(entries, unrelated) != nullptr) {
    std::printf("unrelated prompt unexpectedly matched\n");
    return 1;
  }
  std::printf("all prefix cache checks passed\n");
  return 0;
}

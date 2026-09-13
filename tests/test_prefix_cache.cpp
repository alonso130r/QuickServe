#include "runtime/prefix_cache.hpp"
#include "runtime/environment.hpp"

#include <cstdio>
#include <vector>

int main() {
  if (resolve_prefix_cache_capacity(0, 16) != 16 ||
      resolve_prefix_cache_capacity(64, 16) != 64) {
    std::printf("prefix cache capacity did not resolve independently\n");
    return 1;
  }
  const std::vector<PrefixCacheEntry> entries{
      {{1, 2}, {10}, 0, {}}, {{1, 2, 3, 4}, {20}, 0, {}},
      {{8, 9, 10}, {30}, 0, {}}};
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
  std::vector<PrefixCacheEntry> retained;
  retain_conversation_prefix(retained, {{1, 2}, {10}, 1, "chat-a"}, 2);
  retain_conversation_prefix(retained, {{3}, {20}, 2, "chat-b"}, 2);
  retain_conversation_prefix(retained, {{1, 2, 4}, {30}, 3, "chat-a"}, 2);
  if (retained.size() != 2 || retained[0].conversation_id != "chat-a" ||
      retained[0].tokens != std::vector<Token>({1, 2, 4})) {
    std::printf("newest conversation prefix was not retained in place\n");
    return 1;
  }
  retain_conversation_prefix(retained, {{5}, {40}, 4, "chat-c"}, 2);
  if (retained.size() != 2 || retained[0].conversation_id != "chat-a" ||
      retained[1].conversation_id != "chat-c") {
    std::printf("least recently used conversation was not evicted\n");
    return 1;
  }
  std::printf("all prefix cache checks passed\n");
  return 0;
}

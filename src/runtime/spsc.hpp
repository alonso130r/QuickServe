#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <new>
#include <type_traits>
#include <utility>

template <typename T> class SPSCQueue {
  //  static_assert(std::is_nothrow_move_assignable<T>,
  //                "try_pop requires noexcept move assignment");

public:
  // constructor; capacity must be a power of two so the index wrap is a mask
  explicit SPSCQueue(std::size_t capacity)
      : capacity_(capacity), mask_(capacity - 1),
        storage_(static_cast<Storage *>(::operator new[](
            sizeof(Storage) * capacity, std::align_val_t{alignof(Storage)}))) {
    assert(capacity >= 2 && "Capacity must be at least 2");
    assert((capacity & (capacity - 1)) == 0 &&
           "Capacity must be a power of two");
  }

  // remove copy/move
  SPSCQueue(const SPSCQueue &) = delete;
  SPSCQueue &operator=(const SPSCQueue &) = delete;
  SPSCQueue(const SPSCQueue &&) = delete;
  SPSCQueue &operator=(const SPSCQueue &&) = delete;

  // destructor
  ~SPSCQueue() {
    // get indicies
    const std::size_t read = read_index_.value.load(std::memory_order_relaxed);
    const std::size_t write =
        write_index_.value.load(std::memory_order_relaxed);

    // destroy each slot
    for (std::size_t i = read; i < write; ++i) {
      slot(i)->~T();
    }

    ::operator delete[](storage_, std::align_val_t{alignof(Storage)});
  }

  template <typename... Args> [[nodiscard]] bool try_emplace(Args &&...args) {
    const std::size_t write =
        write_index_.value.load(std::memory_order_relaxed);

    // acquire pairs with the consumer's release store after pop
    const std::size_t read = read_index_.value.load(std::memory_order_acquire);

    if (write - read >= capacity_) {
      return false; // Full.
    }

    new (&storage_[write & mask_]) T(std::forward<Args>(args)...);

    // publish the newly constructed item to the consumer
    write_index_.value.store(write + 1, std::memory_order_release);
    return true;
  }

  [[nodiscard]] bool try_push(const T &value) { return try_emplace(value); }

  [[nodiscard]] bool try_push(T &&value) {
    return try_emplace(std::move(value));
  }

  [[nodiscard]] bool try_pop(T &out) {
    const std::size_t read = read_index_.value.load(std::memory_order_relaxed);

    // acquire pairs with the producer's release store after push
    const std::size_t write =
        write_index_.value.load(std::memory_order_acquire);

    if (read == write) {
      return false; // empty
    }

    T *item = slot(read);
    out = std::move(*item);
    item->~T();

    // publish that this storage slot may be reused by the producer
    read_index_.value.store(read + 1, std::memory_order_release);
    return true;
  }

  [[nodiscard]] bool empty() const {
    const std::size_t read = read_index_.value.load(std::memory_order_acquire);
    const std::size_t write =
        write_index_.value.load(std::memory_order_acquire);

    return read == write;
  }

  [[nodiscard]] std::size_t size_approx() const {
    const std::size_t read = read_index_.value.load(std::memory_order_acquire);
    const std::size_t write =
        write_index_.value.load(std::memory_order_acquire);

    return write - read;
  }

  [[nodiscard]] std::size_t capacity() const { return capacity_; }

private:
  using Storage = std::aligned_storage_t<sizeof(T), alignof(T)>;

  struct alignas(64) PaddedIndex {
    std::atomic<std::size_t> value{0};
  };

  T *slot(std::size_t index) {
    return std::launder(reinterpret_cast<T *>(&storage_[index & mask_]));
  }

  // fixed after construction, so both threads may read them without syncing
  const std::size_t capacity_;
  const std::size_t mask_;
  Storage *const storage_;

  PaddedIndex read_index_;
  PaddedIndex write_index_;
};

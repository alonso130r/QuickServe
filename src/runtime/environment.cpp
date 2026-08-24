#include "environment.hpp"

#include <llama.h>

#include <climits>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
struct BackendGuard { BackendGuard() { llama_backend_init(); } ~BackendGuard() { llama_backend_free(); } };
struct ModelDeleter { void operator()(llama_model *p) const { llama_model_free(p); } };
struct ContextDeleter { void operator()(llama_context *p) const { llama_free(p); } };
struct SamplerDeleter { void operator()(llama_sampler *p) const { llama_sampler_free(p); } };
struct BatchGuard { explicit BatchGuard(llama_batch b) : batch(b) {} ~BatchGuard() { llama_batch_free(batch); } llama_batch batch; };
struct PlanGuard {
  Handoff &handoff;
  Plan *plan;
  ~PlanGuard() { handoff.retire_plan(plan); }
};
using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

struct RequestRecord {
  std::vector<llama_token> prompt;
  std::uint32_t prefill_position = 0;
  std::uint32_t decoded_tokens = 0;
  llama_token last_token = 0;
  llama_seq_id sequence = 0;
  OutputMode output_mode = OutputMode::Natural;
  SamplerPtr sampler;
};

struct PendingAdmission {
  Admission admission;
  std::vector<llama_token> prompt;
  ErrorCode error = ErrorCode::None;
  bool validated = false;
};

template <typename Message, typename Push>
void retry(Message &message, Push push) {
  while (!push(message)) std::this_thread::yield();
}

std::vector<llama_token> tokenize(const llama_vocab *vocab,
                                  const std::string &text, bool add_special,
                                  bool parse_special) {
  if (text.size() > static_cast<std::size_t>(INT32_MAX)) return {};
  const auto size = static_cast<std::int32_t>(text.size());
  const std::int32_t required = llama_tokenize(
      vocab, text.data(), size, nullptr, 0, add_special, parse_special);
  if (required == INT32_MIN || required >= 0) return {};
  std::vector<llama_token> tokens(static_cast<std::size_t>(-required));
  const std::int32_t count = llama_tokenize(
      vocab, text.data(), size, tokens.data(),
      static_cast<std::int32_t>(tokens.size()), add_special, parse_special);
  if (count <= 0) return {};
  tokens.resize(static_cast<std::size_t>(count));
  return tokens;
}

bool to_piece(const llama_vocab *vocab, llama_token token, std::string &piece) {
  std::vector<char> buffer(32);
  std::int32_t size = llama_token_to_piece(vocab, token, buffer.data(), static_cast<std::int32_t>(buffer.size()), 0, false);
  if (size < 0) {
    if (size == INT32_MIN) return false;
    buffer.resize(static_cast<std::size_t>(-size));
    size = llama_token_to_piece(vocab, token, buffer.data(), static_cast<std::int32_t>(buffer.size()), 0, false);
  }
  if (size < 0) return false;
  piece.assign(buffer.data(), static_cast<std::size_t>(size));
  return true;
}

std::optional<llama_token>
choose_synthetic_token(const llama_vocab *vocab) {
  const auto literal = tokenize(vocab, " QuickServe benchmark", false, false);
  for (const llama_token token : literal) {
    if (!llama_vocab_is_eog(vocab, token)) return token;
  }
  const std::int32_t count = llama_vocab_n_tokens(vocab);
  for (std::int32_t token = 0; token < count; ++token) {
    if (llama_vocab_is_eog(vocab, token)) continue;
    std::string piece;
    if (to_piece(vocab, token, piece) && !piece.empty()) return token;
  }
  return std::nullopt;
}
} // namespace

Environment::Environment(Handoff &handoff, EnvironmentConfig config)
    : handoff_(handoff), config_(std::move(config)) {}

EnvironmentStartupResult Environment::wait_for_startup() {
  std::unique_lock<std::mutex> lock(startup_mutex_);
  startup_cv_.wait(lock, [this] { return startup_ready_; });
  return startup_result_;
}

void Environment::publish_startup(EnvironmentStartupResult result) {
  {
    std::lock_guard<std::mutex> lock(startup_mutex_);
    startup_result_ = std::move(result);
    startup_ready_ = true;
  }
  startup_cv_.notify_all();
}

void Environment::run() {
  std::unique_ptr<BackendGuard> backend;
  ModelPtr model;
  ContextPtr context;
  const llama_vocab *vocab = nullptr;
  std::unique_ptr<BatchGuard> storage;
  std::vector<llama_seq_id> free_sequences;
  std::unordered_map<RequestId, RequestRecord> requests;
  std::optional<llama_token> synthetic_token;

  try {
    if (config_.model_path.empty() || config_.context_size == 0 ||
        config_.batch_capacity == 0 || config_.max_sequences == 0 ||
        config_.context_size > static_cast<std::uint32_t>(INT32_MAX) ||
        config_.batch_capacity > static_cast<std::uint32_t>(INT32_MAX) ||
        config_.max_sequences > static_cast<std::uint32_t>(INT32_MAX) ||
        static_cast<std::uint64_t>(config_.context_size) *
                config_.max_sequences >
            UINT32_MAX) {
      publish_startup({false, "invalid environment configuration"});
      return;
    }

    backend = std::make_unique<BackendGuard>();
    model.reset(llama_model_load_from_file(config_.model_path.c_str(),
                                           llama_model_default_params()));
    if (!model) {
      publish_startup({false, "failed to load model"});
      return;
    }
    llama_context_params params = llama_context_default_params();
    params.n_ctx = config_.context_size * config_.max_sequences;
    params.n_batch = config_.batch_capacity;
    params.n_ubatch = config_.batch_capacity;
    params.n_seq_max = config_.max_sequences;
    context.reset(llama_init_from_model(model.get(), params));
    if (!context) {
      publish_startup({false, "failed to create model context"});
      return;
    }
    vocab = llama_model_get_vocab(model.get());
    if (!vocab) {
      publish_startup({false, "model has no vocabulary"});
      return;
    }
    synthetic_token = choose_synthetic_token(vocab);
    storage = std::make_unique<BatchGuard>(llama_batch_init(
        static_cast<std::int32_t>(config_.batch_capacity), 0, 1));
    llama_batch &startup_batch = storage->batch;
    if (!startup_batch.token || !startup_batch.pos ||
        !startup_batch.n_seq_id || !startup_batch.seq_id ||
        !startup_batch.logits) {
      publish_startup({false, "failed to allocate decode batch"});
      return;
    }

    free_sequences.reserve(config_.max_sequences);
    requests.reserve(config_.max_sequences);
    for (std::uint32_t i = config_.max_sequences; i > 0; --i) {
      free_sequences.push_back(static_cast<llama_seq_id>(i - 1));
    }
    publish_startup({true, {}});
  } catch (const std::exception &error) {
    publish_startup({false, error.what()});
    return;
  } catch (...) {
    publish_startup({false, "environment initialization failed"});
    return;
  }

  llama_batch &batch = storage->batch;
  bool fatal = false;
  std::optional<PendingAdmission> pending_admission;
  auto fail_run = [&](ErrorCode error) {
    if (fatal) return;
    fatal = true;
    RunFatal message{error};
    retry(message, [this](const RunFatal &m) { return handoff_.try_report_fatal(m); });
  };

  while (!handoff_.stop_requested()) {
    try {
    bool progressed = false;
    Release release{};
    while (handoff_.try_take_release(release)) {
      progressed = true;
      auto found = requests.find(release.id);
      if (found != requests.end()) {
        const bool removed = llama_memory_seq_rm(
            llama_get_memory(context.get()), found->second.sequence, -1, -1);
        if (!removed) {
          // The pinned API guarantees whole-sequence removal succeeds. Treat
          // a violation as fatal and leave the sequence unavailable; context
          // destruction on this thread performs the remaining llama cleanup.
          fail_run(ErrorCode::DecodeFailed);
        } else {
          free_sequences.push_back(found->second.sequence);
        }
        requests.erase(found);
      }
      ReleaseAck ack{release.id};
      retry(ack, [this](const ReleaseAck &m) { return handoff_.try_acknowledge_release(m); });
    }

    // Keep at most one validated admission outside the bounded handoff queue.
    // Invalid front items may be rejected immediately, but a valid admission
    // waits here when every sequence is occupied and stops further popping.
    for (;;) {
      if (!pending_admission) {
        Admission admission{};
        if (!handoff_.try_take_admission(admission)) break;
        progressed = true;
        pending_admission.emplace();
        pending_admission->admission = std::move(admission);
      }

      PendingAdmission &pending = *pending_admission;
      AdmissionResult result{pending.admission.id, 0, ErrorCode::None};
      if (fatal) {
        result.error = ErrorCode::EnvironmentStopped;
      } else if (!pending.validated) {
        try {
          if (requests.count(pending.admission.id) != 0) {
            pending.error = ErrorCode::ProtocolViolation;
          } else if (pending.admission.synthetic_prompt_tokens) {
            const std::uint32_t count =
                *pending.admission.synthetic_prompt_tokens;
            if (count == 0 || !synthetic_token) {
              pending.error = ErrorCode::TokenizationFailed;
            } else if (validate_synthetic_workload(
                           count, pending.admission.max_output_tokens,
                           config_.context_size) != ErrorCode::None) {
              pending.error = ErrorCode::ContextCapacityExceeded;
            } else {
              pending.prompt.assign(count, *synthetic_token);
            }
          } else {
            pending.prompt = tokenize(vocab, pending.admission.prompt, true,
                                      true);
            if (pending.prompt.empty()) {
              pending.error = ErrorCode::TokenizationFailed;
            }
          }
          if (pending.error == ErrorCode::None &&
              pending.prompt.size() +
                      static_cast<std::uint64_t>(
                          pending.admission.max_output_tokens) >
                  config_.context_size) {
            pending.error = ErrorCode::ContextCapacityExceeded;
          }
          pending.validated = true;
        } catch (...) {
          fail_run(ErrorCode::EnvironmentStopped);
          pending.error = ErrorCode::EnvironmentStopped;
          pending.validated = true;
        }
      }

      if (fatal) result.error = ErrorCode::EnvironmentStopped;
      else result.error = pending.error;

      if (result.error == ErrorCode::None && free_sequences.empty()) break;

      if (result.error == ErrorCode::None) {
        SamplerPtr sampler(llama_sampler_init_greedy());
        if (!sampler) {
          result.error = ErrorCode::SamplingFailed;
        } else {
          RequestRecord record{};
          record.prompt = std::move(pending.prompt);
          record.sequence = free_sequences.back();
          free_sequences.pop_back();
          record.output_mode = pending.admission.output_mode;
          record.sampler = std::move(sampler);
          result.prompt_tokens =
              static_cast<std::uint32_t>(record.prompt.size());
          requests.emplace(pending.admission.id, std::move(record));
        }
      }
      retry(result, [this](const AdmissionResult &m) {
        return handoff_.try_report_admission(m);
      });
      pending_admission.reset();
    }

    if (!fatal) {
      Plan *plan = handoff_.consume_plan();
      if (plan) {
        PlanGuard plan_guard{handoff_, plan};
        progressed = true;
        batch.n_tokens = 0;
        struct Pending { WorkItem work; RequestRecord *record; std::int32_t logits_row; };
        std::vector<Pending> pending;
        std::unordered_set<RequestId> seen;
        bool valid = true;
        for (const WorkItem &work : plan->work) {
          auto found = requests.find(work.id);
          if (found == requests.end() || !seen.insert(work.id).second ||
              work.token_begin >= work.token_end ||
              static_cast<std::uint64_t>(batch.n_tokens) + work.token_count() > config_.batch_capacity) { valid = false; break; }
          RequestRecord &record = found->second;
          const bool prefill = work.kind == WorkKind::Prefill;
          if ((prefill && (work.token_begin != record.prefill_position ||
                           work.token_end > record.prompt.size())) ||
              (!prefill && (work.kind != WorkKind::Decode || record.decoded_tokens == 0 ||
               work.token_begin != record.prompt.size() + record.decoded_tokens - 1 || work.token_end != work.token_begin + 1))) {
            valid = false; break;
          }
          for (std::uint32_t pos = work.token_begin; pos < work.token_end; ++pos) {
            const std::int32_t row = batch.n_tokens++;
            batch.token[row] = prefill ? record.prompt[pos] : record.last_token;
            batch.pos[row] = static_cast<llama_pos>(pos);
            batch.n_seq_id[row] = 1;
            batch.seq_id[row][0] = record.sequence;
            batch.logits[row] = 0;
          }
          const bool output = !prefill || work.token_end == record.prompt.size();
          const std::int32_t row = batch.n_tokens - 1;
          batch.logits[row] = output ? 1 : 0;
          pending.push_back({work, &record, output ? row : -1});
        }
        if (!valid || batch.n_tokens == 0) {
          fail_run(ErrorCode::ProtocolViolation);
          continue;
        }
        if (llama_decode(context.get(), batch) != 0) {
          fail_run(ErrorCode::DecodeFailed);
          continue;
        }
        for (Pending &item : pending) {
          Completion completion{};
          completion.id = item.work.id;
          completion.kind = item.work.kind;
          completion.prefill_position = item.work.kind == WorkKind::Prefill ? item.work.token_end : static_cast<std::uint32_t>(item.record->prompt.size());
          if (item.work.kind == WorkKind::Prefill) {
            item.record->prefill_position = item.work.token_end;
          }
          completion.decoded_tokens = item.record->decoded_tokens;
          if (item.logits_row >= 0) {
            const llama_token token = llama_sampler_sample(item.record->sampler.get(), context.get(), item.logits_row);
            if (token < 0) { fail_run(ErrorCode::SamplingFailed); break; }
            item.record->last_token = token;
            ++item.record->decoded_tokens;
            completion.decoded_tokens = item.record->decoded_tokens;
            completion.token = token;
            completion.generated_token = true;
            completion.eos = llama_vocab_is_eog(vocab, token);
            if (!completion.eos) {
              std::string piece;
              if (!to_piece(vocab, token, piece)) { fail_run(ErrorCode::SamplingFailed); break; }
              OutputPiece output{item.work.id, token, std::move(piece)};
              retry(output, [this](OutputPiece &m) { return handoff_.try_report_output(std::move(m)); });
            }
          }
          retry(completion, [this](const Completion &m) { return handoff_.try_report_completion(m); });
        }
      }
    }
    if (!progressed) std::this_thread::yield();
    } catch (...) {
      fail_run(ErrorCode::EnvironmentStopped);
    }
  }
}

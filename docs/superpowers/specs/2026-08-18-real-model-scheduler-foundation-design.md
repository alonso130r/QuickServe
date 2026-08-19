# Real-Model Scheduler Foundation Design

## Goal

Reach the point where QuickServe can run interchangeable scheduling policies
against a real llama.cpp model from a C++ executable. The first implementation
uses a hardcoded model path and hardcoded prompts so effort stays focused on the
scheduler/environment boundary.

## Scope

The milestone includes a genuine two-thread system, real prompt tokenization,
chunked prefill, iterative decode, sampling, EOS and output-limit handling,
terminal output, and basic per-request timing. A strict non-preemptive FIFO
queue provides the demo policy.

Python bindings, HTTP serving, runtime configuration, cancellation, distributed
execution, speculative decoding, and polished benchmarking are out of scope.

## Ownership

The scheduler thread owns request lifecycle and policy state. It folds in
environment feedback, decides which ready requests run, and publishes plans
that fit the configured batch-token budget. It never owns or calls llama.cpp
objects.

The environment thread owns the model, context, vocabulary, samplers, sequence
IDs, tokenized prompts, and KV-cache lifecycle. Every llama.cpp API call occurs
on this thread. It turns scheduler plans into llama batches and reports absolute
progress back to the scheduler.

The main thread constructs the demo inputs, starts both workers, waits for
completion, reads final results only after the scheduler worker joins, and
performs orderly shutdown. The scheduler thread is the sole terminal-output
writer while work is running, so it can print environment-produced pieces as
they arrive without adding a third cross-thread queue.

## Components

### Protocol types

Protocol types are regular C++ headers rather than implementation files included
as headers. The scheduler allocates each stable numeric request ID before its
worker starts. An admission carries that ID, prompt text, and generation limit
from scheduler to environment. An admission result carries the tokenized prompt
length or an error from environment to scheduler. A release message travels
from scheduler to environment after the scheduler makes a request terminal. A
separate run-fatal message lets environment invalidate the entire run when a
mixed-sequence llama batch cannot be recovered safely.

A plan contains absolute prefill ranges and single-token decode operations.
Absolute positions make feedback idempotent. A completion contains absolute
prefill and decode positions, whether it generated a token, the token ID, EOS,
and an error status. A separate owned output-piece message carries text from
environment to scheduler because detokenization uses the environment-owned
vocabulary.

### Handoff

The existing single-outstanding-plan protocol remains lock-step for the first
milestone. It gains bounded admission and release channels (scheduler producer,
environment consumer), plus admission-result, completion, output-piece, release
acknowledgement, and run-fatal channels (environment producer, scheduler
consumer). Main never touches these SPSC channels. Every producer retains and
retries a message when its queue is full. After publishing a run-fatal message,
environment stops executing plans but stays alive to consume releases and send
release acknowledgements. Shutdown continues draining until pending messages
and releases settle. No correctness-bearing message may be silently dropped.

Scheduler tracks admissions as `NotSent`, `InFlight`, or `EnvironmentOwned`.
An admission error creates no environment-owned resource and becomes terminal
without a release. A successful result moves the request to `EnvironmentOwned`
and always requires release and acknowledgement. The stop signal means “enter
draining mode,” not “exit immediately”: scheduler stops flushing `NotSent`
admissions and fails them locally; environment consumes already queued
`InFlight` admissions but rejects them with `EnvironmentStopped` unless they
already succeeded, stops executing new plans, consumes releases, publishes
pending results and acknowledgements, and returns only when it owns no request
resources. Scheduler waits for a result from every `InFlight` admission and
releases any that raced to success.

### Scheduler

The scheduler base owns admission bookkeeping, completion folding, plan
lifetime, all terminal-state decisions, output-limit handling, and invariant
checks. A policy sees immutable request metadata and fills the next plan. The
base rejects plans unless every request appears at most once, each prefill starts
at its acknowledged position and stays within its prompt, and each decode has
`decoded >= 1` with the exact range `[prompt_length + decoded - 1,
prompt_length + decoded)`. Total plan cost is prefill range lengths plus decode
count and must fit the token budget. It sends a release only after recording a
terminal state.

A scheduler-originated protocol violation uses the same fail-all cleanup path as
a run-fatal environment error. Scheduler stops publishing, marks every request
failed, releases every successfully admitted request, waits for acknowledgements,
and only then finishes. `NotSent` admissions terminate locally; `InFlight`
admissions are resolved as described above so a concurrent success cannot
strand environment state.

The demo policy is a strict FIFO queue. It selects the oldest admitted,
nonterminal request and schedules only that request until it completes. Its
prefill may be split into token-budget-sized chunks; after prefill, each plan
contains one decode step for that same request. Only after release is
acknowledged does the next queued request become active. The demo deliberately
does not perform continuous batching, interleave requests, or preempt work.

### Environment

The environment loads the hardcoded GGUF model on its own thread and reports
startup success or failure before scheduling begins. It tokenizes admissions,
assigns one llama sequence ID per request, and retains prompt tokens.

For each plan it builds one llama batch, executes it, samples requested logits,
updates its environment-owned records, detokenizes newly sampled non-EOS tokens,
and publishes output pieces and completions before retiring the plan. It does
not independently decide that an output limit is terminal. It removes a KV
sequence and sampler only after receiving the scheduler's release message.

The final token of a request's final prefill chunk requests logits. Immediately
after that batch succeeds, the environment samples the first generated token,
so those logits never need to survive another `llama_decode` call. That token's
absolute position is `prompt_length`, and the completion reports `decoded = 1`.
Each later decode work item feeds the most recently generated token at position
`prompt_length + decoded - 1`, requests logits for that row, and samples the
token at position `prompt_length + decoded`. Thus KV contains all prompt tokens
and all generated tokens except the newest sampled token. The invariant is
`next_sample_position = prompt_length + decoded`.

### Demo executable

The demo contains source-level constants for the model path, prompts, context
size, batch-token budget, maximum output tokens, and sampling settings. It
starts the environment, submits all demo prompts, runs until every request is
terminal, prints generated text as it arrives, prints time-to-first-token and
total latency, and joins both threads.

## Data Flow

1. Main submits prompt text and generation limits.
2. Environment tokenizes the prompt and returns its token count.
3. Scheduler marks the request ready for policy decisions.
4. Scheduler publishes a token-budgeted plan.
5. Environment executes the plan through llama.cpp.
6. Environment reports owned text pieces and absolute progress, generated
   tokens, EOS, or errors.
7. Scheduler folds feedback into its authoritative request state.
8. When scheduler marks a request terminal, it sends environment a release.
9. The cycle repeats until all requests are terminal and released.
10. Main reads measurements after join and shuts down both workers.

## Failure Handling

Model-load or context-creation failure prevents the scheduler thread from
starting and returns a nonzero exit code with the llama.cpp error. Tokenization
or per-request capacity failures become request errors. Because one
`llama_decode` call may cover several sequences and does not promise per-sequence
rollback, any batch decode or sampling failure is fatal to the run: environment
publishes a run-fatal message, scheduler assigns that error to every nonterminal
request (including requests absent from the failed plan), releases them, and
both workers stop. Queue saturation applies
backpressure; it never drops messages. Batch-token capacity is enforced per
plan; aggregate context/KV capacity is checked separately at admission and may
reject a request. Shutdown uses an explicit stop signal and always joins both
threads.

## Testing

Protocol and policy tests use a fake environment and require no model. They
cover admission, token-budget enforcement, chunked prefill, multiple requests,
EOS, output limits, errors, backpressure, and shutdown.

Environment unit tests isolate batch construction and sequence bookkeeping where
possible. A separately labeled smoke test accepts a developer-local GGUF path,
runs a tiny prompt through the real environment, requires at least one generated
token, and verifies clean completion. It is not part of default CI because the
model artifact is not stored in the repository.

## Ready-for-Fun Exit Criteria

The milestone is complete when the C++ demo can load a local GGUF model, run at
least two requests through the two-thread scheduler/environment system, generate
text without duplicate positions or KV corruption, respect the batch-token and
output-token limits, cleanly release sequences, print basic latency metrics, and
pass all model-free tests plus the opt-in real-model smoke test.

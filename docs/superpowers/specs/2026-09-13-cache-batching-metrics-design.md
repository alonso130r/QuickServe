# Cache and Continuous-Batching Metrics Design

## Goal

Produce reproducible metrics for three resume claims: continuous-batching
efficiency, conversation-prefix cache effectiveness, and end-to-end cache
impact under matched workloads.

## Scope

Extend the benchmark result pipeline without changing scheduling semantics. The
benchmark will record cache activity and batch utilization, expose an explicit
cache-disable option, and aggregate cache cohorts into `summary.json`.

## Cache telemetry

The environment will attach cache-operation telemetry to the existing
single-producer protocol messages. `AdmissionResult` will carry lookup
eligibility, hit/miss outcome, cached-token count, restore latency, and restore
failure. `ReleaseAck` will carry save latency, save failure, insertion,
replacement, eviction, and the post-operation resident serialized-state bytes.
These fields travel on the existing bounded environment-to-scheduler channels,
so telemetry cannot reorder operations or introduce another producer. The
result writer aggregates only messages already required for protocol progress;
fatal shutdown follows the existing acknowledgement and drain rules.

Counters cover lookups, hits, misses, insertions, replacements, evictions,
state-save failures, state-restore failures, cached tokens, current serialized
state bytes, and peak serialized state bytes. Timing sketches cover attempted
state saves and restores, including failures.

Add an explicit `--disable-prefix-cache` flag. Omission keeps caching enabled
with the existing default of one entry per sequence; a positive
`--prefix-cache-capacity N` overrides that capacity. Metadata records both
`prefix_cache_enabled` and the effective capacity, which is zero when disabled.

Per-request output will retain `cached_input_tokens`. Summary output will add a
`cache` object containing request hit rate, prompt-computation-avoided rate,
prefix-length distribution, memory usage, and state-operation latency. Hit rate
is `successful cache restores / eligible lookups`. Prompt computation avoided
is `cached_input_tokens / logical_input_tokens` over successful eligible
requests, where logical input is cached plus executed input. The cached-prefix
length distribution contains successful cache hits only. Failed and rejected
requests remain in failure counters but do not contribute to these headline
rates.

## Batch telemetry

The existing scheduler-side `BatchOutcome` will additionally record distinct
request IDs from the published plan. Its duration remains the existing
scheduler-observed interval from plan publication until completion is observed;
it is labeled accordingly and is not claimed as isolated `llama_decode` time.
No additional environment-thread transport is required. Each attempted outcome
therefore reports prefill tokens, decode items, work items, distinct requests,
observed duration, and success. Utilization distributions include successful
batches only. Token-budget fill is
`(prefill_tokens + decode_items) / token_budget`; unused token capacity is the
nonnegative difference from that same scheduler budget. Sequence utilization
is `distinct_sequences / max_sequences`. Requests per batch means distinct
request IDs and is therefore identical to distinct sequences in the current
one-sequence-per-request runtime. Failed batch counts remain separate.

## Cohort telemetry

Successful requests will be divided into `hit`, `eligible_miss`, and
`ineligible` cohorts. A request is eligible when caching is enabled and the
admission contains a real, non-synthetic prompt; a lookup without a restorable
prefix is an eligible miss. Each cohort reports count plus mean and
p50/p95/p99 TTFT. The existing `prefill_ns` field is explicitly described as
scheduler-start-to-first-token and will not be presented as backend prefill
execution time. Cohorts describe observed requests only; causal cache impact
comes from matched cache-on/cache-off runs.

## Controlled comparison

The policy-comparison runner will accept repeated cache modes (`off`, `default`,
or a positive explicit capacity) as a matrix dimension. Cache mode appears in
experiment identity, output paths, run keys, report rows, and metadata. It is
excluded from the invariant signature used to pair cache modes; model, trace,
revision, policy build, non-cache capacities, offered load, request count, and
output mode must match. The report emits raw rows; percentage improvements are
computed by a separate analysis step so the benchmark remains policy-neutral.

## Error handling

Telemetry failures must not change inference behavior. Failed state saves and
restores increment explicit counters. Rates with zero denominators serialize as
`null`. Cache-disabled runs record zero activity and zero serialized-state
memory. Memory figures deliberately exclude vector, token, string, and
allocator overhead and are labeled `serialized_state_bytes`.

## Verification

Focused tests will cover cache counter transitions, cache-disabled behavior,
batch-utilization arithmetic, cohort aggregation, JSON output, and comparison
runner argument propagation. Existing benchmark and runtime tests must remain
passing.

# Real llama.cpp Scheduler Backend Design

## Goal

Finish the C++ backend so scheduler policies execute requests end to end against
a real llama.cpp model. FIFO is the first working policy, but the environment
executes every valid work item in a scheduler plan so later policies do not
require environment changes.

## Scope

This milestone includes the existing scheduler/handoff/FIFO work plus one real
llama.cpp environment thread. It loads a GGUF model, tokenizes admissions,
executes prefill and decode work, samples tokens, returns text and progress,
releases request state, and shuts down cleanly.

There is no demo program, fake executor, Python or HTTP API, metrics system,
configuration framework, advanced capacity manager, or broad new test suite.

## Ownership and Flow

- The scheduler thread owns request lifecycle and policy decisions.
- The environment thread owns every llama.cpp object and makes every llama.cpp
  call, including destruction.
- `Handoff` remains the only communication path between the two threads.
- FIFO schedules the oldest active request until it is complete and released.
- The environment supports all work items in a plan, including work from
  multiple request sequences, so FIFO is not baked into the executor.

For each admission, the environment tokenizes the prompt, allocates a llama
sequence, and reports the prompt length. For each plan, it builds one
`llama_batch`, runs `llama_decode`, samples requested logits, emits owned text
pieces and absolute completions, and then retires the plan. The final prefill
token produces the first sampled output token; each decode step feeds the most
recent sampled token and produces the next one.

When the scheduler sends a release, the environment removes that sequence from
the KV cache, destroys its sampler and request state, and acknowledges the
release. Decode or sampling failure is fatal to the run and uses the existing
`RunFatal` cleanup path.

## Capacity and Sampling

The first implementation uses a small fixed maximum sequence count and greedy
sampling. It rejects prompts that cannot fit their requested output within the
configured per-request context. It does not implement dynamic memory planning
or policy-specific admission control.

## Verification

Keep the existing model-free tests. Add one real-model end-to-end test that
runs two requests through the production environment and FIFO policy, verifies
both produce output and become terminal in FIFO order, and confirms both
threads exit. The local GGUF path is supplied when configuring or running that
test; no separate demo or smoke-test framework is needed.

## Done

- Real llama.cpp inference occurs only on the environment thread.
- Two FIFO requests complete end to end and release their environment state.
- The environment is plan-driven rather than FIFO-specific.
- A new scheduler policy can be implemented without modifying the executor.

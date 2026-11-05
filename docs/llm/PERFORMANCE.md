# Task brief: investigate a slow but correct chat

> **For your frontier model.** People: read [the measurements](../MEASUREMENTS.md) instead, and see
> [how to use this brief](README.md). Model: also follow [AGENTS.md](../../AGENTS.md).
> If you cannot open a linked file, ask for it; do not guess.

Read [MEASUREMENTS.md](../MEASUREMENTS.md), [PROJECT.md](../../PROJECT.md),
[PROTOCOL.md](../PROTOCOL.md), [BACKENDS.md](../../BACKENDS.md) and
[the troubleshooting guide](../TROUBLESHOOTING.md). Establish correctness first
with [RUNTIME.md](RUNTIME.md) if any node/reply/numerical check fails.
Use [BENCH.md](../../BENCH.md) for timing boundaries and comparison methods.
Explain the relevant evidence without asking the user to follow the whole
development history.

## Freeze the comparison

Record source/build, model digest, prompt, context, temperature/seed, requested
tokens, API completion tokens, MTP proposal/acceptance and experimental flags.
Record each machine's unit ownership, RAM/SSD split, cache, backend, network
path and launch context. Keep other loads and power conditions comparable.
Do not compile, hash the model or calibrate while measuring chat.

Separate time to first token, full request throughput and decode throughput.
Streaming fragments are not inherently tokens. Report API counts and the exact
timing interval; label older fragment-based results as such. Include cold/warm
behavior and an unchanged control before and after the candidate. If placement
or outputs differ, describe that difference rather than attributing everything
to the new kernel.

## Find the work that actually delays completion

Start from existing coarse reports: dense work, last-arriving participant,
node compute, arrival and local waits. Arrival includes software/queueing and
network effects; it is not cable latency. Parallel node times are not additive.
Trace the critical path before optimizing an already hidden operation.

For SSD work, distinguish cold reads, cache hits, staging copies and GPU compute.
For MTP, distinguish proposal, verification, commitment/rollback and useful
accepted tokens. Deeper speculation is not a gain unless accepted work pays for
its total extra cost. Expert reuse depends on actual routing, not just batch
width. Neuron slicing enables balanced work but does not remove dependencies.

Before ISA/GPU changes, cite primary hardware documentation and state the data
layout, lanes, memory traffic and reduction order. Measure complete backend
calls, including validation, worker handoff and joining. Keep instrumentation
opt-in and compare uninstrumented runs. Do not add permanent per-layer logging
or allocations merely to diagnose a one-off failure.

## Decide using evidence

Keep changes isolated. Use existing reproducible probes and numerical gates,
then a complete bracketed cluster comparison. Report individual observations
and both controls when they drift. A microkernel speedup cannot be added to a
different overlap experiment's speedup. Synthetic token-equivalent expert rates
are not chat rates; ten hypothetical nodes are a scenario, not a benchmark.

Keep a candidate opt-in if it fails correctness or cannot beat its controls.
Preserve negative results and restore ordinary chat. Do not promise 30 tok/s
from extrapolated slice time or an acceptance rate alone.

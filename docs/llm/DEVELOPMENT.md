# Task brief: make a focused contribution

> **For your frontier model.** People: read [where to help](../IMPROVEMENTS.md) instead, and see
> [how to use this brief](README.md). Model: also follow [AGENTS.md](../../AGENTS.md).
> If you cannot open a linked file, ask for it; do not guess.

Read [PROJECT.md](../../PROJECT.md), [CONTRIBUTING.md](../../CONTRIBUTING.md)
and [IMPROVEMENTS.md](../IMPROVEMENTS.md). Choose one question with the user.
Do not turn a small contribution into a new framework or a rewrite of the
whole runtime. Inspect the implementation before proposing a replacement.

## Establish the boundary

Identify the input, output, owner and lifetime of the data being changed.
Find the caller and the test that exercises that boundary. Summarize the current
behavior and the proposed difference with a concrete example. Explain the
hardware or measurement evidence for an optimization; consult primary vendor
documentation before selecting SIMD instructions or a Metal memory layout.

For a node kernel, also read [BACKENDS.md](BACKENDS.md). For scheduling or
transport, read [PROTOCOL.md](PROTOCOL.md) and [RUNTIME.md](RUNTIME.md).
For measurements, use [PERFORMANCE.md](PERFORMANCE.md); for setup failures,
start with [DEBUG.md](DEBUG.md). Keep the user's working installation available.

## Implement the smallest useful experiment

- Keep experimental engines and binaries separate from the ordinary head.
- Preserve the complete-sum, wire identity, numerical and ownership contracts.
  A missing participant cannot become a partial answer.
- Preserve accepted-prefix state and drain outstanding jobs before reusing
  buffers. Candidate rows share immutable weights, not mutable branch state.
- Do not introduce per-layer allocation, logging, compilation or calibration.
  Measure any extra dispatch, queue, copy or synchronization cost.
- Keep existing models and cached calibration. Remeasure compute only when its
  implementation, hardware or applicable profile changes.
- Follow the neighboring code. Explain units, synchronization and non-obvious
  tradeoffs in comments; avoid restating assignments or adding abstractions
  solely to shorten a function.

For a switch experiment, distinguish bytes serialized on the shared uplink from
socket/CPU delay. Report the physical topology. A synthetic ten-node result on
one process is not a ten-machine speed claim. For deeper MTP, report proposal,
verification, recovery, acceptance and total useful-token rate, including widths
that lose. Neither a wider batch nor an idle-looking thread proves spare hardware
capacity.

## Leave a result another person can check

Run the focused regression/conformance checks, then the appropriate integration
test. For a performance claim, compare unchanged controls before and after the
candidate with the same model, prompt, placement and completion-token count.
Name the hardware actually tested and any test that could not run.

Update the guide and its task brief when behavior changes. End with a small
diff, commands and results, the reason to keep or reject the change, and the
next question if unresolved. Use [REPORT.md](REPORT.md) to prepare a reviewed
issue or pull request. Publishing messages or a release requires the user's
authorization; local development does not imply permission to speak for them.

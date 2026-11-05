# Task brief: session, numerical or chat failure

> **For your frontier model.** People: read [the troubleshooting guide](../TROUBLESHOOTING.md) instead, and see
> [how to use this brief](README.md). Model: also follow [AGENTS.md](../../AGENTS.md).
> If you cannot open a linked file, ask for it; do not guess.

Read [the troubleshooting guide](../TROUBLESHOOTING.md),
[PROJECT.md](../../PROJECT.md), [PROTOCOL.md](../PROTOCOL.md), then `src/head.c`,
`node.c`, `plane.h`, `plane.c`, `units.c`, `serve.c`, `runtime.c` and the relevant
backend. For web-only failures inspect `tools/fronk-web.py` and the chat page.

First determine whether the failure is before session load, during the scalar
comparison, in generation, or only in the browser. Do not restart every machine
before collecting the first error and current placement. Identify the actual
executables and environment flags: an old wrapper and the installed binary may
select different paths.

## Follow the result back to its owner

- A session belongs to one authenticated control connection. Use the manager
  lifecycle; manual SESSION/LINK commands can disrupt a live head.
- Each required neuron unit has one owner. Inspect full 20-unit coverage, RAM/
  SSD masks, selected backends and the failing node's session error.
- The node assembles a complete request, publishes it to a persistent worker,
  waits for a complete result and sends tagged partials. The head validates
  source/session/step/layer/width and rejects stale or duplicate fragments.
- A missing-reply bit mask refers to this session's participant order. Resolve
  it from the current plan; do not assume a fixed machine for bit 0 or bit 2.
- Pairing and ping do not prove UDP works from the current launch context.
  Inspect send errors and permissions before blaming arithmetic or the GPU.
- A numerical mismatch needs the model/unit identities, backend, layer/width,
  inputs and measured error. Keep frozen tolerances and the scalar reference.
- If generation works through the API but not the page, inspect HTTP/stream
  completion and browser errors before touching node kernels.

Runtime defaults require complete sums and use GPU event resume. Do not enable
partial sums or GPU polling to make a failure disappear. Experimental deep MTP,
reordered AVX2 reduction, HC and SSD-pipeline switches are not ordinary policy.

## Prove recovery

Run relevant core/plane tests for code changes, then the failed physical boundary.
Require successful load, independent numerical check and a finished generated
response with all participants. HTTP 200 can precede a streaming failure, so
inspect finish/error status and completion counts too. Verify the browser if
the reported problem is graphical. Restore the working runtime after an A/B
test and verify it again from its intended launch context.

Record outputs, exact checks passed, any unavailable hardware and the current
running state. Do not call an isolated kernel or localhost fixture a complete
heterogeneous-cluster validation.

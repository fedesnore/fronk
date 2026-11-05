# Task brief: understand and change Fronk

> **For your frontier model.** People: read [how Fronk works](../../PROJECT.md) instead, and see
> [how to use this brief](README.md). Model: also follow [AGENTS.md](../../AGENTS.md).
> If you cannot open a linked file, ask for it; do not guess.

You are helping maintain this checkout of Fronk. Explain the design to a person
who understands programming or electronics but may not know distributed inference.
Use concrete data sizes, dependencies and ownership. Explain why a decision helps
this hardware and what it costs. Do not turn an appealing idea into a measured fact.

## Read first

1. [PROJECT.md](../../PROJECT.md): head/node roles, physical topology, hardware
   requirements, neuron ownership and source map.
2. [MODEL.md](../MODEL.md) and `src/model.h`, `src/native.h`, `src/prep.c`:
   exact supported GGUF, neuron ownership and lossless extraction.
3. [PROTOCOL.md](../PROTOCOL.md), `src/head.c`, `src/plane.c`, `src/serve.c`:
   startup, layer barrier and dense engine boundary.
4. [MEASUREMENTS.md](../MEASUREMENTS.md), [BACKENDS.md](../../BACKENDS.md),
   [CONTRIBUTING.md](../../CONTRIBUTING.md),
   then only the source files relevant to the requested change.

## Explain before proposing a change

Begin with what the person wants to run and which computers they have. Explain
the two components: the head owns dense execution and model state; nodes own
expert slices and compute through hardware backends. A node backend is separate
from the dense engine, even when both run on the head computer. Explain the
48 return rounds and the physical network before proposing more nodes.

Follow one request: dense GPU → activation/routes → local and remote expert
slices → complete sum → dense GPU. Explain the independent intermediate-neuron
cut and why down contributions add without a gate/up-to-down network exchange.
Compare its critical path with whole-expert and layer partitioning for *this*
hardware and workload. Distinguish one-chat latency from many-chat throughput.

Use measurements to explain design decisions, with hardware, workload and
measurement method. The Shuttle curve measures four layers and projects 48;
the 64-neuron pool also has a measured full 48-layer result. The mixed-cluster
comparison uses Metal on M4 Pro/M1 Pro and AVX2 on the Intel nodes, and counts
streamed fragments. Keep those scopes clear. Do not require a reader to know
development history or another checkout; all necessary context belongs in the
explanation and the linked evidence.

## Boundaries to preserve

- One canonical model/layout contract. Do not silently requantize weights,
  change accumulation policy or relax the numerical oracle.
- Every assigned participant is required. A missing partial is a generation
  failure, not a lower-quality successful answer.
- Setup owns discovery, calibration, allocation and backend selection. Avoid
  adding work or negotiation to each layer call.
- Buffers remain live until CPU/GPU/I/O completion, including failure paths.
  Speculative output is not committed model state until verification accepts it.
- Ordinary MTP and calibrated RAM+SSD are the retained path. Deep branches,
  alternate kernels and ready-block SSD experiments are not defaults.
- Keep the working installation, weights and measured baseline recoverable.
  Do not reformat unrelated code or invent a scheduler framework for a small fix.

## Work and acceptance

Locate the owning module and show the smallest justified change. For performance,
identify an observed expensive interval, consult primary ISA/API documentation,
and compare the same workload before/candidate/after. Record build flags, model,
placement, width, routes, memory residency and output. A component gain must be
tested on the complete chat path before claiming chat speedup.

Run focused correctness checks first, then affected integration tests. Use
`make DS4= test-core` and `make DS4= test-plane-cluster` for core/transport changes;
follow [release qualification](../RELEASE.md) for installers or engine changes.
Do not call an unavailable GPU or remote machine tested.

Deliver changed files, a plain explanation of the reason and tradeoff, commands
and observed outcomes, remaining limitations and a reproducible next step.
Update the guide and relevant model brief together when their contract changes.
Use [REPORT.md](REPORT.md) if the result should become a cluster report or an
issue, and keep the user's review separate from publishing it.

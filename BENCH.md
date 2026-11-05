# Measuring Fronk

[MEASUREMENTS.md](docs/MEASUREMENTS.md) contains the results that explain the
architecture: how neuron width changes CPU cost, why compact slices help, how
a heterogeneous cluster runs and why the head's return link matters. This file
explains how to compare a change. Timings are evidence, not test thresholds.

## Choose the question before the benchmark

| Question | Measure | Keep separate |
|---|---|---|
| Is a backend faster? | Complete `run` time at fixed width, units and expert routing | Inner dot product time |
| Does transport delay the layer? | First send to last required reply, with node compute and arrival times | Ping or spinning echo latency |
| Does an MTP change help? | Wall time per useful accepted token, including proposal, verification and recovery | Proposed rows and accepted tokens |
| Does SSD placement help? | Complete generation with fixed capacity, cold/warm cache and RAM/SSD ownership | Cache-hit-only kernel time |
| Is setup quicker? | Fresh installation and reconnect separately | Cached calibration and a new measurement |

The head can wait for several nodes at once. Do not add their compute times;
find the participant or dense operation that delays the next step. A cheap
operation repeated 48 times can matter more than a conspicuous one-time setup.
[PROTOCOL.md](docs/PROTOCOL.md#reading-a-slow-call-report) explains the reports.

## A comparison another person can repeat

Record the source revision, compiler/flags, hardware, OS, backend, physical
links and unit placement. Include the model digest, prompt, context, sampling,
MTP settings and API completion-token count. Keep the ordinary head binary and
its engine directory intact while building an experimental one.

Run the unchanged control, the candidate, then the unchanged control again.
Use several observations, keep both controls if they drift, and keep builds,
calibration and model hashing outside the timed runs. Check outputs and numerical
state before timing. Use separate runs for expensive verification and profiling.

Report time to first token and decode throughput separately. State the exact
timer boundaries and denominator. SSE messages are not inherently tokens; use
the API's completion count. Report a changed placement rather than attributing
its benefit to a kernel. List individual samples alongside summaries.

For backend checks, use [BACKENDS.md](BACKENDS.md) and its frozen vectors.
For engine variants and diagnostic switches, use [RUN.md](RUN.md). The
[performance brief](docs/llm/PERFORMANCE.md) supplies the same procedure to an
LLM. A numerical pass or a component speedup alone does not qualify a new default.

## Measured RAM plus SSD placement (2026-11-04)

The comparison in [MEASUREMENTS.md](docs/MEASUREMENTS.md)
used ordinary MTP, burst-three planning and the same short prompt. The head's
256 MiB expert cache accompanied two resident units and one streamed unit.
The planner used a measured cold combined cost of 1.952 ms per layer, not the
warm-cache cost of 0.324 ms. Moving one iMac unit to this head share reduced the
load on that participant.

The twelve responses had identical text and 62 streamed fragments. The historical
timer excluded the first fragment and counted fragments, without a separate
tokenizer count. The recorded 15–16 range therefore remains a fragment-based
observation. The RAM controls drifted; the table retains both. Reconnection
reached READY in about 14 seconds using saved calibration. These results support
measured placement; they do not establish a universal throughput or a gain from
deeper speculative trees.

## Measured transport batching (2026-10-29)

The [transport comparison](docs/PROTOCOL.md#what-reduces-transport-overhead-today)
used width two, an M1 Pro over Thunderbolt and a Shuttle over Gigabit Ethernet
through a USB adapter. The head streamed 13–14 units. Batching logical fragments
and retaining connected UDP sockets reduced send/receive overhead. Those costs
include application scheduling under that workload; they are not cable latency.
Do not add that gain to the later RAM + SSD comparison as if they were one run.

## Share the result

Use [SHARE.md](docs/SHARE.md) for a cluster report. For a proposed optimization,
include the hypothesis, smallest patch, numerical checks and complete-workload
comparison. A clear negative result can prevent someone repeating the same
experiment. Keep an experiment out of the ordinary path until its evidence
supports changing the default.

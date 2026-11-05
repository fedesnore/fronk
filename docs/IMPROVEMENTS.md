# Where to help

Fronk is useful when several ordinary computers together have enough memory
for a model that none of them can comfortably run alone. The next gains should
make those computers work together more efficiently. A good contribution can
be a ten-line fix, a backend, or a carefully measured result that changes our
understanding of a bottleneck.

| What you have | What you can do |
|---|---|
| Your computers and an evening | Run the [synthetic demo](INSTALL.md#3-try-the-synthetic-cluster-demo) (no model download) and [share what happened](SHARE.md) |
| A setup that stopped | Open an **Installation help** issue; a clear report already helps |
| One computer and a C compiler | Improve the installer flow, protocol tests or [a backend](../BACKENDS.md), without the model |
| A CPU or GPU not supported yet | [Add a backend](../BACKENDS.md), with a frontier model if you like |
| A complete cluster | Test one of the directions below and publish the numbers |

Start with a machine you have and a question you can answer. You do not need
the example cluster or the full model to work on the protocol tests, installer
flow or backend fixtures. [CONTRIBUTING.md](../CONTRIBUTING.md) explains the
build. [The development brief](llm/DEVELOPMENT.md) helps a frontier model work with you.

## Make the first installation easier

Try the manager on a fresh user account. Note the first instruction that leaves
you unsure which machine to use, what to select, or whether a step finished.
Improve that instruction and verify the same path in terminal and graphical
mode. A clear diagnosis of an unavailable GPU, missing compiler or blocked
network connection can save the next person a long debugging session.

Start in [fronk-install.sh](../tools/install/fronk-install.sh),
[onboard.sh](../tools/install/onboard.sh) and
[manager.py](../tools/install/manager.py). The shell owns the choices; Tk
displays them. Keep one workflow. Use temporary prefixes and the installer
tests; preserve verified models and calibration when resuming.

## Make several token rows cheaper than separate calls

MTP pays when the time saved by accepted tokens exceeds proposal, verification
and recovery costs. A deeper tree alone does not solve this. The ordinary
head remains the reference; experimental engines are separate build targets.

The dense verifier can reuse matrix weights across token rows. On nodes,
`(layer, expert, neuron slice)` identifies the weights that compatible rows can
share. Their activations, gates and output destinations remain distinct. These
are two different kernels with one end-to-end objective: less time per accepted
token. The [native MTP design](PLAN-mtp-native.md) records the current paths,
measurements and hardware references; [RUN.md](../RUN.md) lists the switches.

A useful first experiment is one projection at widths 1, 2 and 4, including
dispatch and output collection. Inspect memory traffic and generated code
before choosing a tile. Compare full calls, then the complete verifier. Do not
assume that a GPU kernel with fewer instructions wins once copies and barriers
are included.

Start in [fronk_dense.metal](../tools/mtp/fronk_dense.metal),
[fronk_mtp_native.inc](../tools/mtp/fronk_mtp_native.inc) and
[serve.c](../src/serve.c). For CPU nodes, start with
[avx2.c](../backends/avx2.c) and [avx2_mtp.inc](../backends/avx2_mtp.inc).
Use the dense and backend numerical checks, including rejected prefixes and
committed recurrent/KV state. A faster verifier that changes the answer is not
an acceptable replacement.

## Spend idle time on work that can survive acceptance

While a remote slice runs, the head may have an independent predicted branch
to advance. While the head computes dense work, a node may have another ready
activation. The exact next layer still depends on the previous complete sum.
Predicted work needs its own state and may be discarded.

Begin by recording which ready job could have run during an observed wait.
Then try a bounded queue of already-ready work, with no waiting to fill a batch.
Account for interference with the real token: a speculative job can occupy the
same GPU, memory bandwidth or CPU workers needed by the critical path.

Start in [fronk_mtp_worker.inc](../tools/mtp/fronk_mtp_worker.inc),
[fronk_mtp_tree.inc](../tools/mtp/fronk_mtp_tree.inc) and the experimental
[wave scheduler](../tools/mtp/fronk_mtp_wave.inc). Exercise cancellation,
partial acceptance and a delayed node. Report useful accepted tokens, wasted
work and wall time together. Keep depth bounded and ordinary chat available.

## Receive many nodes through one switch

With a switch, requests can share a multicast send, but every node must return
its own partial output. Similar-sized jobs may finish together. Their replies
then share the switch's link to the head. A faster head uplink remains important;
software cannot make a 1 Gb/s link carry 10 Gb/s of results.

There is a concrete question to investigate: how much of the last reply's delay
comes from the switch queue, the head's receive queue or scheduling in the head?
Measure one physical switch with several nodes, then compare separate links.
Record uplink/downlink rates, width, reply arrival spread, loss and the time to
the last required result. Loopback tests can establish correctness, but cannot
reproduce a physical switch queue.

Start in [plane.c](../src/plane.c), [fanout.c](../src/fanout.c),
[wire.c](../src/wire.c) and [test_plane_cluster.c](../tests/test_plane_cluster.c).
Possible experiments include draining more ready replies per wakeup, receive
queues per interface, and pacing replies whose measured slack permits it.
These are hypotheses to compare, not a request to add all three. Pacing can
also make the last reply later; extra receive threads can cost more than they
save. Preserve session/step/layer checks, duplicate rejection and complete sums.

The planner could also learn the shared uplink's cost rather than treating each
node's connection as independent. Reuse saved compute profiles and add a short
topology check only where it changes the decision. Read
[PROTOCOL.md](PROTOCOL.md) and [planner.c](../src/planner.c) before changing the
cost model. Reconnecting must remain quick.

## Read an SSD slice once for useful work

RAM + SSD placement already lets Metal machines take a measured streaming
share. The next question is whether ready token rows can reuse the same fetched
expert bytes before the cache evicts them. Prefetching more bytes is useful only
if it shortens a wait without displacing weights needed sooner.

Start in [ssd.c](../src/ssd.c), [units.c](../src/units.c),
[hybrid.c](../src/hybrid.c) and the Metal backend. Measure bytes read, cache hits,
staging copies and full compute time in both cold and warm runs. Keep a staged
buffer alive until its last CPU/GPU reader finishes. Include memory consumed by
queues and staging in the planner's budget; do not count it twice as free RAM.

## Bring another kind of computer

[BACKENDS.md](../BACKENDS.md) and the
[scalar template](../backends/user-template.c) define a small starting point:
load a set of slices, compute their partial result, unload it. You can build and
test that contract without rewriting discovery or the installer. The
[backend brief](llm/BACKENDS.md) gives a frontier model the same constraints.

Start with one correct width and compare it to the oracle. Add SIMD/GPU work
only after reading the hardware documentation and measuring that first path.
Publish the hardware, build flags and complete call timings, including failures
or unsupported cases. Such a result helps the next contributor choose well.

## Close the loop

Describe the question, the change and the result in one issue or pull request.
Keep the simplest implementation that improves the complete workload. A rejected
experiment is useful when its method and cause are clear; it need not become a
permanent runtime option. [SHARE.md](SHARE.md) explains how to report a cluster
or solution, and [PERFORMANCE.md](llm/PERFORMANCE.md) gives the comparison rules.

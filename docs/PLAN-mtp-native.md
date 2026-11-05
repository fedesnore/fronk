# Fronk native MTP head

The project direction is to retain Dwarfstar's C foundation while developing
a Fronk execution engine for deep, distributed MTP. The head's dense runtime
is the immediate focus. Increasing candidate counts in the old scheduler is
insufficient. Success is lower wall time per accepted token on the physical
cluster, with correct committed state and no regression on the working SSD
configuration.

## Hardware contract

Use hardware-specific kernels, explicit layouts and persistent workspaces.
Specialize Metal threadgroups and token tiles; use CPU SIMD intrinsics where
the CPU path needs them. Inspect compiler output where available and measure
actual execution. Prepare pipelines before timed generation. No compilation,
model conversion or calibration belongs in a decode cycle. Avoid per-token
allocation, buffer migration, host readback and global synchronization when
a dependency can stay on the device. This is an engineering constraint, not
a claim that memory access or model dependencies can be made instantaneous.

The first Fronk Metal kernel, `tools/mtp/fronk_dense.metal`, reuses each F16
hyperconnection mixer weight across two, four or eight token rows. Threadgroup
memory holds low-rank inputs and sigmoid factors separately, preserving the
single-token multiply/reduction order. This matters: the inherited pair
kernel precomputes SiLU and contracts multiplication differently. Small
head differences can be amplified by quantized node inputs and routing.

## Slice-aware execution

The useful work key is `(layer, expert, neuron slice, token rows)`. A token
row carries its position, candidate identity, activation, routing weight and
output destination. Weight residency may be shared; mutable activation and
recurrent state may not be shared between incompatible branches.

Neuron slices partition the experts, whose partial output vectors add at the
head. They permit small-grain placement, resident/streamed weight assignment
and reuse of a slice across candidate tokens. Dense head projections need
their own matrix tiles and token batching; expert slicing does not by itself
accelerate those projections. The two backends should consume the same
causal batch identity without discarding row ownership or reduction order.

Batch already-ready compatible activations. Do not wait to fill a batch or
reorder the monotonic completion timeline. Reuse across different expert
ranks needs explicit per-rank outputs and an ordered reduction, with its
scratch traffic measured before adoption. The existing ranked backend groups
same-rank reuse without that extra output materialization.

Start local work and network fan-out early; overlap independent dense work,
local expert slices, SSD reads and remote computation where resources permit.
The next exact layer still requires the current layer's complete reduction.
Any computation that crosses that dependency is speculative and needs separate
state. Sequential predicted tokens within one causal batch do not create
independent branches or eliminate that dependency.

## Native verifier and proposer

The experimental `mtp-native-head` build has one causal verifier graph for
3–8 positions. It shares the prefix and sends a distributed burst per layer.
Snapshots after the first two rows permit early-rejection rollback by swaps.
For later partial rejection, a persistent activation tape records raw and
normalized GDN inputs, recurrence gates and PLE inputs. Only the accepted
suffix's recurrence and history are replayed; the full dense model and remote
experts are not executed again. Recovery time and tape copies remain part of
the cost. Real-weight checks at widths 4, 6 and 8 passed against independent
single-token logits and committed GDN/PLE/KV state. This does not establish
correctness for arbitrary contexts, branches or sampling modes.

The proposer currently reuses the model's existing MTP predictor and recurses
for deeper drafts. This is not a newly trained multi-depth predictor. Proposal
quality, deeper predictor batching and avoiding redundant vocabulary heads
remain separate optimization targets. Do not claim the new proposer is
hardware-optimal or that acceptance will rise merely because depth increases.

The new path is opt-in and greedy-only. Normal chat and installers retain the
verified SSD engine. Diagnostic checks compare all committed logits and
recurrent/cache state against independent single-token execution; they add
substantial work and must be excluded from performance results.

## Choosing depth and placement

Optimize `(proposal + verification + commit/replay time) / accepted tokens`.
Record accepted-prefix distribution, not just acceptance of the first draft.
Reuse the install-time calibration cache, but profile the relevant burst
widths: a 32-neuron slice balanced at T=1 or T=3 may be unbalanced at T=8.
Account for head dense work, local RAM/SSD contention, network fan-out and
remote completion. Overlap savings must be measured, not subtracted twice.

Kernel benchmarks establish local improvements only. The release gate is
correctness on real weights followed by interleaved before/after controls on
the same model, placement, prompts and sampling policy. Report both useful
decode throughput and first-token latency. The target of 30 tok/s remains
unproven until a complete physical-cluster comparison establishes it.

## Evidence gate — 2026-11-04

Before a new execution strategy is implemented, record its primary source,
the applicable hardware and model assumptions, the measured cost it addresses,
and the cheapest experiment that could reject it. A paper is evidence for a
mechanism, not proof that its implementation or speedup transfers to Fronk.
Do not promote a prototype on kernel speed alone. Keep the verified SSD chat
as the reference until an interleaved cluster comparison passes.

The local Dwarfstar base reports commit
`909617d40572d1d14f712d5b57f706ed6fccfc92`; staged sources also contain local
Fronk patches. Online `main` is not a reproducible baseline. Relevant local
implementations are `qwen4_gemv_rows` in `ds4.c`, the Q8 few-row kernels in
`metal/dense.metal`, and `kernel_qwen4_batch_mm_q8` in `metal/qwen4.metal`.
The existing few-row path already shares dequantized weights across rows;
the eight-row SIMD matrix path is a better starting candidate than our Q8
replacement. See the [upstream repository](https://github.com/antirez/ds4).
The numerical/sampling contract was read from a local copy of the engine file
`docs/SPECULATIVE_DECODING.md`; availability of this
local revision at an upstream commit URL has not been verified.

### Sources and transfer limits

| Area | Primary evidence | Fronk decision and limit |
|---|---|---|
| Metal memory and specialization | [Apple: shader best practices](https://developer.apple.com/videos/play/tech-talks/111373/), particularly address spaces and function constants | Specialize fixed token tiles before decode. Test direct/cache-backed loads against explicit threadgroup staging. Apple explains why an additional software cache can lose on family 9; this is not a measured diagnosis of our kernel. |
| Resource pressure and launch geometry | [Apple: Metal Compute on MacBook Pro](https://developer.apple.com/videos/play/tech-talks/10580/) | Inspect threadgroup memory, register use and compiler scheduling, then measure. M1/M4 Apple GPU strategies need independent qualification on Intel/AMD Metal. Our current tools lack `metal` and `xctrace`; timings are available, but we have not measured register spills, occupancy or final GPU ISA. |
| Depth and width selection | [Sequoia, §3.3](https://arxiv.org/html/2402.12374v2#S3.SS3) | Optimize expected accepted tokens divided by measured drafting and verification cost. Extend that cost to Fronk's network, recurrent commit and SSD placement. A single-GPU tree optimum does not determine this cluster's optimum. |
| Deep proposal quality | [EAGLE-3, §3](https://arxiv.org/html/2503.01840v3) | Training on the predictor's own multi-step outputs and using target features are supported approaches to improving deeper proposals. Recursive use of today's predictor is not equivalent. This needs a matched trained drafter and evaluation; changing target dense weights is not a runtime optimization. |
| Recurrent acceptance | [Snakes and Ladders](https://proceedings.mlr.press/v262/wu24a.html) | Activation replay provides a published basis for recording inputs and restoring accepted recurrent state without repeating all weight projections. Its SSM experiments do not validate Qwen's GDN or PLE equations; our independent state checks are still required. |
| Distributed token trees | [SpecInfer](https://arxiv.org/abs/2305.09781) | Share prefix computation and verify candidates with explicit causal identity. Fronk additionally needs branch-specific recurrent state; an attention mask alone is insufficient. Published distributed gains are not forecasts for our Ethernet/Thunderbolt cluster. |
| Pipeline overlap | [FlowSpec](https://arxiv.org/html/2507.02620v1), including Appendix C | Its continuous tree pruning/expansion informs an overlap experiment, but its layer pipeline is different from Fronk's within-layer neuron split. Do not port the scheduler or assume all waits can be hidden. |
| Expert reuse cost | [Expert Bytes, Kernel Bandwidth and the Best Draft Length](https://arxiv.org/html/2610.34440v1) | This recent preprint explicitly compares deduplication on Metal, MLX and CUDA. Fewer requested bytes can lose when effective kernel bandwidth falls. Use the cost accounting as a hypothesis, supported here by our negative Q8 result; its predictions have documented Metal misses and are not a universal law. |
| Cost-aware branch selection | [EcoSpec](https://arxiv.org/html/2607.12696v1) | Consider marginal expert/slice traffic as well as acceptance. Its research prototype uses an additional trained expert predictor; do not add that overhead here without measured benefit. Actual future target routing is unknown until its activations exist. |
| Limits of expert budgeting | [MoE-Spec](https://arxiv.org/html/2602.16052v1) | Useful evidence that deeper verification expands the expert union. Its expert dropping/substitution changes target computation, so it is excluded from Fronk's exact verifier. We can limit speculative candidates, not discard required target experts. |

### What our measurements actually establish

An uninstrumented run of `tests/test_mtp_dense.c`, with before/candidate/after
controls on M4 Pro, produced the following. Values are microseconds per
dispatch, averaged over repeated dispatches in one command submission; they
are **not** complete-layer or chat measurements. The matrices and activations
are synthetic and cache-warm.

| Operation | Existing before | Candidate | Existing after | Decision |
|---|---:|---:|---:|---|
| HC mixer, E=2560, rank=320, T=4 | 103.98 | 46.44 | 100.97 | Keep native candidate; exact T=1 arithmetic checks pass |
| HC mixer, T=8 | 197.20 | 97.93 | 197.48 | Keep native candidate; weight tile shared across eight rows |
| Q8, K=2560, M=4096, T=8, custom shared tile | 172.72 | 210.33 | 166.31 | Reject as default; `FRONK_NATIVE_Q8_REUSE` remains test-only |
| Same Q8, existing SIMD matrix path | 169.69 | 65.16 | 162.41 | Prioritize numerical/full-graph qualification |
| Q8, M=10240, T=8, custom shared tile | 410.00 | 521.64 | 402.59 | Reject as default |
| Same Q8, existing SIMD matrix path | 402.97 | 203.89 | 410.86 | Prioritize numerical/full-graph qualification |

The SIMD matrix outputs have relative L2 differences of `4.15e-7` and
`6.49e-7` from scalar T=1 on these inputs. They are not bit-identical.
The current exact native fallback uses scalar-order reductions and prevents
the generic matrix selector, which costs parallelism. Simply relaxing a
checker would hide that change. A candidate matrix policy must be compared
against the independent target, including routing, greedy continuations,
acceptance and recurrent state across batch boundaries. If a consistent new
arithmetic policy is selected, document its effect on the target separately
from the correctness of rollback under that policy.

HC T=2 had a large difference between its two controls, and its existing pair
kernel was faster after warmup; no T=2 speedup is claimed. Encoder-timeline
results are diagnostic; they are not mixed with uninstrumented timings.
See `build/revision-qa/mtp-dense-uninstrumented.log` and
`build/revision-qa/mtp-dense-timeline.tsv`.

The saved real-cluster trial has one short greedy prose prompt and three warm
repetitions, at head 2 RAM + 1 SSD, iMac 10 RAM, M1 Pro 3 RAM, Shuttle 4 RAM.
Dropping the first request and T=1 startup cycles gives:

| Verifier rows | Mean tokens committed/cycle | Verify ms | Commit ms | Draft ms | Cycle ms | Cycle budget for 30 tokens/s at this acceptance |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 3.10 | 207.79 | 0.22 | 15.21 | 223.23 | 103.33 ms |
| 6 | 3.59 | 313.65 | 1.05 | 23.37 | 338.07 | 119.61 ms |
| 8 | 3.71 | 381.63 | 1.05 | 31.98 | 414.67 | 123.53 ms |

These are engine cycle counters, not HTTP throughput. The historical harness
counted text fragments; its warm median rates were 13.75 / 11.03 / 8.70
fragments/s. The surrounding earlier SSD controls had medians 15.16 and
16.64 fragments/s, but they were not interleaved with these native runs.
There is no demonstrated end-to-end gain from deep native MTP yet.

Recovery events now cost about 1.4–1.5 ms. Optimization effort should move
away from recovery and toward verification. At width 8, even eliminating all
draft time would leave about 383 ms/cycle at the observed acceptance; it
cannot reach the 123.5 ms budget. Even accepting all eight rows at the same
414.7 ms cycle cost gives only 19.3 tokens/s. These are conditional bounds,
not performance forecasts: routing and cost change with different proposals.

Reproduce the analysis without contacting the cluster:

```sh
python3 tools/mtp/summarize-native.py > build/revision-qa/native-evidence-summary.json
```

The report includes source-log SHA-256 hashes. It does not retroactively prove
binary identity for historical runs. `FRONK_PROFILE`'s `gpu_until_ready` is a
CPU wait interval, not a GPU hardware timer; `cluster` includes host tensor
reads and `fronk_moe`, not just wire time. Periodic windows can include prefill
and mixed widths. These fields cannot isolate the current bottleneck between
head, individual nodes, I/O and network.

### Full-path design to qualify

The unit of verification is a causally identified batch. The unit of expert
work is a slice tile with multiple activation rows. These are different
layouts connected by explicit row maps; neither may erase token ownership.

1. **Propose.** Produce token IDs, parent/position IDs and proposal metadata
   into persistent buffers. Measure recursive predictor steps and vocabulary
   projections separately. A trained deeper drafter is a separate decision
   after verification becomes cheap enough to use its outputs.
2. **Dense verify.** Batch independent projections across ready rows; retain
   causal recurrence and attention. Prefer existing vectorized/matrix kernels
   when qualified. Keep projections, route computation and state on GPU until
   the network dependency actually needs CPU-visible activations.
3. **Route by slice.** Build bounded maps keyed by layer, expert, slice and
   compatible rows. Gate/up produces a token-specific intermediate;
   down projection produces a token-specific partial vector. Share weight
   reads/dequantization, never activations. Preserve per-token rank reduction
   order or explicitly qualify a new arithmetic policy.
4. **Dispatch and overlap.** Queue remote sends and local RAM/SSD work as
   soon as dependencies are ready. A node cannot calculate the next exact
   layer from its own partial output: it needs the head's complete reduction.
   Independent branches can overlap only with separate state and events.
   On head, dense and local Metal experts contend for the same GPU/memory;
   shared queues do not create additional bandwidth.
5. **Stream weights.** Key bounded cache entries by layer/expert/slice and
   pin each tile until every consumer completes. Reuse one SSD read for
   compatible rows. Speculative prefetch requires a pollution/traffic budget;
   the predicted route must never replace the actual target route.
6. **Reduce and commit.** Match request epoch, batch, row and slice ownership;
   require all contributions before publication. Commit only an accepted
   causal prefix and restore recurrence using the activation tape. Sibling
   branches need their own snapshots or branch tapes, not the chain's state.
7. **Choose width and placement.** Use cached measured costs by backend,
   32-neuron slice, T, reuse pattern and cold/warm SSD state, including head
   contention. The installed hybrid tables currently cover bursts 1–4;
   width-8 placement is not calibrated. Do not extrapolate from T=3 or
   remeasure every reconnect. Version new install-time tables explicitly.

For one layer, a first lower-bound model is dense-before-route plus the
maximum of local completion and each remote round trip plus ordered reduction.
It is only a lower bound until resource contention, send serialization and
independent work are represented by a measured dependency graph. Adding all
component durations overcounts overlap; subtracting all waits invents it.

### Ordered experiments and stop conditions

| Order | Smallest useful experiment | Required evidence to proceed |
|---|---|---|
| 1 | Separate diagnostic trace of GPU encoders, per-layer expert dispatch/return, local RAM/SSD spans and request row IDs | Join critical-path intervals without adding unsynchronized host clocks. Measure with tracing off for performance. No new per-token production logging. |
| 2 | Qualify existing SIMD matrix Q8 and F16 paths on real projection shapes at T=1/3/4/6/8, including any padding, copies and reduction | Repeatable operation gain and an explicit numerical policy; independent target/state validation. Do not write another Q8 kernel first. |
| 3 | Replay recorded expert routing and synthetic activation shapes through per-row vs ranked-slice backends | Saved bytes, grouping/packing time, output correctness, cold/warm I/O and completion latency on each actual backend. Drop grouping when its total cost loses. |
| 4 | Build a width-aware cost table from those measurements and evaluate depth choices offline | Prediction tracks held-out measured cycle cost and acceptance; no assumption that deeper is always better. |
| 5 | Enable one qualified path at a time in isolated native builds | Alternate SSD baseline/candidate/baseline, same complete placement and weights; prose/code/reasoning and longer contexts; engine token counts, first-token latency, accepted-prefix distribution and tail latency. |
| 6 | Evaluate a trained deeper proposer or a branch-overlap prototype | Verification has enough measured headroom, and added compute/traffic/scratch fits the saved critical-path time. Keep exact target computation. |

An optimization is rejected if it changes required target experts, breaks row
or state ownership, wins only with a different placement/acceptance workload,
or improves one kernel while regressing complete-request latency. Never claim
an ISA-level improvement from source code alone. Hardware counters or compiler
evidence must support that narrower claim; wall-clock measurements establish
the practical result.


### Whole-path evidence after width-separated profiling

The 2026-11-04 diagnostic run now separates actual T=1 and T=4 calls on the
same full hybrid placement. See `BENCH.md` for weighted counts and exact
stage tables. The plane spends 96–97% of its elapsed time waiting/draining
after fanout; preparation plus final reduction/copy is about 1%. GPU-visible
input/output copies total 1.1 us (T=1) / 1.7 us (T=4), far below millisecond
expert completion. The active engine uses the in-process hook, so removing
an imagined inter-process socket copy would optimize a path that is not used.

Head-observed last completion switches from Shuttle (89% of T=1 calls) to
iMac (64% of T=4 calls, Shuttle 36%). Local RAM+SSD compute is already
submitted before fanout and averages 0.62/1.21 ms while the full expert plane
costs 1.09/2.83 ms. This supports investigating width-aware expert/reuse and
network completion first, and rules out unconditional movement of work to
an apparently idle GPU without its measured RAM/SSD and contention costs.
M1's extra reported capacity is SSD-backed, not free resident memory.

The head worker's ready-event wait averages 1.00/1.33 ms. It includes dense
GPU work and scheduling; a CPU interval alone cannot say how much is shader
execution, resume delay or GPU clock ramp. The existing polling-resume path
must remain disabled with local Metal expert work: the prior overlapping
queue experiment showed GPU starvation. Useful independent work would need
separate state and bounded resource pressure, not an unbounded spin kernel.

A real encoder trace identified a repeated 248320-row vocabulary projection
in the proposer. Reusing the existing fixed-eight SIMD matrix kernel there
was tested and rejected: about 4.47 ms versus 2.95 ms scalar in the isolated
complete-submission microprobe. The real-graph scalar observation was about
4.5 ms under different cache/clock conditions. Kernel choice is shape-specific;
the matrix gain on small dense projections does not transfer to this head.

The native catch-up loop also computes a vocabulary head for every accepted
chunk of up to three rows, although only the final chunk's draft is consumed.
`FRONK_NATIVE_SKIP_CATCHUP_HEADS=1` removes those discarded heads, preserving
all recurrent/cache updates and the completion boundary before shared CPU
embedding-buffer reuse. This is independent of target verification width or
arithmetic. Its opt-in scope and measured results are recorded in `BENCH.md`.

### First measured AVX2 benefit from cross-rank slice reuse

The next implemented candidate removes the same-rank restriction when grouping
token activations for a native expert slice. The saved real T=8 capture has
65.224 same-rank groups but only 38.151 unique experts per request. A grouped
slice can reuse its Q4_K unpacking across more rows, then write independent
down-projection results into bounded slots. An ordered second phase reproduces
the original worker FMA chains, so all 192 replayed outputs remain bit-identical.
There is no speculative state commit in this backend: branch acceptance and
cancellation remain owned by the head.

Combined with measured multirow AVX2 kernels, complete resident backend latency
falls about 5% on iMac and 13% on Shuttle. Single kernel wins alone did not
transfer to real routing; union exposed the missing reuse. This is a concrete
benefit of scheduling at slice granularity, not a claim that grouped expert
computation is unique to Fronk. It does not yet overlap different layers or
compute T+2 before its prerequisite activation exists.

The exact reduction has a measurable cost (about 0.337 ms iMac, 0.104 ms
Shuttle in a separate diagnostic pass). Raw compute still dominates, and the
iMac remains slower in this replay. Next qualification must cover held-out
routes, shorter widths, and a fixed full-cluster RAM/SSD placement before
any default change. A width/reuse-aware planner should use measured complete
cost, including reduction, rather than extrapolating from group-count savings.

The prior depth-eight cycle was about 415 ms with about 3.7 accepted tokens.
At that acceptance, 30 tokens/s requires about 123 ms per complete cycle;
even eight accepted tokens at 415 ms would yield only about 19.3 tokens/s.
These node improvements alone do not close that gap. The next end-to-end
decision must combine measured verifier cost, dense work and proposal quality,
without treating idle intervals as free independent speculative computation.

### Head direction: combine batched verification with staggered execution

Source inspection identifies two separate existing mechanisms, not a missing
generic "MTP" switch. `fronk_mtp_native.inc` uses one causal multirow verifier
and amortized projections, but waits for its distributed result before the
next dependent layer. `fronk_mtp_tree.inc` can defer expert reduction while
encoding other lanes, but its lane stages are T=1 and it copies per-lane state.
Neither is a qualified deep multirow overlap path. The Metal adapter also
commits a command buffer and encodes a result-event wait on resume; independent
work placed behind that wait cannot fill the preceding idle interval. The
shared expert already executes after the ready signal while routed experts
are outstanding; moving it there again is not a new optimization.

The candidate architectural experiment is two small causal row groups with
separate activation mailboxes. After group A's dense stage at layer L has
produced its recurrent/KV state and expert request, group B's dense stage at
L can use that state while A's routed experts execute. A may enter L+1 only
after A's expert reduction; B must respect both its own preceding-layer result
and A's same-layer causal-state dependency. Queue only ready stages, rather
than blocking ready GPU work behind an unresolved result event. Start with a
linear proposal chain, not a wider tree or additional predicted routing.

This must retain the measured multirow projection kernels and slice union
inside each group. It introduces a real tradeoff: smaller groups reread some
weights, reduce cross-token expert reuse and add dispatches. M4 local Metal
RAM/SSD experts also compete for that GPU, so dense/expert intervals are not
independent resources. There is no promise of free T+2 work. Rejection must
discard suffix results by explicit prefix/state generation and drain in-flight
mailboxes before reuse; only the verified contiguous prefix may commit.

The first rejection test is measured two-group complete-layer latency versus
one wide group, including dispatch, state handling, local Metal contention and
remote completion. Retain the ordinary path if saved wait is smaller than
lost batching benefit. Only after that passes should a full-graph candidate be
compared on identical accepted prefixes and placement. In the saved T=4 path
profile, pre-request worker wait averaged 1.330 ms and the expert plane 2.828 ms
per layer. The former includes GPU execution/scheduling and is not an isolated
dense-kernel timer. Even an ideal removal of that interval leaves substantial
distributed work; head kernels alone do not demonstrate a route to 30 tokens/s.

Primary support: [FlowSpec](https://arxiv.org/html/2507.02620v1) demonstrates
continuous speculative pipeline scheduling, but partitions whole layers rather
than Fronk's within-layer neuron slices. Its speedups do not transfer directly.
[Apple's shader guidance](https://developer.apple.com/videos/play/tech-talks/111373/)
explains specialization and the occupancy/register/memory tradeoffs behind
kernel selection; explicit shared-memory copies can lose against direct buffer
access. These sources motivate the bounded experiment, not its outcome.

#### Implemented result

`tools/mtp/fronk_mtp_wave.inc` now implements two contiguous causal groups,
using private transient/mailbox storage and the same layer state in ordered
A,B execution. This chain-specific ownership avoids duplicating recurrent
state; it cannot be generalized to incompatible branches by sharing those
pointers. Borrowed pointers are rebound after every rollback. The activation
tape preserves whole-chain row offsets and the final head stays batched.

Real-weight logit/state checks pass and a complete physical-cluster A/B/A
demonstrates a native T=4 median improvement from 11.0–11.2 to 13.6 tokens/s,
with identical prompts, texts, token counts, proposals, accepted prefixes,
placement and RAM/SSD masks. Steady worker waits for the next activation fall
from about 1.35–1.39 ms to 24 us; this is evidence of overlap, not a claim that
dense computation now takes 24 us. Depth eight gives a smaller 8–9% median gain
and remains slower overall. Full evidence and limitations are in `BENCH.md`.

The next 80/20 decision must use this changed critical path. At T=4 the
candidate still needs about 229 ms/cycle for 2.69 accepted tokens, whereas
30 tokens/s would require about 90 ms. More dense-kernel tuning alone cannot
be assumed to erase local RAM/SSD and remote expert completion. Profile those
costs with the new schedule, and evaluate proposal improvements on held-out
prompts. Do not increase depth, add branches, change weights or relax exact
verification merely to raise the nominal number of speculative tokens.

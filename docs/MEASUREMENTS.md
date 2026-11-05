# Measurements behind the design

These measurements explain three choices in Fronk: dividing experts into
neuron slices, storing those slices compactly, and balancing work across
different computers. The [architecture guide](../PROJECT.md) explains the
system; this page gives the workloads and numbers behind those choices.

## How much work should a small CPU receive?

A Shuttle with an Intel i5-10210U evaluated ten selected experts per layer,
using different neuron widths. All 512 experts of layers 12–15 were resident.
The workload used 24 captured activation/routing states; after warmup, each
width had 40 timed four-layer calls. The CPU used an eight-thread OpenMP team,
FP32 activations, prepared fused gate/up weights and an MXFP4 down kernel.
The measurement date was 2026-10-16.

| Neurons per expert | Share of each expert | Measured four-layer time, ms | Projected 48-layer expert time, ms | Equivalent expert passes/s |
|---|---:|---:|---:|---:|
| 32 | 5% | 0.501005 | 6.012 | 166.33 |
| 64 | 10% | 0.780655 | 9.368 | 106.75 |
| 96 | 15% | 1.094638 | 13.136 | 76.13 |
| 128 | 20% | 1.384935 | 16.619 | 60.17 |
| 160 | 25% | 1.701930 | 20.423 | 48.96 |
| 320 | 50% | 3.228792 | 38.746 | 25.81 |
| 576 | 90% | 5.681079 | 68.173 | 14.67 |
| 608 | 95% | 6.089018 | 73.068 | 13.69 |
| 640 | 100% | 6.318240 | 75.819 | 13.19 |

The 640-neuron row measures whole-width expert computation; it was not
extrapolated from a smaller slice. The timer covers ten active experts per
layer, not a single expert. Four resident layers allow this comparison even
when the full-width, 48-layer pool would exceed the Shuttle's memory.

The last two columns are projections: multiply the measured time by 12, then
divide 1,000 by the resulting milliseconds. They express the expert work in
token-equivalent passes. Dense computation, communication and joining results
are additional costs, so these columns are **not generated chat tokens/s**.

A 64-neuron share gives this CPU much less work per layer, leaving the remaining
576 neurons for other machines to compute concurrently. This is the reason for
small ownership units: the planner can give each computer an appropriate share
instead of requiring it to finish a whole expert.

## Why extract compact slices?

The same experiment compared two layouts with the same kernels: indexing the
needed neurons inside full matrices, and packing those neurons into compact
matrices. Runs alternated layouts in ABBA order to reduce ordering bias.

| Neurons per expert | Indexed from full matrices, ms | Compact, ms | Component speedup |
|---|---:|---:|---:|
| 32 | 1.051609 | 0.501005 | 2.10× |
| 64 | 1.688895 | 0.780655 | 2.16× |

These are four-layer times. At 64 neurons, the down-matrix row stride shrank
from 340 to 34 bytes. The compact layout keeps useful bytes together. No
cache-counter measurement was taken, so the result establishes the layout
benefit without attributing it to a particular cache mechanism. At full width,
the layout change gave no gain.

Outputs matched bit for bit between the two layouts. Independently computed
slices rejoined the full CPU output within relative L2 `2.24e-7`. A separate
64-neuron test loaded all 48 layers and all 512 routed experts per layer:

| Full 48-layer, 64-neuron workload | Result |
|---|---:|
| Resident compiled weight pool | 6.328 GiB |
| Timed passes after warmup | 304 |
| Mean expert-compute time | 9.658185 ms |
| p95 expert-compute time | 10.089042 ms |
| Equivalent expert passes/s from the mean | 103.54 |

The full-layer test had no measured physical reads or major page faults. It
confirms that the small share can remain resident and run across the complete
layer set on this machine.

The [raw records and method](evidence/SHUTTLE.md) include all widths and the
numerical/residency checks. This CPU experiment used prepared weights and FP32
activations. The released AVX2 backend uses a different activation path and
native unit format, so the installer measures its actual costs for planning.

## What does a complete mixed cluster do?

The 2026-11-04 RAM + SSD comparison used the supported Q4 model and ordinary
MTP, with a planning width of three. The cluster combined:

| Machine | CPU / GPU used | Installed memory | Link to head | Hybrid placement |
|---|---|---:|---|---|
| MacBook Pro M4 Pro | Apple GPU, Metal dense head and local expert worker | 24 GiB | Local | 2 RAM units + 1 SSD unit |
| Intel iMac | i5-7600K, 4 cores, AVX2 | 48 GiB | Thunderbolt | 10 RAM units |
| MacBook Pro M1 Pro | Apple GPU, Metal expert worker | 16 GiB | Thunderbolt | 3 RAM units |
| Shuttle | i5-10210U, 4 cores/8 threads, AVX2, Debian 13 | About 16 GB | Gigabit Ethernet through a USB adapter on the head | 4 RAM units |

All 20 units participated. The RAM-only control placed 11 units on the iMac
and two on the head. The hybrid plan moved one iMac unit to head SSD, using a
256 MiB expert cache. The planner used the measured cold combined cost of
1.952 ms per layer, rather than assuming the warm-cache cost of 0.324 ms.

| Run order | Three warm decode observations, fragments/s | Median |
|---|---|---:|
| RAM control before | 14.32, 14.44, 15.07 | 14.44 |
| RAM + SSD | 15.27, 15.72, 15.50 | 15.50 |
| RAM + SSD after reconnect | 14.47, 15.35, 16.07 | 15.35 |
| RAM control after | 13.40, 13.46, 13.75 | 13.46 |

All twelve responses had identical text and 62 streamed fragments. The timer
excluded time to the first fragment; the benchmark counted fragments, without
an independent tokenizer count. Both controls are shown because the baseline
drifted. This is one short-prompt placement comparison, not a prediction for
every prompt or cluster. The [measurement entry](../BENCH.md#measured-ram-plus-ssd-placement-2026-11-04)
records the comparison.

A separate [six-request native baseline](evidence/native-chat-baseline.csv)
also retained the API token counts. Its fastest short response generated
62 completion tokens at 15.79 tokens/s during warm decode, excluding time to
the first token. The six observations ranged from 10.25 to 15.79 tokens/s on
three prompts; their median was 13.37. This is the basis for the demo's
rounded **best recorded ≈16 tokens/s** caption, not an expected rate for every
request. The video identifies its current live capture separately. The source
record SHA-256 is
`08f8b7b6fcae7cfe59cba76d969d945111296e8eb3ffb9cbc11ff94d02a5f486`.

The result shows why a mixed cluster is useful: modest CPUs provide resident
expert capacity, Apple GPUs handle other shares and dense work, and a measured
SSD share can relieve a busy participant. A GPU is not required on every node.

## Why the return path matters

Each node returns 5,248 UDP payload bytes per activation row. Ten nodes in a
width-three call therefore return `10 × 3 × 5,248 = 157,440` payload bytes.
Serializing that amount onto a 1 Gb/s head link alone takes at least 1.26 ms,
or about 60.5 ms across 48 rounds. A 10 Gb/s head link cuts that aggregate
serialization bound by ten. Individual node links, protocol overhead and compute
still take time. This is a byte/rate calculation, not a ten-node benchmark.

Software overhead also matters. A width-two transport comparison reduced head
send time from 676 to 121 µs per request by batching logical fragments and using
connected sockets. Those runs used a different placement, with 13–14 head SSD
units. The [protocol measurements](PROTOCOL.md#what-reduces-transport-overhead-today)
give the complete intervals and conditions; the gain should not be added to the
RAM + SSD comparison as if both were the same experiment.

## From a slice time to a chat estimate

Expert passes/s are useful for sizing work, but the head and cables take time
too. A simple single-token estimate is
`1,000 / (head_ms + slowest_node_compute_ms + communication_ms)`.
This treats the slowest node as the limit and assumes the other nodes finish
their assigned shares within that time. It does not model overlap or MTP gains.

For the animated explanation, we use a **58 ms head budget** from an M4 Pro
single-token trace and an **assumed 20 ms communication budget** per 48-layer
pass. Combining these with the earlier Shuttle width experiment gives:

| Neurons per expert on each Shuttle | Projected expert time, ms | Single-token estimate, tokens/s |
|---|---:|---:|
| 32 | 6.012 | 11.9 |
| 64 | 9.368 | 11.4 |
| 128 | 16.619 | 10.6 |
| 640 | 75.819 | 6.5 |

These are examples of a timing budget, not measurements of four complete
clusters. The Shuttle data used prepared FP32 kernels; the head data came from
the native Metal runtime. This cross-experiment calculation explains the
trade-off, and must not replace the installer's measurements of the actual
backends and links. A whole-width 48-layer expert pool does not fit in one
16 GB Shuttle. Smaller shares need enough other nodes to cover all 640 neurons
and enough aggregate memory for the roughly 67 GB expert pool. MTP changes
both work per pass and accepted tokens, so its real chat rate must be measured.

The [head trace extract](evidence/m4-head-single-token.csv) contains all ten
480-layer reporting windows with `avg_T=1.00` from the retained single-token
test. For each window, its head-path budget is
`48 × (idle + gpu_until_ready) / 1,000 + boundaries × boundary_mean / 10,000`
milliseconds. The median is 57.755 ms, rounded to 58 above. This includes
observed inter-pass overhead; it is not a pure GPU-kernel measurement.
The original trace SHA-256 is
`abbe517d3c045a1477d1d21e0963d8fc68d683d80b4398f92c08aca3b8867e18`.

## Compare your cluster

Run the normal setup demo first, then try a real prompt. Keep the demo estimate,
time to first token and real generation rate separate. Record the chosen
backends, unit placement and physical links with the result. API completion
token counts are preferable to counting streaming fragments.

The console already shows per-node compute and arrival costs. Begin there if
chat is slow: node times overlap, and the last required result determines when
the head can continue. [Share your cluster](SHARE.md) gives a small report format;
the [performance brief](llm/PERFORMANCE.md) helps a frontier model investigate further.

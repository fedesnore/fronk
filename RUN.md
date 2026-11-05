# Development runs and experiments

For a downloaded release, use [INSTALL.md](docs/INSTALL.md). The commands here
are for contributors working in a checkout. Start with [PROJECT.md](PROJECT.md)
and keep experiments separate from the ordinary installed head.

## Build a head from this checkout

On Apple Silicon:

```sh
make head
```

This reconstructs the pinned engine and stages it under `build/head-engine`.
Copy `configs/head.conf` to `fronk.conf`, set `model` to the supported GGUF and
configure your interfaces. Generate a secret on the head with
`openssl rand -hex 32`; set `FRONK_SECRET` to that value in the terminal on
each computer. Use the same `burst` on head and nodes. Keep the secret out of
Git and shared logs. The examples use relative paths from the checkout root.
See [MODEL.md](docs/MODEL.md) for exact file identity and
[NETWORK.md](docs/NETWORK.md) for cabling and addresses.

```sh
./fronk head --config fronk.conf
# Or, from the same checkout:
python3 tools/fronk-web.py --open
```

`src/runtime.c` supplies the ordinary head defaults: complete expert sums,
event resume, and no GPU keepalive on the head. Nodes retain their own measured
policy. Explicit environment overrides are for diagnostics. An ordinary head
must fit all twenty units through resident or explicitly measured streaming
capacity. Review PLAN, proceed, and wait for the real generation test.

Build a node without the dense engine:

```sh
make DS4=
./fronk node --config configs/node.conf
```

Use separate checkouts/build directories for different engine variants. A
running head's binary and Metal source directory must remain intact. The
installer is the recommended way to manage persistent services, pairing,
models and uninstall.

## Backend of each machine

Each machine measures its compatible backends and keeps the fastest, unless you
choose one. On that machine, run:

```sh
./fronk backends --choose --config configs/node.conf
```

It lists the backends, marks the recommended one (Enter takes it), saves the
choice in its state-dir and measures it there. The short first-time sweep is
bounded by currently usable memory; widths 1–4 share each fixture. SSD
calibration retains resident fixtures between points and fills only missing
profiles. Every head after that reads those times instead of measuring. After
a compute change, measure again with the node stopped:

```sh
./fronk backends --measure --config configs/node.conf
```

`--set auto` goes back to racing the compatible backends. Writing your own
backend for a machine, and a prompt for an AI model to write it: BACKENDS.md.

## Expert blocks on the head

`head-blocks N` in `fronk.conf` (the **Blocks on this Mac** field) asks the
head to hold N expert blocks (3.35 GB each) on its GPU next to ds4; the nodes
get the rest. Free memory still caps it: at plan time the head keeps
`head-reserve-gb` (default 10) for ds4, whose dense weights page in after the
first request. When it caps the count, the raw output says so (`PLAN this Mac:
head-blocks 3 asked, memory allows 2 ...`). Leave this unset for automatic
placement. More resident blocks can slow generation if they force dense weights
or context into paging; use the measured plan rather than installed RAM alone.

## Notes

### Experimental MTP interleaving on an Apple Silicon head

`make mtp-head` builds `build/fronk-mtp` from a private copy of the ds4 checkout
selected by `DS4`. Start it with the same head config and nodes used by the
ordinary head, after stopping the current head.

```sh
FRONK_MTP_BRANCHES=3 FRONK_EVENT_RESUME=1 FRONK_METAL_KEEPALIVE=0 \
  ./build/fronk-mtp head --config fronk.conf
```

`FRONK_MTP_BRANCHES` accepts 2, 3 or 4 as the maximum number of sibling
candidates. Each sibling has one predicted continuation. An 80% predictor-mass
budget reduces the active width on confident tokens. The common token is
evaluated once. Each layer forks its attention and recurrent state before the
child's dense work is queued alongside the parent's remote expert work.
Accepted prefixes retain their state, including partial accepts.

The mode requires complete expert replies, supports greedy and exact stochastic
target sampling, and preserves the existing context-copy accelerator. It uses
private workspaces up to 2,048 context positions. Longer contexts, allocation
failure, vision and steering use the existing verifier. The default is the
existing MTP mode. The November 4 repeated measurements found similar speed
with two siblings and lower speed with three, so wider branching remains an
opt-in experiment.

`FRONK_PROFILE=1` reports dense waits and expert time. The console displays
branch acceptance and overlapping expert calls. `FRONK_MTP_VERIFY=1` additionally
replays each accepted token through the ordinary single-token forward function
and checks all logits. Use separate runs for validation and timing because
this check repeats the computation. `FRONK_MTP_COALESCE=1` enables the experimental
bounded reuse scheduler. It merges at most eight adjacent, already-ready,
single-row tree jobs with the same layer and identical ordered routed experts.
Each row retains its own activations, gate weights, output mailbox and completion
value. The shared expert stays on the head. No queue reordering or wait to fill a
batch is allowed. Partial expert overlap and permutations stay separate: the
older unrestricted prototype changed reduction order and failed numerical replay.
The scheduler defaults to zero pending repeatable performance gains. With
`FRONK_PROFILE=1`, `FRONK_SCHED` reports cumulative candidates, merged jobs,
route mismatches, and the scheduling time per worker call over the last reporting
window. These timings include route reads and queue locking, but exclude compute;
only an end-to-end comparison can establish a speedup.

The partial-reuse variant is built separately with `make mtp-ranked-head`.
Every participant must use a binary compiled with `-DFR_EXPERT_RANKED=1`;
for a node, build a separate executable with
`make DS4= FRONK_TARGET=build/fronk-ranked CFLAGS='-O2 -DFR_EXPERT_RANKED=1' build/fronk-ranked`.
Use `FRONK_MTP_COALESCE=2` with the ranked head and branching enabled.
The head checks authenticated `EXPERT-ORDER` responses at setup and refuses a
mixed cluster. This query never runs in the generation loop. New binaries have
new calibration keys; keep their measured curves separate from legacy builds.

Ranked kernels group expert work by routing position, preserving each token's
single-row sum order. AVX2 also retains that row's original worker assignment.
Weights are shared when two rows use the same expert at the same rank; reuse
across different ranks is deliberately left separate. No extra output matrix,
hot allocation or second GPU reduction pass is introduced. The scheduler only
admits a ready adjacent row when it shares at least two ranks with an admitted
row; it never waits to fill a batch. This is an experimental prerequisite for
deeper verification, not a claim that deeper MTP is already faster.

- `FRONK_PROFILE=1 ./fronk head` adds per-call head timings to ds4.log.
- `FRONK_DS4_MTP= ./fronk head` turns MTP off.
- ds4 problems: `runs/state-head/ds4.log`.

## Experimental native MTP head

`make mtp-native-head` stages a private Dwarfstar copy and builds
`build/fronk-mtp-native`. `FRONK_NATIVE_MTP=3..8` opts into the causal batch
prototype; it currently supports greedy generation. This is not the default
installer/chat engine and has not beaten the working SSD reference.
Use the matching ranked node binaries and setup capability check described
above (`FRONK_MTP_COALESCE=2`); keep `FRONK_MTP_BRANCHES=0` for native-chain
experiments. Point `FRONK_DS4_DIR` to the private `build/mtp-native-engine`.

`FRONK_NATIVE_VERIFY=1` performs independent single-token logit and committed
state checks. It repeats target computation and must be disabled for timing.
`FRONK_PROFILE=1` reports verification, recovery and proposal cycle times.
Partial rejection uses saved recurrent activations rather than replaying the
whole model. Arbitrary tree-state recovery is not provided by that chain tape.

`make test-mtp-dense` checks standalone Metal kernels and recovery without
model weights or network access. `./build/test-mtp-dense --timeline` adds
encoder timestamps; compare performance separately without that flag.
The test deliberately enables `FRONK_NATIVE_Q8_REUSE` to preserve a negative
experiment: this custom Q8 path was slower on M4 Pro and should remain unset
for normal generation. `./build/test-mtp-dense --vocab` separately measures
the full 248320-row vocabulary projection. Padding this single-row operation
to eight SIMD matrix columns was slower and is not enabled in the proposer.

`FRONK_NATIVE_MATRIX8=1` opts into the qualified fixed-eight Q8 matrix policy.
The independent single-token oracle uses the same arithmetic policy: this
does not claim bit identity with the older scalar summation order. Real
logit/state checks passed, but fixed depth four still regresses low-acceptance
prompts; it remains experimental.

`FRONK_NATIVE_SKIP_CATCHUP_HEADS=1` omits intermediate vocabulary projections
whose argmax is overwritten by the next predictor catch-up chunk. It retains
all predictor-state updates and the GPU completion boundary before the CPU
reuses the embedding buffer. This is an opt-in native experiment, independent
of the matrix policy. Profile logs include catch-up/chain time, skipped heads
and a draft-ID hash for matched comparisons.

`make mtp-path-profile-head` builds a separate `build/fronk-path-profile` and
`build/mtp-path-engine`. Set `FRONK_PROFILE=1` and point `FRONK_DS4_DIR` at that
engine. For a single-token control, unset `FRONK_NATIVE_MTP` and set
`FRONK_DS4_MTP=`. `FRONK_PATH`, `FRONK_PATH_NODE` and `FRONK_PATH_WORKER` lines
aggregate 480 calls, separated by actual burst width. Additional timers and
counters are compiled out of ordinary builds. Analyze saved logs with
`python3 tools/mtp/summarize-path.py LOG...`; node times overlap and must not
be added to the serial stage total. Diagnostic rates are not acceptance
criteria for a performance change: repeat with profiling compiled out.

The [native MTP plan](docs/PLAN-mtp-native.md) records primary sources,
measurements, hardware limits and the required experiment gates.
`python3 tools/mtp/summarize-native.py` summarizes saved local QA logs only;
it never starts or changes the cluster.

## Saved RAM and SSD calibration

On Apple Silicon, the planner can choose `metal` or `metal-ssd` independently
for the head and each node. The latter keeps a measured number of 32-neuron
units resident and streams up to two additional units. It uses the same native
Metal arithmetic and ordered reduction. Resident GPU work runs while eight I/O
workers fetch only the selected expert slices. An owned 256 MiB cache and one
native-layer staging buffer (about 70 MB) per streamed unit bound the extra RAM;
the streamed 3.35 GB unit is not registered as resident GPU memory.

Installation runs the existing `fronk backends --measure` path before joining
the cluster. Metal additionally records combined RAM+SSD timings at bursts
1–4, with both empty and populated expert caches. The synthetic I/O fixture is
physically written on the unit-cache filesystem, with filesystem caching
disabled before writing and reading. It needs about 70 MB of temporary disk
space and no GGUF. These are synthetic estimates; generation can still differ
because of routing, thermal state, memory pressure and other applications.

Profiles remain in `STATE/bench`. Their identity includes the compute build,
hardware, OS, filesystem, cache size, I/O policy, resident share and burst.
Discovery forwards compatible saved profiles to the head; it does not run the
SSD calibration again. Current available RAM still gates every placement.
Missing or incompatible SSD calibration leaves the machine in resident mode
and prints the setup command. To calibrate after a relevant change, stop that
machine's Fronk process and run:

```sh
./fronk backends --measure --config fronk.conf
```

`--refresh` explicitly replaces existing measurements. A repeated measure with
unchanged inputs reads the saved tables. The head-side installation demo also
saves its actual network overhead, keyed by peer identity, address and burst.
A new path uses the connection probe until a valid timed pass is available.

The planner minimizes the slowest complete reply: saved combined compute/I/O
cost plus measured network overhead for remote nodes, and the combined local
cost for the head. Local compute starts before fanout. Cold-cache timings govern
SSD admission; fully warm timings are displayed separately, not assumed as the
normal hit rate. No extra streamed share is extrapolated beyond its measured
points. PLAN and the console show RAM/SSD units, cache budget, burst, cold/warm
costs and the predicted reply time.

New built-in Metal head installations use burst 3, matching ordinary MTP with
two draft tokens. Nodes inherit it through the pairing code. Existing configs
keep their burst; use 3 for that verifier and 1 for the single-row branching
experiment when comparing plans. Batching deduplicates selected experts across
rows, and the expert cache can reuse weights across calls, but MTP acceptance
and a resulting chat speedup still require real generation measurements.

For a resident-only comparison without deleting calibration, start each
participating Metal process with `FRONK_SSD_STREAMING=0`. Unset it to let the
planner consider the hybrid mode. Head and Metal nodes need the updated runtime
to exchange hybrid profiles; older CPU nodes can continue sending their ordinary
resident profiles. Intel/AMD dense-head support remains the existing Vulkan
path; this streamed-expert mode currently targets Apple Silicon Metal.

### Experimental AVX2 multi-token slice reuse

The cross-rank union and multirow Q4_K kernels are opt-in and have not yet
demonstrated an end-to-end chat speedup. Their real-request replay results and
limits are in `BENCH.md`. Use a separate checkout/build directory on an AVX2
machine; leave the installed worker unchanged while qualifying the candidate.

```sh
make DS4= CFLAGS='-O2 -DFR_EXPERT_RANKED=1' build/avx2-mtp-probe
./build/avx2-mtp-probe
./build/avx2-mtp-probe --backend UNIT_DIRECTORY UNIT_IDS WORKERS
./build/avx2-mtp-probe --replay UNIT_DIRECTORY UNIT_IDS WORKERS CAPTURE_FILE 100
```

`UNIT_IDS` is a comma-separated list of native unit numbers. The probe maps
existing `unit-NN.bin` files read-only. Replay policy 99 selects union only;
100 combines union with the measured row policy (eight shared rows for clang,
three for GCC). Policies 3 and 8 select row kernels only. These compiler-based
crossover choices are qualified only on the tested iMac and Shuttle.

A diagnostic head built with `FRONK_PATH_PROFILE` can capture a bounded
fixture using `FRONK_EXPERT_CAPTURE=/absolute/path/requests.bin` and
`FRONK_EXPERT_CAPTURE_WIDTH=8`. It records at most 192 selected-width requests
after the first layer-zero single-token verifier call, writes rounded node
inputs, and flushes at the limit. Do not measure throughput with capture on.
Replay files use a checked local C ABI and endianness marker; they are not a
portable network format. Normal builds contain no capture path.

To include these candidates in a separate worker build, use compile flags
`-DFR_EXPERT_RANKED=1 -DFR_AVX2_UNION=1 -DFR_AVX2_MTP_ROWS=1`, then set
`FRONK_AVX2_UNION=1 FRONK_AVX2_MTP_ROWS=1` at process startup. Union is eligible
only for routed-only native batches wider than one. A union build reserves
15.625 MiB of scratch at setup; the ordinary build contains neither that
allocation nor the new execution path. Neither switch is enabled by default.

Do not reuse old calibration as if it measured the changed backend. Qualify
correctness and complete-request latency first, then measure a versioned cost
table before allowing a new build to change normal placement. End-to-end A/B/A
must hold unit placement, memory/SSD policy, prompts and acceptance work fixed.

### Experimental head overlap for native MTP

```sh
make mtp-wave-head
```

This creates `build/fronk-mtp-wave` and a private `build/mtp-wave-engine`.
It preserves the installed executable. For a separately configured experimental
head, select that engine with `FRONK_DS4_DIR` pointing to its absolute path and
set `FRONK_NATIVE_MTP=4 FRONK_NATIVE_MATRIX8=1 FRONK_NATIVE_WAVE=1`.
The measured trials also use `FRONK_NATIVE_SKIP_CATCHUP_HEADS=1`,
`FRONK_REQUIRE_ALL=1`, `FRONK_EVENT_RESUME=1`, `FRONK_METAL_KEEPALIVE=0`,
`FRONK_MTP_BRANCHES=0` and `FRONK_SSD_STREAMING=1`. Every participant must use
the already-qualified ranked expert arithmetic (`FR_EXPERT_RANKED=1`).
Do not launch a second head against units owned by the running head.

The flag is read at startup. Wave 0/unset retains the single-batch native
verifier; wave 1 uses two groups only at widths 4–8. Smaller final batches
retain the existing path. Greedy decoding and the other native prerequisites
still apply. This is a chain verifier, not independent branch execution.

Before measuring a new model/configuration, `FRONK_NATIVE_VERIFY=1` compares
committed logits and state to independent single-token execution under the
same arithmetic policy. Remove that flag for timing. See `BENCH.md` for
tested widths, short-context limits, the 21–24% native T=4 median speedup and
the current gap to 30 tokens/s. Ordinary chat and installers retain the
previous default; depth eight was slower than depth four in this workload.

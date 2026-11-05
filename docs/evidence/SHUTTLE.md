# The Shuttle neuron-width experiment

This CPU experiment measures how expert computation changes with neuron width
and weight layout. [MEASUREMENTS.md](../MEASUREMENTS.md) gives the results and
their use in the design; this note describes the workload and the raw records.

On 2026-10-16, one Intel i5-10210U Shuttle ran the 125B-backbone model's routed
expert computation with 32, 64, 96, 128, 160, 320, 576, 608 and 640 neurons.
Gate/up used the retained fused lane8 CPU kernel and prepared compact weights;
down used the native MXFP4 kernel, with an eight-thread OpenMP team over the ten
selected experts. Activations were FP32. No GPU participated in this Shuttle
component test. This does not describe the complete later Metal-head cluster.

The four-layer experiment used layers 12–15, all 512 experts resident and 24
changing captured states. States 0–3 were warmup. Logical cropping and physical
compaction were compared in ABBA order, leaving 40 timed four-layer samples per
width/layout. The 3,456 layer records include warmup. A 640-neuron result is real
full-width routed-expert compute, not an extrapolation from smaller slices.
The four-layer timer includes ten selected experts per layer and their weighted
mixture, not a single expert and not a complete model token.

## Files and provenance

- [shuttle-neurons-20261016.csv](shuttle-neurons-20261016.csv): all original
  per-layer timing rows. `layout=0` means logical crop, `layout=1` physical
  compaction; `width` is neurons per expert, **not** MTP token width.
- [shuttle-neurons-20261016.json](shuttle-neurons-20261016.json):
  summaries, paired comparisons, residency/numerical checks, limitations,
  full 48-layer/64-neuron result and SHA-256 identities of the original evidence.

The JSON retains the experiment's source/archive identities and original record
hashes. These identify the measured implementation; source archives and model
bytes are not included in this evidence directory. The raw stdout, guard and
telemetry hashes were verified against the analysis. All 18 layout/width means
were independently recomputed from the included CSV:
discard warmup rows, sum the four layers for each `(width, layout, replica,
position)`, then average the 40 sums. Microseconds are converted to milliseconds.

## What is measured and what is derived

The four-layer means are measured. Multiplying them by 12 estimates 48-layer
**expert-only** cost under comparable conditions; its reciprocal is an equivalent
expert-pass rate. It excludes dense work, network gaps, rejoining and memory
admission of a full pool. It is not generated-token throughput. At full width,
6.318240 ms over four layers projects to 75.819 ms, or 13.19 equivalent passes/s.
At 64 neurons, 0.780655 ms projects to 9.368 ms, or 106.75 equivalent passes/s.

The separate 64-neuron test really loaded all 24,576 partial expert instances
(48 × 512), using 6,794,772,480 bytes of compiled routed weights. It replayed
16 genuine states at occupied contexts 8192–8207 for 20 passes, dropping pass 0.
Its 304 complete measurements give 9.658185 ms mean, 10.089042 ms p95 and 11.277903
ms maximum. The corresponding 103.54 passes/s is a measured component capacity.
No emitted tokens, network or dense-head execution belong to that timer.

Physical compaction at 64 neurons was 2.16× faster than the logical crop using
the same kernels. This is stronger evidence than dividing a full-width timing
by ten: the physical arrangement itself was compared. Partial outputs matched
bit for bit between layouts; joining independent partials had worst relative
L2 `2.23078e-7`, within the original `1e-5` gate. The full-width layout control
showed no gain. Timed disk reads and major faults were zero; residency and
thermal/swap guards passed. These checks cover the expert calculation; complete
model generation is measured separately in [MEASUREMENTS.md](../MEASUREMENTS.md).

The prepared pool in this experiment differs from Fronk's native units. Do not
substitute its bytes, arithmetic tolerances or timings into today's planner.
The AVX2 node also uses different activation handling. Use this evidence to
explain the design, then measure the actual installed backend for placement.

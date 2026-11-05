# Node backends

A node backend computes that machine's share of the experts. It receives
activations and expert IDs, reads the assigned neuron slices and returns a
partial result. Fronk handles networking, weight preparation and planning
around it. The head's local expert worker uses this same interface; the head's
dense GPU engine is a separate component.

| Backend | Hardware / purpose |
|---|---|
| `avx2` | x86 CPUs with AVX2, FMA and F16C |
| `neon` | ARM CPUs with NEON |
| `metal` | Apple Silicon GPUs |
| `metal-ssd` | Measured RAM + SSD expert placement using Metal |
| `ref`, `ref-q8` | Portable scalar implementations for reference and validation |

You can add a backend for your hardware starting from
[user-template.c](backends/user-template.c). It gives you a complete scalar
calculation to optimize and compare against. A node-only backend does not
require a new head protocol. For help from a frontier model, use the
[backend task brief](docs/llm/BACKENDS.md).

## Choosing a machine's backend

The installer offers the compatible backends and a recommendation. You can list
an installed node's backends at any time. Before choosing or measuring one,
stop its worker through the manager so it does not compete with calibration.
Use the node's configuration. With the default installation:

```sh
cd ~/fronk/src
./fronk backends --config fronk.conf          # list; one is marked recommended
./fronk backends --choose --config fronk.conf # choose and calibrate
./fronk backends --set avx2 --config fronk.conf
./fronk backends --set auto --config fronk.conf
```

Use your chosen prefix if it differs from `~/fronk`. Start the node again from
its manager after making a change, then reconnect from the head.

- The recommendation is a rule, with no measuring. On Apple Silicon it's
  `metal`. Otherwise it's the CPU's own kernel (`avx2` on x86, `neon` on ARM),
  and `ref` when there is neither.
- The choice is saved in that machine's state directory (`~/fronk/state/backend`
  for the default installation).
  Installer upgrades retain that choice.
- A `backend NAME` line in the config wins over the saved choice.
- **auto** (the default) self-tests every compatible backend, probes them for a
  moment and keeps the fastest.
- **With a chosen backend**, only that one runs. Its SELFTEST still runs once, and
  a chosen backend that fails it stops the node instead of falling back.
- Saving a choice (`--choose` or `--set`) also measures it right away: the
  1–5 block sweep the planner splits the 20 blocks by. It's cached in the
  state-dir. Missing points use 256 samples at each width and are bounded by
  current usable memory. Hybrid Metal calibration records actual combined RAM
  and SSD work. Reconnect reads compatible profiles without rerunning them.
  `./fronk backends --measure --config fronk.conf` fills missing profiles;
  compute-code or build-option changes can invalidate saved costs. `--no-measure` skips
  it. The first head then measures missing RAM profiles; combined RAM + SSD
  calibration still requires `fronk backends --measure` with its configuration.
- The installer pairs a node with the head's width (ordinary Metal head: 3).
  New short profiles measure widths 1–4 together; `--burst N` selects which saved
  width to report. A width is a batch size, not an MTP acceptance guarantee.
  A head config names the dense engine even before a model is installed. It is
  measured with the head's reserve, `head-blocks` and configured burst.

## Writing your own

Create a named source module, then validate and measure it:

```sh
python3 tools/backend-module.py new mycpu
make DS4= fronk
./fronk backends --check mycpu
./fronk backends --test mycpu
```

[BACKEND_MODULES.md](docs/BACKEND_MODULES.md) covers the manifest, supported
platforms, multiple modules, installer selection, upgrades and contributions.
The generated scalar implementation is a readable starting point, not a speed
claim. Optimize it in a separate checkout; keep the active service recoverable.

The head doesn't need your backend compiled in. It only reads the measured times
the node reports under your backend's name.

### The contract (`src/backend.h`)

```c
typedef struct fr_backend {
    const char *name;
    int (*probe)(fr_hwinfo *out);           /* FR_UNSUPPORTED if this machine can't run it */
    int (*load)(int block, const void *data, size_t len);
    int (*unload)(int block);
    int (*run)(const fr_req *req, fr_part *out);
    int (*bench)(int burst_max, fr_costmodel *out);
    int (*self_test)(const char *vectors_dir);
} fr_backend;
```

- **`load`** gets one unit: `FR_NATIVE_BLOCK_BYTES` (3.35 GB) of read-only,
  memory-mapped bytes for block 0–19.
  - Borrow the pointer until `unload`. Don't copy, convert or scan it.
  - Loading an occupied slot, or unloading an empty one, returns `FR_STATE`.
  - A node may hold any subset of the 20 blocks.
- **`run`** gets one layer for 1–8 tokens.
  - For each token: `expert[t][0..9]` are the 10 routed expert ids (distinct,
    0–511), and `gate[t][0..9]` are their weights.
  - `gate[t][10]` is the shared expert's weight. When it is 0, skip the shared
    expert: the head computes it.
  - `x[t]` holds 2560 fp32 values.
  - Write `y[t]`: the sum over those experts and over every loaded block of
    `weight * down(silu(gate·x) * (up·x))`. Leave unused tokens zero.
  - For a resident backend, allocate nothing, do no I/O, and leave `*out` untouched
    on any error. The built-in `metal-ssd` mode is an explicit separately measured
    streaming policy; do not hide paging inside a resident backend.
  - Calls are serialized; threads inside `run` are yours to manage.
  - Errors: `FR_INVALID` for bad requests, `FR_NUMERIC` for non-finite values,
    `FR_STATE` when nothing is loaded.
- **`self_test`**: `fr_native_self_test(&fr_module_mycpu)` compares your `run` with the
  reference. It covers bursts 1–8, distinct and repeated experts, the shared
  expert and corrupt scales, within frozen tolerances: relative L2 0.025,
  normalized max 0.030, cosine ≥ 0.9998 per token.
  It must refuse with `FR_STATE` while blocks are loaded.
- **`bench`**: return `fr_bench_execute_backend(&fr_module_mycpu, burst, out)`
  (substitute your module's name).
- **`probe`**: check what you need (an ISA flag, a GPU) and return
  `FR_UNSUPPORTED` without it. `hw_probe` fills `fr_hwinfo`: os, arch, cpu,
  cores, isa, metal, memory.

### The unit layout (`src/native.h`)

These are lossless slices of the GGUF: no requantization and no padding.

- **A unit** holds 48 layers. Each layer stores 512 routed slices, then 1 shared
  slice: `fr_native_offset(layer, expert)` gives the slice offset, and the shared
  expert's id is 512.
- **A routed slice** (32 neurons of one expert):
  - gate: 32 rows × 2560 weights in Q4_K, 1440 bytes per row;
  - up: the same, right after gate;
  - down: 2560 rows × 32 weights in MXFP4, 17 bytes per row.
- **The shared slice:** gate, up and down in Q8_0. Gate and up are 32 rows ×
  2720 bytes; down is 2560 rows × 34 bytes.
- **The quantization formats** are the standard GGUF/ggml blocks:
  - Q4_K: 144 bytes per 256 weights;
  - MXFP4: 17 bytes per 32;
  - Q8_0: 34 bytes per 32.
- `fr_native_dot()` in `src/native.c` defines the scalar decoding for all three.
  Decode the formats exactly; optimized accumulation must satisfy the frozen
  conformance tolerances. Bit-identical arithmetic is a stronger, separate claim.
- `fr_native_row(type, n)` gives the bytes of one row of n weights.

### What makes it fast here

- A call reads 10–11 experts × 3 projections from each loaded block, about
  1.6 MB per token with shared work. Memory movement, quantized decoding and
  worker/submission overhead can each dominate; measure the actual machine.
  The [Shuttle width/layout experiment](docs/evidence/SHUTTLE.md) demonstrates why
  physically compact data matters, without claiming measured cache behavior.
- The planner gives a machine more blocks when its time per call is lower.
  **p99 matters too:** the head waits for the slowest machine at each of the 48
  layers.
- Calls can contain 1–8 tokens. Ordinary MTP and prefill use different widths;
  deep native MTP is experimental. Reuse decoded weights when tokens select the
  same expert, but measure the cost for distinct routes too.
- Spread the experts × blocks over a persistent worker team, leaving room for
  the receiver. More workers can hurt a small CPU. Node GPU keepalive is a
  measured policy; it is disabled on the head because it competes with dense
  work. Do not copy the node setting to the head without a complete comparison.
- Examples: `backends/neon.c` (threads, int8 dot products), `backends/avx2.c`
  (threads, prefetch, AVX2 decoding), `backends/metal.m` (GPU, zero-copy units).

### Limits

- Capacity is the system's free memory minus the margin, as for a CPU backend.
  A discrete GPU with its own memory can't yet say how much it holds.
- Up to ten named source modules per build. The legacy `fr_user` slot remains
  available for older single-file ports.
- `run` is synchronous. fronk overlaps the network itself.

## Give the work to a frontier model

Use [the backend task brief](docs/llm/BACKENDS.md) with an assistant that can read
this checkout. It contains the reading order, hardware evidence required before
choosing instructions, invariants, build commands and acceptance tests. Give it
your CPU/GPU and OS if it cannot inspect them. Ask for a working source file and
actual numerical/timing results, not a proposed kernel alone.

For example: “Read docs/llm/BACKENDS.md. Implement a backend for this machine in
a separate build, explain the hardware decisions and compare it with the current
backend. Keep the working installation recoverable.”

The [model brief index](docs/llm/README.md) also has briefs for architecture,
communication, installation troubleshooting and complete cluster setup.

## Put a backend in an installer

Put a module under `backends/modules/<name>/` in the tree passed to `make dist`,
or choose **add-module** in the downloaded manager. A compatible module is
compiled and appears in the same selection menu as the built-in backends.
The [module guide](docs/BACKEND_MODULES.md) explains the complete procedure.

Share a working port with its hardware requirements, build command, numerical
checks and timings. If a port fails, the smallest reproducing input and the
first mismatch are useful too. [SHARE.md](docs/SHARE.md) covers reports and
[CONTRIBUTING.md](CONTRIBUTING.md) covers a pull request.

# Task brief: implement a specialized compute backend

> **For your frontier model.** People: read [the backend guide](../../BACKENDS.md) instead, and see
> [how to use this brief](README.md). Model: also follow [AGENTS.md](../../AGENTS.md).
> If you cannot open a linked file, ask for it; do not guess.

Implement a backend for the user's actual CPU/GPU. First inspect the hardware,
OS, compiler and available ISA/API; ask only for missing access or preferences.
Choose a short backend name and preserve the working binary. Do not assume a
marketing name proves an instruction is available in the running OS/process.

## Read the small contract

1. [BACKENDS.md](../../BACKENDS.md), [BACKEND_MODULES.md](../BACKEND_MODULES.md) and `src/backend.h`.
2. `src/model.h`, `src/native.h`, `src/native.c`: exact native quantized layout.
3. `backends/user-template.c`: the executable scalar specification.
4. The relevant existing backend (`neon.c`, `avx2.c`, `metal.m`), then
   `src/bench.c`, `src/selection.c` and the Makefile registration/build rules.
5. [PROJECT.md](../../PROJECT.md) for why the neuron cut exists, and
   [CONTRIBUTING.md](../../CONTRIBUTING.md) for numerical/performance evidence.

The backend computes a partial; it does not own sockets, routing decisions,
model download, token commitment or cluster planning. Do not add those systems.

## State the hardware case before writing intrinsics

Cite primary vendor ISA/GPU documentation for the operations you intend to use.
Explain data layout, alignment, decoding, lane assignment, register/threadgroup
storage, memory traffic and reduction order. Estimate the likely bottleneck,
then measure it. Wider SIMD, matrix engines or extra threads are hypotheses,
not automatic improvements. Do not claim ISA-level optimization from C syntax.

Create a named source module with `python3 tools/backend-module.py new NAME`.
Do not overwrite an existing module or built-in backend. Keep the first port
inside `backends/modules/NAME/`; if
the interface cannot express the hardware requirement, explain that limitation
and propose the smallest explicit extension instead of silently abusing it.

## Preserve these invariants

- `probe` rejects unsupported hardware before executing unsupported instructions.
  Scope architecture-specific flags to its kernel, not the entire executable.
- `load` borrows immutable unit bytes until `unload`; validate IDs, lengths and
  slot state. No weight conversion, copying or eager multi-GB scan in this contract.
- `run` accepts layers 0–47, widths 1–8, ten distinct routed IDs and eleven gates.
  Skip shared work when its gate is zero; handle a nonzero shared gate correctly.
- Native routed gate/up is Q4_K; down is MXFP4; shared weights are Q8_0. Decode
  exactly. Do not confuse the legacy Q4_0 oracle layout with production units.
- No allocation, I/O or thread creation in a resident `run`. Preallocate worker
  resources outside the timed path. Calls are externally serialized.
- Success publishes a complete result with unused tokens zeroed; every error
  leaves the caller's output untouched. Drain submitted GPU/worker work on failure.
- Do not loosen the frozen conformance tolerances, remove invalid-input checks
  or update golden outputs merely to make the candidate pass.
- Explicit RAM+SSD streaming has its own measurements/policy. Do not disguise
  paging as resident computation. Dedicated VRAM capacity is not fully represented
  by the current module interface; disclose this before promising support.

## Build, validate, then measure

Use a separate checkout/build. Export `fr_module_NAME`, matching the folder and
manifest name, then run `make DS4= fronk`, `./fronk backends --check NAME` and
`./fronk backends --test NAME`. Restrict manifest platforms where needed and
verify that `probe` rejects unsupported ISA before executing it. Never apply an
ISA flag to the whole executable. Do not replace the active service binary.

Test adding the folder with the downloaded manager's **add-module** choice or
`--backend-module PATH --backend NAME`, then upgrade and uninstall under an
isolated prefix. A failing SELFTEST must preserve the prior binary. The source
under `PREFIX/modules/` must survive uninstall. Registry/installer changes also
run `python3 tests/test_backend_modules.py` and `bash tests/backend-modules.sh`.

Run relevant sanitizer and native-oracle checks, including unsupported ISA,
invalid requests, corrupted scales, multiple/nonadjacent units, shared work,
widths 1–8 and unchanged output on failure. Use real resident weight storage for
timings; sparse correctness fixtures are not bandwidth benchmarks.

Measure widths 1–4 (validate 5–8), distinct/reused experts and all feasible shares
up to five units. Include validation, worker handoff and final reduction in
`run` time. Report p50/p99, memory residency, compiler flags and a bracketed
control/candidate/control. A component win still needs a complete cluster test
with identical model/output/placement before being called a chat speedup.

If two valid comparisons show no benefit, retain the evidence and working
backend; propose a different measured cause. Do not iterate indefinitely or
change precision to manufacture a win. Wire the selected backend into the
existing saved calibration and test the installer path described in the human
guide. The head need not compile a node-only backend.

Deliver the source, hardware rationale and references, build/use/rollback
commands, numerical and timing evidence, and explicit unsupported cases.

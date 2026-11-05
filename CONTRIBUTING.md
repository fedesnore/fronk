# Contributing

You can contribute by trying Fronk on your machines, reporting where setup
stopped, sharing a solution, improving instructions or changing the code.

## Start with what you found

For a blocked installation, open an **Installation help** issue with the role,
OS, step and error. For a working combination of machines, choose **Cluster
result**. [SHARE.md](docs/SHARE.md) gives an example and explains which details
help. A small report is enough to begin; you do not need to diagnose the whole
system first.

If you solve the problem, reply to the issue with the cause if known, the change
and the check that passed afterward. Keep the solution with the report so
others can find it. A documentation correction or a clearer error message is
as welcome as a kernel change. A frontier model can help investigate with
[DEBUG.md](docs/llm/DEBUG.md), or prepare a reviewed report with
[REPORT.md](docs/llm/REPORT.md).

## Work on the code

Read [PROJECT.md](PROJECT.md) for the roles, data flow and source map. Fronk
favors a small amount of explicit code that can be followed with an editor and
a C debugger. Start from the module that owns the behavior you want to change.
[Where to help](docs/IMPROVEMENTS.md) offers small first experiments and larger
directions, with files and checks for each. The
[development brief](docs/llm/DEVELOPMENT.md) helps a frontier model keep the work focused.

Write for someone who can program or understand hardware but has not seen this
project. Explain the reason for a boundary, then follow its data. Name the units
and show one concrete example. Prefer an explicit function over an abstraction
that hides ownership. The reader should be able to predict what happens next.

Keep guides and [model briefs](docs/llm/README.md) paired. A guide explains
why and how; its brief points an assistant to code, invariants and acceptance
checks. Do not duplicate wire/layout constants into a competing
specification. Update both when behavior changes.

## Build and test

A C11 compiler, make and Python 3.9 or newer build the node/CLI. On macOS install
Xcode command line tools; on Linux install the platform's C development tools.
Python also runs the console and installer helpers. No model is needed for
normal tests.

```sh
make DS4=
make DS4= test-core
make test-plane-cluster
```

`test-core` uses AddressSanitizer/UndefinedBehaviorSanitizer and isolated local
nodes. GPU tests need an actual supported GPU; unavailable hardware is reported,
not emulated as a performance pass. `make test` also needs `redis-cli` and `nc`
for interoperability checks. `make test-installer` builds the downloads first
and exercises installations under temporary prefixes. See `docs/RELEASE.md`
for its platform/build prerequisites.

For installer changes, the source workflow can be tested without assembling
the GPU download bundle:

```sh
FRONK_TEST_MANAGER="$PWD/tools/install/fronk-install.sh" bash tests/installer.sh
```

This uses isolated prefixes and local test ports. Release qualification still
tests the actual downloaded manager and a physical head/node cluster.

For a backend contribution, use the conformance and timing loop in
[BACKENDS.md](BACKENDS.md). The scalar template is an executable description of
the slice math. Do not change the frozen tolerances or golden vectors to make a
new kernel pass.

## Code style and boundaries

- Use C11 for the core and CPU backends, Objective-C/Metal where the GPU API
  requires it, small shell scripts for installation, and standard-library
  Python for UI/control helpers. Discuss a new dependency before introducing it.
- Follow the surrounding file's style. Prefer named helpers with one job to
  deeply nested conditionals. Avoid a global formatting diff mixed with logic.
- Explain ownership, units, synchronization, error publication and the reason
  for an unusual optimization. Do not narrate obvious assignments.
- At a module's entrance, name its job and a useful reading order. At a tricky
  boundary, explain what the next function can rely on. Prefer a small example
  over a catalogue of abstractions. Remove stale milestone notes when they no
  longer describe the code; keep historical results in the measurement notebook.
- Keep setup work out of `run` and the layer transport loop. Allocate persistent
  buffers/threads during load/setup. Preserve the existing Metal framework
  command-object exception and measure submission overhead.
- Return errors without publishing partial output. Release/drain work on every
  failure path. A missing node is not a valid model result.
- Preserve weight bytes and arithmetic policy. Reassociation, FMA changes,
  quantization, expert ordering and reduced precision need numerical evidence.
- Keep user-facing strings and documentation in English.

## Show the evidence

For correctness changes, describe the failure and add a focused test that fails
before the fix. Use real integration tests for protocol, lifecycle and installer
changes. Do not create a test that only repeats the implementation.

For performance changes, report hardware, build flags, model hash, prompt,
context, accepted token counts, unit placement, RAM/SSD split and network path.
Measure an unchanged control before and after the candidate. Use API completion
token counts, not SSE fragment counts. Report individual samples and both
controls if they drift. Distinguish kernel time, full layer time, prompt latency
and end-to-end decode throughput. A faster microkernel can slow the chat.

Keep experiments opt-in until their numerical checks and complete-workload
measurements justify changing the default. Do not combine unrelated optimizations
in the first comparison. Do not claim 30 tokens/s from a component extrapolation.

## Submitting a change

Choose the module in [MODULES.md](docs/MODULES.md). Named backend ports follow
[BACKEND_MODULES.md](docs/BACKEND_MODULES.md) and appear in installer choices.
One part per pull request, please: `python3 tools/pr-scope.py` shows what yours touches.
Pull requests receive scope labels; the maintainer assigns priorities.

Open an issue for a large design change, or send a focused pull request with:

1. The concrete problem and resulting behavior.
2. The files/contract affected and any compatibility change.
3. The tests run and their outcomes; explicitly name hardware not tested.
4. Before/after evidence for performance claims.

Keep generated binaries, models, credentials, machine state and private logs out
of the change. Use temporary prefixes/ports when testing installers. Review the
actual diff before publishing. Contributions to Fronk are provided under its
[MIT license](LICENSE); keep upstream notices with imported code.

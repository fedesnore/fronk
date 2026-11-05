# Building and qualifying a release

A release is a set of source/install artifacts plus evidence about what was
actually tested. It must not depend on a maintainer's model directories, SSH
configuration, private engine checkout or cached build products.

## Prepare the public repository

Export an explicit set of files rather than pushing a development checkout's
entire history. From this checkout, choose a new destination:

```sh
python3 tools/release/export-source.py build/public/fronk-0.3.0
```

The exporter combines the installer allowlist with
`tools/release/source-files`: GitHub templates/CI, example configurations, the
download page and source provenance. It creates a clean source tree, a tarball,
an archive checksum and a per-file source manifest. It refuses an existing
destination. Local configs, weights, logs, SSH scripts and Git history are not
copied. The working installation is left in place.

Run the checks below **inside the export**. Only that qualified directory should
be initialized as the public Git repository. Use `configs/head.conf` and
`configs/node.conf` as examples; the manager creates users' real configs.
Review the exported files and notices before the first commit.

The repository already contains issue templates for installation help, cluster
results and development, a short pull-request template and a Linux/macOS CI
workflow. At publication, describe the project as “Run a large local model across
heterogeneous computers using neuron-sliced experts.” Link the README, keep
Issues enabled and attach the managers, source archive, checksums and verification
report to one tagged prerelease. Use [0.3.0.md](releases/0.3.0.md) as the release
body, preserving its qualification limits. A completed local test does not mean
hosted CI, signing or an untested platform has passed.

## Reconstruct the engine

```sh
make engine
make head                    # Apple Silicon Metal head
```

`tools/engine/prepare.sh` fetches ds4 commit
`8db1d1d155cb0400a86a86b9c62d0defb3a6148b` from its public repository and applies
`tools/engine/ds4-fronk.patch`. `tools/mtp/stage-ds4.sh` copies the explicit
`source-files` manifest into a private build directory and applies the worker/
MTP integration. No Git index from a private checkout is needed.

To use an already-fetched upstream repository without network access, set
`DS4_UPSTREAM=/absolute/path/to/ds4` when preparing a new destination. This still
archives the pinned commit and applies the committed patch. `DS4=PATH` selects
an explicitly supplied compatible source tree for advanced builds. Do not edit
an engine used by a running head; stage another directory.

The ordinary release keeps native deep MTP/wave, alternative AVX2 ordering,
ready-block SSD pipelining and the new HC mixer disabled. Default head startup
policy lives in `src/runtime.c`; no personal wrapper should be needed to obtain
complete sums and GPU event resume.

## Build the managers

```sh
make dist
```

On macOS the multiplatform release also builds the Intel Vulkan runtime. The
builder needs CMake, Git, glslc, Vulkan headers and SPIRV-Headers in addition to
the normal C/Apple tools. `tools/gpu-head/macos-dist.sh` downloads and verifies
the pinned MoltenVK release. `LLAMA_CPP`, `CMAKE` and `VULKAN_SDK` can select
existing dependencies; the llama.cpp commit remains pinned.

A Linux build can provide its native Vulkan bundle with `make gpu-head`. A
multiplatform release assembled on Linux must also be supplied the Intel macOS
runtime built on macOS. The packager refuses to silently omit that required
runtime. Platforms without a bundled Linux runtime build the pinned adapter
locally; document this prerequisite in the download notes.

Outputs are under `build/site/downloads/`:

- `fronk-manager-macos.zip`: Finder app, graphical manager.
- `fronk-manager-linux.tar.gz`: desktop launcher and the same manager.
- `fronk-manager.sh`: portable terminal/GUI self-extracting manager.
- `fronk-source-0.3.0.tar.gz`: the public source tree, examples and GitHub files.
- `SHA256SUMS`: checksums for those downloadable artifacts.

Publish these artifacts together only after qualification. Signing/notarization is separate; do not describe the
current ad-hoc/unsigned package as notarized.

From the exported source directory, verify the complete download set:

```sh
python3 tools/release/verify-artifacts.py build/site/downloads --source . > build/artifacts.json
```

This checks every archive checksum, both graphical launchers' embedded manager,
the payload inventory and its shared files against the reviewed source export.
It also rejects a source archive from a different revision. Keep the generated
report with the test results. If a shipped file changes, create a new export,
rebuild and repeat qualification; a report for an earlier archive is not valid
for its replacement.

`tools/install/runtime-files` is the allowlist. It must include build inputs,
documentation and licenses but no state, models, private configs, developer SSH
scripts or logs. The package includes a per-file source manifest. Adding a
runtime dependency means updating the allowlist and testing an extracted copy.

## Update releases

There is one version and one set of packages, because head and nodes run one binary from
one source tree. A release that concerns one part is still a complete release: its notes
say which machines need it, and the others can stay as they are.

```sh
python3 tools/release/new-release.py                 # what changed, who updates, which version
python3 tools/release/new-release.py --write 0.3.1   # set the version and draft the notes
```

The report compares with the newest `vX.Y.Z` tag, using the parts in
[modules.json](../.github/modules.json). If only documentation, tests or build files
changed, no machine runs them: merge them and do not release. Edit the drafted
`docs/releases/VERSION.md`, name the release after its part (`Fronk 0.3.1: console`), then
test what the change can affect:

| Changed | Before releasing |
|---|---|
| Console, installer | That part's checks from [MODULES.md](MODULES.md), then a packaged install |
| Planner, head | Also a real cluster run, compared with the previous release |
| Backend | Also SELFTEST and a real cluster run on that hardware |
| Transport, model | All of the above on every platform; head and nodes must be updated together |

Then `make dist`, verify the artifacts as described above, tag `vVERSION`, and attach the
four downloads and `SHA256SUMS` to a GitHub release using the notes as its body. An
installed machine updates by opening the new manager and choosing upgrade, which keeps its
configuration and verified weights.

After the first publication, work in a clone of the public repository: a branch per pull
request, tags on `main`. The export step above is for creating that first repository.

## Qualification

Run in a fresh checkout/export and use isolated prefixes for installation:

Put real-cluster test prefixes on persistent storage. Linux `/tmp` can be a
RAM-backed filesystem smaller than the incoming units; do not let a test cache
consume the RAM being measured. On macOS, launch through the same authorized
Terminal or installed service the user will use. SSH administration does not
prove the launched process has local-network permission.

```sh
make DS4= test-core
make test-plane-cluster
make test-installer
```

CI also runs the install/upgrade/uninstall lifecycle directly from the exported
source, without building the experimental Vulkan distribution. That catches
workflow regressions on both operating systems; it does not replace the
packaged-installer check above or the physical cluster qualification below.

Then install the packaged **head** into a temporary prefix with `--no-measure
--demo-later --yes`, without selecting/downloading a model. This checks that its
engine, shader files, Python UI and launchers are present. Check the packaged
node in another temporary prefix. Test both GUI and terminal handoffs, resume,
missing peers, wrong model identity, upgrade, keep-state uninstall and fresh
reinstall. Automated headless checks do not replace opening the real GUI.

For model layout changes, run the read-only full-file audit separately:

```sh
make build/fronk-model-check
./build/fronk-model-check /absolute/path/Qwen3.8-Flash-Next-Q4.gguf tools/install/model-units.sha256
```

This verifies every extracted byte against source tensor coordinates and
recomputes all twenty unit digests. The **whole GGUF SHA-256** must also match
`docs/MODEL.md`; the unit audit alone does not hash dense/ngram tensors. The normal
installer performs the whole-file check and preparation/transfers check units.

Finally run a real complete cluster with the packaged ordinary policy and
compare against the retained baseline. Use identical prompts, context, model,
placement, arithmetic and completion token counts. Record timings before and
after the candidate; do not turn a single fast prompt into a universal throughput
claim. Keep the original working binary until this comparison passes.

For that complete run, enable the manager's normal calibration (including
Metal RAM + SSD), then run the synthetic demo. The earlier `--no-measure` smoke
test alone does not qualify SSD placement or data-plane access. Stop at any
failed stage and preserve its logs; do not label service authentication as a
successful generation test.

Record source/artifact hashes, compiler/platform, test summaries, model identity,
node list, RAM/SSD placement and limitations in the release notes. macOS ARM,
Intel macOS and Linux have separate qualification status. A cross-compiled
binary or successful GPU probe is not an end-to-end chat test on that platform.

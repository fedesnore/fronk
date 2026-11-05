# Add a backend module

A module is a folder of source files with a small manifest. Give it a unique
name, put it under `backends/modules/`, and build. Fronk adds it to the compiled
registry and the installer's choices. Several modules can coexist. Nothing is
loaded or discovered while an inference request runs.

This interface serves a node's expert slices and the head's local expert worker.
The dense head engine is separate. Read [BACKENDS.md](../BACKENDS.md) for the
arithmetic, memory ownership and measurements before optimizing a kernel.

## Start with a working implementation

In a separate source checkout, with a C11 compiler, make and Python 3:

```sh
python3 tools/backend-module.py new mycpu
make DS4= fronk
./fronk backends --check mycpu
./fronk backends --test mycpu
```

The first check compares numerical results with the reference. The second also
times a one-block request and the recommended backend. The generated kernel is
portable scalar code: correct, deliberately simple, and a starting point for
optimization. It is not presented as a fast backend.

The new folder contains `backend.c`, `backend.json` and a README. Its manifest is:

```json
{
  "api": 1,
  "name": "mycpu",
  "sources": ["backend.c"],
  "systems": ["Darwin", "Linux"],
  "architectures": ["aarch64", "arm64", "x86_64"]
}
```

The folder name, manifest name and exported C table must agree:

```c
const fr_backend fr_module_mycpu = {
    "mycpu", probe, load, unload, run, bench, self_test
};
```

`src/backend.h` defines those six functions. Include it as `"backend.h"`;
the build supplies the include path. Use static names for private functions
and state so two modules cannot accidentally share symbols.

## Describe the hardware honestly

- Names use 1–15 lowercase letters, digits or underscores, starting with a
  letter. Built-in names and names starting with `ref` are reserved.
- `sources` lists C or Objective-C files relative to the module folder. Keep
  headers, tests and a README there too. Symlinks and paths outside the folder
  are rejected. At most ten source modules can be compiled together.
- `systems` and `architectures` filter compilation. Omit them to allow all
  currently supported platforms. Limit them when the compiler cannot build
  the source elsewhere; a filtered-out module is not offered on that machine.
- `probe` checks the actual ISA, device and driver at runtime. Return
  `FR_UNSUPPORTED` before executing an unavailable instruction. A successful
  build alone does not prove that every CPU of that architecture can run it.
- Optional `ldlibs` is a list of linker arguments, for example
  `["-framework", "Accelerate"]`. List extra prerequisites in your README.
  There are no manifest shell hooks or global ISA flags. Keep specialized
  instructions inside guarded kernel functions, as the AVX2 backend does.

The manifest's `api` is the source contract version, not a dynamic-library ABI.
The build generates `build/backend-modules/registry.h` and `modules.mk`.
Do not edit or commit these files. Source/manifest changes invalidate the saved
compute calibration. Registration adds no per-request lookup or loader.

## Offer it in the installer

For your own machine, open the downloaded manager, choose **add-module** at the
expert-backend step and select the module folder. Both the GUI and terminal use
this flow. The equivalent unattended node command is:

```sh
bash fronk-manager.sh --role node --name node-1 \
  --backend-module /absolute/path/to/mycpu --backend mycpu --secret-stdin
```

Supply the pairing code from the head on standard input. Repeat
`--backend-module` to add more modules. Without `--backend`, choose from the
compatible list. A module bundled under `backends/modules/` is also discovered
automatically; `make dist` includes its source in every manager.

The manager compiles in a temporary directory and runs each added compatible
module's SELFTEST before replacing the installed binary. A failed build or
SELFTEST leaves the old program in place. It then offers the backend and saves
its measured costs through the same calibration path as built-in backends.
The head reads the node's name and costs; it need not contain that kernel.

Added source is kept in `~/fronk/modules/` (or your installation prefix) and
reused on upgrade. Replacing a module keeps one `.name.previous` source copy.
Uninstall, including a fresh-state uninstall, preserves this user-owned folder.
To stop using a module, choose a built-in backend. To remove it from the next
build, stop Fronk, move its folder outside `modules/`, then upgrade using the
downloaded manager. Keep your
source elsewhere before deleting it. Modules are native program code: review
the source you choose just as you would a patch to Fronk itself.

The older single-file `backends/user.c` / `USER_BACKEND=...` interface still
works with its `fr_user` table. Use named modules for new contributions.

## Send a focused contribution

Put the module, hardware requirements, numerical checks and timings in one
pull request. Include unsupported cases and compare complete request times
with the existing backend. Do not alter the frozen tolerances to obtain a pass.
Run `python3 tests/test_backend_modules.py` and `bash tests/backend-modules.sh`
for registry/installer changes, plus the relevant core tests for kernel changes.

[MODULES.md](MODULES.md) explains how pull requests are grouped for review.
[The backend brief](llm/BACKENDS.md) gives an assistant the reading order,
hardware evidence and acceptance checks for a port.

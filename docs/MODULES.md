# Find a place to contribute

A useful pull request can change one kernel, one installer question or one
error message. Choose the part that owns the behavior. The boundaries below
help a reviewer understand the change without reading the whole engine.

| Area label | Responsibility and first files | Useful validation |
|---|---|---|
| `area:backend` | Expert slice math: `src/backend.h`, `src/native.c`, `backends/` | SELFTEST, invalid inputs, widths 1–8, full-request timings |
| `area:head` | Dense engine and token commitment: `src/head.c`, `src/serve.c`, `tools/engine/`, `tools/mtp/` | Numeric checks, accepted tokens, unchanged control/candidate/control |
| `area:transport` | Control/data planes and worker lifecycle: `src/connection.c`, `src/plane.c`, `src/node.c` | Core + plane-cluster tests, loss/reconnect/partial-output handling |
| `area:planner` | Measured placement and RAM/SSD costs: `src/selection.c`, `src/planner.c`, `src/hybrid.c` | Capacity boundaries, cached profiles, measured whole-layer costs |
| `area:model` | GGUF identity, extraction and unit integrity: `src/prep.c`, `src/units.c` | Byte comparisons, malformed/truncated inputs, model hash |
| `area:installer` | Head-first setup and local services: `tools/install/` | Isolated install/upgrade/uninstall/reinstall, GUI and terminal |
| `area:console` | Run view, chat and diagnostics: `tools/fronk-web.py`, HTML files | Real head transitions and browser verification |
| `area:docs` | Explanations and paired model briefs | `python3 tools/check-docs.py` and a first-reader check |
| `area:tests` | Tests, fixtures and regressions | A failure before the fix; a pass afterward |
| `area:build` | Reproducible source, packages and CI | Clean source export and build |

For a new architecture, follow [BACKEND_MODULES.md](BACKEND_MODULES.md).
The module appears in the installer's compatible choices; a node-only port
does not require a head rewrite. For other work, [IMPROVEMENTS.md](IMPROVEMENTS.md)
lists concrete experiments, including deeper MTP and shared-switch scheduling.

## Make a pull request easy to assess

Name the main module in the title, for example `backend/mycpu: reuse decoded
weights across a token batch` or `installer: retain a local model after uninstall`.
Explain the problem, the resulting behavior, the tests and any remaining limits.
Include the matching guide/model brief when a contract changes. **One part per
pull request is what we ask for:** it can be reviewed, accepted and released on its own.
Check before opening it:

```sh
python3 tools/pr-scope.py
```

It prints the labels the pull request will receive. Documentation and tests that go with
the change do not count as another part. A change that really must cross parts is welcome;
explain the boundary it crosses. If the parts can land separately, send them separately.

After CI finishes, the repository classifies the changed paths using
[modules.json](../.github/modules.json). It adds `area:*` labels and, for backend
changes, `backend:<name>`. A change spanning runtime areas gets
`scope:cross-module`; supporting documentation and tests do not make it cross-module.
Labels describe scope, not quality or urgency.

The maintainer can add `priority:high`, `priority:normal` or `priority:later`.
Automation preserves these and other manually assigned status labels. Useful
GitHub pull-request searches include:

```text
is:pr is:open label:"area:backend"
is:pr is:open label:"backend:mycpu"
is:pr is:open label:"priority:high"
is:pr is:open label:"scope:cross-module"
```

## Maintainer setup

Keep both workflows on the default branch and enable GitHub Actions. The
classifier runs after **Core and documentation**, including failed runs, or
manually through **Classify pull requests** with a PR number. A first-time fork
may need its CI run approved before classification. The first classification
creates missing module and priority labels. No third-party labeling service or
personal token is needed.

The privileged job reads only default-branch code and changed-file names through
the API. It never checks out a PR or reads its build artifacts. This follows
GitHub's separation between [workflow_run events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run)
and [label API operations](https://docs.github.com/en/rest/issues/labels).
The path rules and classifier are tested locally; the hosted workflow must also
be checked after the repository is published.

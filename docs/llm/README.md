# Get help from a frontier model

Fronk has two kinds of documentation:

| | Written for | Where | You should |
|---|---|---|---|
| **Guides** | You | [`docs/`](../README.md), `README.md`, `PROJECT.md` | Read them |
| **Briefs** | Your AI model | `docs/llm/`, `AGENTS.md` | Hand them over; no need to read them |

A brief tells the model which code to read, what must stay true and how to check
its work, so it does not have to guess at this project. Use it when something
breaks, when you are setting up your machines or when you want to build something.

## How to use one

**With a coding agent that can read files:** open it in your Fronk folder and paste
a request below. The folder is `~/fronk/src` after installation, or a clone of this
repository. The agent finds [`AGENTS.md`](../../AGENTS.md) and the right brief.

**With a chat model that cannot read your files:** attach the brief named in the
request, or paste its GitHub link if the model can browse. It will ask for anything
else it needs. Use the files from the release you installed.

## Requests to copy

Something stopped or fails:

```text
Read docs/llm/DEBUG.md. Fronk stopped at [step] on [head/node], running [OS],
with this error: [exact message]. Help me find the cause, one check at a time, and
tell me which computer to use. Keep my configuration and verified weights.
```

Setting up from scratch:

```text
Follow docs/llm/SETUP.md. Connect my computers and install Fronk, asking only for
what needs my hands or my yes.
```

Chat works but feels slow:

```text
Read docs/llm/PERFORMANCE.md. My cluster is [machines and cables] and the console
reports [result]. Find what limits it before changing anything.
```

Add support for your hardware:

```text
Read docs/llm/BACKENDS.md. Inspect this machine and implement a backend for it in a
separate build. Explain the hardware decisions, measure it against the current
backend and keep my working installation recoverable.
```

Contribute something else:

```text
Read docs/llm/DEVELOPMENT.md and docs/IMPROVEMENTS.md. I have [machines] and about
[time]. Suggest one question I could answer, then help me make the smallest change
and measure it.
```

Share what happened:

```text
Read docs/llm/REPORT.md. Help me turn this setup and result into an issue: [notes].
Remove private information and show me the draft before anything is posted.
```

## All the briefs

| You want to | Give it | Guide for you |
|---|---|---|
| Find why something stopped | [`DEBUG.md`](DEBUG.md) | [Troubleshooting](../TROUBLESHOOTING.md) |
| Connect computers and set up a cluster | [`NETWORK.md`](NETWORK.md) | [Network](../NETWORK.md) |
| Install, resume, repair or uninstall | [`INSTALL.md`](INSTALL.md) | [Installation](../INSTALL.md) |
| Understand a measurement, capacity or plan | [`CALIBRATION.md`](CALIBRATION.md) | [Startup](../../PROJECT.md#startup-head-first-then-nodes) |
| Solve a model, slice or transfer error | [`MODEL.md`](MODEL.md) | [Model](../MODEL.md) |
| Solve a loading, numerical or chat failure | [`RUNTIME.md`](RUNTIME.md) | [Troubleshooting](../TROUBLESHOOTING.md) |
| Investigate a slow chat | [`PERFORMANCE.md`](PERFORMANCE.md) | [Measurements](../MEASUREMENTS.md) |
| Connect a chat client or agent | [`API.md`](API.md) | [API](../API.md) |
| Add a backend for a CPU or GPU | [`BACKENDS.md`](BACKENDS.md) | [Node backends](../../BACKENDS.md) |
| Make a focused contribution | [`DEVELOPMENT.md`](DEVELOPMENT.md) | [Where to help](../IMPROVEMENTS.md) |
| Understand or change the architecture | [`PROJECT.md`](PROJECT.md) | [How Fronk works](../../PROJECT.md) |
| Change head/node communication | [`PROTOCOL.md`](PROTOCOL.md) | [Protocol](../PROTOCOL.md) |
| Share a result, a problem or a solution | [`REPORT.md`](REPORT.md) | [Share your cluster](../SHARE.md) |

## Stay in control

- **Keep private things out.** Do not paste pairing codes, `cluster.secret`, SSH keys
  or private conversations. Attach small excerpts, never the model or a whole state
  directory.
- **Models make mistakes.** Read a change before allowing it, especially on network
  settings. A brief improves the odds; it does not guarantee a fix.
- **Models have limits.** They cannot plug in cables or reach computers you have not
  given them access to. Without access, they should give you one step at a time.
- **Pass on what you learn.** If it solved your problem, add the cause and the fix to
  an issue. It saves the next person the same work ([how](../SHARE.md)).

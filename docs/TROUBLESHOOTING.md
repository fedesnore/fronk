# When a step does not finish

Keep the error message and note which computer showed it. Start with that step:
you can usually keep the installation, verified weights and saved measurements
while fixing the problem.

## Continue from where you stopped

Reopen the downloaded manager on the affected computer, using the same
installation folder. For the default installation, the installed manager is
`~/fronk/bin/fronk-manager`.

| Where you stopped | Where to continue |
|---|---|
| Some nodes still need installation or pairing | On the head, choose **join-nodes** and follow the next named node |
| The demo passed; the model is not configured yet | On the head, choose **configure-model** |
| A download was interrupted | Choose the same download folder to resume and verify the file |
| Chat was working; the console was closed | On the head, choose **web-console** |
| An installed node is stopped | On that node, choose **run**, then return to the head for verification |
| The graphical manager will not open | Run `bash fronk-manager.sh --terminal`; it follows the same setup |

For a custom prefix, pass the same `--prefix` as before. If the first build did
not finish, reopen the downloaded manager and select the same role and folder.
Keep the first error if it fails again; it usually gives a better starting
point than later messages caused by that failure.

## Ask a frontier model for help

When a step fails, the graphical manager and the head console offer a button that
copies this request with the error filled in. Check it for private names, then give
your model the request and [DEBUG.md](llm/DEBUG.md) from this release:

> Help me continue Fronk setup. I am on the [head/node], running [OS].
> The procedure stopped at [step]. The error is: [paste the message].
> Read the debugging brief, explain the next check simply, and tell me which
> computer to use. Keep my existing configuration and verified weights.

That is enough to begin. The assistant can ask for a particular log or setting
when it needs one. If it can inspect your files or run commands, let it use the
existing manager; otherwise it should give you one action at a time. The briefs
are also installed under `~/fronk/src/docs/llm/` once installation has reached
that stage.

Remove pairing codes, keys and private prompts before sharing an excerpt. If
you prefer help through GitHub, open an **Installation help** issue using the
same information. [SHARE.md](SHARE.md) explains how to report a problem and
return with its solution.

## Find your step

Use this table when you want to investigate directly. A node service starting
successfully is one step; the head's demo, numerical check and finished chat
verify the rest of the path.

| Where it stops | What should happen | What to check first | Model brief |
|---|---|---|---|
| Open/build the manager | A native executable and a usable GPU head | Compiler/Python/Tk, architecture, build/GPU log | [Install](llm/INSTALL.md) |
| Connect / Verify | Every named node authenticates and has a working path | Names, running service, pairing, cables/addresses, launch permissions | [Network](llm/NETWORK.md), [protocol](llm/PROTOCOL.md) |
| Measure | Reuse saved costs or measure missing points once | Cache identity, memory available, active backend and width | [Calibration](llm/CALIBRATION.md) |
| Plan / synthetic demo | All 20 units have measured capacity; every participant replies | RAM/SSD split and per-node arrival time | [Calibration](llm/CALIBRATION.md) |
| Select/download the GGUF | Exact supported file, verified in place | File name, pinned size/hash, drive access and free disk space | [Weights](llm/MODEL.md) |
| Load | Missing units arrive intact and the selected backend loads them | Named node's disk/error, unit hash, backend/session log | [Weights](llm/MODEL.md) |
| Check | Distributed output agrees with the independent reference | Model/unit identity, backend, numerical difference | [Runtime](llm/RUNTIME.md) |
| Test / Chat | A complete response arrives through all required slices | Missing-reply mask, node log, dense engine log | [Runtime](llm/RUNTIME.md) |
| Chat works but is slow | Measure the actual limiting work | Context, accepted tokens, placement, compute versus arrival | [Performance](llm/PERFORMANCE.md) |

## Collect a small, useful report

Write down the release version, computer/role, operating system, current step
and exact error. Include the last successful step and whether the problem
started after an upgrade, restart, moved drive or changed cable. For a cluster
problem, list which computer connects to which port or switch.

On the affected node, `~/fronk/bin/fronk-manager --status` checks the installed
worker. Its log is normally `~/fronk/state/node.log`. On the head, expand **Raw
head output** in the console and copy the first error with a few lines before
it. The console's placement and report describe which machines actually took
part. `state/ds4.log` contains dense-engine messages. Installation logs include
`build.log`, `build-ds4.log`, `build-gpu.log` and `head-gpu.log`; the last two apply only to a Vulkan GPU head.
Use your chosen prefix if it differs from `~/fronk`.

Do not share `cluster.secret`, the pairing code, SSH keys or an unreviewed full
state directory. Redact private paths, addresses and prompts before posting a
public issue. A short excerpt and a physical topology sketch are usually more
useful than a large archive.

## Common cases

**Pairing succeeds but generation loses a Mac.** Authentication uses TCP;
expert replies use UDP. Inspect the node's reply error, route and local-network
permission. A process started through SSH may have a different permission
context from the authorized Terminal or installed service. Restart through the
intended user workflow and repeat the demo. Ping alone cannot validate this.

**Storage is rejected or a unit cannot be received.** Use a persistent disk.
Linux `/tmp` may be RAM-backed. The installer checks this and the node checks
space again before each unit. A unit is about 3.35 GB; receiving a replacement
also needs temporary space, because the old file remains until verification.
Move the installation through a deliberate reinstall, or free reviewed unused
data. Do not delete the active unit cache blindly or remove the original GGUF.

**Calibration appears again.** A new kernel/compiler, different workload width
or incompatible cache can require a measurement. Current short profiles cover
widths 1–4 together. Reconnect should reuse compatible profiles; do not select
“measure again” merely because discovery takes a few seconds. Current available
RAM and network paths are still checked. See [backend costs](../BACKENDS.md).

**The demo says the model does not fit.** Add capacity or free memory, then run
the demo again. A rate for only the placed blocks is not a complete-model rate.
Measured RAM + SSD capacity is different from unmeasured cold spill. Do not
interpret the estimate as a promise of real chat speed.

**A hash or numerical check fails.** Stop at that boundary. A similar filename
does not identify a compatible model. Recheck the exact file, transfer and
backend before changing arithmetic. Never relax a tolerance or forge a cache
stamp to get past the check.

## Know when the repair is complete

Repeat the failed step, then the next boundary it feeds. For a cluster repair,
require all intended nodes, a complete 20-unit plan, successful numerical checks
and an actual finished chat response. For maintenance, verify that the original
GGUF remains, the intended service is running, and the manager can reopen it.

If the cause remains unknown, keep the evidence and the working version. Record
what is proved, what is only suspected and the next useful check. When you find
the solution, add it to the issue with the step that now passes. A short note
about a cable, permission or unclear instruction can save the next person the
same investigation.

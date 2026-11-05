# Task brief: install or repair Fronk

> **For your frontier model.** People: read [the installation guide](../INSTALL.md) instead, and see
> [how to use this brief](README.md). Model: also follow [AGENTS.md](../../AGENTS.md).
> If you cannot open a linked file, ask for it; do not guess.

Prefer the terminal manager: it is the recommended path and the graphical manager
is experimental. Both use the same commands. Read
[INSTALL.md](../INSTALL.md), [NETWORK.md](../NETWORK.md), [MODEL.md](../MODEL.md)
and `tools/install/fronk-install.sh`, `onboard.sh`, `service.sh`, `model.conf`.
For a new physical cluster, start with [NETWORK.md](NETWORK.md) in this directory.

## Inspect before acting

Determine which computer you are on, its intended role, installation prefix,
release version and current manager step. Inspect saved configuration/status
without printing `cluster.secret`. Identify any running head/node and existing
weights. Preserve the functioning instance; use a separate prefix for tests.
Only claim access to machines you can actually inspect.

Begin with the role, OS, current step and error if there is one. Gather other
details when they affect the next action. Reuse the existing prefix, downloaded
bytes, verified weights and valid calibration. A failed setup step is not a
reason to purge the installation or repeat every benchmark.

Keep an action log with computer, step, command, result and next step. Ask for
missing choices as short options. For ordinary use prefer the manager's role,
node count, assigned-name and backend menus; GUI file/folder pickers avoid typing
model paths. Use explicit CLI flags for unattended, authorized work.

## Complete the existing workflow

1. Head: check compiler/Python/GPU prerequisites, install, choose the local
   expert backend and create a roster. Dense execution must pass the GPU probe.
2. Each node, in head order: install, select its assigned name, pair privately,
   select/calibrate its backend and start its service. Return to the head for
   Verify; the last check must include every intended node.
3. Head: run the synthetic demo. Read capacity, compute and communication costs.
   If the roster or paths fail, diagnose them before a large model download.
4. Select the exact file from a local folder/external SSD or download it from
   the pinned source in `model.conf`. Resolve identity against `MODEL.md`; do
   not pick a similarly named GGUF or reconstruct the URL from memory.
5. Check size and full SHA-256. Reuse only the installer's valid file-identity
   cache. Never forge a verification stamp to skip a failed hash. Preparation
   extracts 20 native units losslessly and transfers only missing verified units.
6. Start the head/console, review the complete plan, proceed, require the real
   generation check to pass, then verify an actual browser chat response.

Do not claim end-to-end success from a build, GPU self-test, synthetic demo or
HTTP 200 alone. Distinguish cached-weight testing from a fresh transfer and a
local model from a tested download. Record any unavailable platform separately.

## Diagnose the first failing boundary

| Symptom | First evidence to inspect |
|---|---|
| Manager does not open | Python/Tk availability; terminal `--help`; launcher error |
| Build/GPU probe fails | `state/build.log`, `build-ds4.log`, `head-gpu.log`; actual architecture/driver |
| Node not found | Service/status, matching private pairing, IPv4 route, local-network permission, chosen interface |
| Storage rejected or transfer cannot fit | Check the actual state filesystem and free bytes on the named node. Linux tmpfs/ramfs is not persistent cache storage. Use a disk-backed prefix; preserve original GGUFs and the working installation. |
| Pairing works but generation loses a Mac | Check UDP reply errors and launch context. An SSH-started process may lack the authorized Terminal/service's local-network access; restore through that context, then retest the complete session. |
| Long calibration | Cache/build key, worker `BENCH-PROFILE` capability, current width and memory; not an automatic refresh |
| Model unavailable | Exact path/size/hash, SSD connected, local macOS file permission |
| Unit/transfer failure | Expected digest, destination disk space, unfinished temporary transfer |
| Generation fails | All 20 units, node session/backend errors, missing replies, dense engine log |
| Slow chat | Placement and RAM/SSD split, per-node compute/arrival, head cost and context; not just nominal link speed |

Gather the smallest relevant log excerpt, redact secrets/private prompts and
paths when sharing, and change one cause at a time. Never bypass authentication,
hash checks, numerical tests, all-node completion or the download demo gate to
make a setup screen green. Do not change power/thermal protections or delete
weights to mask a failure.

## Drive the installation over SSH

Use the terminal manager with explicit flags. SSH is optional: use it only on hosts
the person named and already opened or approved, and never ask for passwords or keys.
Name the computer for each command, and ask before changing a machine.

- Node install: `ssh -t HOST 'bash fronk-manager.sh --terminal --role node --name NAME'`,
  so the person types the pairing code into the private prompt. Never read, print or
  pass the code as an argument.
- Mac nodes must run as the installed LaunchAgent, not `--no-service` or `nohup` from
  SSH: a process left behind by a closed SSH session can fail with `EPIPE` or
  `No route to host` on replies. Ask the person to accept Local Network on that screen.
- Runtime traffic uses Fronk's own protocols, so the cluster must not depend on SSH.
- Return to the head and run Verify after each node, then continue the workflow above.

## Maintenance and recovery

Use manager upgrade/uninstall actions. Stop the owned head before changing its
installation; honor its lock. Explain keep-state versus `--purge`: purging removes
managed state/pairing and requires re-pairing, while model files are preserved.
Do not remove another installation's service or an arbitrary user folder. Test
fresh reinstall in an isolated prefix before proposing changes to working state.

Finish with what is running on which computer, verified model identity, chosen
paths/placement, commands to reopen chat or resume setup, tests and remaining
blockers. If a full chat was not tested, state that plainly.
If the user wants to share the result or a solution, follow [REPORT.md](REPORT.md)
and prepare a small draft they can review.

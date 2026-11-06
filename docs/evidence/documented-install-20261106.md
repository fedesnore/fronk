# Documented manager installation on a mixed cluster, 2026-11-06

This audit separates the published manager from the new SSH wrapper and from
machine administration. It uses the official v0.3.0 portable manager, downloaded
from the GitHub release, with SHA-256
`633eb2a73b084b10954af6e3598dc384574c782021b12d46c6bbfbcc6c746e05`.
Its checksum matched that release's `SHA256SUMS`. No replacement dense runtime,
CPU parking override, skipped calibration or bypassed check was used.

## Machines and paths

| Role | Machine | OS | Runtime/backend | Connection to head |
|---|---|---|---|---|
| head | Intel i5-7600K, 48 GiB, Radeon Pro 580 8 GiB | macOS 13.7.8 | Packaged Vulkan/MoltenVK; AVX2 local experts | Head Ethernet at 1 Gb/s |
| node-1 | M4 Pro, 24 GiB | macOS 26.6.2 | Metal | Direct Thunderbolt |
| node-2 | M1 Pro, 16 GiB | macOS 15.7.1 | Metal; RAM/SSD placement varies with available memory | 1 Gb/s Ethernet switch |
| node-3 | i5-10210U, 16 GiB | Debian 13 | AVX2 | Same 1 Gb/s Ethernet switch |

The manager selected actual paths and capacity. Hardware names, addresses and
unit placement were not encoded in the wrapper. The model remained on the
head's attached SSD. The head used a new dedicated installation prefix; nodes
used `~/fronk-fresh`. Existing node caches were retained as described below.
An initial cache-reuse attempt was stopped before qualification. At the user's
request, every managed unit cache was then emptied, including the head's partial
output. The definitive run repeats the documented synthetic demo before slicing
the GGUF on the external SSD and transferring fresh units to every node. It is
a fresh head installation and node reinstall with empty unit caches; compatible
calibration and pairing are retained. Model download, a Linux GPU head and a
five-node switch were not tested.

## Commands and ownership

Below, `H` means the head's absolute installation prefix, `N` the absolute
prefix on the named node. Paths and SSH destinations are normalized to remove
personal information. Each command is executed on the indicated machine.

| Machine | Command | Source/purpose |
|---|---|---|
| Head | `curl -fLO https://github.com/fedesnore/fronk/releases/download/v0.3.0/fronk-manager.sh` | Published setup brief |
| Head | `curl -fLO https://github.com/fedesnore/fronk/releases/download/v0.3.0/SHA256SUMS` | Published setup brief |
| Head | `grep ' fronk-manager.sh$' SHA256SUMS \| shasum -a 256 -c` | Published integrity check |
| Head | `bash fronk-manager.sh --help` | Verify available manager options |
| Head | `bash fronk-manager.sh --yes --role head --prefix "$H"` | Published head install, custom prefix |
| Each node, via head SSH | `bash ~/fronk-manager.sh --yes --role node --name NAME --prefix "$N" --secret-stdin` | Published node install; stdin is piped from head manager `--pairing-code` |
| Each node | `"$N/bin/fronk-manager" --prefix "$N" --status` | Verify owned service and authenticated identity |
| Head | `"$H/bin/fronk-manager" --join-nodes --yes --nodes 'node-1 node-2 node-3'` | Published simultaneous verification and demo |
| Head | `"$H/bin/fronk-manager" --configure-model --yes --model /attached-ssd/Qwen3.8-Flash-Next-Q4.gguf` | Published exact local-model verification |
| Head | `"$H/bin/fronk-manager" --run` | Published start; wrapper removes controlling terminal using Python |

The wrapper also checks SSH prerequisites, copies the manager with `scp`, and
checks its SHA-256 using Python's standard library. It does not introduce a
different install/calibration/discovery/transfer path. It was first exercised
with `--through install`, then `--resume --through model`. All three nodes
started through the normal LaunchAgent/systemd user service and authenticated.
The initial head calibration and compatible subsequent cached calibration were
both observed. SSH access used existing authenticated master connections; no
password was embedded in the script or pairing code saved to a log.

After emptying all unit caches, the recorded sequence was `--resume --through
demo`, followed by `--resume --model /attached-ssd/Qwen3.8-Flash-Next-Q4.gguf`.
The latter repeats the demo before starting. An already-configured model is
also revalidated by the manager after `--join-nodes`; that does not extract or
transfer units. Extraction and transfer happen after the real plan is accepted.

## Prerequisites and maintenance outside the Fronk commands

- **M1 Ethernet addressing:** the connected adapter had no suitable address on
  the isolated switch. A temporary alias did not survive a reboot. With the
  user's authorization, `sudo networksetup -setmanual "USB 10/100/1000 LAN"
  NODE_ADDRESS 255.255.255.0 0.0.0.0` set a persistent address while keeping Wi-Fi
  available. `networksetup -getinfo "USB 10/100/1000 LAN"` and SSH on Ethernet
  verified the setting after the user's reboot. The previous DHCP setting can
  be restored with `sudo networksetup -setdhcp "USB 10/100/1000 LAN"`.
- **macOS permissions:** the user enabled Local Network access on the M4 Pro.
  The setup script cannot accept this prompt remotely. macOS users were logged
  in for their LaunchAgents.
- **Old installations:** the user explicitly requested their removal. Owned
  old services were stopped through their managers before migration. Explicitly
  inventoried old build/test directories were removed; original model files
  were protected. Retained native units were hard-linked into the final managed
  state, checking expected size and recorded digest against the manifest, then
  leaving normal runtime verification enabled. This preliminary metadata check
  is not a substitute for hashing the actual bytes. No stamp was fabricated.
  Cleanup reclaimed approximately 1.79 GB on M4, 3.58 GB on M1 and 23.78 GB on
  Shuttle (decimal disk-free deltas, before the new installs). This cleanup was
  additional authorized administration, not a feature of the published guide
  or of the new wrapper.
- **Empty-cache qualification:** after stopping head and services, all generated
  `unit-00` through `unit-19` files (`.bin`, `.bin.tmp`, `.sha256`, `.ok`) were
  removed from the four final state directories. Additional inventoried old
  Fronk backups, containing no GGUF files, were removed from the nodes. The
  original GGUF, its valid identity check, pairing and calibration were retained.
  All four state directories had zero unit files before the next synthetic
  demo. Services were restarted using their installed managers' `--run` action.

## Guide gaps and observed behavior

| Finding | Evidence and documentation change |
|---|---|
| No single SSH orchestration script | The guide contained individual unattended commands. The new wrapper sequences those commands and retains each check; it is not present in the published v0.3.0 download. |
| A second prefix does not isolate a node service | The service name is fixed per user. The revised guide describes ownership and migration rather than recommending a second node prefix without qualification. |
| `nohup` is not enough to select automatic console startup | Manager `--run` checks `/dev/tty`. A shell background process can retain it. The wrapper uses `start_new_session=True`, then waits for generation and API checks. |
| Persistent network configuration was not shown as a macOS CLI recipe | The revised network guide adds the service-level command, inspection and rollback; Fronk itself still assigns no IP addresses. |
| Intel Radeon head did not need a manual GPU build | The official package passed its Q8_0 GPU probe on Radeon Pro 580 and built the AVX2 head. This is evidence for this machine, not general Linux GPU qualification. |
| M1 multicast membership refresh failed | Node log reported `Cannot allocate memory`. Earlier connection checks used unicast fallback; the definitive fresh run selected multicast on the shared Ethernet interface and later missed the M1 at self-test layer 0. Reconnection subsequently reported no multicast reply from M1, selected unicast and passed the numerical/generation checks. This is consistent with a multicast-path problem but does not establish its root cause; no OS protection was disabled. |
| A path-probe reply was missing | One connection pass recorded 1 missing reply out of 1,000 M1 probes at 5,312 bytes; its other two probe sizes recorded none. This is distinct from synthetic barrier completion and rules out a blanket claim of zero network loss. |
| Chat context selector and active size disagree | On the unmodified release, Options displayed 8K while the completed chat reported 4096 active tokens. No context change was applied; this is a UI discrepancy, not a verified 8K run. |
| Protocol guide said no per-layer retransmission | Code already retries a whole incomplete generation expert call once (500 ms then 1 s, new step ID). Revised wording distinguishes that from absent UDP fragment retransmission. |

## Verification record

- Official manager checksum, head GPU probe, normal calibration and all three
  authenticated service starts: passed.
- Simultaneous synthetic demo: 40 passes × 48 layers, every node answered,
  zero missing replies; all 20 units fit (18 resident, 2 SSD on M1 in this demo).
  Its 2.3 tok/s estimate is synthetic, not a generated-model measurement.
- Complete model hash: passed. Live generation/API/browser checks passed
  after recovery from the initial self-test failure described below.
  Model size: 177,280,286,720 bytes; SHA-256:
  `680944460a8cbe93ba8b6d7b6107213ffb7e22320bd913000e563ca0a0f25a8a`.
- Repeated synthetic demo after emptying caches: all 20 blocks fit; zero
  missing barrier replies; estimate 4.6 tok/s. Memory availability changed
  placement between runs; these estimates are not a performance comparison.
- Fresh extraction and transfer: all 20 required units were regenerated from
  the attached-SSD GGUF, with runtime SHA-256 verification. Initial placement
  was M1: 3, Shuttle: 4, M4: 5, head: 8, all resident. Preparation took
  1008.9 seconds. The subsequent check failed with `SELFTEST warm-up failed at
  layer 0; check compute and replies on node-2`. This is a failed end-to-end
  run, despite successful installation, demo and transfer. The wrapper exited
  with status 1 and preserved services for diagnosis. Its displayed failure
  text came from the console and misleadingly selected the informational AVX2
  backend line; the raw log identifies the actual self-test failure. The wrapper
  now prefers the self-test error when the console provides one.
- Recovery uses the console's normal **modified** (`m`) action. Available-memory
  planning selected 2 units on M1 and 9 locally, requiring fresh local extraction
  of unit 02. No placement, timeout or numerical tolerance was overridden.
  CONNECTION reported no multicast reply from M1 and automatically selected
  unicast. The numerical comparison passed with relative L2 error 0.00672.
  Real generation completed 4058 expert calls with complete sums, zero missing
  replies and zero retries; wall time was 75.313 seconds. This qualifies the
  recovered path, not the initial multicast path. The unmodified installed
  v0.3.0 `tools/api-check.py` then passed model discovery,
  JSON completion, SSE completion plus usage, and malformed-request HTTP 400.
  Both replies finished normally, with 44 completion tokens, including
  reasoning: the old `think:false` field did not disable it on llama.cpp.
  The paired API guides now show the llama.cpp template field explicitly.
  Browser chat then completed a short Italian answer with thinking disabled;
  the UI reported 26 tokens and a completed turn. All three SSH master
  connections were then closed. After 61 seconds of inactivity, the checkout's
  API checker also passed JSON, complete SSE with usage, and HTTP 400 handling.
  The final non-streaming response ended with `finish_reason: stop`.
- Final state: head running in REPORT/ready, all three node dataplanes active.
  Each node reported 20,278 requests, zero invalid frames, zero duplicates and
  zero compute errors for the recovered session. These counters do not prove
  zero loss in every network probe or repair the initial multicast failure.
  The cluster was left running; SSH is not required for its inference traffic.
- Final free storage after installing fresh slices: M4 15.97 GB, M1 22.59 GB,
  Shuttle 43.12 GB (decimal). Old inventoried Fronk installs/backups were removed;
  the active installation and downloaded manager remain. A small model directory
  on M4 was left intact. Final M1 cache includes fresh unit 02 from the initial
  placement even though recovery moved its active ownership to the head.
- During development, editing the wrapper while Bash was still reading it
  caused an EOF parse error after successful model verification. This was an
  audit execution mistake, not a failure of the release installer. The stable
  wrapper passes `bash -n`; rerunning reused the valid model identity check.
- The wrapper rejects a mismatched manager checksum and duplicate roster names
  before SSH or installation. Public documentation links pass their checker.

Private raw logs, personal SSH configuration, pairing state, model bytes and
cleanup inventory are deliberately excluded from this public report.

# Installer and console follow-up, 2026-11-06

Base: public commit `60bb08b`. These changes contain installation orchestration,
console fixes, regression tests and paired guides. No dense engine, expert
kernel, transport, GPU runtime or DwarfStar experiment is changed. The console
fixes land in their own pull request.

The [earlier audit](documented-install-20261106.md) records the **published
v0.3.0** manager on the physical cluster, including the synthetic demo, fresh
slices cut from the external SSD and complete generation. It remains historical
evidence; these fixes were not inside that downloaded asset.

## Differences found and addressed

| Boundary | Previous behavior / missing instruction | Correction / files |
|---|---|---|
| Mac node installation | SSH could work without the graphical login needed by a LaunchAgent; installation reached a late service failure | `tools/install/fronk-install.sh` checks the GUI session before building; paired installation guides explain the distinction |
| Pause after pairing | `--demo-later` could continue into model setup when a previous demo stamp existed | `tools/install/onboard.sh` honors the explicit pause even on resume; regression in `tests/installer-choices.sh` |
| SSH orchestration | Manual copying/pairing, SSH identities and service prerequisites were scattered across expert and LLM guides | `setup-cluster.sh` calls the existing manager; `ssh-prerequisites.sh` checks every node before installation; both guides show the same staged commands and SSH config |
| Portable manager | Copying the source entry point alone leaves its dependencies behind | Wrapper checks the checksum and portable payload marker before SSH; docs distinguish source, manager payload and release |
| Console start | Omitting `head_blocks` erased saved placement and could cause new slicing | Preserve on omission; explicit empty requests automatic placement; reject invalid values with HTTP 400 |
| Console settings | Invalid values could be silently ignored or saved as `None`; writes truncated config in place | Validate the complete update, reject invalid input, replace the config atomically, serialize lifecycle edits, reject settings changes while a head runs |
| Context | Installer saved 4096 but the menu started at 8192; missing Vulkan context file was reported as 32768 | Include 4096; use Vulkan launcher's 8192 fallback, keeping native default 32768; distinguish saved from active context |
| Failure report | Backend startup or a normal unicast fallback could obscure a SELFTEST failure | Report the concrete SELFTEST failure; do not mark normal fallback as an inference failure |
| API check | Native `think: false` alone did not disable reasoning in llama.cpp, exhausting a short answer budget | Also send `chat_template_kwargs.enable_thinking: false` in `tools/api-check.py` |

Main human instructions: `docs/INSTALL.md`, `docs/TROUBLESHOOTING.md`,
`docs/NETWORK.md`, `docs/PROTOCOL.md`, `docs/API.md`. Agent counterparts:
`docs/llm/INSTALL.md`, `SETUP.md`, `DEBUG.md`, `NETWORK.md`, `PROTOCOL.md`, `API.md`.

## Commands used for source verification

Run from the clean checkout on the head. Test services use isolated ports and
temporary installation prefixes; they do not pair with the physical cluster.

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
bash -n tools/install/fronk-install.sh tools/install/setup-cluster.sh tools/install/ssh-prerequisites.sh
python3 tools/check-docs.py
git diff --check
make DS4=
make DS4= test-core
make DS4= test-plane-cluster
FRONK_TEST_MANAGER="$PWD/tools/install/fronk-install.sh" bash tests/installer.sh
python3 tools/pr-scope.py 60bb08b
```

The Python suite passed 37 tests, including 18 new console/SSH tests. The console
suite was also run against the original source and reproduced the failures.
Core tests passed; plane-cluster passed 27 checks. The source manager passed
install, authenticated head handoff, absent-peer rejection, download gate,
upgrade, uninstall and fresh reinstall. Missing GUI/systemd sessions are tested
with controlled command fixtures, not by logging out a live physical node.

## Physical-cluster commands and deployment boundary

Commands below use normalized paths/hosts. Authentication used temporary SSH
master connections; no passwords, private keys or pairing codes were recorded.

| Machine | Command/action | Result |
|---|---|---|
| M4 node-1 | `ssh -T HOST '~/fronk-fresh/bin/fronk-manager --status'` | Existing service authenticated |
| M1 node-2 | Same command, using its known Ethernet host key/address | Existing service authenticated |
| Linux node-3 | Same command | Existing service authenticated |
| Head | Wrapper command below | Checksum and all three node prerequisite checks passed; no installs or restarts |
| Head | Back up changed installed scripts/docs and package manifest; copy the tested files; preserve binaries and config | Development update, explicitly not a new downloaded-manager qualification |
| Head | Console `POST /stop`; terminate only the verified installed console PID; restart `python3 tools/fronk-web.py` | Console updated with rollback copies retained locally |
| Head | Console `POST /start` with `{"refresh":false}` | Saved `head-blocks 9` preserved, confirmed by unchanged configuration hash |
| Head | `ssh -F SSH_CONFIG -O exit HOST` for each test master | SSH sessions closed; node services kept running |

```sh
# Read-only preflight against already-installed nodes; no installation phase runs.
bash tools/install/setup-cluster.sh \
  --manager DOWNLOAD/fronk-manager.sh --checksums DOWNLOAD/SHA256SUMS \
  --head-prefix HEAD_PREFIX --nodes NODES_FILE --ssh-config SSH_CONFIG \
  --resume --through install
```

The running coordinator/GPU binary hashes were unchanged. Existing local CPU
parking settings were retained operationally; they are outside these changes. Node
binaries and services were not replaced. On reconnect, available node RAM changed
the automatic placement: head 9, M1 3, Linux 4 and M4 4 blocks (one M4 block using
its measured SSD profile). The head prepared the missing local unit from the
original SSD model; verified cached units were retained.

After the restart, SELFTEST passed with relative L2 `0.00673`. The initial
generation report completed 4106 barriers with zero deadline misses, HTTP 200
and 64 generated tokens. READY took 418.4 seconds including preparing the
missing unit. Its initial rate was 3.33 tok/s; these changes make no
throughput claim. Available RAM and placement differed from the earlier run.

`python3 tools/api-check.py --output LOCAL_RESULT.json` passed model discovery,
non-streaming completion, SSE with usage and malformed-message rejection.
The JSON completion reported 4.37 tok/s after startup. After about 72 seconds
idle and with SSH masters closed,
the console's authenticated `POST /chat` returned a complete Italian answer
about an Ethernet switch, `[DONE]`, and usage for 26 completion / 27 prompt
tokens in 11.23 seconds including prompt processing (4.76 decode tok/s reported
by the engine). The console served `/chat`; its context endpoint returned
current 4096 with a selectable 4096 entry. Config and runtime hashes stayed
unchanged through the update and omitted-placement start.

The source export also passed its package-manifest and documentation checks.
It includes both SSH helpers and the regression tests without local state.

## Changes made while splitting for review

The installer and console changes were then separated into two pull requests,
which can land in either order. Two small edits followed the runs above. The
console's failure field again shows the generic cycle message when nothing more
specific was logged, with a new regression test. The wrapper's final stage now
ignores a comment on a `fronk.conf` line, as its earlier port check already did.

On an M4 Pro, each branch and their merge then passed the Python suite and the
documentation check, and the installer branch passed the installer lifecycle
test. The physical cluster was not rerun after these two edits.

## Limits

- A new packaged release with these fixes has not been built and installed
  end-to-end on all physical machines. Source installer tests and a live console
  update are separate from that release qualification.
- The original multicast-refresh problem is not fixed here. The M1 uses the
  normal unicast path. Recovery after a long extraction is documented; no checks
  or complete-sum requirements were relaxed.
- A Linux GPU head, five-node switch, model download and GUI installer wizard
  were not qualified in this follow-up.
- Browser automation was unavailable. Live console endpoints and the chat
  streaming route are checked separately from a visual browser inspection.

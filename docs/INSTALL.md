# Install a Fronk cluster

Begin on the **head**, which runs the dense part of the model on its GPU and
coordinates chat. Keep its manager open while visiting the nodes. A **node**
holds and computes expert slices through its CPU or GPU backend; it receives
only its own weights. No SSH access or remote shell is required.

For guided help from a frontier model, give it the [installation/repair brief](llm/INSTALL.md).
If you first need help connecting the computers, use the
[complete network/setup brief](llm/NETWORK.md). Both follow this manager's flow.
If a step stops, use [continue from where you stopped](TROUBLESHOOTING.md#continue-from-where-you-stopped).
Keep the existing installation folder so verified files and calibration can
be reused.

## What to have ready

- [ ] **One head.** Tested: Apple Silicon with Metal. Other GPUs are experimental.
- [ ] **One or more nodes,** running macOS or Linux. Windows is not supported.
- [ ] **Memory:** about 67 GB available across the machines, plus about 10 GB the
      head keeps for itself. Close memory-heavy applications first.
- [ ] **Disk on the head:** 177.28 GB for the model and up to 66.94 GB for prepared
      slices, on an SSD. The model can stay on an external drive; keep it attached.
- [ ] **Disk on each node:** 3.35 GB per slice it receives, plus 1 GiB. Use a
      persistent disk: Linux `/tmp` may be held in RAM.
- [ ] **Cables connected,** and every machine has an address on the same trusted
      network. If this is unfamiliar, follow [Connect the head and nodes](NETWORK.md).
- [ ] **Machines awake and on power** for the whole setup.
- [ ] **Compiler tools** installed on every machine (see the table below).

Things to keep in mind:

- **Start on the head and keep its manager open.** It tells you when to go to each
  node and when to come back.
- **You do not need the model yet.** The demo runs first, and it reports whether
  the model fits before you download 177 GB.
- **Allow time.** The first run can include the download, one calibration per
  machine, a full hash of the model and sending each node its slices. Later starts
  reuse all of this.
- **Use a network you trust.** Cluster traffic is authenticated but not encrypted.
  Do not expose its ports to the Internet, and keep the pairing code private.
- **Allow local-network access when macOS asks.** A node without it may pair and
  still be unable to reply. [Network guide](NETWORK.md#macos-configure-each-end).
- **Fronk changes no system settings.** It does not configure cables, assign IP
  addresses, disable firewalls or install GPU drivers.

## Before starting

| Machine | Requirements |
|---|---|
| Tested head: Apple Silicon / Metal | macOS, working Metal GPU, Xcode command line tools, Python 3.9 or newer |
| Experimental head: Intel Mac | macOS, compatible GPU passing the bundled Vulkan probe, Xcode tools, Python 3.9 or newer |
| Experimental head: Linux GPU | C compiler/make, Python 3.9 or newer, Vulkan driver; source GPU builds also need CMake, Git, glslc and Vulkan/SPIR-V headers |
| macOS node | Xcode command line tools, Python 3.9 or newer; Metal on Apple Silicon or a compatible CPU |
| Linux node | C compiler and make, Python 3.9 or newer; supported x86/ARM CPU or a custom backend |
| Graphical installer | Python 3.9 or newer with Tk and a graphical login session |

On macOS, `xcode-select --install` installs the compiler tools. The Python
installer from python.org includes Tk; with Homebrew, install the matching
Python/Tk packages. On Debian/Ubuntu, `build-essential python3 python3-tk curl
openssl` covers the ordinary node/manager prerequisites. GPU driver and build
packages depend on the distribution. The terminal manager does not require Tk.

An expert unit is **3,347,005,440 bytes**, and there are 20 of them. Each needs
resident memory or measured SSD capacity. The head's default memory reserve for
the dense model and context is 10 GB. The demo uses current available memory, not
the sticker on the machine. Do not assume the entire model must fit in the head RAM.

The manager rejects Linux `tmpfs`/`ramfs` prefixes and checks free disk space
before building. Each transfer checks space again and reports the affected node
and missing capacity. A service being authenticated means it has started, not
that the full cluster is ready: finish the head's demo and generation check.

The [protocol guide](PROTOCOL.md) explains control/data traffic and default ports.

## 1. Open the manager on the head

Download `fronk-manager-macos.zip`, `fronk-manager-linux.tar.gz`, or the portable
`fronk-manager.sh` from the project's release assets. Check `SHA256SUMS` from
the same release:

```sh
shasum -a 256 fronk-manager-macos.zip      # macOS
sha256sum fronk-manager-linux.tar.gz      # Linux
```

Compare the result with the release checksum. The macOS app is not notarized;
if macOS blocks it, use the per-app **Open Anyway** action in Privacy & Security
only after checking the source/checksum. Do not disable Gatekeeper globally.

Extract the graphical archive and open the manager. The terminal equivalent is:

```sh
bash fronk-manager.sh --terminal
```

Choose **head**. Installation is per-user in `~/fronk`; a dedicated custom
absolute prefix is supported with `--prefix`. Prefixes cannot contain spaces.
The model itself may live in a path containing spaces, including an external
drive. Install on the local machine as your normal user, not with `sudo`.

The manager builds Fronk and checks the head GPU. Apple Silicon uses the Metal
engine. Other heads use the Vulkan adapter, which requires at least 5.40 GB of
free GPU memory for dense/output weights, plus context/scratch memory. The probe
is a prerequisite, not a guarantee that every GPU/driver can run this model.
If it fails, that computer can still be used as a supported CPU node.

Choose the backend for the head's **local expert slices**. This does not change
the dense GPU engine. Accept the recommendation unless you have measured a
better backend. Calibration is saved for later connections.

The advanced `--no-measure` option skips setup calibration. Missing RAM profiles
are measured on the first connection, but Metal RAM + SSD capacity needs
`~/fronk/src/fronk backends --measure --config ~/fronk/src/fronk.conf` before
starting the program. Keep normal calibration enabled for a first installation.

For a custom expert kernel, choose **add-module** and select its source folder.
It is checked before installation and added to the compatible backend choices.
Most users should keep **included**. See [BACKEND_MODULES.md](BACKEND_MODULES.md).

## 2. Visit each node when the head asks

On the head, choose the number of node computers (excluding the head). The
manager assigns `node-1`, `node-2`, and so on. Choose **custom-names** only if
you want your own names; **keep-current** retains an existing roster. The head
displays a private pairing code and tells you which machine to visit. The code
includes the workload width used for calibration. The terminal uses numbered
menus; the GUI uses selection lists.

On the named node:

1. Open the same downloaded manager and choose **node**.
2. Select the name displayed on the head; use **custom-name** for a custom roster.
3. Paste the pairing code into the private prompt.
4. Choose/accept a backend. Wait for its self-test and short saved calibration.
5. Start the node service, allow local-network access, and wait for the
   **service started and authenticated** message.

Terminal equivalent, with interactive private pairing:

```sh
bash fronk-manager.sh --terminal --role node --name node-1
```

Then **return to the head** and choose **Verify**. Repeat for the next named
machine. The last verification checks all planned nodes in one authenticated
pass, so a previously connected node cannot silently disappear from setup.

macOS uses a per-user LaunchAgent. Linux uses a systemd user service. On a
Linux machine without a systemd user session, use `--no-service`, then run
`~/fronk/bin/fronk-node` and leave that terminal open. For unattended nodes,
`--secret-stdin` accepts the pairing code over standard input; do not put the
secret in command-line arguments or public logs.

## 3. Try the synthetic cluster demo

Stay on the head and keep the nodes running. Choose **demo**. It uses synthetic
units and real computation/network traffic; it does not assign or transfer the
real model slices yet. Read:

- How many of the 20 units fit, including explicitly measured SSD capacity.
- Per-machine compute and communication time, and the slowest reply.
- Estimated throughput and whether the estimate covers all required units.

The demo first warms each layer and request width, then times the work with the
same 500 ms reply limit used by ordinary chat. Slow replies count toward the
measured time. A missing reply fails the demo; no partial result is accepted.

These are estimates, not generated-model tokens/s. If capacity is insufficient,
add nodes or free memory before getting the weights. The report is saved under
`~/fronk/state/install-demo.log`. Compatible compute calibration is reused;
current memory and network paths are checked again. Choose **later** to pause;
reopen the head manager and choose **join-nodes** to resume.

## 4. Select the exact model on the head

Choose **local** to select a file, a folder containing it, or an external SSD.
Choose **download** to use the pinned source. The file must be exactly
`Qwen3.8-Flash-Next-Q4.gguf`, as identified in [MODEL.md](MODEL.md). A Q2 GGUF,
a different conversion or another model with a similar name is not compatible.

The manager checks size and SHA-256, verifies in place, and prepares the unit
manifest. It does not copy an already-downloaded GGUF. The first full hash can
take several minutes; later checks reuse it only if the file identity and
metadata are unchanged. Keep the drive attached. On macOS, grant the manager
access to removable volumes; permission granted to a Terminal session does not
necessarily apply to a different process.

A download can resume. Select the same download folder on retry. If verification
fails, do not bypass it: compare with the pinned model identity and replace the
bad file. Original model weights are separately licensed and not bundled.

## 5. Start, review, chat

On the head, choose **start**. In the Run view, review the actual placement and
choose **Proceed**. Fronk transfers missing units, checks their hashes, starts
sessions and performs a real generation check. Leave the nodes running.

The console is at `http://127.0.0.1:7730` on the head with the default ports.
The local model API is at port 7710. Use the chat link in the console. Both GUI
and terminal starts use the ordinary runtime defaults in `src/runtime.c`.

Installed commands:

```sh
~/fronk/bin/fronk-console       # browser dashboard and chat
~/fronk/bin/fronk-head          # terminal head; same configuration/engine
~/fronk/bin/fronk-manager       # continue setup or maintain this machine
```

Use one head per installation/cluster. The console can follow a head started
from the terminal; the state-directory lock prevents a second head from resetting
its sessions. Reconnect reuses verified unit files and compatible measurements.

## Upgrade, uninstall and start fresh

Stop the head before changing its installation. On each machine, reopen the
**downloaded** manager and choose **upgrade**. Keep the downloaded manager:
uninstall removes the installed launcher itself.

- **keep-state** removes programs while preserving identity, pairing secret,
  configuration and unit cache for reinstall.
- **fresh-install** also removes managed identity/secret/calibration/unit state.
  Original GGUF files and unrelated files are preserved. A new head creates a
  new pairing code, so install the new head first and pair each node again.

Terminal equivalents:

```sh
~/fronk/bin/fronk-uninstall                  # prompts for keep/fresh
bash fronk-manager.sh --uninstall --purge    # explicit fresh state, with confirmation
bash fronk-manager.sh --role head           # begin the new cluster on its head
```

For a custom prefix, pass the same `--prefix` on maintenance commands. Do not
remove the directory manually if it contains a model or other user files.

## If a step fails

[Troubleshooting](TROUBLESHOOTING.md) follows each stage from installation to a
finished response. If you use an assistant, give it [docs/llm/DEBUG.md](llm/DEBUG.md)
and the exact error. These guides are installed under `~/fronk/src/docs/`
(or your chosen prefix); they do not require a separate download.

| Symptom | Where to act |
|---|---|
| Node absent in Verify | On that node: check name, pairing, running service, cable/IP and local-network permission; retry on head |
| GPU probe fails | On that machine: check the driver/device and free GPU memory; use a supported node role if necessary |
| GUI cannot open | Install Python/Tk or run `--terminal` |
| Insufficient capacity | Free RAM or add nodes; rerun the head demo; never count swap as resident RAM |
| No compatible SSD profile | Stop Fronk on that machine and run `fronk backends --measure --config ~/fronk/src/fronk.conf` |
| External disk cannot be read | On head: keep it mounted and grant the launching application access |
| Build fails | Read `~/fronk/state/build.log`, `build-ds4.log`, `build-gpu.log` or `head-gpu.log` |
| Generation fails | Keep all nodes online; inspect head `ds4.log`, console report and node `node.log`; do not accept missing slices |

Use `fronk-cli --config PATH INFO` against a node for authenticated diagnostics.
The control/data transport is intended for trusted LANs and is not encrypted.
Do not forward its ports to the public Internet.

Once you have tried it, [share your cluster or the problem you met](SHARE.md).
If a change to these instructions helped, include it so the guide can improve.

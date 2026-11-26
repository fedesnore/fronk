# Node sources of the November 2026 evaluation

The tree tagged 0.7.0 is the head's: the program and the engine that ran on the
head during the evaluation of 15–16 November 2026. The nodes did not run a
build of that exact tree. Each node had its own copy of the source and built
its program from it on 11 November, a few hours around the head's last build. This page records what each node ran, as a patch against the head's tree.

| Node | Program built | Patch | Files | What is different |
|---|---|---|---:|---|
| HP laptop (Linux) | 2026-11-11 18:49 | [hp.patch](node-sources-20261116/hp.patch) | 15 | An earlier revision of the transport and of the GSQ kernels; no expert-cache code |
| Shuttle (Linux) | 2026-11-11 18:49 | [shuttle.patch](node-sources-20261116/shuttle.patch) | 15 | As the HP, plus two environment switches for idle worker threads in `backends/gsq.c`, unset by default |
| iMac (macOS) | 2026-11-11 18:49 | [imac.patch](node-sources-20261116/imac.patch) | 32 | As the HP, plus changes to configuration, hardware detection and benchmarking, and a backend module for its Radeon GPU, present in the tree while the node was configured to use the `gsq` backend |
| MacBook Pro M1 Pro | 2026-11-11 21:27 | none needed | 0 | Built from a copy whose `Makefile`, `src/` and `backends/` are identical, file by file, to the head's tree |

The patches cover the inputs of the node program: `Makefile`, `src/` and
`backends/`. The node copies also held older versions of head-only files (the
engine patch, the launcher, the console), which a node never runs; those are
left out.

## How to obtain a node's source

The patches are relative to commit `54b3261`, the first commit of the 0.7.0
change, which is the head's tree as it ran:

```sh
git checkout 54b3261
patch -p1 < docs/evidence/node-sources-20261116/hp.patch
```

Each patch was checked this way: applied to that commit, it gives files
identical to the node's.

## What this means

Head and nodes spoke the same protocol with different revisions of the code:
the nodes had the layer call as it was on 10–11 November, the head had the
later changes to it and the expert-cache code. The head serves only after its
start-up test, which compares a layer computed by the nodes with an oracle from
the model file, so that test passed with this mix whenever the cluster ran.

So the tag reproduces the head and the engine exactly, and each node within
the patch listed above. A cluster built entirely from the tagged tree is not
the cluster that was measured; it has not been run.

## Provenance

The node sources come from a backup made on 14 November 2026, one archive per
machine, each matching the SHA-256 recorded with the backup. In every archive
no source file is newer than the program binary beside it. Checked again on the
machines on 26 November 2026:

- iMac: the binary in the backup is identical to the one its node service
  still runs (SHA-256 `25c36fda…`).
- HP: the binary in the backup (`911af639…`) is identical to the copies the HP
  kept of the program it ran until 18 November.
- M1 Pro: its sources were read on the machine and compared by hash with the
  head's tree; the program built from them (`8e597575…`) is the copy the M1
  kept of what it ran until 18 November.
- Shuttle: the machine has since left the cluster, so its binary
  (`b3bbe3f7…`) is known from the backup only.

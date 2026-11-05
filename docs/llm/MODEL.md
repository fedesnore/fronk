# Task brief: model identity, slices and transfer failures

> **For your frontier model.** People: read [the model guide](../MODEL.md) instead, and see
> [how to use this brief](README.md). Model: also follow [AGENTS.md](../../AGENTS.md).
> If you cannot open a linked file, ask for it; do not guess.

Read [MODEL.md](../MODEL.md), [PROTOCOL.md](../PROTOCOL.md) and
[troubleshooting](../TROUBLESHOOTING.md). Inspect `tools/install/model.conf`,
`model-units.sha256`, `verify-model.py`, `src/model.h`, `native.h`, `prep.c`,
`units.c`, the transfer code in `head.c` and `RECV` in `node.c`.

The supported file is the exact pinned `Qwen3.8-Flash-Next-Q4.gguf`. Resolve its
URL, size and digest from this release; do not rely on a filename or memory.
The head can read it in place from an external SSD. Nodes need only their units.
The manager downloads only after its synthetic demo; explicit local verification
can be done offline. Never upload model bytes to diagnose an installation.

## Locate the failed boundary

1. Can the launching process read the original file? A Terminal permission on
   macOS may not apply to another app or SSH process. Check the actual context.
2. Does size/full SHA-256 match? Use the manager's identity cache only when
   valid; do not repeatedly scan 177 GB or forge a verification stamp.
3. Did GGUF parsing validate tensor names, shapes, types, alignment and bounds?
   Do not accept a different quantization by changing only a config constant.
4. Is extraction correct? Gate/up select neuron rows; down selects the matching
   32-neuron column blocks. This gathers bytes without requantization.
5. Did the receiving node have room for the temporary unit and reserve? Check
   its persistent filesystem and the specific remote error, not just head disk.
6. Did the final unit digest match, and can the selected backend load it?

Twenty units cover all neuron ranges across all distributed experts/layers.
Each unit has exactly 3,347,005,440 bytes. A `.tmp` transfer is not a valid unit.
The `.ok` stamp binds digest to file identity; copying to another filesystem may
invalidate it even when bytes match. Let the normal verifier handle that case.
Do not edit stamps to make an unverified cache appear usable.

## Validate a format or extraction change

Use the independent full audit in `tools/fronk-model-check.c`, as described in
[release qualification](../RELEASE.md). It compares every extracted byte with
source tensor coordinates and recomputes all unit digests. The original GGUF
hash is a separate check because dense/ngram tensors are not all in the units.
Run malformed-GGUF tests as well; a happy-path digest does not validate bounds.

Changing dimensions, quantization or model revision requires coordinated
extractor, manifest, dense-adapter and backend work, numerical evidence and a
real complete generation. Never weaken a check to accept mismatched weights.
Report whether data was downloaded, pre-existing, freshly transferred or reused.

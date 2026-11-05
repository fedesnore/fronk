# Preserved Shuttle AVX2 port sources

This archive preserves the CPU kernels used while developing Fronk's AVX2
backend. It is historical source evidence, not a build or runtime dependency.
For the current implementation and contribution contract, read
[avx2.c](../../backends/avx2.c) and [BACKENDS.md](../../BACKENDS.md).

`source-20261023.tar.gz` SHA-256:
`549a3453c992617295b29c6de963ee65106ae11ffef2761b1c6ac57c14d1817a`.
`SHA256SUMS` records every archived file's original bytes (66 files).
The 55 files in the captured run's original `source_sha256` manifest were
independently verified against that manifest after transfer.

The archive contains:

- The source and original manifest of
  `r4/native125/sim64-mac-compute-20261023-fusions-abba-a/`.
  `source/phase5/runtime/q4panel.h` implements the four-position shared-weight
  path with explicit AVX2 accumulators, Q8 activations, integer dot products,
  per-group correction, scale multiplication and FMA. Its original SHA-256 is
  `e99513f8bb2f297fc076f5eacf2acd11e9b653d1e7881d8325b57c896dc0e20c`.
- The older selected `r4/native/src/` implementation, `build-v2.sh`, and
  `r4/native/build/selected-v3.json`, together with its retained dependencies
  `r3/src/native_block.cpp`, `r3/src/block_execution.h`, `phase6/src/worker.cpp`
  and `phase6/src/panel4.h`. The older fixed path explicitly uses Q4_K blocks
  of 256 weights/144 bytes and FP32 reduction. Its existing build flags include
  `-O3 -std=c++17 -mavx2 -mfma -mf16c -fopenmp -ffp-contract=off`.

No weights, model blobs, libraries, executables, credentials or SSH state are
included. The preserved source is C++/Python from the earlier project; fronk's
actual port must obey its C11 and dependency constraints.

When comparing these kernels with the current backend, check layout and
arithmetic before borrowing an optimization. Run the canonical Q4_0 checks
as well as the native-model checks.
The existing panel execution layout and older Q4_K path are not proof of
canonical block-32 Q4_0 compatibility. Do not silently change either format or
requantize weights to make them fit.
The archive makes no new speed or backend-conformance claim.

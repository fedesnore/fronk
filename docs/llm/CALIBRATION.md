# Task brief: calibration, planning and the synthetic demo

> **For your frontier model.** People: read [how startup works](../../PROJECT.md#startup-head-first-then-nodes) instead, and see
> [how to use this brief](README.md). Model: also follow [AGENTS.md](../../AGENTS.md).
> If you cannot open a linked file, ask for it; do not guess.

Read [PROJECT.md](../../PROJECT.md), [BACKENDS.md](../../BACKENDS.md),
[troubleshooting](../TROUBLESHOOTING.md), then `src/setup.c`, `selection.c`,
`hybrid.c`, `planner.c`, `head.c` and the demo in `serve.c`.

Keep these quantities separate: available RAM now, saved resident compute cost,
combined RAM + SSD cost, network cost, and dense-head cost. A 32-neuron unit is
not an MTP token. A burst is the number of activation rows in one call, not the
number of tokens ultimately accepted. The head is also a possible expert worker.

## Diagnose

- Inspect selected backend, configured/effective width, reserve, head-blocks,
  capacity and cache messages. Do not race every backend if one was selected.
- Inspect the compute fingerprint and cache schema. Short profiles load a
  resident fixture once per feasible block count and measure widths 1–4.
  A UI change alone should not require fresh kernel timings.
- Check the actual state filesystem and available memory. A RAM-backed test
  cache consumes the very capacity being measured. Disk space is separate.
- Metal SSD placement needs a matching combined profile for RAM count, width,
  filesystem, cache and pipeline policy. `--no-measure` does not produce it.
  The normal installer calibrates it; discovery restores it.
- Network paths are re-probed, while width-specific runtime network reports
  and compatible compute profiles are reused. Old workers may lack the short
  profile capability; do not silently switch their measurement width.

Do not run calibration alongside inference/timing or when a backend still owns
loaded units. If measurement is needed, stop the affected program through its
manager, use `fronk backends --measure --config PATH`, then reconnect. Do not
refresh merely to make a slow or failed measurement disappear.

## Evaluate the plan honestly

Follow `selection_cost` and `compute_shares`: each participant's measured compute
plus link cost gives a candidate completion time, and the planner minimizes the
slowest participant subject to capacity. Costs beyond the bounded sweep are
estimates. Physical shared-uplink contention is not fully modeled. A new plan
can change slice ownership and transfers without changing model arithmetic.

The demo runs synthetic units on real participants and links. Require every
intended node and every reply, and inspect whether all 20 units are covered.
`serve.c` warms both measured widths before resetting counters. Timed requests
share the ordinary complete-sum chat's 500 ms failure bound; slow SSD reads stay
in the measured duration. Do not replace that with an arbitrary shorter network
deadline, discard slow samples, or accept missing sums to make setup pass.
Its dense workload and acceptance range are estimates; it is not real-model
token throughput. Preserve the distinction between calibrated hybrid capacity
and a fallback estimate for missing resident blocks.

For a planner change, use fixed saved curves and the planner/selection tests
before a complete cluster comparison. Preserve ownership coverage, nonoverlap,
capacity gates and measured-versus-estimated labels. A setup-time improvement
does not establish a decoding speedup. Report elapsed calibration, cache hits,
plan, model fit and remaining limitations separately.

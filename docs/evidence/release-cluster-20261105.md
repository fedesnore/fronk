# My Fronk cluster

Fronk version: 0.3.0
Model: Qwen3.8-Flash-Next-Q4.gguf

## Machines and allocation

| Machine | CPU | RAM (GiB) | OS / architecture | Expert backend | RAM / SSD units | Link |
|---|---|---:|---|---|---|---|
| Head | Apple M4 Pro | 24.0 | Darwin / arm64 | metal-ssd | 1 / 1 | local |
| Node 1 | Apple M1 Pro | 16.0 | macos / arm64 | metal | 3 / 0 | thunderbolt |
| Node 2 | Intel(R) Core(TM) i5-7600K CPU @ 3.80GHz | 48.0 | macos / x86_64 | avx2 | 11 / 0 | thunderbolt |
| Node 3 | Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz | 15.4 | linux / x86_64 | avx2 | 4 / 0 | ethernet |

Dense head engine / GPU: [fill in; separate from its expert backend]
Cables: [separate ports, or switch + head uplink speed + node port speeds]

## Console test result

- Reported decode rate (tok/s): 15.61
- Time to first token (s): 1.46
- Whole request (s): 5.368
- Reported output count: 62
- Deadline misses: 0
- Slowest participant: Node 2
- Source: the console's test report; not a synthetic estimate or a guarantee.

| Machine | Compute (µs) | Link/return (µs) | Last to reply (count) |
|---|---:|---:|---:|
| Head | 1024 | 0 | 4 |
| Node 1 | 729 | 580 | 16 |
| Node 2 | 1822 | 459 | 1009 |
| Node 3 | 808 | 1230 | 267 |

## What I tried

Cold / warm run: [fill in]
Prompt and context length: [a harmless reproduction, if sharing one]
What worked, failed, or helped: [fill in]

This file omits hostnames, addresses, pairing codes, paths and chat text.
Review it, fill in what you know, and paste it into a GitHub Cluster result issue.
Nothing has been uploaded. Unknown fields are fine.

# What if the nodes were more alike?

These are **conditional estimates**, not benchmarks of clusters we own.
The head stays a **24 GiB M4 Pro running dense computation on Metal**. Only
the expert nodes change. This makes it possible to ask what removing the
memory-heavy iMac might buy before collecting ten computers.

The head takes no expert units in these examples:

| Hypothetical nodes | Allocation | Total coverage |
|---|---|---|
| 10 Shuttle i5-10210U machines | 2 units = 64 neurons per expert on each | 20 units, all 640 neurons |
| 7 M1 Pro machines, 16 GiB each | Six with 3 units (96 neurons); one with 2 units (64 neurons) | 20 units, all 640 neurons |

For a like-for-like 64-neuron share, ten M1 Pro nodes are another possibility,
but the seven-node arrangement uses their measured three-unit capacity and
needs fewer links. All-M1 here means the **nodes**, not replacing the head's
GPU with an M1 Pro. None of these arrangements requires expert SSD streaming.

## Start from measured pieces

Saved installer profiles from the real Shuttle and M1 Pro cover widths one
through four, with 256 samples per width. The following values use the slower
median among the retained profiles, rather than choosing the best sample:

| Node's share | One activation, per layer | Three activations, per layer | Three-activation work over 48 layers |
|---|---:|---:|---:|
| Shuttle, 64 neurons, AVX2 | 0.1855 ms | 0.4812 ms | 23.10 ms |
| M1 Pro, 96 neurons, Metal | 0.4678 ms | 0.7672 ms | 36.83 ms |
| M1 Pro, 64 neurons, Metal | 0.3851 ms | 0.5879 ms | 28.22 ms |

These are native synthetic backend timings, without a network, not copies of
the older prepared-weight Shuttle experiment. The slower M1 share sets the
seven-node example's compute budget. Small workloads do not automatically
favor a GPU: fixed submission and memory costs matter. The measured three-token
Metal cost grows less than three times its single-token cost.

The [saved profile values](evidence/homogeneous-node-profiles.csv) retain the
medians, p99 values, widths and cache record identities. They do not establish
that every other Shuttle or M1 Pro will behave the same way.

For the head, five pure width-three windows in a real M4 Pro trace give a
median **63.74 ms per pass** outside the distributed expert wait. Round that
to **64 ms**. The [head trace extract](evidence/m4-head-width-three.csv) includes
dense work up to each barrier and between-pass work. Mixed-width prompt windows
are excluded. This is an instrumented budget, not a guarantee after removing
the local worker or changing the network.

## A range worth testing

Let `A` be the **mean number of output tokens committed per three-activation
pass**. It is not the probability of accepting one proposal. Rejections, partial
acceptance and any extra work have to count against it.

```text
estimated decode tokens/s = 1,000 × A / (64 + C + N)

C = slowest node's three-activation expert work across 48 layers, in ms
N = additional critical-path communication, fanout and join cost per pass, in ms
```

`N` is the part that remains after overlap; it must not be double-counted from
an already end-to-end measurement. Twenty, forty and eighty milliseconds below
are **assumed scenarios**, not measured ten-node network times.

| Nodes | Assumed network/coordination budget | If A = 2 | If A = 2.5 |
|---|---:|---:|---:|
| 10 Shuttle, 64 neurons each | 20 ms/pass | **18.7 tok/s** | **23.3 tok/s** |
| 10 Shuttle, 64 neurons each | 40 ms/pass | **15.7 tok/s** | **19.7 tok/s** |
| 10 Shuttle, 64 neurons each | 80 ms/pass | **12.0 tok/s** | **15.0 tok/s** |
| 7 M1 Pro, 96/64 neurons | 20 ms/pass | **16.6 tok/s** | **20.7 tok/s** |
| 7 M1 Pro, 96/64 neurons | 40 ms/pass | **14.2 tok/s** | **17.8 tok/s** |
| 7 M1 Pro, 96/64 neurons | 80 ms/pass | **11.1 tok/s** | **13.8 tok/s** |

With ten M1 Pro nodes at 64 neurons, the same 20 ms assumption gives **17.8–22.3
tok/s** for `A = 2–2.5`; at 40 ms it gives **15.1–18.9 tok/s**. That extra
parallelism is useful only if adding three network participants does not spend
the compute saving.

The table says there is a plausible experiment beyond the current 15–16 range.
It does not demonstrate a speedup. If only one token commits per pass, halve
the `A = 2` column. Even committing all three gives only about **28.0 tok/s**
for ten Shuttles in the optimistic 20 ms case. With these budgets, 30 tok/s
would also need cheaper head work, a smaller communication remainder or a
different multi-token schedule. More nodes alone do not remove the head cost.

Without MTP, using the measured single-token head budget of 58 ms and an assumed
20 ms communication remainder gives about **11.5 tok/s** for ten Shuttles and
**10.0 tok/s** for seven M1 Pros. Those are different workloads from the
three-activation table; do not compare them as if the MTP acceptance were free.

## The switch can decide the result

Seven or ten nodes cannot all have separate direct Thunderbolt ports on this
head. A practical test needs Ethernet, an appropriately fast head uplink, or a
carefully measured mixed topology. A 10 Gb/s head uplink with 1 Gb/s per node is
the relevant starting point, not ten nodes sharing one 1 Gb/s head port.

At width three, a node returns `3 × 4 × 1,312 = 15,744` encoded UDP bytes per
layer. Ten nodes return **157,440 bytes per layer**, regardless of their neuron
shares. Serializing just those bytes across a shared 1 Gb/s head link takes
at least **1.26 ms/layer**, or **60.46 ms over 48 layers**. At 10 Gb/s the same
aggregate serialization floor is **6.05 ms per pass**. Ethernet/IP headers,
request traffic, forwarding, jitter, host work and loss are additional; packets
can overlap on independent ports. These are byte-rate lower bounds, not a
prediction of the complete network remainder.

The installer currently measures individual paths; shared return congestion
still needs a real simultaneous test. To test these estimates, record the
allocation, head uplink, per-node ports, width distribution, committed tokens
per pass, p50/p99 arrival times and misses. Repeat identical short prompts,
include time to first token separately, and then try a longer conversation.
[Save and share your cluster report](SHARE.md), including a result that falls
below the table. Finding the actual limit is the point.

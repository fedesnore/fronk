# Communication between the head and nodes

The **control plane** establishes who participates and what each machine owns.
The **data plane** carries the activations and partial sums during inference.
They are two software paths over the same cables, not two physical networks.
The installer, terminal and web console drive the existing Fronk lifecycle.
They do not introduce another distributed inference protocol.

For an assistant investigating a connection problem or changing the transport,
use the companion [communication task brief](llm/PROTOCOL.md).

Start with [the project overview](../PROJECT.md#communication-control-plane-and-data-plane)
and [physical network setup](NETWORK.md) if these concepts are new. Contributors
should read [resp.h](../src/resp.h), [probe.h](../src/probe.h),
[wire.h](../src/wire.h), [connection.h](../src/connection.h) and
[plane.h](../src/plane.h) before changing a field.

## Trust boundary and ports

| Flow | Transport and default endpoint | Contents |
|---|---|---|
| Head → node control server | TCP 7700, RESP2 | Authentication, hardware, benchmarks, placement and session commands |
| Head → node weight receiver | Same authenticated TCP connection | `RECV`, then exactly one raw native unit, then acknowledgment |
| Head → local network discovery | UDP 7702, IPv4 broadcast (or explicit configured peers) | `FRH2` HELLO; node sends unicast announcement back |
| Head → individual node | UDP 7701 | `FRK1` probes or REQ activation frames |
| Head → eligible wired group | UDP 239.7.7.7:7702 | Multicast probes and REQ frames |
| Node → head | UDP to head's dynamically allocated session address/port | PONG probes or PART output frames |
| Browser/client → head locally | HTTP 127.0.0.1:7710 | Model API; port is `control-port + 10` |
| Browser → head locally | HTTP 127.0.0.1:7730 | Web console/chat proxy |

Ports and the data multicast address are configuration values, not constants
to assume in a new client. The head binds session sockets to selected source
addresses/interfaces. It also creates persistent connected send sockets;
"connected UDP" sets a socket's destination without adding a TCP handshake or
reliability. Loopback discovery tests use 239.7.7.6 because macOS loopback does
not deliver ordinary broadcast. That is distinct from the data multicast group.

This is a **trusted-LAN protocol**. Control authentication uses a fresh nonce
and HMAC-SHA256. Layer traffic is not encrypted or authenticated with a MAC;
source/session validation is not protection against a hostile LAN peer. Do not
expose control, discovery, data or the model API to the Internet. Pairing codes
are cluster secrets, not machine-login passwords.

RESP commands, weight bytes and UDP tensors have no TLS or transport encryption.
The HMAC challenge proves a client's knowledge of the secret to the node; it
does not provide a mutually authenticated encrypted channel. Neither discovery
cookies nor SHA-256 file checks are substitutes for that. Restrict access to
trusted local machines. There is no SSH dependency, RDMA transport or peer-to-peer
expert exchange between nodes.

## Control plane

`node.c` implements bounded RESP2 parsing and command dispatch. RESP is a
length-prefixed request/reply format familiar from Redis; Fronk implements its
own commands and does not run a Redis server. Normal clients send arrays of
bulk strings. The parser also accepts simple whitespace-separated inline
commands for debugging, without quoting/escaping. It limits messages to 65,536
bytes, 128 parsed values and nesting depth eight; command dispatch accepts up
to 16 string arguments. The multi-gigabyte unit stream is a separate transfer
state and does not fit inside a RESP message.

`fronk-cli`
performs NONCE/AUTH before protected commands when given the configuration.
PING, NONCE and AUTH are the only unauthenticated commands. AUTH consumes the
challenge even on failure; reauthentication revokes the old link/session.

The node generates a random 16-byte challenge. NONCE returns its 32 hex digits;
the client decodes them and sends `AUTH` with the 64-hex-digit
`HMAC-SHA256(cluster_secret, raw_nonce_bytes)`. A nonce belongs to one TCP
connection and each authentication attempt consumes it. Neither secret nor
proof should be logged. On disconnect the owned link/session is released.

Useful inspection commands:

```sh
fronk-cli -h NODE_IP --config NODE_CONFIG INFO
fronk-cli -h NODE_IP --config NODE_CONFIG INFO backend
fronk-cli -h NODE_IP --config NODE_CONFIG INFO dataplane
fronk-cli -h NODE_IP --config NODE_CONFIG HW
fronk-cli -h NODE_IP --config NODE_CONFIG UNITS
```

| Command family | Purpose |
|---|---|
| `PING`, `NONCE`, `AUTH` | Liveness and single-use authentication challenge |
| `INFO`, `HW` | Control/data-plane counters, fresh hardware and memory information |
| `LINK session head-udp-port node-index` | Bind authenticated control owner to a session and UDP peer |
| `LINK PATH`, `LINK REPORT`, `LINK RELEASE` | Record chosen path and measured RTT distributions, or release ownership |
| `BACKEND`, `BENCH`, `BENCH-PROFILE` | Backend choice, cancellable calibration, short-profile capability query |
| `UNITS`, `BLOCKS` | Verified cached digests and currently loaded unit mask |
| `RECV block sha256` | Receive a missing native weight unit |
| `SESSION START id unit-mask burst deadline-us`, `SESSION STOP` | Start/stop a selected backend and its data-plane receiver |
| `EXPERT-ORDER` | Advertise the current expert-order capability for compatible research paths |

This is a lifecycle map, not a promise that every command takes arbitrary
arguments; exact validation is in `node.c` and `tests/test_node.c`. A linked
session has one control owner. Another client cannot silently replace a live
session. The node's event loop remains available for inspection while a worker
runs a benchmark; long computation is not performed inside RESP dispatch.

`RECV` accepts a unit index and expected lowercase SHA-256, acknowledges readiness,
then consumes exactly **3,347,005,440 raw bytes** on that TCP connection. It hashes
while writing a temporary file, checks the digest, renames/publishes the unit
and returns success. A failed transfer must not be treated as a usable unit.
TCP supplies ordered delivery/retransmission for setup; unit hashes check model
identity and storage integrity. Reconnect reuses verified units instead of
transferring all weights again.

Before acknowledging, the node checks space for that unit plus a 1 GiB reserve.
An insufficient-space error includes required and available bytes. The head
preserves the node's error in its transfer report, so the user knows which disk
needs attention. This is a setup check, not work added to layer inference. Free
space can still change after the check; write errors and hash failures remain
fatal to that transfer.

Do not call SESSION or LINK manually against a live head's nodes: those commands
change ownership. Use the head/manager lifecycle. Stopping a session normally
retains loaded units for reconnect; a backend/share change or calibration may
release them. Debug-only synthetic session commands are for tests, not real
model data.

## Discovery and planning

A persistent 64-bit ID identifies a machine; multiple addresses are candidate
paths. `FRH2` discovery uses a 24-byte HELLO and a 100-byte announcement with
the echoed 128-bit run cookie, machine ID, name, control port and data port.
The cookie rejects unrelated discovery replies; TCP authentication follows.

The head probes source/interface-bound authenticated paths at three sizes:
64 bytes, 5,312 bytes (width one) and 21,248 bytes (width four). These probe
trains concatenate their logical fragments into one UDP datagram, as inference
does. The complete width-four p99 ranks paths; missing probes cannot win.
**Probe timing still does not include expert computation** and uses the node's
probe handler rather than an active inference worker. The planner also has
width-specific runtime link measurements.

Wired groups can multicast only after successful eligibility/reception checks.
A failed multicast check falls back to unicast on that interface; Wi-Fi always
uses unicast. One-member groups unicast too. Multicast sends one request for a
group; it does not combine replies. Every participant still returns its own
partial. A reconnect rechecks paths while reusing compatible backend measurements
and verified unit files.

The planner's network cache is width-specific. Mixed-width generation reports
are not saved as a measurement of one fixed width. Older workers without the
short-profile capability keep the configured width; a width change must not
silently trigger their old long calibration sweep.

The planner currently adds per-node link cost to compute cost. Shared-uplink
contention and many simultaneous replies are not fully modeled; ten nodes on a
single Gigabit head port cannot be estimated as ten independent links.

## Lifecycle across both planes

```text
HEAD                                      NODE
HELLO (UDP) ----------------------------> announce identity
             <--------------------------- ANNOUNCE (UDP)
connect + NONCE/AUTH (TCP) --------------> authenticate control client
LINK + probes --------------------------> bind owner; measure UDP path
HW / BACKEND / BENCH / UNITS (TCP) ------> report capacity, costs, cache
RECV + missing unit bytes (TCP) ---------> verify and publish files
SESSION START (TCP) ---------------------> load slices; start receiver

repeat for the 48 distributed layers of a batch:
  dense work produces activations/routes
  start local slice; REQ (UDP) ----------> assemble full request
                                          backend computes partial
                       <----------------- PART (UDP)
  validate + sum every assigned partial
  resume dense graph

SESSION STOP / LINK RELEASE (TCP) -------> drain work; retain reusable cache
```

Several nodes execute the right-hand side concurrently. The critical path is
the last required completion, not the sum of all node compute times. Built-in
MTP can include several token activations in one call, amortizing barriers over
accepted tokens. Width is a batch size, not an acceptance guarantee. The normal
protocol does not independently let a node invent/commit the next token or
continue the next layer without the head's dense computation and routing.

## One layer call

REQ carries a session ID, step, layer, width, FP16 activations and the routing
record for each token. The ordinary backend contract accepts widths 1–8. Each
token has four logical fragments of 640 dimensions. A common header occupies
24 bytes and a range header 8 bytes. The first REQ fragment for a token also
contains its 64-byte routing record. All integer/float bit fields are big endian.

The 24-byte `FRK1` header has this layout (offsets in bytes):

| Offset | Size | Field |
|---|---:|---|
| 0 | 4 | Magic `FRK1` |
| 4 | 1 | Type: 1 REQ, 2 PART, 3 probe PING, 4 probe PONG |
| 5 | 1 | Node index; 255 for head-originated requests |
| 6 | 1 | Burst width |
| 7 | 1 | Reserved, zero |
| 8 | 4 | Session ID |
| 12 | 4 | Step ID |
| 16 | 2 | Layer |
| 18 | 2 | Logical fragment index |
| 20 | 4 | Bytes after this common header |

An eight-byte range header follows: token index, starting dimension, dimension
count and a zero reserved field, each two bytes. REQ/PART carry 640 binary16
values per fragment. Fragment `4 * token` of REQ additionally contains ten
16-bit routed expert IDs and eleven FP32 gate values (the shared gate is last).
The other three fragments contain only the next activation ranges. The supported
model has 512 routed experts; selected IDs must be distinct and in range.
Non-finite values, invalid shapes and FP16 overflow are errors, not silent clamps.

| Payload | Bytes per token |
|---|---:|
| Request | 5,312 |
| One node reply | 5,248 |

The logical fragments are concatenated into **one UDP datagram per call** for
each destination/group; the kernel may IP-fragment it. The 1,376-byte maximum
logical fragment is not a claim that every transmitted datagram fits the MTU.
This batching reduced per-send overhead on measured macOS hosts. Changing it
requires loss/latency and complete-generation measurements, not just packet-rate
benchmarks.

Thus a width-three request is 15,936 bytes and each reply 15,744 bytes; at width
eight they are 42,496 and 41,984. These sizes exclude UDP/IP/Ethernet overhead.
Keeping the logical codec fragments below MTU does **not** prevent IP
fragmentation of the concatenated datagram. Jumbo frames are not an installer
requirement. Losing one IP fragment can lose the complete UDP datagram.

The head validates and encodes everything before sending any request. It starts
its local expert worker before fanout, overlaps local work with network traffic,
and receives all selected interfaces. The receiver assembles a complete request
before publishing it to the backend. No allocation/connection setup belongs in
the layer loop.

Replies are checked against source/interface, session, step, layer, width and
fragment bitmap. Duplicates and stale fragments cannot contribute twice. Local
computation uses the same FP16-rounded activations as remote nodes. The ordinary
head requires all assigned replies and drains submitted work on error; partial
sums must not be returned as successful generation.

There is no per-fragment acknowledgment or retransmission in the UDP plane.
A lost fragment can leave a request or reply incomplete at its deadline. The
ordinary generation path in `src/serve.c` waits up to 500 ms for a complete sum;
if a participant is missing, it retries the **whole expert call once**, with a
new step ID and a 1 s deadline. The new step excludes late packets from the
first attempt and replaces the entire sum; it does not add a duplicate result.
If the retry is still incomplete, generation fails instead of using a partial
sum. The synthetic demo reports missing replies as a failure, without this
generation retry. Diagnostics distinguish invalid, late, duplicate and missing
data. Keep these finite deadlines and clear error behavior when experimenting
with another transport; they are not a guarantee of UDP delivery.

## What reduces transport overhead today

| Implemented mechanism | Work it avoids | Tradeoff / boundary |
|---|---|---|
| Session-owned buffers and persistent compute/RX workers | Per-layer allocation, thread creation and connection setup | Ownership and completion must be explicit before buffer reuse |
| Connected UDP send sockets | Repeated destination/route setup on each send | Still UDP; no delivery guarantee |
| One datagram per call/destination | One send syscall for each logical fragment | IP fragmentation can amplify packet loss |
| Wired multicast after testing reception | Repeated copies of the same request on one head interface | All node replies still converge independently |
| Binary16 activations and partial outputs | Roughly half the tensor bytes versus FP32 | Rounding is part of the numerical contract |
| Head-local compute submitted before fanout | Waiting for sends to finish before local work begins | Dense GPU and local Metal work can still contend |
| Cached units and compatible compute profiles | Weight transfer and calibration on every reconnect | Current memory and network paths must still be checked |

The historical width-two comparison in [BENCH.md](../BENCH.md) measured:

| Interval, µs per request | 8+8 small datagrams | Batched, connected head send | Also connected node reply |
|---|---:|---:|---:|
| Head send | 676 | 171 | 121 |
| Shuttle receive / reply | 682 / 92 | 137 / 58 | 139 / 57 |
| M1 Pro receive / reply | 576 / 1,062 | 67 / 309 | 60 / 219 |
| Shuttle reported network | 1,689 | 1,153 | 1,141 |
| M1 Pro reported network | 2,166 | 966 | 801 |

These were real transport runs with 13–14 head-SSD units, a USB Ethernet adapter
to Shuttle and Thunderbolt to M1 Pro. They establish a benefit from batching
and socket reuse; they are not isolated cable latency or a prediction for the
later 15–16 tok/s configuration. A separate spinning UDP echo using similar
message sizes measured p50 round trips of 176 µs to M1 Pro and 506 µs to Shuttle.
CPU scheduling, encode/decode, the network stack and shared-link contention
explain why application intervals must be measured separately from that floor.

Recent path instrumentation with another placement found preparation plus
final reduction at about 1% of the expert call. This is a reason to measure the
last-arriving participant and head GPU time before rewriting the codec. It is
not proof that every cluster has the same bottleneck.

## Reading a slow-call report

`INFO dataplane` and the head's reports expose completed calls, backend time,
queue/receive/reply time, missing replies, duplicates, late and invalid data.
Optional path instrumentation attributes arrival time by node and width. Use
the exact field definitions in `plane.c`, `node.c` and `serve.c`: "network" is
an observed interval/residual, not a hardware timestamp of cable transit.

- A node's **compute time** is the backend interval; memory pressure, SSD reads
  and worker contention can affect it depending on the selected backend.
- Its **arrival time** is observed on the head and also includes sends, queues,
  receive scheduling and communication. Concurrent nodes' arrival times overlap;
  adding them invents latency.
- **Deadline misses** mean a required result did not arrive in time. The ordinary
  release requires every assigned participant and fails generation if a result
  is still missing after its one whole-call retry; legacy partial-sum diagnostic
  settings are not the release policy.
- **Invalid/late/duplicate** counters distinguish rejected input from useful
  work. A TCP `PING` success does not prove that the UDP data path is healthy.
- `request_gaps` counts gaps in received step IDs, not lost IP packets. The
  transition from self-test to generation can itself skip IDs; compare a delta
  within the same phase with head misses/retries before inferring network loss.
- Compare the same **width, placement, context, output and acceptance rate**.
  A faster packet microbenchmark is not automatically faster chat.

## The dense engine's local boundary

On Apple Silicon, the patched Metal engine calls the in-process expert hook
in `serve.c`. The persistent expert worker hands the request to `plane.c` and
signals completion so the dense graph can resume. The normal path uses GPU
event resume; it does not send an HTTP or RESP request for each layer.

The Vulkan adapter uses a local socket bridge defined in
[fronk_client.h](../client/fronk_client.h) to reach the same expert service.
That local bridge is distinct from the remote `FRK1` protocol. In both cases the
backend sees `fr_req`/`fr_part` with FP32 arrays after network conversion. A new
CPU/GPU expert backend implements that compute contract, not discovery, sockets
or model downloading. See [BACKENDS.md](../BACKENDS.md).

## Compatibility and changes

The direct Metal hook and the Vulkan socket adapter feed the same `fr_req` /
`fr_part` contract. A new expert backend does not require a new head protocol:
it reports its name and measured costs and computes its assigned share.

Run `test-wire`, `test-fanout`, `test-plane`, `test-node`, `test-connection`, and
`test-plane-cluster` after transport changes. Include wrong sessions/sources,
duplicates, out-of-order fragments, loss, deadlines, mixed widths and failed
publication. Retain the control oracle before making performance comparisons.

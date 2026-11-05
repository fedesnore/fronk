# Task brief: inspect or improve head/node communication

> **For your frontier model.** People: read [the protocol guide](../PROTOCOL.md) instead, and see
> [how to use this brief](README.md). Model: also follow [AGENTS.md](../../AGENTS.md).
> If you cannot open a linked file, ask for it; do not guess.

Read [the protocol guide](../PROTOCOL.md), [network setup](../NETWORK.md),
`src/resp.h`, `src/probe.h`, `src/wire.h`, `src/connection.h`, `src/plane.h`,
then the relevant implementations. Read `src/node.c` for ownership/control and
`src/serve.c` for the local dense-engine boundary. Explain a problem in terms of
the bytes, owner, event and deadline involved, not only a protocol name.

## Establish what is running

Record head/node versions, interfaces, addresses/ports, physical topology,
workload width and unit placement. Distinguish control reachability from data
delivery. HMAC-authenticated TCP/RESP controls setup; UDP `FRK1` carries layer
requests/partials. `FRH2` discovery is not authenticated model traffic. The HTTP
API is local user access, not the per-layer transport.

Use read-only INFO/HW observations before intervention. Do not issue LINK or
SESSION commands against a live owned session as a diagnostic shortcut. Do not
print pairing secrets or expose local services to the Internet.

## Existing work to preserve

- Persistent connected UDP sends; preallocated buffers; persistent RX/compute
  workers; local expert submission before fanout.
- One datagram per call/destination containing logical fragments. Logical
  fragments fit MTU, but the combined datagram can be IP-fragmented.
- Multicast only for tested eligible wired groups; unicast otherwise. Replies
  are independent and share the head's receiving capacity.
- Session/source/interface/step/layer/width checks and duplicate bitmaps.
- FP16 wire values, equal local-input rounding, frozen accumulation policy.
- Complete required sums, finite deadlines, no UDP layer retransmission today.
- TCP unit transfer into temporary files, exact byte count and SHA-256 before
  publication; compatible calibration and unit caches reused at reconnect.

Do not remove validation to make a benchmark faster. Keep output unpublished
on failure and drain submitted local/GPU/I/O work before buffer reuse. Source
checks are not encryption or a defense against a hostile trusted-LAN peer.

## Diagnose and test

Attribute encoding/send, remote receive/compute/reply, head arrival/merge and
dense resume separately. Arrival times overlap: use the last required completion,
not the sum across nodes. RTT probes are batched too, but do not include backend
computation. Width-specific measured reports must not be replaced by mixed-width
averages. Shared-uplink contention is not fully modeled by the planner.

Before an optimization, state a measured costly interval, the mechanism and a
bounded comparison. Test wrong sessions/sources, malformed frames, duplicates,
out-of-order input, loss/deadlines, mixed widths and all error publication paths.
Run `make DS4= test-wire test-fanout test-plane test-node test-connection` and
`make DS4= test-plane-cluster`; include control/auth tests for lifecycle changes.
Use isolated nodes/ports first, then the real topology if authorized.

Compare complete generation with unchanged model, context, placement, output and
accepted-token count. Report syscall/packet component gains separately. Update
both the human protocol guide and this brief if a wire field or lifecycle changes;
document compatibility rather than letting old nodes misinterpret a new frame.

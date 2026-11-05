# Task brief: connect the computers and set up a cluster

> **For your frontier model.** People: read [the network guide](../NETWORK.md) instead, and see
> [how to use this brief](README.md). Model: also follow [AGENTS.md](../../AGENTS.md).
> If you cannot open a linked file, ask for it; do not guess.

Help the user get a complete Fronk cluster working. Assume no networking
experience unless they show otherwise. Begin on the head and use
the existing installer manager. Read [NETWORK.md](../NETWORK.md),
[INSTALL.md](../INSTALL.md), [MODEL.md](../MODEL.md) and
[PROTOCOL.md](../PROTOCOL.md). Do not create another installer or discovery system.

## Establish the physical facts

Inspect accessible machines using read-only tools. For machines you cannot
access, ask only for missing facts. Record a small table: computer label, OS,
CPU/GPU, RAM, intended role, physical port/adapter, cable, switch port, current
IP/interface and whether the user has opened its installer. Never guess a
machine's address or assume a USB-C cable provides Thunderbolt networking.

Offer concrete choices: separate direct Thunderbolt links, Ethernet switch, or
mixed topology. Explain why all replies converge on the head 48 times per batch.
A switch should have a faster head uplink, e.g. 10 Gb/s versus 1 Gb/s nodes,
with suitable head adapter/cable/port. This is a design recommendation, not a
measured speed guarantee. Check actual link negotiation and later Fronk timings.

## Turn that into a reviewable network plan

1. Prefer the existing trusted LAN's DHCP if it works. Otherwise propose unique
   private IPv4 addresses and masks for isolated links, checking existing
   Wi-Fi/VPN/routes for overlap.
2. On macOS, account for Thunderbolt Bridge grouping physical ports. Do not
   assign overlapping subnets to separate unbridged interfaces. Use distinct
   subnets for separate Ethernet and Thunderbolt networks when appropriate.
3. Keep the Internet/default route intact. An isolated cluster link needs no
   gateway or DNS; do not enable Internet Sharing or forwarding without a reason.
4. Before changing persistent network settings, show the exact interface,
   address, commands and rollback. Obtain authorization if it is not already
   given. Do not cut off the SSH route you are using to make the change.
5. Apply only the authorized changes on machines you can access. For physical
   actions, name the computer and port explicitly, then verify the result.

Use the platform's existing network manager. On Linux inspect `ip -br address`,
`ip route` and, where installed, `nmcli device status`; on macOS inspect
`networksetup -listallhardwareports`, `ifconfig` and `route -n get NODE_IP`.
Adapt commands to observed interfaces. Do not install a second network manager,
disable firewalls/VPNs globally or expose ports through the Internet router.

## Drive setup to completion

Start the head installer. Let the user select a node count and assigned names.
Guide node 1 through role/name/private pairing/backend/service, then return to
the head for Verify. Repeat, and check all intended nodes together. Pairing
secrets stay in private prompts/local state; do not put them in this conversation,
command arguments or logs. SSH is optional, not an installation prerequisite.

Run the synthetic demo before any large download. Explain compute, communication,
capacity and whether all 20 units fit. A component estimate is not actual tok/s.
Do not rerun long calibration on every reconnect; compatible profiles are saved.
Then follow [the installation task brief](INSTALL.md) through exact weights,
unit verification, generation check and browser chat. If a machine is inaccessible,
say precisely which action the user must perform there and continue independent
work; do not report its setup as complete.

Deliver the final physical/IP diagram, selected paths, node roster, working
launch/resume instructions, tests actually passed and any unresolved limitation.

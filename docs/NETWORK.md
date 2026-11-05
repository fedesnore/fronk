# Connect the head and nodes

Start on the **head**. Connect and configure the network before the installer's
node verification. Fronk discovers reachable machines and measures their paths;
it does not assign IP addresses or change operating-system network settings.

To have an assistant guide or perform these steps with you, give it the
[network/setup task brief](llm/NETWORK.md). It starts by identifying the actual
computers, ports and cables, then follows the head's installer through the nodes.

## Choose the physical layout

The head waits for partial results at 48 layer barriers per forward/verification
batch. A fast Internet connection is irrelevant to these local exchanges.
What matters is latency between computers and the capacity of the links returning
results to the head. See [PROJECT.md](../PROJECT.md#physical-topology-is-part-of-the-design)
for the diagram and explanation of the repeated layer exchanges.

- **Direct Thunderbolt:** connect each compatible node to a separate head port
  using a Thunderbolt cable. A USB-C connector alone does not establish
  Thunderbolt networking. A hub's one upstream cable remains a shared path.
- **Ethernet switch:** connect the head to a faster switch port, for example
  10 Gb/s, and nodes to 1 Gb/s ports. Check the head adapter, cable and port's
  negotiated speed. Ten 1 Gb/s node ports still share a bottleneck if the head
  itself has only one 1 Gb/s port.
- **Mixed:** Thunderbolt nodes and Ethernet nodes can coexist. The head needs
  an address on each network. Nodes only need to reach the head; they do not
  need to exchange model activations with one another.

A uniform Gigabit network can work, but will have a different performance
ceiling. Wi-Fi is supported via unicast, with less predictable latency. Do not
expect a faster link to fix a CPU-bound node. Keep machines awake and on power.

## Addresses, subnets and gateways in plain language

An IP address identifies a network interface, not a Fronk role. A subnet groups
addresses reachable directly over that network. For the examples below, `/24`
means mask `255.255.255.0`: addresses sharing the first three numbers are on
the same subnet. Every computer/interface on that subnet needs a distinct last
number. Do not reuse an address or overlap a subnet already used by Wi-Fi/VPN.

A router is a gateway to other networks. An isolated cable/switch cluster needs
no gateway or DNS server. Keep your existing Wi-Fi/Internet gateway for downloads.
A switch alone generally does not hand out addresses; a router's DHCP service
can do that. If a trusted existing LAN supplies DHCP, retain it and let Fronk
discover the addresses. Do not replace a working LAN configuration unnecessarily.

Example for an isolated Ethernet switch, with no DHCP:

| Computer | IPv4 address | Mask | Router/gateway and DNS |
|---|---|---|---|
| Head Ethernet | 192.168.50.1 | 255.255.255.0 | Leave empty |
| Node 1 Ethernet | 192.168.50.2 | 255.255.255.0 | Leave empty |
| Node 2 Ethernet | 192.168.50.3 | 255.255.255.0 | Leave empty |

If the head also has a separate Thunderbolt network, use another unused subnet,
for example `192.168.51.1` on its Thunderbolt Bridge and `.2`, `.3` on the nodes'
bridges. macOS normally combines Thunderbolt ports into one bridge: set its
address once, not one overlapping address for each member port. Separate
unbridged point-to-point interfaces instead require distinct subnets. Apple's
[Thunderbolt networking note](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
explains this bridge layout; Fronk uses its ordinary IP path, not the RDMA API.

## macOS: configure each end

On the head, open **System Settings → Network**, select **Ethernet** or
**Thunderbolt Bridge**, then its **Details → TCP/IP** settings. For the isolated
example choose manual IPv4, enter the head address and mask, and leave Router
empty. Repeat on each node with its distinct address. If Thunderbolt Bridge is
missing, add the Thunderbolt Bridge network service. System Preferences on older
macOS releases has the equivalent Network controls. Apple's
[IP over Thunderbolt guide](https://support.apple.com/guide/mac-help/mchld53dd2f5/mac)
describes the service and manual-address setup.

Allow local-network access for the installer/Terminal/Fronk when requested.
Allow Fronk's incoming connections in the firewall; do not turn the firewall
off globally. Remote Login/SSH is optional and is not required by the manager.

Permission belongs to the launch context. A worker that works from an authorized
Terminal can lose access when restarted through SSH. Successful pairing or ping
does not prove that its UDP replies work. If the log reports `EPIPE` or
`No route to host` while the link and address are still correct, inspect local
network permission for the program that actually started the worker. Use the
manager/service or the authorized Terminal, restart the session, and repeat
the full demo/generation check. Do not disable macOS protections to bypass it.

These commands only inspect settings; run them in Terminal:

```sh
networksetup -listallhardwareports
ifconfig
route -n get 192.168.50.2     # on head: route to node 1
ping -c 4 192.168.50.2
```

On node 1, test `192.168.50.1` instead. For Thunderbolt use that subnet's actual
addresses. A successful ping checks basic reachability, not the Fronk password,
UDP permissions or application-sized latency.

## Linux: configure the cluster interface

In the desktop network settings, choose the **wired** interface connected to the
cluster, set IPv4 to Manual, and use the same table. Leave gateway/DNS empty on
an isolated link and choose the setting that limits it to its own network, if
present. Do not alter the interface providing your current Internet/SSH route.

First inspect the machine:

```sh
ip -br address
ip route
nmcli device status          # if NetworkManager manages this machine
```

With NetworkManager, this example creates a dedicated **node 1** profile. Replace
`enp3s0` with the actual unused cluster interface shown above. Run locally on
that machine; bringing up a profile changes that interface's active settings.
Use `.1` instead of `.2` on a Linux head and a different address on every node.

```sh
sudo nmcli connection add type ethernet ifname enp3s0 con-name fronk-cluster \
  ipv4.method manual ipv4.addresses 192.168.50.2/24 ipv4.never-default yes \
  ipv6.method disabled
sudo nmcli connection up fronk-cluster
ip route get 192.168.50.1
ping -c 4 192.168.50.1
```

The profile persists across reboots. `sudo nmcli connection delete fronk-cluster`
removes this example profile; then reactivate the previous connection if needed.
The official [NetworkManager examples](https://networkmanager.pages.freedesktop.org/NetworkManager/NetworkManager/nmcli-examples.html)
describe static profiles, and its [settings reference](https://networkmanager.pages.freedesktop.org/NetworkManager/NetworkManager/nm-settings-nmcli.html)
defines `never-default`. If `nmcli` is absent or the interface is unmanaged,
use the distribution's existing network manager instead of installing a second
one. Linux Thunderbolt networking also needs the OS to expose a working network
interface; connecting a cable alone is not sufficient evidence of support.

## Verify in Fronk, on the head

1. Start the head manager, choose the intended nodes, and follow its pairing
   instructions on each node. Allow TCP 7700 and UDP 7701/7702 from the trusted
   cluster network to the nodes. Permit replies to the head's session UDP ports;
   those source-bound ports are allocated dynamically. Restricted firewalls
   should permit the app on that interface/subnet, not only destination 7701.
2. Return to the head and choose **Verify** for each node. Discovery is local
   IPv4 broadcast; routed/VLAN-isolated networks may need an explicit peer
   configuration. A working Internet route is not enough.
3. Run the weight-free demo. Read the chosen path/interface, node compute,
   communication cost, missing replies and memory fit. A saved compute profile
   does not eliminate the need to check the current cables/routes.
4. Continue to model setup only when all intended nodes participate. The web
   console's path report lets you check that Ethernet/Thunderbolt won over a
   slower Wi-Fi path. Persistent machine IDs prevent counting two interfaces
   of one node as two computers.

No router port forwarding is needed. The chat API and console stay on the
head's loopback interface by default. Do not publish the pairing code or expose
cluster ports to the Internet: authentication does not encrypt the traffic.
The [protocol guide](PROTOCOL.md) lists ports and security boundaries precisely.

If a node is missing, check power, its running service, unique name, matching
pairing code, IPv4 address/mask, route and local-network/firewall permission.
If it is slow, inspect link speed and Fronk's per-node compute/arrival timings.
VPNs, USB adapters, shared switch uplinks, memory pressure and competing CPU
work are distinct possible costs; measure before changing several at once.

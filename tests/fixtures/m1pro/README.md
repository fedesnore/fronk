# Recorded M1 Pro hardware

Captured over authorized SSH on 2026-10-28.
Files are raw `uname -a`, `sw_vers`, `vm_stat`, `cc --version`, `make --version`,
and `sysctl hw.machine hw.memsize hw.physicalcpu hw.logicalcpu machdep.cpu.brand_string`.

The machine is an eight-core Apple M1 Pro with 17,179,869,184 RAM bytes running
macOS 15.7.1. `test_hw` parses these files, including free + inactive + purgeable
pages: `(16644 + 391503 + 7627) * 16384` bytes at capture time. Live Metal
working-set capacity is queried separately and may further limit capacity.
Memory observations are not reservations.

The installed SDK reports a larger `HOST_VM_INFO64_COUNT` than macOS 15 returns
(40 versus 38). The probe accepts the returned structure when it contains all
fields actually consumed. `test_hw_vm` covers this compatibility case and
rejects replies missing required fields.

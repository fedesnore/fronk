# Recorded iMac hardware

Captured over authorized SSH on 2026-10-28.
The ED25519 fingerprint was explicitly confirmed by the user before login.
Raw command output is retained; credentials are not fixtures.

| File | Command |
| --- | --- |
| uname.txt | `uname -a` |
| hostname.txt | `hostname -s` |
| sysctl.txt | `sysctl hw.machine hw.memsize hw.physicalcpu hw.logicalcpu hw.cpufrequency_max machdep.cpu.brand_string` |
| sysctl-optional.txt | `sysctl sysctl.proc_translated iogpu.wired_limit_mb 2>&1` |
| vm_stat.txt | `vm_stat` |
| sw_vers.txt | `sw_vers` |
| gpu.txt | `system_profiler SPDisplaysDataType` |
| compiler.txt | `cc --version` |
| make.txt | `make --version` |
| cpuid.txt | `cc -std=c11 -Wall -Wextra -Werror tests/capture_cpuid.c -o build/capture_cpuid && ./build/capture_cpuid` in the isolated validation directory |

The optional sysctl command reports missing keys; that is raw evidence, not a
fabricated value. The machine is an i5-7600K, four physical/logical cores,
51,539,607,552 RAM bytes, macOS 13.7.8, Radeon Pro 580. The captured available
memory calculation is `(4143925 + 3117437 + 143853) * 4096` bytes; subtract
3,000,000,000 bytes for the default node cap. Memory changes between samples.
`test_hw` parses the recorded sysctl, VM and OS-version outputs; ISA eligibility
uses actual CPUID/XCR0 register fixtures, never sysctl feature strings.

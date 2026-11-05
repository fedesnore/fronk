# Shuttle hardware capture

Captured read-only over the user-provided SSH connection on 2026-10-28.
Remote hostname: netlab-shuttle09. Raw outputs are stored without rewriting
values or normalizing whitespace. Commands completed successfully:

| File | Remote command |
| --- | --- |
| uname.txt | `uname -a` |
| hostname.txt | `hostname -s` |
| os-release.txt | `cat /etc/os-release` |
| cpuinfo.txt | `cat /proc/cpuinfo` |
| meminfo.txt | `cat /proc/meminfo` |
| lscpu.txt | `LC_ALL=C lscpu` |
| cpuinfo_max_freq.txt | `cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq` |

Identity: Debian GNU/Linux 13, Linux 6.12.107+deb13-amd64, Intel Core i5-10210U,
4 physical cores and 8 logical CPUs. cpuinfo_max_freq is in kHz: 4,200,000.
The captured memory inputs are 16,512,028,672 total bytes and 15,838,007,296
available bytes. With the newly specified Linux margin of 1,500,000,000 bytes,
the expected cap is 14,338,007,296 bytes, or four whole Q4_0 ownership blocks.
These are historical observations, not promises about current free memory.

The CPU feature strings are retained as raw evidence but are not used to
decide AVX2 eligibility. `command -v cpuid` exited 1: that utility is absent.
No package was installed, source compiled, service enabled, or setting changed
on the Shuttle. CPUID decoder tests use saved register values separately.
The existing optimized AVX2 engine remains the source to port at milestone 7;
it was not modified or executed during this capture.

During the subsequently authorized isolated validation builds, actual register
values were captured in `cpuid.txt` with
`cc -std=c11 -Wall -Wextra -Werror tests/capture_cpuid.c -o build/capture_cpuid && ./build/capture_cpuid`.
`test_hw` now exercises the ISA decoder with those register values. The existing
optimized engine is preserved separately in `port_sources/shuttle-avx2/`, with
its original provenance and per-file hashes; the remote sources remain intact.

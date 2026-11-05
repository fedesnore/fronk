# Hardware fixture provenance

`macos-rosetta.txt` is an actual report captured on this MacBook Pro M4 Pro
(macOS 26.6.2) on 2026-10-27 by the x86_64 sanitizer build:

```sh
cc -arch x86_64 -std=c11 -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE -Wall -Wextra -Werror -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all tests/test_hw.c src/hw.c -o build/test/test_hw-x86_64
arch -x86_64 ./build/test/test_hw-x86_64 tests/fixtures/hw/macos-rosetta.txt
```

It records a translated process on Apple hardware, not an Intel Mac. Preserve
this snapshot; its available-memory number is historical, not a test of current
free memory. An independent temporary `cpuid.h`/`xgetbv` diagnostic recorded:

```text
leaf1_ecx=0x0298220f leaf7_ebx=0x00000000 xcr0=0x0000000000000000
```

XGETBV was not executed because XSAVE/OSXSAVE were not advertised. The fixed
historical backend rule selected ref. The binding revision now selects `none`
and excludes translated processes irrespective of instruction flags. Earlier direct AVX2/FMA/F16C execution
checks passed under Rosetta; those checks and conservative CPUID-based selection
measure different things. No instruction probing or forced override is added
to production discovery.

Linux `/proc/cpuinfo` and `/proc/meminfo` fixtures in test_hw.c are explicitly
synthetic. They cover SMT topology, missing topology, repeated IDs, memory units,
missing keys and overflow. They are not measurements of the Shuttle. The Intel
Mac/backend policy tests likewise use synthetic typed inputs. The original capture preceded remote validation. The milestone 4 revision
adds real raw fixtures and CPUID captures in `../shuttle/`, `../imac/`, and
`../m1pro/`, alongside native execution on those authorized SSH nodes.
`MemAvailable` is an OS estimate of memory available for starting applications,
not simply MemFree; see the [Linux kernel /proc documentation](https://docs.kernel.org/filesystems/proc.html).

On macOS, runtime discovery reads sysctl values and Mach VM counters directly.
Metal's device name and recommendedMaxWorkingSetSize are queried through the
SDK APIs in backends/metal.m, without a shader, allocation, or setting change.

# One native build and a separate sanitizer build, using only the system C tools.
# Keep make compatible with macOS GNU make 3.81.

VERSION := 0.3.0
SYSTEM := $(shell uname -s)
ARCH := $(shell uname -m)
ifeq ($(SYSTEM),Darwin)
  OS := macos
  PLATFORM_FLAGS := -D_DARWIN_C_SOURCE
  ifeq ($(filter $(ARCH),arm64 x86_64),)
    $(error Unsupported macOS architecture: $(ARCH))
  endif
else ifeq ($(SYSTEM),Linux)
  OS := linux
  PLATFORM_FLAGS := -D_DEFAULT_SOURCE
  ifeq ($(filter $(ARCH),aarch64 arm64 x86_64),)
    $(error Unsupported Linux architecture: $(ARCH))
  endif
else
  $(error Unsupported platform: $(SYSTEM) $(ARCH))
endif

CFLAGS ?= -O2
REQUIRED_FLAGS := -std=c11 -Wall -Wextra -Werror -g
BUILD_FLAGS := -D_POSIX_C_SOURCE=200809L $(PLATFORM_FLAGS) -DFRONK_DEBUG=1 \
               -DFR_VERSION='"$(VERSION)"' \
               -DFR_BUILD_OS='"$(OS)"' -DFR_BUILD_ARCH='"$(ARCH)"'
SANITIZERS := -O1 -fno-omit-frame-pointer -fsanitize=address,undefined \
              -fno-sanitize-recover=all
TESTS := runtime mtp-worker prep head plane selection planner fanout wire interfaces model test resp ae net udp config sha256 auth auth-entropy stats control probe hw hw-failure identity q4 native backend vectors bench ssd
HW := src/hw.c
ifeq ($(OS),macos)
  TESTS += hw-vm
  HW += backends/metal.m
  HW_LIBS := -framework Foundation -framework Metal -framework SystemConfiguration -framework CoreFoundation -framework IOKit
  ifeq ($(ARCH),arm64)
    COMPUTE_FLAGS := -DFR_METAL_COMPUTE=1
    TESTS += metal
  endif
endif
CRYPTO := src/sha256.c src/auth.c
CONNECTION := src/interfaces.c src/connection.c src/wire.c src/stats.c src/control.c src/identity.c
DISCOVERY := src/discovery.c src/planner.c src/selection.c $(HW)
BACKEND := src/backend.c src/q4.c src/native.c src/vectors.c src/bench.c src/ssd.c backends/ref.c
COMPUTE_FLAGS += -DFR_BENCH_RUNTIME=1
ifneq ($(filter $(ARCH),arm64 aarch64),)
  COMPUTE_FLAGS += -DFR_NEON_COMPUTE=1
  BACKEND += backends/neon.c
  TESTS += neon
endif
ifeq ($(ARCH),x86_64)
  COMPUTE_FLAGS += -DFR_AVX2_COMPUTE=1
  BACKEND += backends/avx2.c
  TESTS += avx2
endif
# Your own backend: backends/user.c defines fr_user (BACKENDS.md). Build another
# file in its place with make USER_BACKEND=backends/user-template.c.
USER_BACKEND ?= $(wildcard backends/user.c)
ifneq ($(USER_BACKEND),)
  COMPUTE_FLAGS += -DFR_USER_BACKEND=1
  BACKEND += $(USER_BACKEND)
endif
# Generate static registration from small source manifests. Rewriting only when
# inputs change also makes adding/removing a module invalidate an old binary.
MODULE_CONFIGURED := $(shell python3 tools/backend-module.py configure --system $(SYSTEM) --arch $(ARCH) && echo ready)
ifneq ($(MODULE_CONFIGURED),ready)
  $(error Backend module configuration failed; see the diagnostic above)
endif
include build/backend-modules/modules.mk
COMPUTE_FLAGS += -DFR_BACKEND_MODULES=1 -Ibuild/backend-modules -Isrc
BACKEND += $(FR_MODULE_SOURCES)
LDLIBS += $(FR_MODULE_LDLIBS)
BACKEND_DEPS := $(BACKEND) $(HW) src/sha256.c
PLANE := src/plane.c src/fanout.c src/units.c src/prep.c
CORE := src/ae.c src/resp.c src/net.c src/config.c src/runtime.c src/node.c src/head.c src/serve.c src/setup.c src/hybrid.c $(PLANE) $(CONNECTION) $(DISCOVERY) $(CRYPTO) $(BACKEND)
LDLIBS += -lm -pthread
CLI := tools/fronk-cli.c src/control.c src/net.c src/resp.c src/config.c $(CRYPTO)
HEADERS := $(wildcard src/*.h) $(wildcard backends/*.inc) $(FR_MODULE_INPUTS) build/backend-modules/registry.h build/backend-modules/modules.mk

.PHONY: all dist gpu-head test-installer fronk-extract test test-core $(addprefix test-,$(TESTS)) test-main test-node test-cli test-local-cluster test-interop test-connection test-discovery test-bench-control test-topology clean
# fronk-extract (PDF and photo text for a head's chat) needs macOS and swiftc.
ALL := fronk fronk-cli
ifeq ($(OS),macos)
  ALL += $(if $(wildcard tools/native/extract.swift),fronk-extract)
endif
all: $(ALL)

# Generation is deliberately not a dependency of tests: committed vectors are
# read-only inputs. Use an empty, existing directory with the explicit tool.
build/fronk-vectors: tools/fronk-vectors.c $(BACKEND_DEPS) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) tools/fronk-vectors.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/fronk-vectors: tools/fronk-vectors.c $(BACKEND_DEPS) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) $(SANITIZERS) tools/fronk-vectors.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_mtp-worker: tests/test_mtp-worker.c tools/mtp/fronk_mtp_worker.inc tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_runtime: tests/test_runtime.c src/runtime.c src/runtime.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_runtime.c src/runtime.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_fanout: tests/test_fanout.c src/fanout.c src/interfaces.c src/wire.c src/net.c $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_fanout.c src/fanout.c src/interfaces.c src/wire.c src/net.c $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_planner: tests/test_planner.c src/planner.c src/selection.c $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_planner.c src/planner.c src/selection.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_selection: tests/test_selection.c src/selection.c $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_selection.c src/selection.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_wire: tests/test_wire.c src/wire.c $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_wire.c src/wire.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_plane: tests/test_plane.c src/plane.c src/fanout.c src/interfaces.c src/wire.c src/net.c $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_plane.c src/plane.c src/fanout.c src/interfaces.c src/wire.c src/net.c $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_interfaces: tests/test_interfaces.c src/interfaces.c $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_interfaces.c src/interfaces.c $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_prep: tests/test_prep.c src/prep.c src/sha256.c $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_prep.c src/prep.c src/sha256.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_head: tests/test_head.c $(CORE) $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) $(SANITIZERS) tests/test_head.c $(CORE) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_q4: tests/test_q4.c src/q4.c $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_q4.c src/q4.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_bench: tests/test_bench.c src/bench.c $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_bench.c src/bench.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_backend: tests/test_backend.c $(BACKEND_DEPS) $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) $(SANITIZERS) tests/test_backend.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_native: tests/test_native.c $(BACKEND_DEPS) $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) $(SANITIZERS) tests/test_native.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_ssd: tests/test_ssd.c $(BACKEND_DEPS) $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) $(SANITIZERS) tests/test_ssd.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_vectors: tests/test_vectors.c $(BACKEND_DEPS) $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) $(SANITIZERS) tests/test_vectors.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_metal: tests/test_metal.c $(BACKEND_DEPS) $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) $(SANITIZERS) tests/test_metal.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_neon: tests/test_neon.c $(BACKEND_DEPS) $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) $(SANITIZERS) tests/test_neon.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_avx2: tests/test_avx2.c $(BACKEND_DEPS) $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) $(SANITIZERS) tests/test_avx2.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/fronk-bench: tools/fronk-bench.c $(BACKEND_DEPS) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) tools/fronk-bench.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@
build/fronk-gap-bench: tools/fronk-gap-bench.c $(BACKEND_DEPS) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) tools/fronk-gap-bench.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/fronk-plane-test: tests/test_plane_cluster.c $(PLANE) $(CONNECTION) src/net.c src/resp.c src/config.c $(CRYPTO) $(BACKEND_DEPS) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) tests/test_plane_cluster.c $(PLANE) $(CONNECTION) src/net.c src/resp.c src/config.c $(CRYPTO) $(filter-out src/sha256.c,$(BACKEND_DEPS)) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

test-plane-cluster: build/fronk-plane-test fronk fronk-cli
	FRONK_BASE_PORT=27700 FRONK_BIN="$(CURDIR)/fronk" FRONK_CLI="$(CURDIR)/fronk-cli" FRONK_CONNECTION_TEST="$(CURDIR)/build/fronk-plane-test" ./tests/local_cluster.sh 3 sh ./tests/connection.sh

build/avx2-probe: tools/avx2-probe.c $(BACKEND_DEPS) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) tools/avx2-probe.c $(filter-out backends/avx2.c,$(BACKEND_DEPS)) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/avx2-mtp-probe: tools/avx2-mtp-probe.c $(BACKEND_DEPS) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) tools/avx2-mtp-probe.c $(filter-out backends/avx2.c,$(BACKEND_DEPS)) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/fronk-metal-bench: tools/fronk-metal-bench.c $(BACKEND_DEPS) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) tools/fronk-metal-bench.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

# Experiment only: no registration in backend.c and no fixed-rule change.
build/neon-probe: tools/neon-probe.c $(BACKEND_DEPS) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) -O3 -march=armv8.2-a+dotprod+fp16 $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) tools/neon-probe.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/neon-probe: tools/neon-probe.c $(BACKEND_DEPS) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) -march=armv8.2-a+dotprod+fp16 $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) $(SANITIZERS) tools/neon-probe.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

# Text from PDFs and photos for the chat (PDFKit, Vision), macOS only.
fronk-extract: runs/bin/fronk-extract
runs/bin/fronk-extract: tools/native/extract.swift
	@if [ "$$(uname)" = Darwin ] && command -v swiftc >/dev/null 2>&1; then mkdir -p runs/bin && swiftc -O $< -o $@; \
	else echo "fronk-extract: skipped (macOS with swiftc only; a head's chat reads PDFs and photos with it)"; fi

# ds4 linked into the head (`ds4 builtin` in fronk.conf): DS4 names a ds4-fronk
# tree where `make libds4-fronk.a` has run. Without it the head starts
# an explicitly configured dense engine as a child process (the Vulkan adapter
# uses this interface). `make head` builds the ordinary in-process Metal engine.
DS4 ?= $(CURDIR)/build/ds4-source
ifneq ($(wildcard $(DS4)/libds4-fronk.a),)
  DS4_FLAGS := -DFR_DS4_INPROCESS=1 -DFR_DS4_DIR='"$(DS4)"'
  DS4_LIBS := $(DS4)/libds4-fronk.a -framework Foundation -framework Metal
  DS4_DEPS := $(DS4)/libds4-fronk.a
endif
# Only compute code, its build inputs and self-test vectors determine whether a
# cached measurement still applies. Rebuilding the web UI or the head orchestration
# must not charge every machine for another sweep. In particular plane/control
# structs and planner policy do not alter a backend measurement. Cache schema
# versions live in the measurement code, which remains fingerprinted here.
BENCH_HEADERS := src/backend.h src/bench.h src/hw.h src/model.h src/native.h src/q4.h src/metal.h src/neon.h src/avx2.h src/ssd.h src/sha256.h src/vectors.h
BENCH_INPUTS := $(sort $(BACKEND_DEPS) src/hybrid.c $(BENCH_HEADERS) $(wildcard backends/*.inc) $(wildcard tests/vectors/*.txt) $(FR_MODULE_INPUTS) build/backend-modules/registry.h build/backend-modules/modules.mk)
BENCH_DIGEST = $(shell { $(CC) --version; printf '%s\n' "$(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(COMPUTE_FLAGS) $(LDFLAGS) $(PLATFORM_FLAGS) $(ARCH)"; cat $(BENCH_INPUTS); } | $(if $(filter macos,$(OS)),shasum -a 256,sha256sum) | cut -d' ' -f1)
FRONK_TARGET ?= fronk
$(FRONK_TARGET): src/main.c $(CORE) $(HEADERS) Makefile $(DS4_DEPS) $(BENCH_INPUTS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) -DFR_BENCH_BUILD='"$(BENCH_DIGEST)"' $(DS4_FLAGS) src/main.c $(CORE) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) $(DS4_LIBS) -o $@

fronk-cli: $(CLI) $(HEADERS) Makefile
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(CLI) $(LDFLAGS) $(LDLIBS) -o $@

build/test:
	mkdir -p $@

build/test/fronk: src/main.c $(CORE) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) $(SANITIZERS) src/main.c $(CORE) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/fronk-cli: $(CLI) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) $(CLI) $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_%: tests/test_%.c tests/test.h src/model.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_resp: tests/test_resp.c src/resp.c src/resp.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_resp.c src/resp.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_ae: tests/test_ae.c src/ae.c src/ae.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_ae.c src/ae.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_net: tests/test_net.c src/net.c src/net.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_net.c src/net.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_udp: tests/test_udp.c src/net.c src/net.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_udp.c src/net.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_stats: tests/test_stats.c src/stats.c src/stats.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_stats.c src/stats.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_identity: tests/test_identity.c src/identity.c src/identity.h $(CRYPTO) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_identity.c src/identity.c $(CRYPTO) $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_hw: tests/test_hw.c tests/fixtures/hw/macos-rosetta.txt tests/fixtures/shuttle/cpuinfo.txt tests/fixtures/shuttle/meminfo.txt $(HW) src/hw.h src/identity.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_hw.c $(HW) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_hw-failure: tests/test_hw_failure.c $(HW) src/hw.h src/identity.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) -Duname=fr_test_uname tests/test_hw_failure.c $(HW) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_hw-vm: tests/test_hw_vm.c $(HW) src/hw.h src/identity.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) -Dhost_statistics64=fr_test_host_statistics64 tests/test_hw_vm.c $(HW) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test/test_probe: tests/test_probe.c src/wire.c $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_probe.c src/wire.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_control: tests/test_control.c src/control.c src/net.c src/resp.c $(CRYPTO) $(HEADERS) tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_control.c src/control.c src/net.c src/resp.c $(CRYPTO) $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_config: tests/test_config.c src/config.c src/config.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_config.c src/config.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_sha256: tests/test_sha256.c src/sha256.c src/sha256.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_sha256.c src/sha256.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_auth: tests/test_auth.c src/auth.c src/auth.h src/sha256.c src/sha256.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_auth.c src/auth.c src/sha256.c $(LDFLAGS) $(LDLIBS) -o $@

build/test/test_auth-entropy: tests/test_auth_entropy.c src/auth.c src/auth.h src/sha256.c src/sha256.h tests/test.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) -Dgetentropy=fr_test_getentropy -Dgetrandom=fr_test_getrandom tests/test_auth_entropy.c src/auth.c src/sha256.c $(LDFLAGS) $(LDLIBS) -o $@

$(addprefix test-,$(TESTS)): test-%: build/test/test_%
	./$<

test-main: build/test/test_main build/test/fronk
	./build/test/test_main ./build/test/fronk

build/test/test_node: tests/test_node.c tests/server.h tests/test.h src/net.c src/resp.c src/wire.c $(CRYPTO) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_node.c src/net.c src/resp.c src/wire.c $(CRYPTO) $(LDFLAGS) $(LDLIBS) -o $@

test-node: build/test/test_node build/test/fronk
	./build/test/test_node ./build/test/fronk

build/test/test_cli: tests/test_cli.c tests/server.h tests/test.h src/net.c src/net.h Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_cli.c src/net.c $(LDFLAGS) $(LDLIBS) -o $@

test-cli: build/test/test_cli build/test/fronk build/test/fronk-cli
	./build/test/test_cli ./build/test/fronk ./build/test/fronk-cli

test-local-cluster: build/test/test_local_cluster
	./build/test/test_local_cluster

build/test/test_redis: tests/test_redis.c tests/test.h src/net.c $(CRYPTO) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_redis.c src/net.c $(CRYPTO) $(LDFLAGS) $(LDLIBS) -o $@

test-interop: build/test/fronk build/test/fronk-cli build/test/test_redis
	FRONK_BIN="$(CURDIR)/build/test/fronk" FRONK_CLI="$(CURDIR)/build/test/fronk-cli" FRONK_REDIS_TEST="$(CURDIR)/build/test/test_redis" ./tests/local_cluster.sh 3 sh ./tests/interop.sh

build/test/test_connection: tests/test_connection.c tests/test.h $(CONNECTION) src/net.c src/resp.c src/config.c $(CRYPTO) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_connection.c $(CONNECTION) src/net.c src/resp.c src/config.c $(CRYPTO) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

test-connection: build/test/test_connection build/test/fronk build/test/fronk-cli
	FRONK_BASE_PORT=18700 FRONK_BIN="$(CURDIR)/build/test/fronk" FRONK_CLI="$(CURDIR)/build/test/fronk-cli" FRONK_CONNECTION_TEST="$(CURDIR)/build/test/test_connection" ./tests/local_cluster.sh 3 sh ./tests/connection.sh

build/test/test_discovery: tests/test_discovery.c tests/test.h $(DISCOVERY) $(CONNECTION) src/bench.c src/net.c src/resp.c src/config.c $(CRYPTO) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_discovery.c $(DISCOVERY) $(CONNECTION) src/bench.c src/net.c src/resp.c src/config.c $(CRYPTO) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

test-discovery: build/test/test_discovery build/test/fronk build/test/fronk-cli
	FRONK_BASE_PORT=20700 FRONK_BIN="$(CURDIR)/build/test/fronk" FRONK_CLI="$(CURDIR)/build/test/fronk-cli" FRONK_CONNECTION_TEST="$(CURDIR)/build/test/test_discovery" ./tests/local_cluster.sh 3 sh ./tests/connection.sh

# Opt-in full-resident GPU experiment; do not allocate 10 GB in every unit run.
build/test/test_selection_control: tests/test_selection_control.c tests/test.h tests/server.h $(DISCOVERY) $(CONNECTION) src/bench.c src/net.c src/resp.c src/config.c $(CRYPTO) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_selection_control.c $(DISCOVERY) $(CONNECTION) src/bench.c src/net.c src/resp.c src/config.c $(CRYPTO) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

test-selection-control: build/test/test_selection_control fronk
	./build/test/test_selection_control ./fronk

build/test/test_bench_control: tests/test_bench_control.c tests/test.h $(DISCOVERY) $(CONNECTION) src/bench.c src/net.c src/resp.c src/config.c $(CRYPTO) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_bench_control.c $(DISCOVERY) $(CONNECTION) src/bench.c src/net.c src/resp.c src/config.c $(CRYPTO) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

test-bench-control: build/test/test_bench_control build/test/fronk build/test/fronk-cli
	FRONK_BASE_PORT=24700 FRONK_BIN="$(CURDIR)/build/test/fronk" FRONK_CLI="$(CURDIR)/build/test/fronk-cli" FRONK_CONNECTION_TEST="$(CURDIR)/build/test/test_bench_control" ./tests/local_cluster.sh 1 sh ./tests/connection.sh

build/test/test_topology: tests/test_topology.c tests/test.h $(DISCOVERY) $(CONNECTION) src/bench.c src/net.c src/resp.c src/config.c $(CRYPTO) $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(SANITIZERS) tests/test_topology.c $(DISCOVERY) $(CONNECTION) src/bench.c src/net.c src/resp.c src/config.c $(CRYPTO) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

test-topology: build/test/test_topology build/test/test_fanout build/test/fronk build/test/fronk-cli
	sh ./tests/topology.sh

test-core: $(addprefix test-,$(TESTS)) test-main test-local-cluster test-node test-cli test-connection test-discovery
	@echo "all core tests passed"

test: test-core test-interop
	@echo "all tests passed"

clean:
	rm -rf build fronk fronk-cli

# Native release engines. Apple Silicon keeps its built-in ds4 path. The Intel
# Mac runtime is cross-built once and distributed inside the same manager.
.PHONY: engine head
engine:
	@if [ -f "$(DS4)/.fronk-source" ] || [ ! -f "$(DS4)/Makefile" ]; then bash tools/engine/prepare.sh "$(DS4)"; \
	else echo "Using externally supplied engine source: $(DS4)"; fi

# A public checkout can build the same ordinary Metal head as the installer.
head: engine
	bash tools/mtp/stage-ds4.sh "$(DS4)" build/head-engine
	$(MAKE) -C build/head-engine libds4-fronk.a
	$(MAKE) DS4="$(CURDIR)/build/head-engine" fronk fronk-cli

dist: engine

ifeq ($(OS),macos)
gpu-head: build/gpu-runtime/Darwin-x86_64/.built

build/gpu-runtime/Darwin-x86_64/.built: $(filter-out tools/gpu-head/run.sh,$(wildcard tools/gpu-head/*)) client/fronk_client.h
	tools/gpu-head/macos-dist.sh
	touch $@

dist: gpu-head
else
gpu-head:
	tools/gpu-head/build.sh build/gpu-runtime/Linux-$(ARCH)
endif

# Build an opt-in Apple Silicon engine beside the working head.
.PHONY: mtp-head
mtp-head: build/mtp-engine/.prepared
	$(MAKE) -C build/mtp-engine libds4-fronk.a
	$(MAKE) FRONK_TARGET=build/fronk-mtp DS4="$(CURDIR)/build/mtp-engine" build/fronk-mtp

build/mtp-engine/.prepared: tools/mtp/stage-ds4.sh tools/mtp/ds4-tree.patch tools/mtp/fronk_mtp_tree.inc tools/mtp/fronk_mtp_worker.inc $(DS4)/Makefile $(DS4)/ds4.c $(DS4)/ds4_metal.m $(DS4)/ds4_gpu.h tools/mtp/patch-head.py tools/mtp/fronk_dense.metal
	tools/mtp/stage-ds4.sh "$(DS4)" build/mtp-engine
	touch $@

test-installer: dist
	bash tests/installer.sh

dist:
	DS4="$(DS4)" tools/install/make-dist.sh

# Partial-reuse experiments use their own engine objects and executable so
# changing this compile-time arithmetic policy cannot contaminate mtp-head.
.PHONY: mtp-ranked-head
mtp-ranked-head: build/mtp-ranked-engine/.prepared
	$(MAKE) -C build/mtp-ranked-engine CC="$(CC) -DFR_EXPERT_RANKED=1" libds4-fronk.a
	$(MAKE) FRONK_TARGET=build/fronk-mtp-ranked DS4="$(CURDIR)/build/mtp-ranked-engine" CFLAGS="$(CFLAGS) -DFR_EXPERT_RANKED=1" build/fronk-mtp-ranked

build/mtp-ranked-engine/.prepared: tools/mtp/stage-ds4.sh tools/mtp/ds4-tree.patch tools/mtp/fronk_mtp_tree.inc tools/mtp/fronk_mtp_worker.inc $(DS4)/Makefile $(DS4)/ds4.c $(DS4)/ds4_metal.m $(DS4)/ds4_gpu.h tools/mtp/patch-head.py tools/mtp/fronk_dense.metal
	tools/mtp/stage-ds4.sh "$(DS4)" build/mtp-ranked-engine
	touch $@

.PHONY: mtp-native-head mtp-path-profile-head
mtp-native-head: build/mtp-native-engine/.prepared build/mtp-native-engine/fronk_mtp_native.inc
	$(MAKE) -C build/mtp-native-engine CC="$(CC) -DFR_EXPERT_RANKED=1" libds4-fronk.a
	$(MAKE) FRONK_TARGET=build/fronk-mtp-native DS4="$(CURDIR)/build/mtp-native-engine" CFLAGS="$(CFLAGS) -DFR_EXPERT_RANKED=1" build/fronk-mtp-native

build/mtp-native-engine/.prepared: tools/mtp/stage-native.sh tools/mtp/patch-native.py tools/mtp/fronk_dense.metal tools/mtp/fronk_q8.metal tools/mtp/fronk_mtp_wave.inc tools/mtp/stage-ds4.sh tools/mtp/ds4-tree.patch tools/mtp/fronk_mtp_tree.inc tools/mtp/fronk_mtp_worker.inc $(DS4)/Makefile $(DS4)/ds4.c $(DS4)/ds4_metal.m $(DS4)/ds4_gpu.h tools/mtp/patch-head.py
	tools/mtp/stage-native.sh "$(DS4)" build/mtp-native-engine
	touch $@

build/mtp-native-engine/fronk_mtp_native.inc: tools/mtp/fronk_mtp_native.inc | build/mtp-native-engine/.prepared
	cp $< $@

# Separate diagnostic engine and executable. No extra path timers/counters in
# the ordinary builds; enable FRONK_PROFILE=1 when running this executable.
mtp-path-profile-head: build/mtp-path-engine/.prepared build/mtp-path-engine/fronk_mtp_native.inc
	$(MAKE) -C build/mtp-path-engine CC="$(CC) -DFR_EXPERT_RANKED=1 -DFRONK_PATH_PROFILE=1" libds4-fronk.a
	$(MAKE) FRONK_TARGET=build/fronk-path-profile DS4="$(CURDIR)/build/mtp-path-engine" CFLAGS="$(CFLAGS) -DFR_EXPERT_RANKED=1 -DFRONK_PATH_PROFILE=1" build/fronk-path-profile

build/mtp-path-engine/.prepared: tools/mtp/stage-native.sh tools/mtp/patch-native.py tools/mtp/fronk_dense.metal tools/mtp/fronk_q8.metal tools/mtp/fronk_mtp_wave.inc tools/mtp/stage-ds4.sh tools/mtp/ds4-tree.patch tools/mtp/fronk_mtp_tree.inc tools/mtp/fronk_mtp_worker.inc $(DS4)/Makefile $(DS4)/ds4.c $(DS4)/ds4_metal.m $(DS4)/ds4_gpu.h tools/mtp/patch-head.py
	tools/mtp/stage-native.sh "$(DS4)" build/mtp-path-engine
	touch $@

build/mtp-path-engine/fronk_mtp_native.inc: tools/mtp/fronk_mtp_native.inc | build/mtp-path-engine/.prepared
	cp $< $@

# Keep the two-group head trial separate from both the stable and qualified
# native binaries. FRONK_NATIVE_WAVE=1 selects it at process startup.
.PHONY: mtp-wave-head
mtp-wave-head: build/mtp-wave-engine/.prepared build/mtp-wave-engine/fronk_mtp_native.inc
	$(MAKE) -C build/mtp-wave-engine CC="$(CC) -DFR_EXPERT_RANKED=1" libds4-fronk.a
	$(MAKE) FRONK_TARGET=build/fronk-mtp-wave DS4="$(CURDIR)/build/mtp-wave-engine" CFLAGS="$(CFLAGS) -DFR_EXPERT_RANKED=1" build/fronk-mtp-wave

build/mtp-wave-engine/.prepared: tools/mtp/stage-native.sh tools/mtp/patch-native.py tools/mtp/fronk_dense.metal tools/mtp/fronk_q8.metal tools/mtp/fronk_mtp_wave.inc tools/mtp/stage-ds4.sh tools/mtp/ds4-tree.patch tools/mtp/fronk_mtp_tree.inc tools/mtp/fronk_mtp_worker.inc $(DS4)/Makefile $(DS4)/ds4.c $(DS4)/ds4_metal.m $(DS4)/ds4_gpu.h tools/mtp/patch-head.py
	tools/mtp/stage-native.sh "$(DS4)" build/mtp-wave-engine
	touch $@

build/mtp-wave-engine/fronk_mtp_native.inc: tools/mtp/fronk_mtp_native.inc | build/mtp-wave-engine/.prepared
	cp $< $@

.PHONY: test-mtp-dense
.PHONY: head-cost-head
head-cost-head: build/head-cost-engine/.prepared
	$(MAKE) -C build/head-cost-engine libds4-fronk.a
	$(MAKE) FRONK_TARGET=build/fronk-head-cost DS4="$(CURDIR)/build/head-cost-engine" build/fronk-head-cost

build/head-cost-engine/.prepared: tools/mtp/stage-head.sh tools/mtp/patch-head.py tools/mtp/fronk_dense.metal tools/mtp/stage-ds4.sh tools/mtp/ds4-tree.patch tools/mtp/fronk_mtp_tree.inc tools/mtp/fronk_mtp_worker.inc $(DS4)/Makefile $(DS4)/ds4.c $(DS4)/ds4_metal.m $(DS4)/ds4_gpu.h
	bash tools/mtp/stage-head.sh "$(DS4)" build/head-cost-engine
	touch $@

build/ssd-pipeline-probe: tools/ssd-pipeline-probe.c $(BACKEND_DEPS) $(HEADERS) Makefile
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) $(COMPUTE_FLAGS) tools/ssd-pipeline-probe.c $(BACKEND_DEPS) $(LDFLAGS) $(LDLIBS) $(HW_LIBS) -o $@

build/test-head-hc: tests/test_head_hc.c build/head-cost-engine/ds4_metal.o build/head-cost-engine/ds4_image.o
	$(CC) -O2 -Ibuild/head-cost-engine tests/test_head_hc.c build/head-cost-engine/ds4_metal.o build/head-cost-engine/ds4_image.o -framework Foundation -framework Metal -lm -o $@

test-mtp-dense: mtp-native-head
	$(CC) -O2 -Ibuild/mtp-native-engine tests/test_mtp_dense.c build/mtp-native-engine/ds4_metal.o build/mtp-native-engine/ds4_image.o -framework Foundation -framework Metal -lm -o build/test-mtp-dense
	./build/test-mtp-dense

# Read-only full-model release audit; deliberately separate from fast tests.
build/fronk-model-check: tools/fronk-model-check.c src/prep.c src/sha256.c $(HEADERS) Makefile | build/test
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_FLAGS) $(BUILD_FLAGS) tools/fronk-model-check.c src/prep.c src/sha256.c $(LDFLAGS) $(LDLIBS) -o $@

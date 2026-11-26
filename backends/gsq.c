/* gsq.c - the backend for GSQ units (layout.h: gsq; gsq.h: the unit format).
 *
 * CPU only: ggml's IQ2/IQ3 gate/up kernels (vendored, third_party/ggml), the
 * multi-token decode-once versions (src/gsq_multi.c, bit-identical to ggml) and
 * the down kernels (src/gsq_down.c), on x86 with AVX2/FMA/F16C or on ARM NEON.
 *
 * run(): x is quantized once per token (Q8_K, as llama.cpp's CPU path does); the
 * request is grouped by expert, so a slice that several tokens route to is read
 * and decoded once for all of them; a persistent pool splits the (expert, unit)
 * items and every worker adds into its own partial, summed at the end. Between
 * layers the workers spin (up to 50 ms idle, then they sleep): a condvar wake-up
 * costs up to milliseconds on macOS, 48 times per token. Units contain routed
 * experts only (the head computes the shared one), so a request with a shared
 * gate other than 0 is refused. run allocates no memory and does no I/O.
 */
#include "../src/backend.h"
#include "../src/bench.h"
#include "../src/gsq.h"
#include "../src/gsq_multi.h"
#include "../src/hw.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <pthread/qos.h>
#include <sys/sysctl.h>
#endif

#define MAXTH 64
#define MAXQ (FR_MAX_BURST * FR_ACTIVE_ROUTED)

static const unsigned char *unit[FR_BLOCKS];
/* A cache (hot.c) keeps whole experts in slots, not at their place in a unit: when
 * set, slice (layer, expert) of every loaded unit is at slots[layer][expert]. */
_Static_assert(FR_ROUTED_EXPERTS==512,"gsq.h declares the slot table with 512 experts a layer");
static const uint64_t (*slots)[FR_ROUTED_EXPERTS];
void fr_gsq_backend_slots(const uint64_t (*table)[FR_ROUTED_EXPERTS]) { slots = table; }
static uint32_t loaded_mask;
static unsigned n_loaded, loaded_ids[FR_BLOCKS];

static struct {
    unsigned layer, burst, n_uniq, total;
    uint16_t uniq[MAXQ];
    unsigned char n_use[MAXQ], use_t[MAXQ][FR_MAX_BURST];
    float use_w[MAXQ][FR_MAX_BURST];
    unsigned char xq[FR_MAX_BURST][FR_GSQ_Q8K_BYTES];
    atomic_uint next, failed;   /* failed: a slice was refused; the call must not return a partial sum */
} job;
static float partial[MAXTH][FR_MAX_BURST][FR_HIDDEN];
static unsigned n_threads;
static int pool_started;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t go = PTHREAD_COND_INITIALIZER;
static atomic_uint generation, finished, quitting;
static pthread_t threads[MAXTH];

static inline void relax(void) {
#if defined(__aarch64__)
    __asm__ volatile("yield");
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void fast_thread(void) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

static void run_share(unsigned me) {
    memset(partial[me], 0, sizeof(float) * job.burst * FR_HIDDEN);
    for (;;) {
        unsigned i = atomic_fetch_add(&job.next, 1);
        if (i >= job.total) break;
        unsigned q = i / n_loaded, u = loaded_ids[i % n_loaded];
        const uint64_t at = slots ? slots[job.layer][job.uniq[q]] : fr_gsq_offset(job.layer, job.uniq[q]);
        if (at == UINT64_MAX) { atomic_store(&job.failed, 1); continue; }   /* asked for an expert the cache does not hold */
        const unsigned char *slice = unit[u] + at;
        const void *xs[FR_MAX_BURST]; float *ys[FR_MAX_BURST];
        for (unsigned k = 0; k < job.n_use[q]; ++k) { xs[k] = job.xq[job.use_t[q][k]]; ys[k] = partial[me][job.use_t[q][k]]; }
        /* The kernel takes at most FR_GSQ_MULTI_MAX tokens per call and refuses more:
         * a call of FR_MAX_BURST tokens can route more than that to one expert. */
        for (unsigned k = 0; k < job.n_use[q]; k += FR_GSQ_MULTI_MAX) {
            const unsigned n = job.n_use[q] - k < FR_GSQ_MULTI_MAX ? job.n_use[q] - k : FR_GSQ_MULTI_MAX;
            if (fr_gsq_fast_slice_multi(job.layer, slice, xs + k, job.use_w[q] + k, ys + k, n)) atomic_store(&job.failed, 1);
        }
    }
}
static void *worker(void *arg) {
    unsigned me = (unsigned)(uintptr_t)arg, seen = 0;
    fast_thread();
    for (;;) {
        double t0 = now(); unsigned g, spins = 0;
        while ((g = atomic_load_explicit(&generation, memory_order_acquire)) == seen) {
            if (atomic_load(&quitting)) return NULL;
            relax();
            if (++spins % 1024 == 0 && now() - t0 > 0.05) {
                pthread_mutex_lock(&mu);
                while (atomic_load(&generation) == seen && !atomic_load(&quitting)) pthread_cond_wait(&go, &mu);
                pthread_mutex_unlock(&mu);
            }
        }
        seen = g;
        run_share(me);
        atomic_fetch_add_explicit(&finished, 1, memory_order_release);
    }
}
static unsigned default_threads(void) {
    const char *v = getenv("FRONK_GSQ_THREADS");
    if (v && atoi(v) > 0) return (unsigned)atoi(v) > MAXTH ? MAXTH : (unsigned)atoi(v);
#if defined(__APPLE__) && defined(__aarch64__)
    int n = 0; size_t len = sizeof(n);  /* performance cores only on Apple Silicon; the receiver uses an efficiency core */
    if (!sysctlbyname("hw.perflevel0.logicalcpu", &n, &len, NULL, 0) && n > 0) return (unsigned)n > MAXTH ? MAXTH : (unsigned)n;
#endif
    /* One core stays free for the node's receiver thread, which spins between
     * layers for low latency (plane.c): with every core computing, the receiver
     * and a worker took turns and the iMac's layers took 2.4x their BENCH time. */
    long n2 = sysconf(_SC_NPROCESSORS_ONLN) - 1;
    return n2 < 1 ? 1 : n2 > MAXTH ? MAXTH : (unsigned)n2;
}
static int pool_start(void) {
    if (pool_started) return FR_OK;
    fr_gsq_init();
    n_threads = default_threads();
    atomic_store(&quitting, 0);
    for (unsigned k = 1; k < n_threads; ++k)
        if (pthread_create(&threads[k], NULL, worker, (void *)(uintptr_t)k)) {
            atomic_store(&quitting, 1);
            pthread_mutex_lock(&mu); pthread_cond_broadcast(&go); pthread_mutex_unlock(&mu);
            for (unsigned j = 1; j < k; ++j) pthread_join(threads[j], NULL);
            return FR_STATE;
        }
    pool_started = 1;
    return FR_OK;
}
/* Stop the pool; refuses while units are borrowed (as fr_avx2_close does). */
int fr_gsq_backend_close(void) {
    if (loaded_mask) return FR_STATE;
    if (!pool_started) return FR_OK;
    atomic_store(&quitting, 1);
    pthread_mutex_lock(&mu); pthread_cond_broadcast(&go); pthread_mutex_unlock(&mu);
    for (unsigned k = 1; k < n_threads; ++k) pthread_join(threads[k], NULL);
    pool_started = 0;
    return FR_OK;
}

static int probe(fr_hwinfo *out) {
    if (!out) return FR_INVALID;
    fr_hwinfo h;
    if (hw_probe(&h)) return FR_IO;
#if defined(__x86_64__)
    if ((h.isa & FR_ISA_USABLE) != FR_ISA_USABLE) return FR_UNSUPPORTED;  /* the kernels need AVX2/FMA/F16C */
#elif !defined(__aarch64__)
    return FR_UNSUPPORTED;
#endif
    if (h.translated) return FR_UNSUPPORTED;
    *out = h; return FR_OK;
}
static int load(int block, const void *data, size_t len) {
    if (block < 0 || block >= FR_BLOCKS || !data || (uint64_t)len != fr_gsq_unit_bytes()) return FR_INVALID;
    if (loaded_mask >> block & 1) return FR_STATE;
    int rc = pool_start(); if (rc) return rc;
    unit[block] = data; loaded_mask |= 1u << block;
    n_loaded = 0; for (unsigned b = 0; b < FR_BLOCKS; ++b) if (loaded_mask >> b & 1) loaded_ids[n_loaded++] = b;
    return FR_OK;
}
static int unload(int block) {
    if (block < 0 || block >= FR_BLOCKS || !(loaded_mask >> block & 1)) return FR_STATE;
    unit[block] = NULL; loaded_mask &= ~(1u << block);
    n_loaded = 0; for (unsigned b = 0; b < FR_BLOCKS; ++b) if (loaded_mask >> b & 1) loaded_ids[n_loaded++] = b;
    return FR_OK;
}
static int run_impl(const fr_req *req, const unsigned char *xq, fr_part *out) {
    if (!req || !out || req->layer >= FR_LAYERS || !req->burst || req->burst > FR_MAX_BURST) return FR_INVALID;
    if (!loaded_mask || !pool_started) return FR_STATE;
    const unsigned b = req->burst;
    for (unsigned t = 0; t < b; ++t) {
        if (req->gate[t][FR_ACTIVE_ROUTED] != 0.0f) return FR_UNSUPPORTED;  /* the shared expert is the head's */
        for (unsigned j = 0; j < FR_ACTIVE_ROUTED; ++j) {
            if (req->expert[t][j] >= FR_ROUTED_EXPERTS || !isfinite(req->gate[t][j]) || req->gate[t][j] < 0) return FR_INVALID;
            for (unsigned k = 0; k < j; ++k) if (req->expert[t][k] == req->expert[t][j]) return FR_INVALID;
        }
        if (!xq) for (unsigned d = 0; d < FR_HIDDEN; ++d) if (!isfinite(req->x[t][d])) return FR_NUMERIC;
        if (xq) for (unsigned k = 0; k < FR_HIDDEN/256; ++k) {
            float scale; memcpy(&scale,xq+t*FR_GSQ_Q8K_BYTES+k*292,4);
            if (!isfinite(scale)) return FR_NUMERIC;
        }
    }
    job.layer = req->layer; job.burst = b; job.n_uniq = 0;
    for (unsigned t = 0; t < b; ++t) {
        if (xq) memcpy(job.xq[t],xq+t*FR_GSQ_Q8K_BYTES,FR_GSQ_Q8K_BYTES);
        else fr_gsq_quantize_x(req->x[t], job.xq[t]);
        for (unsigned j = 0; j < FR_ACTIVE_ROUTED; ++j) {
            unsigned e = req->expert[t][j], q = 0;
            if (req->gate[t][j] == 0.0f) continue;  /* adds nothing; the head sends 0 for experts it computes itself (hot.h) */
            while (q < job.n_uniq && job.uniq[q] != e) ++q;
            if (q == job.n_uniq) { job.uniq[q] = (uint16_t)e; job.n_use[q] = 0; ++job.n_uniq; }
            job.use_t[q][job.n_use[q]] = (unsigned char)t; job.use_w[q][job.n_use[q]] = req->gate[t][j]; ++job.n_use[q];
        }
    }
    job.total = job.n_uniq * n_loaded;
    atomic_store(&job.next, 0); atomic_store(&job.failed, 0);
    atomic_store(&finished, 0);
    atomic_fetch_add_explicit(&generation, 1, memory_order_release);
    pthread_mutex_lock(&mu); pthread_cond_broadcast(&go); pthread_mutex_unlock(&mu);
    run_share(0);
    while (atomic_load_explicit(&finished, memory_order_acquire) < n_threads - 1) relax();
    if (atomic_load(&job.failed)) return FR_STATE;
    for (unsigned t = 0; t < b; ++t)
        for (unsigned d = 0; d < FR_HIDDEN; ++d) {
            float s = 0;
            for (unsigned k = 0; k < n_threads; ++k) s += partial[k][t][d];
            if (!isfinite(s)) return FR_NUMERIC;
            out->y[t][d] = s;
        }
    /* Unused tokens are zero; a decode call must not pay for FR_MAX_BURST rows. */
    memset(out->y[b], 0, (FR_MAX_BURST - b) * sizeof(out->y[0]));
    out->layer = req->layer; out->burst = b;
    return FR_OK;
}
static int run(const fr_req *req,fr_part *out) { return run_impl(req,NULL,out); }
int fr_gsq_run_quantized(const fr_req *req,const void *xq,fr_part *out) {
    if (!xq) return FR_INVALID;
    return run_impl(req,xq,out);
}
static int bench(int burst_max, fr_costmodel *out) { return fr_bench_execute_backend(&fr_gsq, burst_max, out); }

/* Self-test: synthetic slices of every layer type (scales finite, codes random),
 * the fast path against gsq.c's scalar oracle (ggml's reference dequantizers,
 * double accumulation). Compared, as the contract defines it (backend.h), on a
 * token's whole partial: 10 routed experts, three tokens sharing some of them.
 * Fronk's backend tolerance, not a looser one. */
static int self_test(const char *vectors_dir) {
    (void)vectors_dir;
    if (loaded_mask) return FR_STATE;
    fr_gsq_init();
    enum { T = 3, E = FR_ACTIVE_ROUTED, S = E + 5 };  /* 15 distinct slices; token 1 shares 5 with token 0 */
    size_t cap = 0; for (unsigned l = 0; l < FR_LAYERS; ++l) if (fr_gsq_slice_bytes(l) > cap) cap = fr_gsq_slice_bytes(l);
    unsigned char *slices = malloc(cap * S); float *x = malloc(sizeof(float) * T * FR_HIDDEN), *y = malloc(sizeof(float) * T * FR_HIDDEN);
    double *ref = malloc(sizeof(double) * FR_HIDDEN); float *reff = malloc(sizeof(float) * FR_HIDDEN);
    unsigned char *xq = malloc((size_t)T * FR_GSQ_Q8K_BYTES);
    int rc = slices && x && y && ref && reff && xq ? FR_OK : FR_IO;
    uint32_t seed = 12345;
    for (unsigned l = 0; !rc && l < FR_LAYERS; l += 3) {  /* every gate and down type appears in this subset */
        for (unsigned k = 0; k < S; ++k) fr_gsq_fill_slice(l, slices + k * cap, seed + 31 * l + k);
        for (unsigned i = 0; i < T * FR_HIDDEN; ++i) { seed = seed * 1664525u + 1013904223u; x[i] = ((float)(seed >> 8) / 16777216.0f - 0.5f) * 4.0f; }
        for (unsigned t = 0; t < T; ++t) fr_gsq_quantize_x(x + t * FR_HIDDEN, xq + t * FR_GSQ_Q8K_BYTES);
        memset(y, 0, sizeof(float) * T * FR_HIDDEN);
        for (unsigned k = 0; k < S; ++k) {  /* slice k: token 0 uses k < 10, token 1 uses 5..14, token 2 uses even k */
            const void *xs[T]; float *ys[T]; float g[T]; unsigned n = 0;
            for (unsigned t = 0; t < T; ++t) {
                int uses = t == 0 ? k < E : t == 1 ? k >= 5 : k % 2 == 0;
                if (uses) { xs[n] = xq + t * FR_GSQ_Q8K_BYTES; ys[n] = y + t * FR_HIDDEN; g[n] = 0.05f + 0.01f * (float)((k + t) % 7); ++n; }
            }
            if (n && fr_gsq_fast_slice_multi(l, slices + k * cap, xs, g, ys, n)) { rc = FR_STATE; break; }
        }
        for (unsigned t = 0; !rc && t < T; ++t) {
            memset(ref, 0, sizeof(double) * FR_HIDDEN);
            for (unsigned k = 0; k < S; ++k) {
                int uses = t == 0 ? k < E : t == 1 ? k >= 5 : k % 2 == 0;
                if (uses && fr_gsq_ref_slice(l, slices + k * cap, x + t * FR_HIDDEN, 0.05f + 0.01f * (float)((k + t) % 7), ref)) rc = FR_STATE;
            }
            for (unsigned d = 0; d < FR_HIDDEN; ++d) reff[d] = (float)ref[d];
            fr_error e;
            if (!rc && (fr_compare(reff, y + t * FR_HIDDEN, FR_HIDDEN, &e) || !fr_conforms(&e, &fr_backend_tolerance))) rc = FR_NUMERIC;
        }
    }
    free(slices); free(x); free(y); free(ref); free(reff); free(xq);
    return rc;
}

const fr_backend fr_gsq = {"gsq", probe, load, unload, run, bench, self_test};

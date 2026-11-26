/* hot.c - see hot.h. */
#include "hot.h"
#include "gsq.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

_Atomic uint8_t fr_hot_owner[FR_LAYERS][FR_ROUTED_EXPERTS];

/* ---- store ---- */
#define HOT_CHUNK ((uint64_t)1<<20)   /* a unit's budget is a whole number of these; a multiple of the page size */
#define HOT_WIRE (16*HOT_CHUNK)       /* memory wired at a time in each unit, always before it is written */
static struct {
    const fr_gguf *g; unsigned char *unit[FR_BLOCKS]; size_t unit_bytes; uint32_t loaded; int open,held;
    /* One layout for the 20 units: the slice of an expert sits at the same offset in
     * each of them, in a slot. Slots fill a unit from its start, one after the other.
     * The memory behind them is wired once, a piece at a time before the first write,
     * and stays wired until the store closes: an expert that leaves only frees its
     * slot for another one. No page goes back to the system and comes again. (Giving
     * pages back does not work here: after mlock on parts of a mapping, macOS ignores
     * madvise(MADV_FREE) on it, the pages stay, lose their wiring and are compressed
     * and swapped: measured, 17 GB written to swap in one chat.) */
    uint64_t at[FR_LAYERS][FR_ROUTED_EXPERTS];     /* the expert's slot in every unit; UINT64_MAX: not here */
    uint32_t size[FR_LAYERS][FR_ROUTED_EXPERTS];   /* that slot's size */
    uint64_t free_at[FR_LAYERS*FR_ROUTED_EXPERTS]; uint32_t free_size[FR_LAYERS*FR_ROUTED_EXPERTS]; unsigned free_n;
    uint64_t capacity,top,wired;   /* per unit: the budget, the end of the slots, the end of the wired memory */
    uint64_t limit;                /* per unit: where the system refused to wire more (0: nowhere yet) */
} S;
int fr_hot_open(const fr_gguf *g,uint64_t budget_bytes) {
    if (S.open || !g || !budget_bytes) return -1;
    S.g=g; S.loaded=0; S.unit_bytes=(size_t)fr_gsq_unit_bytes(); S.held=1; S.free_n=0; S.top=S.wired=S.limit=0;
    S.capacity=budget_bytes/FR_BLOCKS/HOT_CHUNK*HOT_CHUNK;
    if (S.capacity>S.unit_bytes/HOT_CHUNK*HOT_CHUNK) S.capacity=S.unit_bytes/HOT_CHUNK*HOT_CHUNK;
    memset(S.at,0xff,sizeof(S.at));
    S.open=1;
    for (unsigned b=0;b<FR_BLOCKS;++b) {
        /* Address space only: memory comes when a piece is wired. */
        void *p=mmap(NULL,S.unit_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
        if (p==MAP_FAILED || fr_gsq.load((int)b,p,S.unit_bytes)) { if (p!=MAP_FAILED) munmap(p,S.unit_bytes); fr_hot_close(); return -1; }
        S.unit[b]=p; S.loaded|=1u<<b;
    }
    fr_gsq_backend_slots(S.at);
    return 0;
}
void fr_hot_close(void) {
    if (!S.open) return;
    fr_gsq_backend_slots(NULL);
    for (unsigned b=0;b<FR_BLOCKS;++b) if (S.loaded>>b&1) { fr_gsq.unload((int)b); munmap(S.unit[b],S.unit_bytes); S.unit[b]=NULL; }
    S.loaded=0; S.open=0;
}
int fr_hot_admit(unsigned layer,unsigned expert) {
    if (!S.open || layer>=FR_LAYERS || expert>=FR_ROUTED_EXPERTS) return -1;
    if (S.at[layer][expert]!=UINT64_MAX) return 0;
    const uint32_t need=(uint32_t)fr_gsq_slice_bytes(layer); uint64_t at; uint32_t size=need; int reuse=-1;
    for (unsigned i=0;i<S.free_n;++i)   /* a freed slot that fits, the tightest */
        if (S.free_size[i]>=need && (reuse<0 || S.free_size[i]<S.free_size[reuse])) { reuse=(int)i; if (S.free_size[i]==need) break; }
    if (reuse>=0) { at=S.free_at[reuse]; size=S.free_size[reuse]; }
    else {
        if (S.top+need>S.capacity) return 1;   /* the budget is full */
        at=S.top;
        /* Wired before a byte is written: a page written under memory pressure and wired
         * afterwards keeps a compressed twin (measured: 13 GB more, in swap). */
        while (at+need>S.wired) {
            const uint64_t step=S.capacity-S.wired<HOT_WIRE?S.capacity-S.wired:HOT_WIRE; unsigned b=0;
            while (b<FR_BLOCKS && !mlock(S.unit[b]+S.wired,(size_t)step)) ++b;
            if (b<FR_BLOCKS) {
                /* The system allows no more wired memory (the engine's own counts too). The
                 * cache ends here, all of it wired: memory that is not wired would be
                 * compressed and swapped. What this step had taken goes back untouched. */
                const int e=errno;
                while (b--) {
                    (void)munlock(S.unit[b]+S.wired,(size_t)step);
                    (void)mmap(S.unit[b]+S.wired,(size_t)step,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON|MAP_FIXED,-1,0);
                }
                S.capacity=S.wired;
                if (S.limit!=S.wired) fprintf(stderr,"fronk: the cache stops at %.1f GB: the system allows no more wired memory (mlock: %s)\n",(double)FR_BLOCKS*(double)S.wired/1e9,strerror(e));
                S.limit=S.wired;
                return 1;
            }
            S.wired+=step;
        }
    }
    /* An expert is routed here only after this returns: no request reads the slot
     * while it is being written. */
    for (unsigned b=0;b<FR_BLOCKS;++b) if (fr_gsq_slice(&S.g->gsq,b,layer,expert,S.unit[b]+at)) return -1;
    fr_gsq_forget(&S.g->gsq,layer,expert);   /* the copy is here: the file's pages are free to go */
    if (reuse>=0) { --S.free_n; S.free_at[reuse]=S.free_at[S.free_n]; S.free_size[reuse]=S.free_size[S.free_n]; }
    else S.top+=need;
    S.size[layer][expert]=size; S.at[layer][expert]=at;
    return 0;
}
/* The caller no longer sends requests for this expert (hot.h). Its slot is free for
 * another expert; the memory stays wired. */
int fr_hot_evict(unsigned layer,unsigned expert) {
    if (!S.open || layer>=FR_LAYERS || expert>=FR_ROUTED_EXPERTS) return -1;
    const uint64_t at=S.at[layer][expert];
    if (at==UINT64_MAX) return 0;
    S.at[layer][expert]=UINT64_MAX;
    S.free_at[S.free_n]=at; S.free_size[S.free_n++]=S.size[layer][expert];
    return 0;
}
static uint64_t per_unit(uint64_t budget) {
    const uint64_t c=budget/FR_BLOCKS/HOT_CHUNK*HOT_CHUNK,most=S.unit_bytes/HOT_CHUNK*HOT_CHUNK;
    return c>most?most:c;
}
uint64_t fr_hot_capacity(void) { return S.open?(uint64_t)FR_BLOCKS*S.capacity:0; }
int fr_hot_grow(uint64_t budget_bytes) {
    if (!S.open) return -1;
    uint64_t c=per_unit(budget_bytes);
    if (S.limit && c>S.limit) c=S.limit;   /* the system refused to wire beyond this */
    if (c>S.capacity) S.capacity=c;
    return 0;
}
unsigned fr_hot_above(uint64_t budget_bytes,uint16_t (*who)[2],unsigned max) {
    if (!S.open) return 0;
    const uint64_t cut=per_unit(budget_bytes); unsigned n=0;
    for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e)
        if (S.at[l][e]!=UINT64_MAX && S.at[l][e]+S.size[l][e]>cut) { if (n<max) { who[n][0]=(uint16_t)l; who[n][1]=(uint16_t)e; } ++n; }
    return n;
}
/* The pages above the new budget are unlocked and their mapping replaced by a fresh
 * one: the system has them back before this returns. (madvise would not do it, see
 * above; a replaced mapping does, and what was in it is not worth a swap write.) */
int fr_hot_shrink(uint64_t budget_bytes) {
    if (!S.open) return -1;
    const uint64_t cut=per_unit(budget_bytes); uint64_t top=0; unsigned kept=0; int rc=0;
    if (cut>=S.capacity) return 0;
    for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e) if (S.at[l][e]!=UINT64_MAX) {
        const uint64_t end=S.at[l][e]+S.size[l][e];
        if (end>cut) return -1;
        if (end>top) top=end;
    }
    for (unsigned i=0;i<S.free_n;++i) if (S.free_at[i]+S.free_size[i]<=cut) {
        if (S.free_at[i]+S.free_size[i]>top) top=S.free_at[i]+S.free_size[i];
        S.free_at[kept]=S.free_at[i]; S.free_size[kept++]=S.free_size[i];
    }
    S.free_n=kept;
    if (S.top>cut) S.top=top;   /* the next new slot starts after the last one that stays */
    if (S.wired>cut) {
        for (unsigned b=0;b<FR_BLOCKS;++b) {
            (void)munlock(S.unit[b]+cut,(size_t)(S.wired-cut));
            if (mmap(S.unit[b]+cut,(size_t)(S.wired-cut),PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON|MAP_FIXED,-1,0)==MAP_FAILED) rc=-1;
        }
        S.wired=cut;
    }
    S.capacity=cut; S.limit=0;   /* what the system allows may have changed by the next time */
    return rc;
}
void fr_hot_usage(uint64_t *bytes,int *held) { *bytes=S.open?(uint64_t)FR_BLOCKS*S.wired:0; *held=S.open && S.held; }
const unsigned char *fr_hot_peek(unsigned block,unsigned layer,unsigned expert) {
    return S.open && block<FR_BLOCKS && layer<FR_LAYERS && expert<FR_ROUTED_EXPERTS && S.at[layer][expert]!=UINT64_MAX?S.unit[block]+S.at[layer][expert]:NULL;
}

/* ---- policy ---- */
#define HOT_MAGIC 0x31544f48u   /* "HOT1" */
#define HOME_MAGIC 0x314d4f48u  /* "HOM1" */
#define HOT_ONE 256u            /* one use, in the fixed point of the recent scores */
#define HOT_MIN (2*HOT_ONE)     /* recent uses before an expert may enter a cache, and the margin of a replacement */
#define HOT_AGE_TOKENS 12       /* every that many tokens a score loses a sixteenth: it halves in about 130 tokens */
#define HOT_SEED 185u           /* the most used expert of earlier runs starts like one used at every token */
#define HOT_WINDOW 96           /* calls judged at a time: two decode passes */
#define HOT_SPARE 0.10          /* a cache last in fewer calls than this has time to spare: fill it quickly */
#define HOT_LOW 0.35            /* ... than this: give it more */
#define HOT_HIGH 0.55           /* ... in more than this, twice in a row: give it less */
#define HOT_LOADS 3.0           /* replacements per token of work, all caches together (state-dir/hot-loads-per-token) */
#define HOT_BATCH 256           /* replacements in one turn of one cache */
/* An elastic cache and the machine's spare memory, counted as the engine's guard
 * counts it (free, inactive, speculative and purgeable pages over all RAM; the guard
 * warns under 12% and stops the engine under 6%). */
#define HOT_MEM_LOW 0.11        /* less spare than this: the cache gives memory back at once ... */
#define HOT_MEM_TARGET 0.125    /* ... enough to have this much spare again */
/* 0.22 until 2026-11-11: the Mac generated with 16% spare all day and never swapped, so the
 * cache never grew; it now grows to the spare the Mac has had, not below (LOW is unchanged). */
/* 14/17.5% until the evening of 2026-11-11. Spare counts only free and inactive pages; the
 * Mac also holds 2-3 GB of active file pages it can drop without swapping, so 11% spare is
 * about 5 GB that can be given back. The engine's guard still stops everything at 6%. */
#define HOT_MEM_HIGH 0.14       /* more than this for two seconds: it may take a step more */
#define HOT_MEM_STEP (256ull<<20)
#define HOT_MEM_REST 60         /* seconds without growth after giving memory back: what grows is read from the disk */
typedef struct {
    char name[32]; double gb; fr_hot_admit_fn admit; fr_hot_evict_fn evict; void *ctx;
    const _Atomic uint64_t *calls,*last;
    _Atomic uint64_t held[FR_LAYERS][FR_ROUTED_EXPERTS/64];   /* what it has in memory */
    atomic_uint experts; atomic_int disabled;                 /* routed to it; it stopped answering */
    uint64_t served,calls0,last0; unsigned busy,swaps,catalogue; int full; double allowance;
    _Atomic long retry_at; unsigned backoff;   /* a silent cache is tried again: after 20 s, then twice as long, up to 5 minutes */
    uint64_t floor,ceiling; unsigned calm; long rest_until; _Atomic uint64_t asked;   /* elastic (this process's store): its bounds in bytes; checks in a row with memory to spare; no growth before this time */
} hot_cache;
typedef struct { uint32_t score; uint16_t layer,expert; } hot_pick;
static struct {
    hot_cache c[FR_HOT_CACHES]; unsigned caches; pthread_mutex_t lock;   /* one change at a time */
    const _Atomic uint64_t *begun,*done;
    uint32_t count[FR_LAYERS][FR_ROUTED_EXPERTS];             /* this run plus half of earlier runs: saved, seeds the next start */
    uint32_t recent[FR_LAYERS][FR_ROUTED_EXPERTS];            /* uses that fade (HOT_AGE_TOKENS): what the caches follow */
    uint8_t home[FR_LAYERS][FR_ROUTED_EXPERTS];               /* the one cache that may hold each expert (its catalogue) */
    uint64_t uses,since,uses0; atomic_int age_due; unsigned resume; double loads;
    /* state-dir/hot-trace present at start: every call's routing is kept in memory
     * (layer, tokens, then ten experts per token) and written to hot-trace.bin when
     * the head stops. For choosing the policy from real work; off otherwise. */
    unsigned char *trace; size_t trace_at,trace_cap; char trace_path[PATH_MAX+32];
    hot_pick waiting[FR_LAYERS*FR_ROUTED_EXPERTS],inside[FR_LAYERS*FR_ROUTED_EXPERTS];   /* the balance thread's lists */
    unsigned char taken[FR_LAYERS*FR_ROUTED_EXPERTS];
    pthread_t filler; atomic_int stop; int running,going; char path[PATH_MAX+32],homes[PATH_MAX+32];
} P={.lock=PTHREAD_MUTEX_INITIALIZER};
static int held(unsigned k,unsigned layer,unsigned expert) { return (int)(atomic_load(&P.c[k].held[layer][expert/64])>>(expert%64)&1); }
/* 0: routed to cache k, or already routed somewhere; 1: k's memory is full; -1: error. */
static int promote(unsigned k,unsigned layer,unsigned expert) {
    hot_cache *c=&P.c[k]; const uint64_t bit=1ull<<(expert%64); int rc=0;
    pthread_mutex_lock(&P.lock);
    if (atomic_load(&c->disabled)) rc=-1;
    else if (!fr_hot_where(layer,expert)) {
        if (!(atomic_load(&c->held[layer][expert/64])&bit) && !(rc=c->admit(c->ctx,layer,expert))) atomic_fetch_or(&c->held[layer][expert/64],bit);
        if (!rc) { atomic_store_explicit(&fr_hot_owner[layer][expert],(uint8_t)(k+1),memory_order_release); atomic_fetch_add(&c->experts,1); }
    }
    pthread_mutex_unlock(&P.lock);
    return rc;
}
static void demote(unsigned layer,unsigned expert) {   /* the slices compute it again from the next call */
    pthread_mutex_lock(&P.lock);
    const unsigned w=fr_hot_where(layer,expert);
    if (w) { atomic_store_explicit(&fr_hot_owner[layer][expert],0,memory_order_release); atomic_fetch_sub(&P.c[w-1].experts,1); }
    pthread_mutex_unlock(&P.lock);
}
/* An expert leaves a cache only when no call can still be asking for it: it is no
 * longer routed there (demote), and every call that began before has finished. A
 * call builds its views of the routing after it is counted as begun. */
static void settle(void) {
    const uint64_t begun=atomic_load(P.begun);
    while (atomic_load(P.done)<begun && !atomic_load(&P.stop)) usleep(500);
}
static int evict(unsigned k,unsigned layer,unsigned expert) {
    hot_cache *c=&P.c[k]; const uint64_t bit=1ull<<(expert%64); int rc=0;
    pthread_mutex_lock(&P.lock);
    if (fr_hot_where(layer,expert)==k+1 || !c->evict) rc=-1;
    else if ((atomic_load(&c->held[layer][expert/64])&bit) && !(rc=c->evict(c->ctx,layer,expert))) atomic_fetch_and(&c->held[layer][expert/64],~bit);
    pthread_mutex_unlock(&P.lock);
    return rc;
}
int fr_hot_policy_evict(unsigned layer,unsigned expert) {
    if (!P.running || layer>=FR_LAYERS || expert>=FR_ROUTED_EXPERTS) return -1;
    demote(layer,expert); settle();
    if (atomic_load(&P.stop)) return -1;
    int rc=0;
    for (unsigned k=0;k<P.caches;++k) if (held(k,layer,expert)) rc|=evict(k,layer,expert);
    return rc;
}
void fr_hot_policy_disable(unsigned k) {
    if (!P.running || k>=P.caches || atomic_exchange(&P.c[k].disabled,1)) return;
    /* One late reply (the machine was busy for a second) must not cost the cache for
     * the rest of the run: the balance thread routes experts to it again later. They
     * are still in its memory. If it is really gone, each try costs one late call. */
    if (P.c[k].backoff<20) P.c[k].backoff=20;
    atomic_store(&P.c[k].retry_at,(long)time(NULL)+(long)P.c[k].backoff);
    if (P.c[k].backoff<300) P.c[k].backoff*=2;
    pthread_mutex_lock(&P.lock);   /* no promotion is half done */
    for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e)
        if (fr_hot_where(l,e)==k+1) atomic_store_explicit(&fr_hot_owner[l][e],0,memory_order_release);
    atomic_store(&P.c[k].experts,0);
    pthread_mutex_unlock(&P.lock);
}
int fr_hot_policy_admit(unsigned k,unsigned layer,unsigned expert) {
    if (!P.running || k>=P.caches || layer>=FR_LAYERS || expert>=FR_ROUTED_EXPERTS) return -1;
    const int rc=promote(k,layer,expert);
    return rc?rc:fr_hot_where(layer,expert)==k+1?0:-1;
}
static int score_down(const void *a,const void *b) { const hot_pick *x=a,*y=b; return x->score<y->score?1:x->score>y->score?-1:0; }
static int score_up(const void *a,const void *b) { return score_down(b,a); }
/* One turn of cache k. Scores are read without the caller's lock: a stale value
 * only shifts a choice.
 *   room:  up to that many of its most used experts that wait are copied in while
 *          its memory has room.
 *   swaps: up to that many of its least used experts give their places to experts
 *          that wait and were used more, as one batch: route the cold ones back to
 *          the slices, wait once for the calls under way, free them, copy the hot
 *          ones in from the cache's own model file.
 *   back:  only the experts already in its memory (a cache tried again after a silence). */
static void turn(unsigned k,unsigned room,unsigned swaps,int back) {
    hot_cache *c=&P.c[k]; unsigned waiting=0,inside=0,next=0;
    for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e)
        if (P.home[l][e]==k && !fr_hot_has(l,e) && (back?held(k,l,e):P.recent[l][e]>=HOT_MIN)) P.waiting[waiting++]=(hot_pick){P.recent[l][e],(uint16_t)l,(uint16_t)e};
    if (!waiting) return;
    qsort(P.waiting,waiting,sizeof(P.waiting[0]),score_down);
    for (;next<waiting && next<room && !atomic_load(&P.stop);++next) {
        const int rc=promote(k,P.waiting[next].layer,P.waiting[next].expert);
        if (rc<0) { if (!atomic_load(&c->disabled)) fprintf(stderr,"fronk: cache %s refused an expert; the slice nodes keep computing it\n",c->name); return; }
        if (rc>0) { if (!c->full) { c->full=1; printf("HOT cache %s full at %u experts\n",c->name,atomic_load(&c->experts)); fflush(stdout); } break; }
    }
    if (!swaps || !c->evict || !c->full || next>=waiting || atomic_load(&P.stop)) return;
    for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e)
        if (held(k,l,e)) P.inside[inside++]=(hot_pick){P.recent[l][e],(uint16_t)l,(uint16_t)e};
    qsort(P.inside,inside,sizeof(P.inside[0]),score_up);
    /* Each expert that waits takes the slot of the least used one inside that is used
     * less (by HOT_MIN) and whose slice is as large: the same size when there is one,
     * so that no slot is left half empty; a larger one otherwise. */
    static unsigned cold[HOT_BATCH],hot[HOT_BATCH]; unsigned pairs=0;
    memset(P.taken,0,inside);
    for (unsigned i=next;i<waiting && pairs<swaps && pairs<HOT_BATCH;++i) {
        const size_t need=fr_gsq_slice_bytes(P.waiting[i].layer); int same=-1,larger=-1,any=0;
        for (unsigned j=0;j<inside && (uint64_t)P.inside[j].score+HOT_MIN<=P.waiting[i].score;++j) {
            any=1;
            if (P.taken[j]) continue;
            const size_t has=fr_gsq_slice_bytes(P.inside[j].layer);
            if (has==need) { same=(int)j; break; }
            if (has>need && larger<0) larger=(int)j;
        }
        if (!any) break;   /* nothing inside is used less than this one, nor than those after it */
        const int victim=same>=0?same:larger;
        if (victim<0) continue;
        P.taken[victim]=1; cold[pairs]=(unsigned)victim; hot[pairs++]=i;
    }
    if (!pairs) return;
    for (unsigned i=0;i<pairs;++i) if (fr_hot_where(P.inside[cold[i]].layer,P.inside[cold[i]].expert)==k+1) demote(P.inside[cold[i]].layer,P.inside[cold[i]].expert);
    settle();
    if (atomic_load(&P.stop)) return;
    for (unsigned i=0;i<pairs;++i) if (evict(k,P.inside[cold[i]].layer,P.inside[cold[i]].expert)) {
        c->evict=NULL; fprintf(stderr,"fronk: cache %s cannot give experts back (an older fronk on it?); it keeps the ones it has\n",c->name); return;
    }
    for (unsigned i=0;i<pairs && !atomic_load(&P.stop);++i) {
        const int rc=promote(k,P.waiting[hot[i]].layer,P.waiting[hot[i]].expert);
        if (rc<0) return;
        if (!rc) ++c->swaps;
    }
}
/* Exclusive catalogues: every expert has one cache that may hold it, so the head
 * knows where an expert can be without asking, and each cache machine needs on its
 * disk only its own part of the model. Walking the experts from the most used of
 * earlier runs to the least, each goes to the cache whose catalogue is furthest
 * below its share (its GB): every catalogue gets its part of the hot ones. Kept in
 * state-dir/hot-homes.bin so that a catalogue does not move between runs; written
 * once, again only when the caches or their sizes change. */
static void catalogues(void) {
    memset(P.home,0,sizeof(P.home));
    for (unsigned k=0;k<P.caches;++k) P.c[k].catalogue=k?0:FR_LAYERS*FR_ROUTED_EXPERTS;
    if (P.caches<2) return;
    struct { uint32_t magic,caches; char name[FR_HOT_CACHES][32]; uint32_t mgb[FR_HOT_CACHES]; } want,got;
    memset(&want,0,sizeof(want)); want.magic=HOME_MAGIC; want.caches=P.caches;
    for (unsigned k=0;k<P.caches;++k) { memcpy(want.name[k],P.c[k].name,sizeof(want.name[k])); want.mgb[k]=(uint32_t)(P.c[k].gb*1000); }
    FILE *f=fopen(P.homes,"rb"); int kept=0;
    if (f) { kept=fread(&got,sizeof(got),1,f)==1 && !memcmp(&got,&want,sizeof(want)) && fread(P.home,sizeof(P.home),1,f)==1; fclose(f); }
    if (!kept) {
        unsigned n=0; double got_n[FR_HOT_CACHES]={0};
        for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e) P.waiting[n++]=(hot_pick){P.count[l][e],(uint16_t)l,(uint16_t)e};
        qsort(P.waiting,n,sizeof(P.waiting[0]),score_down);
        for (unsigned i=0;i<n;++i) {
            unsigned to=0; for (unsigned k=1;k<P.caches;++k) if (got_n[k]/P.c[k].gb<got_n[to]/P.c[to].gb) to=k;
            P.home[P.waiting[i].layer][P.waiting[i].expert]=(uint8_t)to; got_n[to]+=1;
        }
        if ((f=fopen(P.homes,"wb"))) { if (fwrite(&want,sizeof(want),1,f)!=1 || fwrite(P.home,sizeof(P.home),1,f)!=1) fputs("fronk: could not write hot-homes.bin\n",stderr); fclose(f); }
    }
    for (unsigned k=0;k<P.caches;++k) P.c[k].catalogue=0;
    for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e) { if (P.home[l][e]>=P.caches) P.home[l][e]=0; ++P.c[P.home[l][e]].catalogue; }
    fputs("HOT catalogues (exclusive):",stdout);
    for (unsigned k=0;k<P.caches;++k) printf(" %s %u experts;",P.c[k].name,P.c[k].catalogue);
    printf(" %s\n",kept?"as in hot-homes.bin":"written to hot-homes.bin"); fflush(stdout);
}
/* The share of RAM the system can still hand out, and whether it is already short
 * (it says so, or it has swapped since the last look). -1: unknown here. */
static double memory_spare(int *alarm,uint64_t *total_bytes) {
#ifdef __APPLE__
    static uint64_t total,swapouts; static mach_port_t host; static int seen;
    vm_statistics64_data_t v; mach_msg_type_number_t n=HOST_VM_INFO64_COUNT; vm_size_t page=0;
    if (!host) host=mach_host_self();
    if (!total) { size_t len=sizeof(total); if (sysctlbyname("hw.memsize",&total,&len,NULL,0)) total=0; }
    if (!total || host_page_size(host,&page)!=KERN_SUCCESS || host_statistics64(host,HOST_VM_INFO64,(host_info64_t)&v,&n)!=KERN_SUCCESS) return -1;
    int level=1; size_t len=sizeof(level);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level",&level,&len,NULL,0)) level=1;
    *alarm=level>=2 || (seen && v.swapouts>swapouts);
    swapouts=v.swapouts; seen=1; *total_bytes=total;
    return (double)((uint64_t)v.free_count+v.inactive_count+v.purgeable_count)*(double)page/(double)total;
#else
    FILE *f=fopen("/proc/meminfo","r"); char line[128]; unsigned long long total=0,available=0,v;
    if (!f) return -1;
    while (fgets(line,sizeof(line),f)) {
        if (sscanf(line,"MemTotal: %llu kB",&v)==1) total=v;
        else if (sscanf(line,"MemAvailable: %llu kB",&v)==1) available=v;
    }
    fclose(f);
    if (!total) return -1;
    *alarm=0; *total_bytes=(uint64_t)total<<10;
    return (double)available/(double)total;
#endif
}
/* Cache k (elastic) down to `to` bytes: the experts in the slots above go back to the
 * slices, the calls under way finish, the memory returns to the system. */
static void give_back(unsigned k,uint64_t to,const char *why) {
    static uint16_t gone[FR_LAYERS*FR_ROUTED_EXPERTS][2];
    hot_cache *c=&P.c[k]; const uint64_t from=fr_hot_capacity();
    const unsigned n=fr_hot_above(to,gone,FR_LAYERS*FR_ROUTED_EXPERTS);
    for (unsigned i=0;i<n;++i) if (fr_hot_where(gone[i][0],gone[i][1])==k+1) demote(gone[i][0],gone[i][1]);
    settle();
    if (atomic_load(&P.stop)) return;
    for (unsigned i=0;i<n;++i) (void)evict(k,gone[i][0],gone[i][1]);
    pthread_mutex_lock(&P.lock); const int rc=fr_hot_shrink(to); pthread_mutex_unlock(&P.lock);
    c->full=1;
    printf("HOT cache %s: %.1f -> %.1f GB, %u experts back to the slices (%s)%s\n",c->name,(double)from/1e9,(double)fr_hot_capacity()/1e9,n,why,rc?"; some slots were still in use":"");
    fflush(stdout);
}
static void elastic(void) {
    for (unsigned k=0;k<P.caches;++k) {
        hot_cache *c=&P.c[k]; int alarm=0; uint64_t total=0;
        if (!c->ceiling || atomic_load(&c->disabled)) continue;
        const double spare=memory_spare(&alarm,&total);
        if (spare<0) return;
        const uint64_t now=fr_hot_capacity(),asked=atomic_exchange(&c->asked,0);
        if (asked) {   /* fr_hot_policy_resize: a size on request, within the bounds */
            const uint64_t to=asked<c->floor?c->floor:asked>c->ceiling?c->ceiling:asked;
            if (to<now) give_back(k,to,"asked");
            else if (to>now) { pthread_mutex_lock(&P.lock); (void)fr_hot_grow(to); pthread_mutex_unlock(&P.lock); if (fr_hot_capacity()>now) c->full=0; }
            c->calm=0;
            continue;
        }
        if ((alarm || spare<HOT_MEM_LOW) && now>c->floor) {
            uint64_t give=alarm?now-c->floor:(uint64_t)((HOT_MEM_TARGET-spare)*(double)total);
            if (give<HOT_MEM_STEP) give=HOT_MEM_STEP;
            give_back(k,now>c->floor+give?now-give:c->floor,alarm?"the system is short of memory":"memory is needed elsewhere");
            c->calm=0; c->rest_until=(long)time(NULL)+HOT_MEM_REST;
        } else if (!alarm && spare>HOT_MEM_HIGH && c->full && now<c->ceiling && (long)time(NULL)>=c->rest_until) {
            if (++c->calm<4) continue;
            uint64_t more=(uint64_t)((spare-(HOT_MEM_HIGH-0.02))*(double)total);
            if (more>HOT_MEM_STEP) more=HOT_MEM_STEP;
            const uint64_t to=now+more<c->ceiling?now+more:c->ceiling;
            pthread_mutex_lock(&P.lock); (void)fr_hot_grow(to); pthread_mutex_unlock(&P.lock);
            c->calm=0;
            const uint64_t got=fr_hot_capacity();
            if (got==now) continue;   /* the system wires no more than this */
            c->full=0;
            if (got/(1ull<<30)!=now/(1ull<<30) || got>=c->ceiling) { printf("HOT cache %s: room for %.1f GB now (memory to spare)\n",c->name,(double)got/1e9); fflush(stdout); }
        } else c->calm=0;
    }
}
static void *balance(void *arg) {
    (void)arg; unsigned changes=0;
    /* Where the last run stood: each cache takes the most used experts of its catalogue. */
    /* (A run that ended with a cache silent saved 0: then each cache fills its memory.) */
    for (unsigned k=0;k<P.caches && !atomic_load(&P.stop);++k)
        turn(k,P.resume?(unsigned)((double)P.resume*P.c[k].catalogue/(FR_LAYERS*FR_ROUTED_EXPERTS)):P.c[k].catalogue,0,0);
    for (unsigned k=0;k<P.caches;++k) {
        hot_cache *c=&P.c[k]; c->calls0=atomic_load(c->calls); c->last0=atomic_load(c->last);
        printf("HOT cache %s: %u experts routed to it from earlier use\n",c->name,atomic_load(&c->experts));
    }
    fflush(stdout); P.uses0=P.uses;
    while (!atomic_load(&P.stop)) {
        for (unsigned i=0;i<5 && !atomic_load(&P.stop);++i) usleep(100000);
        elastic();
        for (int n=atomic_exchange(&P.age_due,0);n>0;--n)
            for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e) P.recent[l][e]-=P.recent[l][e]>>4;
        const uint64_t uses=P.uses; const double tokens=(double)(uses-P.uses0)/(FR_LAYERS*FR_ACTIVE_ROUTED); P.uses0=uses;
        int more[FR_HOT_CACHES]={0},less[FR_HOT_CACHES]={0},judged=0; double share[FR_HOT_CACHES]={0}; unsigned before=0,swaps0=0;
        for (unsigned k=0;k<P.caches;++k) {
            hot_cache *c=&P.c[k]; before+=atomic_load(&c->experts); swaps0+=c->swaps;
            /* Replacements follow the work done: so many per token, shared by catalogue size. */
            c->allowance+=P.loads*tokens*c->catalogue/(FR_LAYERS*FR_ROUTED_EXPERTS); if (c->allowance>HOT_BATCH) c->allowance=HOT_BATCH;
            if (atomic_load(&c->disabled)) {
                const long at=atomic_load(&c->retry_at);
                if (!at || (long)time(NULL)<at) continue;
                atomic_store(&c->disabled,0); c->calls0=atomic_load(c->calls); c->last0=atomic_load(c->last); c->busy=0;
                printf("HOT cache %s: trying it again\n",c->name); fflush(stdout);
                turn(k,FR_LAYERS*FR_ROUTED_EXPERTS,0,1);   /* its experts are still in its memory: route them back */
                continue;
            }
            const uint64_t calls=atomic_load(c->calls),last=atomic_load(c->last);
            if (calls-c->calls0<HOT_WINDOW) continue;   /* idle or too little to judge */
            share[k]=(double)(last-c->last0)/(double)(calls-c->calls0); c->calls0=calls; c->last0=last; judged=1;
            /* Far from the balance a cache is nearly idle: a quarter more at a time fills
             * its memory in seconds; near the balance, small steps. One busy window is
             * noise (a wide read call, a reply asked again). */
            if (share[k]<HOT_LOW) { c->busy=0; more[k]=share[k]<HOT_SPARE?2:1; }
            else if (share[k]>HOT_HIGH) less[k]=++c->busy>=2;
            else c->busy=0;
        }
        if (!judged) continue;
        for (unsigned k=0;k<P.caches;++k) {
            hot_cache *c=&P.c[k]; const unsigned has=atomic_load(&c->experts);
            if (atomic_load(&c->disabled)) continue;
            if (less[k]) {   /* its least used routed experts go back to the slices; they stay in its memory */
                unsigned n=0,step=has/32; if (step<32) step=32;
                for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e)
                    if (fr_hot_where(l,e)==k+1) P.inside[n++]=(hot_pick){P.recent[l][e],(uint16_t)l,(uint16_t)e};
                qsort(P.inside,n,sizeof(P.inside[0]),score_up);
                for (unsigned i=0;i<step && i<n;++i) demote(P.inside[i].layer,P.inside[i].expert);
                continue;
            }
            unsigned room=!more[k]?0:more[k]>1?has/4:has/16; if (more[k] && room<64) room=64;
            const unsigned swaps=share[k]>HOT_HIGH?0:(unsigned)c->allowance;
            const unsigned done0=c->swaps;
            turn(k,room,swaps,0);
            c->allowance-=(double)(c->swaps-done0);
        }
        unsigned now=0,swaps=0;
        for (unsigned k=0;k<P.caches;++k) { now+=atomic_load(&P.c[k].experts); swaps+=P.c[k].swaps; }
        if ((now!=before || swaps!=swaps0) && ++changes%8==1) {   /* a line now and then, not one per second */
            fputs("HOT balance:",stdout);
            for (unsigned k=0;k<P.caches;++k) printf(" %s %u experts (%.0f%% of expert uses, last in %.0f%% of recent calls);",P.c[k].name,
                atomic_load(&P.c[k].experts),P.uses?100.0*(double)P.c[k].served/(double)P.uses:0,100*share[k]);
            printf(" %u replaced so far\n",swaps); fflush(stdout);
        }
    }
    return NULL;
}
/* The counts reach the disk when the head stops, never while it works: a session
 * does not write the SSD. open/write/rename only, so a signal handler may call it. */
void fr_hot_save(void) {
    if (!P.running || !P.uses) return;
    char tmp[PATH_MAX+40]; const size_t n=strlen(P.path);
    if (n+5>sizeof(tmp)) return;
    memcpy(tmp,P.path,n); memcpy(tmp+n,".tmp",5);
    const int fd=open(tmp,O_WRONLY|O_CREAT|O_TRUNC,0600);
    if (fd<0) return;
    uint32_t routed=0; for (unsigned k=0;k<P.caches;++k) routed+=atomic_load(&P.c[k].experts);
    const uint32_t head[4]={HOT_MAGIC,FR_LAYERS,FR_ROUTED_EXPERTS,routed};
    int ok=write(fd,head,sizeof(head))==(ssize_t)sizeof(head);
    const unsigned char *p=(const unsigned char *)P.count; size_t left=sizeof(P.count);
    while (ok && left) { const ssize_t w=write(fd,p,left); if (w<=0) ok=0; else { p+=w; left-=(size_t)w; } }
    ok=!close(fd) && ok;
    if (ok) rename(tmp,P.path); else unlink(tmp);
    if (P.trace && P.trace_at) {
        const int td=open(P.trace_path,O_WRONLY|O_CREAT|O_TRUNC,0600);
        if (td>=0) {
            const unsigned char *q=P.trace; size_t rest=P.trace_at;
            while (rest) { const ssize_t w=write(td,q,rest); if (w<=0) break; q+=w; rest-=(size_t)w; }
            close(td);
        }
        P.trace_at=0;
    }
}
int fr_hot_policy_start(const char *state_dir,const _Atomic uint64_t *begun,const _Atomic uint64_t *done) {
    if (P.running || !state_dir || !begun || !done) return -1;
    memset(P.count,0,sizeof(P.count)); memset(P.recent,0,sizeof(P.recent)); memset(P.home,0,sizeof(P.home)); memset(fr_hot_owner,0,sizeof(fr_hot_owner));
    P.caches=0; P.begun=begun; P.done=done; P.uses=P.since=P.uses0=0; P.resume=0; P.going=0; P.loads=HOT_LOADS;
    atomic_store(&P.stop,0); atomic_store(&P.age_due,0);
    { char path[PATH_MAX+32]; double loads=0; snprintf(path,sizeof(path),"%s/hot-loads-per-token",state_dir);
      FILE *t=fopen(path,"r"); if (t) { if (fscanf(t,"%lf",&loads)==1 && loads>=0 && loads<=64) P.loads=loads; fclose(t); }
      snprintf(path,sizeof(path),"%s/hot-trace",state_dir); snprintf(P.trace_path,sizeof(P.trace_path),"%s/hot-trace.bin",state_dir);
      free(P.trace); P.trace=NULL; P.trace_at=0; P.trace_cap=0;
      if (!access(path,F_OK) && (P.trace=malloc(64u<<20))) { P.trace_cap=64u<<20; puts("HOT trace on: routing kept in memory, written to hot-trace.bin when the head stops"); } }
    snprintf(P.path,sizeof(P.path),"%s/hot-experts.bin",state_dir); snprintf(P.homes,sizeof(P.homes),"%s/hot-homes.bin",state_dir);
    FILE *f=fopen(P.path,"rb"); uint32_t head[4];
    if (f) {
        if (fread(head,sizeof(head),1,f)!=1 || head[0]!=HOT_MAGIC || head[1]!=FR_LAYERS || head[2]!=FR_ROUTED_EXPERTS ||
            fread(P.count,sizeof(P.count),1,f)!=1) memset(P.count,0,sizeof(P.count));
        else P.resume=head[3];
        fclose(f);
        /* Earlier runs fade: each start halves what was known before. */
        uint32_t top=1;
        for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e) { P.count[l][e]=P.count[l][e]/2+(P.count[l][e]&1); if (P.count[l][e]>top) top=P.count[l][e]; }
        /* The caches start from that history, at the scale of recent use: the work of
         * this run overtakes it within some tens of tokens. */
        for (unsigned l=0;l<FR_LAYERS;++l) for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e) P.recent[l][e]=(uint32_t)((uint64_t)P.count[l][e]*HOT_SEED*HOT_ONE/top);
    }
    P.running=1;
    return 0;
}
int fr_hot_policy_add(const char *name,double gb,fr_hot_admit_fn admit,fr_hot_evict_fn evict_fn,void *ctx,
                      const _Atomic uint64_t *calls,const _Atomic uint64_t *last) {
    if (!P.running || P.going || P.caches>=FR_HOT_CACHES || !name || !(gb>0) || !admit || !calls || !last) return -1;
    hot_cache *c=&P.c[P.caches];
    memset(c->held,0,sizeof(c->held)); snprintf(c->name,sizeof(c->name),"%s",name);
    c->gb=gb; c->admit=admit; c->evict=evict_fn; c->ctx=ctx; c->calls=calls; c->last=last;
    c->served=c->calls0=c->last0=0; c->busy=c->swaps=0; c->full=0; c->allowance=0; c->catalogue=0; c->backoff=0; atomic_store(&c->retry_at,0);
    c->floor=c->ceiling=0; c->calm=0; c->rest_until=0; atomic_store(&c->asked,0);
    atomic_store(&c->experts,0); atomic_store(&c->disabled,0);
    return (int)P.caches++;
}
int fr_hot_policy_elastic(unsigned k,uint64_t floor_bytes,uint64_t ceiling_bytes) {
    if (!P.running || P.going || k>=P.caches || !floor_bytes || ceiling_bytes<floor_bytes) return -1;
    P.c[k].floor=floor_bytes; P.c[k].ceiling=ceiling_bytes; P.c[k].calm=0; P.c[k].rest_until=0; atomic_store(&P.c[k].asked,0);
    return 0;
}
int fr_hot_policy_resize(unsigned k,uint64_t bytes) {
    if (!P.running || k>=P.caches || !P.c[k].ceiling || !bytes) return -1;
    atomic_store(&P.c[k].asked,bytes);
    return 0;
}
int fr_hot_policy_go(void) {
    if (!P.running || P.going || !P.caches) return -1;
    catalogues();
    if (pthread_create(&P.filler,NULL,balance,NULL)) return -1;
    P.going=1;
    return 0;
}
void fr_hot_policy_stop(void) {
    if (!P.running) return;
    atomic_store(&P.stop,1);
    if (P.going) pthread_join(P.filler,NULL);
    fr_hot_save(); memset(fr_hot_owner,0,sizeof(fr_hot_owner)); P.running=P.going=0; P.caches=0;
}
void fr_hot_count(const fr_req *req) {
    if (!P.running) return;
    for (unsigned t=0;t<req->burst;++t) for (unsigned j=0;j<FR_ACTIVE_ROUTED;++j) {
        const unsigned e=req->expert[t][j],w=fr_hot_where(req->layer,e);
        if (P.count[req->layer][e]<UINT32_MAX) ++P.count[req->layer][e];
        if (P.recent[req->layer][e]<UINT32_MAX-HOT_ONE) P.recent[req->layer][e]+=HOT_ONE;
        ++P.uses; if (w && w<=P.caches) ++P.c[w-1].served;
    }
    P.since+=(uint64_t)req->burst*FR_ACTIVE_ROUTED;
    while (P.since>=(uint64_t)HOT_AGE_TOKENS*FR_LAYERS*FR_ACTIVE_ROUTED) { P.since-=(uint64_t)HOT_AGE_TOKENS*FR_LAYERS*FR_ACTIVE_ROUTED; atomic_fetch_add(&P.age_due,1); }
    if (P.trace && P.trace_at+2+(size_t)req->burst*FR_ACTIVE_ROUTED*2<=P.trace_cap) {
        unsigned char *q=P.trace+P.trace_at; *q++=(unsigned char)req->layer; *q++=(unsigned char)req->burst;
        for (unsigned t=0;t<req->burst;++t) for (unsigned j=0;j<FR_ACTIVE_ROUTED;++j) { const uint16_t e=req->expert[t][j]; memcpy(q,&e,2); q+=2; }
        P.trace_at=(size_t)(q-P.trace);
    }
}
void fr_hot_report(unsigned k,unsigned *experts,double *served) {
    if (k>=P.caches) { *experts=0; *served=0; return; }
    *experts=atomic_load(&P.c[k].experts); *served=P.uses?(double)P.c[k].served/(double)P.uses:0;
}

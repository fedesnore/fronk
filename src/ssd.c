/* Selected native slices, read concurrently into a bounded owned cache. The
 * caller copies into GPU staging before dispatch, so eviction never mutates
 * buffers referenced by running kernels. No allocation in fr_ssd_load. */
#include "ssd.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define KEYS (FR_BLOCKS*FR_LAYERS*FR_ROUTED_EXPERTS)
#define JOBS (FR_SSD_MAX_BLOCKS*(FR_MAX_BURST*FR_ACTIVE_ROUTED+1))
typedef struct { unsigned key; uint64_t age; } entry;
typedef struct { int fd,slot; off_t offset; size_t bytes; unsigned char *target; unsigned block; unsigned char *stage; } read_job;
struct fr_ssd {
    unsigned char *data;
    entry *entry;
    int *index,fd[FR_BLOCKS];
    unsigned file_layers[FR_BLOCKS];
    unsigned slots,workers,started,count,next,pending;
    unsigned block_pending[FR_BLOCKS];
    uint64_t clock;
    pthread_t thread[FR_SSD_WORKERS];
    pthread_mutex_t mutex;
    pthread_cond_t ready,done;
    int stop,error;
    read_job job[JOBS];
    fr_ssd_sample sample;
};
static uint64_t now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000000+t.tv_nsec; }
static void *reader(void *arg) {
    fr_ssd *s=arg;
    pthread_mutex_lock(&s->mutex);
    for (;;) {
        while (!s->stop && s->next==s->count) pthread_cond_wait(&s->ready,&s->mutex);
        if (s->stop) break;
        read_job j=s->job[s->next++];
        pthread_mutex_unlock(&s->mutex);
        size_t used=0; int error=0;
        while (used<j.bytes) {
            ssize_t n=pread(j.fd,j.target+used,j.bytes-used,j.offset+(off_t)used);
            if (n<0 && errno==EINTR) continue;
            if (n<=0) { error=1; break; }
            used+=(size_t)n;
        }
        /* Pipelined loads copy disjoint slices while the other workers read.
         * Completion publishes staging only after this copy is finished. */
        if (!error && j.stage) memcpy(j.stage,j.target,j.bytes);
        pthread_mutex_lock(&s->mutex);
        s->error|=error;
        --s->pending;
        if (!--s->block_pending[j.block] || !s->pending) pthread_cond_signal(&s->done);
    }
    pthread_mutex_unlock(&s->mutex); return NULL;
}
fr_ssd *fr_ssd_open(uint64_t bytes,unsigned workers) {
    if (!workers || workers>FR_SSD_WORKERS || bytes<2*JOBS*FR_NATIVE_SLICE_BYTES || bytes>(UINT64_C(2)<<30)) return NULL;
    fr_ssd *s=calloc(1,sizeof(*s)); if (!s) return NULL;
    for (unsigned b=0;b<FR_BLOCKS;++b) s->fd[b]=-1;
    s->slots=(unsigned)(bytes/FR_NATIVE_SLICE_BYTES); s->workers=workers;
    s->data=malloc((size_t)s->slots*FR_NATIVE_SLICE_BYTES); s->entry=calloc(s->slots,sizeof(entry)); s->index=malloc(KEYS*sizeof(int));
    if (!s->data || !s->entry || !s->index) goto memory_failed;
    memset(s->index,0xff,KEYS*sizeof(int));
    memset(s->data,0,(size_t)s->slots*FR_NATIVE_SLICE_BYTES);
    if (pthread_mutex_init(&s->mutex,NULL)) goto memory_failed;
    if (pthread_cond_init(&s->ready,NULL)) { pthread_mutex_destroy(&s->mutex); goto memory_failed; }
    if (pthread_cond_init(&s->done,NULL)) { pthread_cond_destroy(&s->ready); pthread_mutex_destroy(&s->mutex); goto memory_failed; }
    for (;s->started<workers;++s->started) if (pthread_create(&s->thread[s->started],NULL,reader,s)) { fr_ssd_close(s); return NULL; }
    return s;
memory_failed:
    free(s->data); free(s->entry); free(s->index); free(s); return NULL;
}
void fr_ssd_clear(fr_ssd *s) {
    memset(s->index,0xff,KEYS*sizeof(int)); memset(s->entry,0,s->slots*sizeof(entry)); s->clock=0;
}
static int attach(fr_ssd *s,unsigned block,int fd,unsigned layers) {
    struct stat st;
    if (!s || block>=FR_BLOCKS || s->fd[block]>=0 || fstat(fd,&st) || !S_ISREG(st.st_mode) || (uint64_t)st.st_size!=layers*FR_NATIVE_LAYER_BYTES) return FR_INVALID;
    unsigned count=0; for (unsigned b=0;b<FR_BLOCKS;++b) count+=s->fd[b]>=0;
    if (count>=FR_SSD_MAX_BLOCKS) return FR_INVALID;
    int own=dup(fd); if (own<0) return FR_IO;
#ifdef __APPLE__
    /* Avoid a second unbounded file cache displacing resident dense weights. */
    if (fcntl(own,F_NOCACHE,1) || fcntl(own,F_RDAHEAD,0)) { close(own); return FR_IO; }
#endif
    s->fd[block]=own; s->file_layers[block]=layers; return FR_OK;
}
int fr_ssd_attach(fr_ssd *s,unsigned block,int fd) { return attach(s,block,fd,FR_LAYERS); }
int fr_ssd_attach_fixture(fr_ssd *s,unsigned block,int fd) { return attach(s,block,fd,1); }
int fr_ssd_detach(fr_ssd *s,unsigned block) {
    if (!s || block>=FR_BLOCKS || s->fd[block]<0) return FR_INVALID;
    int rc=close(s->fd[block]); s->fd[block]=-1;
    for (unsigned i=0;i<s->slots;++i) if (s->entry[i].age && s->entry[i].key/(FR_LAYERS*FR_ROUTED_EXPERTS)==block) {
        s->index[s->entry[i].key]=-1; s->entry[i].age=0;
    }
    return rc?FR_IO:FR_OK;
}
int fr_ssd_load_ready(fr_ssd *s,const fr_req *r,uint32_t mask,void *layers[FR_BLOCKS],fr_ssd_ready_fn ready,void *context) {
    if (!s || !r || !layers || !mask || mask>=(1u<<FR_BLOCKS) || r->layer>=FR_LAYERS || !r->burst || r->burst>FR_MAX_BURST) return FR_INVALID;
    unsigned char selected[FR_ROUTED_EXPERTS]={0}; int shared=0;
    for (unsigned t=0;t<r->burst;++t) {
        for (unsigned e=0;e<FR_ACTIVE_ROUTED;++e) {
            unsigned id=r->expert[t][e]; if (id>=FR_ROUTED_EXPERTS) return FR_INVALID;
            selected[id]=1;
        }
        shared|=r->gate[t][FR_ACTIVE_ROUTED]!=0;
    }
    unsigned blocks=0;
    for (unsigned b=0;b<FR_BLOCKS;++b) if (mask>>b&1) {
        if (s->fd[b]<0 || !layers[b] || ++blocks>FR_SSD_MAX_BLOCKS) return FR_INVALID;
    }
    uint64_t begin=now_ns(); memset(&s->sample,0,sizeof(s->sample));
    if (!++s->clock) { fr_ssd_clear(s); ++s->clock; }
    pthread_mutex_lock(&s->mutex);
    s->count=s->next=s->pending=0; s->error=0;
    memset(s->block_pending,0,sizeof(s->block_pending));
    /* Copy hits first. Missing entries may then evict them safely: GPU staging
     * is separate, and every new slot is pinned for this entire read batch. */
    for (unsigned b=0;b<FR_BLOCKS;++b) if (mask>>b&1) for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e) if (selected[e]) {
        unsigned key=(b*FR_LAYERS+r->layer)*FR_ROUTED_EXPERTS+e; int at=s->index[key];
        if (at>=0) {
            memcpy((unsigned char *)layers[b]+e*FR_NATIVE_SLICE_BYTES,s->data+(size_t)at*FR_NATIVE_SLICE_BYTES,FR_NATIVE_SLICE_BYTES);
            s->entry[at].age=s->clock; ++s->sample.hits;
        }
    }
    for (unsigned b=0;b<FR_BLOCKS;++b) if (mask>>b&1) {
        for (unsigned e=0;e<FR_ROUTED_EXPERTS;++e) if (selected[e]) {
            unsigned key=(b*FR_LAYERS+r->layer)*FR_ROUTED_EXPERTS+e;
            if (s->index[key]>=0) continue;
            unsigned at=0; for (unsigned i=1;i<s->slots;++i) if (s->entry[i].age<s->entry[at].age) at=i;
            if (s->entry[at].age==s->clock || s->count>=JOBS) goto failed;
            if (s->entry[at].age) s->index[s->entry[at].key]=-1;
            s->entry[at]=(entry){key,s->clock}; /* publish index only after every read succeeds */
            s->job[s->count++]=(read_job){s->fd[b],(int)at,(off_t)fr_native_offset(r->layer%s->file_layers[b],e),FR_NATIVE_SLICE_BYTES,s->data+(size_t)at*FR_NATIVE_SLICE_BYTES,b,
                ready?(unsigned char *)layers[b]+e*FR_NATIVE_SLICE_BYTES:NULL};
            ++s->block_pending[b];
            ++s->sample.misses; s->sample.bytes+=FR_NATIVE_SLICE_BYTES;
        }
        if (shared) {
            if (s->count>=JOBS) goto failed;
            s->job[s->count++]=(read_job){s->fd[b],-1,(off_t)fr_native_offset(r->layer%s->file_layers[b],FR_ROUTED_EXPERTS),FR_NATIVE_SHARED_BYTES,(unsigned char *)layers[b]+FR_ROUTED_EXPERTS*FR_NATIVE_SLICE_BYTES,b,NULL};
            ++s->block_pending[b];
            s->sample.bytes+=FR_NATIVE_SHARED_BYTES;
        }
    }
    uint64_t reads=now_ns();
    s->pending=s->count;
    pthread_cond_broadcast(&s->ready);
    uint32_t delivered=0; int callback_error=0;
    for (;;) {
        uint32_t complete=0;
        if (ready && !s->error && !callback_error) {
            for (unsigned b=0;b<FR_BLOCKS;++b) if ((mask&~delivered)>>b&1 && !s->block_pending[b]) complete|=1u<<b;
        }
        if (complete) {
            delivered|=complete;
            pthread_mutex_unlock(&s->mutex);
            callback_error=ready(context,complete);
            pthread_mutex_lock(&s->mutex);
            continue;
        }
        if (!s->pending) break;
        pthread_cond_wait(&s->done,&s->mutex);
    }
    pthread_mutex_unlock(&s->mutex);
    s->sample.read_ns=now_ns()-reads;
    for (unsigned i=0;i<s->count;++i) if (s->job[i].slot>=0) {
        unsigned at=(unsigned)s->job[i].slot,key=s->entry[at].key;
        if (s->error || callback_error) { s->entry[at].age=0; continue; }
        s->index[key]=(int)at;
        unsigned b=key/(FR_LAYERS*FR_ROUTED_EXPERTS),e=key%FR_ROUTED_EXPERTS;
        if (!ready) memcpy((unsigned char *)layers[b]+e*FR_NATIVE_SLICE_BYTES,s->job[i].target,FR_NATIVE_SLICE_BYTES);
    }
    s->sample.total_ns=now_ns()-begin; return s->error?FR_IO:callback_error;
failed:
    for (unsigned i=0;i<s->count;++i) if (s->job[i].slot>=0) s->entry[s->job[i].slot].age=0;
    s->count=s->next=0; pthread_mutex_unlock(&s->mutex); return FR_STATE;
}
int fr_ssd_load(fr_ssd *s,const fr_req *r,uint32_t mask,void *layers[FR_BLOCKS]) {
    return fr_ssd_load_ready(s,r,mask,layers,NULL,NULL);
}
void fr_ssd_sample_get(const fr_ssd *s,fr_ssd_sample *out) { if (s && out) *out=s->sample; }
void fr_ssd_close(fr_ssd *s) {
    if (!s) return;
    pthread_mutex_lock(&s->mutex); s->stop=1; pthread_cond_broadcast(&s->ready); pthread_mutex_unlock(&s->mutex);
    for (unsigned i=0;i<s->started;++i) pthread_join(s->thread[i],NULL);
    pthread_cond_destroy(&s->ready); pthread_cond_destroy(&s->done); pthread_mutex_destroy(&s->mutex);
    for (unsigned b=0;b<FR_BLOCKS;++b) if (s->fd[b]>=0) close(s->fd[b]);
    free(s->data); free(s->entry); free(s->index); free(s);
}
int fr_ssd_fixture_file(const char *dir) {
    char path[700]; int n=snprintf(path,sizeof(path),"%s/hybrid-io.XXXXXX",dir);
    if (n<0 || (size_t)n>=sizeof(path)) return -1;
    int fd=mkstemp(path); if (fd<0) return -1; unlink(path);
#ifdef __APPLE__
    /* Set this before writing, not only before reading: otherwise the freshly
     * written fixture remains in the filesystem cache despite F_NOCACHE. */
    if (fcntl(fd,F_NOCACHE,1) || fcntl(fd,F_RDAHEAD,0)) { close(fd); return -1; }
#endif
    unsigned char *data=malloc(FR_NATIVE_LAYER_BYTES); if (!data) { close(fd); return -1; }
    memset(data,0x11,FR_NATIVE_LAYER_BYTES); size_t at=0;
    while (at<FR_NATIVE_LAYER_BYTES) {
        ssize_t got=write(fd,data+at,FR_NATIVE_LAYER_BYTES-at);
        if (got<0 && errno==EINTR) continue;
        if (got<=0) break;
        at+=(size_t)got;
    }
    free(data); if (at!=FR_NATIVE_LAYER_BYTES || fsync(fd)) { close(fd); return -1; }
    return fd;
}

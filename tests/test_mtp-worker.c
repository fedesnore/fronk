/* Exercise the actual worker with immutable GPU mailboxes and a recording
 * transport. Failures must drain fences without publishing stale results. */
#include "test.h"
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#define FR_EXPERT_RANKED 1
#define DS4_N_EMBD 16
#define DS4_N_EXPERT_USED 10
#define ROWS 8

typedef struct { _Alignas(float) unsigned char data[ROWS*DS4_N_EMBD*4]; bool fail; } tensor;
typedef struct {
    tensor *mixed,*selected,*weights,*sh_gate_logit,*part;
    bool fronk_defer_reduce;
} ds4_qwen4_gpu_graph;
typedef struct { ds4_qwen4_gpu_graph *g; unsigned layer; uint32_t T; uint64_t ready,resume; bool shared_local; } qwen4_fronk_job;
static struct { pthread_mutex_t mu; pthread_cond_t cv; qwen4_fronk_job q[512]; unsigned head,tail; int error; } qwen4_fronk;
static uint64_t fronk_tree_overlaps;
static unsigned enabled, nsignals, nwaits, ncalls, nrows[16], stop_after, fail_transport, missing_transport;
static uint64_t ready_value, signals[16];
static void *allocated[6]; static unsigned nallocated;
static void *tracked_realloc(void *old,size_t bytes) {
    unsigned i=0; while(i<nallocated && allocated[i]!=old) ++i;
    void *p=realloc(old,bytes); if(p) { allocated[i]=p; if(i==nallocated) ++nallocated; } return p;
}
static int qwen4_fronk_env(const char *key,int fallback) { (void)key;(void)fallback;return enabled; }
static double qwen4_fronk_now(void) { return 1; }
static int ds4_gpu_fronk_wait(uint64_t value,double timeout) { (void)value;(void)timeout;++nwaits;return 1; }
static uint64_t ds4_gpu_fronk_ready_value(void) { return ready_value; }
static int ds4_gpu_tensor_read(tensor *t,uint64_t offset,void *out,uint64_t bytes) {
    if(t->fail || offset+bytes>sizeof(t->data)) return 0;
    memcpy(out,t->data+offset,bytes);return 1;
}
static int ds4_gpu_tensor_write(tensor *t,uint64_t offset,const void *in,uint64_t bytes) {
    if(t->fail || offset+bytes>sizeof(t->data)) return 0;
    memcpy(t->data+offset,in,bytes);return 1;
}
static int fronk_moe(unsigned layer,unsigned rows,const float *x,const int32_t *ids,const float *w,const float *sh,float *y,uint32_t *missing) {
    (void)layer;(void)sh;nrows[ncalls++]=rows;*missing=missing_transport;
    for(unsigned t=0;t<rows;++t) for(unsigned d=0;d<DS4_N_EMBD;++d)
        y[t*DS4_N_EMBD+d]=x[t*DS4_N_EMBD+d]+w[t*DS4_N_EXPERT_USED]+ids[t*DS4_N_EXPERT_USED];
    return fail_transport ? -1 : 0;
}
static int ds4_gpu_fronk_signal(uint64_t value) {
    signals[nsignals++]=value;
    if(nsignals==stop_after) pthread_exit(NULL);
    return 1;
}
#define realloc tracked_realloc
#include "../tools/mtp/fronk_mtp_worker.inc"
#undef realloc

static tensor buffers[9][5]; static ds4_qwen4_gpu_graph graphs[9];
static void setup(unsigned count) {
    unsetenv("FRONK_NATIVE_MTP");
    memset(&qwen4_fronk,0,sizeof(qwen4_fronk));
    pthread_mutex_init(&qwen4_fronk.mu,NULL);pthread_cond_init(&qwen4_fronk.cv,NULL);
    memset(buffers,0,sizeof(buffers)); memset(nrows,0,sizeof(nrows));
    enabled=1;ready_value=99;nsignals=nwaits=ncalls=0;fail_transport=missing_transport=0;
    stop_after=count;fronk_tree_coalesced=0;
    qwen4_fronk.head=510;qwen4_fronk.tail=510+count;
    for(unsigned i=0;i<count;++i) {
        graphs[i]=(ds4_qwen4_gpu_graph){&buffers[i][0],&buffers[i][1],&buffers[i][2],&buffers[i][3],&buffers[i][4],true};
        for(unsigned j=0;j<DS4_N_EXPERT_USED;++j) {
            ((int32_t *)buffers[i][1].data)[j]=10+j;
            ((float *)buffers[i][2].data)[j]=.25f+i;
        }
        for(unsigned d=0;d<DS4_N_EMBD;++d) ((float *)buffers[i][0].data)[d]=i*100+d;
        memset(buffers[i][4].data,0x5a,sizeof(buffers[i][4].data));
        qwen4_fronk.q[(510+i)%512]=(qwen4_fronk_job){&graphs[i],7,1,2*i+1,2*i+2,true};
    }
}
static void run(void) {
    pthread_t thread;pthread_create(&thread,NULL,qwen4_fronk_worker,NULL);pthread_join(thread,NULL);
    bool order=true; for(unsigned i=0;i<stop_after;++i) if(signals[i]!=2*i+2) order=false;
    check(order && qwen4_fronk.head==qwen4_fronk.tail,"every mailbox consumed once, monotonic completion across queue wrap");
    for(unsigned i=0;i<nallocated;++i) free(allocated[i]);
    nallocated=0;
    pthread_mutex_destroy(&qwen4_fronk.mu);pthread_cond_destroy(&qwen4_fronk.cv);
}
static bool outputs(void) {
    for(unsigned i=0;i<stop_after;++i) for(unsigned d=0;d<DS4_N_EMBD;++d)
        if(((float *)buffers[i][4].data)[d]!=i*100+d+.25f+i+10) return false;
    return true;
}
static bool unpublished(unsigned count) {
    for(unsigned i=0;i<count;++i) for(unsigned j=0;j<sizeof(buffers[i][4].data);++j)
        if(buffers[i][4].data[j]!=0x5a) return false;
    return true;
}
static void grouped(unsigned first,unsigned second) {
    setup(2);setenv("FRONK_NATIVE_MTP","8",1);enabled=2;
    unsigned widths[2]={first,second};
    for(unsigned i=0;i<2;i++) {
        qwen4_fronk.q[(510+i)%512].T=widths[i];
        for(unsigned t=0;t<widths[i];t++) {
            for(unsigned j=0;j<DS4_N_EXPERT_USED;j++) {
                ((int32_t *)buffers[i][1].data)[t*DS4_N_EXPERT_USED+j]=10+j+t;
                ((float *)buffers[i][2].data)[t*DS4_N_EXPERT_USED+j]=.25f+i+t;
            }
            for(unsigned d=0;d<DS4_N_EMBD;d++) ((float *)buffers[i][0].data)[t*DS4_N_EMBD+d]=i*100+t*10+d;
        }
    }
}
static bool grouped_outputs(unsigned first,unsigned second) {
    unsigned widths[2]={first,second};
    for(unsigned i=0;i<2;i++) {
        for(unsigned t=0;t<widths[i];t++) for(unsigned d=0;d<DS4_N_EMBD;d++)
            if(((float *)buffers[i][4].data)[t*DS4_N_EMBD+d]!=i*100+t*10+d+.25f+i+t+10+t) return false;
        for(unsigned j=widths[i]*DS4_N_EMBD*sizeof(float);j<sizeof(buffers[i][4].data);j++)
            if(buffers[i][4].data[j]!=0x5a) return false;
    }
    return true;
}
int main(void) {
    test_begin("mtp-worker");
    setup(3);run();check(ncalls==1 && nrows[0]==3 && outputs(),"shared routing batches distinct activations/gates and scatters to correct lanes");
    setup(9);run();check(ncalls==2 && nrows[0]==8 && nrows[1]==1 && outputs(),"bounded eight-row batch, no loss at capacity");
    setup(3);enabled=0;run();check(ncalls==3 && outputs(),"disabled scheduler retains separate calls");
    setup(3);ready_value=1;run();check(ncalls==3 && nwaits==3 && outputs(),"no waiting for a future activation to fill a batch");
    setup(3);qwen4_fronk.q[511].layer=8;run();check(ncalls==3 && outputs(),"no layer crossing or skipping front job");
    setup(3);((int32_t *)buffers[1][1].data)[1]=12;((int32_t *)buffers[1][1].data)[2]=11;
    run();check(ncalls==3 && outputs(),"same expert set in different order stays separate");
    setup(3);((int32_t *)buffers[1][1].data)[9]=400;
    run();check(ncalls==3 && outputs(),"partial reuse cannot change worker assignment");
    setup(3);enabled=2;((int32_t *)buffers[1][1].data)[9]=400;
    run();check(ncalls==1 && nrows[0]==3 && outputs(),"ranked mode batches partial overlap with distinct row ownership");
    setup(3);enabled=2;for(unsigned j=1;j<DS4_N_EXPERT_USED;++j) ((int32_t *)buffers[1][1].data)[j]=400+j;
    run();check(ncalls==3 && outputs(),"ranked mode does not batch insufficient reuse or skip the front job");
    setup(3);fail_transport=1;run();check(qwen4_fronk.error && unpublished(3),"transport failure drains all fences without result publication");
    setup(3);missing_transport=1;run();check(qwen4_fronk.error && unpublished(3),"missing shard fails speculative batch closed");
    setup(3);setenv("FRONK_NATIVE_MTP","8",1);missing_transport=1;
    for(unsigned i=0;i<3;i++) graphs[i].fronk_defer_reduce=false;
    run();check(qwen4_fronk.error && unpublished(3),"missing shard fails native ordinary mailboxes closed and drains every fence");
    setup(3);setenv("FRONK_NATIVE_MTP","8",1);
    for(unsigned i=0;i<3;i++) graphs[i].fronk_defer_reduce=false;
    run();check(!qwen4_fronk.error && ncalls==3 && outputs(),"complete native ordinary mailboxes preserve every activation and output");
    setup(3);buffers[1][1].fail=true;run();check(qwen4_fronk.error && unpublished(3),"failed ready-mailbox read preserves every output and drains queue");
    grouped(4,4);run();check(ncalls==2 && nrows[0]==4 && nrows[1]==4 && grouped_outputs(4,4),"two wave groups keep distinct row destinations and untouched output guards");
    grouped(2,3);run();check(ncalls==2 && nrows[0]==2 && nrows[1]==3 && grouped_outputs(2,3),"unequal wave tail retains each group's width and routing");
    grouped(4,4);missing_transport=1;run();check(qwen4_fronk.error && unpublished(2),"wave groups with missing shards drain both fences and publish no rows");
    return test_end();
}

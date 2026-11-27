/* fronk_client.h - header-only client for the head's expert service (like
 * hiredis, but one call). The engine passes the routed-expert inputs of one
 * layer and receives the gated sum of routed + shared experts. It knows
 * nothing about ds4. Enabled when FRONK_MOE_SOCKET names the head's socket.
 * Wire format: see src/serve.h. Bursts above FRONK_BURST tokens are sent in chunks.
 */
#ifndef FRONK_CLIENT_H
#define FRONK_CLIENT_H
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#define FRONK_HIDDEN 2560
#define FRONK_ROUTED 10
#define FRONK_BURST 32  /* = FR_MAX_BURST: two calls per layer for a 64-token prompt batch */
static int fronk_fd=-1;
static int fronk_enabled(void) { const char *p=getenv("FRONK_MOE_SOCKET"); return p && *p; }
static int fronk_io(void *p,size_t n,int out) {
    for (char *c=(char *)p;n;) { ssize_t k=out?send(fronk_fd,c,n,0):recv(fronk_fd,c,n,0); if (k<=0) return -1; c+=k; n-=(size_t)k; }
    return 0;
}
static int fronk_connect(void) {
    if (fronk_fd>=0) return 0;
    struct sockaddr_un a; memset(&a,0,sizeof(a)); a.sun_family=AF_UNIX;
    const char *path=getenv("FRONK_MOE_SOCKET"); if (!path || strlen(path)>=sizeof(a.sun_path)) return -1;
    strcpy(a.sun_path,path);
    if ((fronk_fd=socket(AF_UNIX,SOCK_STREAM,0))<0) return -1;
    if (connect(fronk_fd,(struct sockaddr *)&a,sizeof(a))) { close(fronk_fd); fronk_fd=-1; return -1; }
    return 0;
}
/* A split call owns the socket until collect. The request is kept here: an
 * expert sum has no side effects, so when the head or a node stops answering the
 * same call is sent again once the head is back. The engine pauses instead of
 * dying, and the chat's context stays in memory. Only a head that cannot even be
 * connected to for FRONK_GIVE_UP_S seconds (it was stopped) ends the wait. */
#define FRONK_GIVE_UP_S 30
static struct {
    uint32_t head[3]; uint16_t e[FRONK_BURST][FRONK_ROUTED]; float g[FRONK_BURST][FRONK_ROUTED+1];
    float x[FRONK_BURST][FRONK_HIDDEN];
} fronk_call;
static unsigned fronk_pending_tokens=0,fronk_pending_layer=0;
static void fronk_drop(void) { if (fronk_fd>=0) close(fronk_fd); fronk_fd=-1; }
static int fronk_fail(void) { fronk_drop(); fronk_pending_tokens=0; return -1; }
static int fronk_send_call(void) {
    const unsigned b=fronk_call.head[2];
    if (fronk_connect()) return -1;
    if (fronk_io(fronk_call.head,sizeof(fronk_call.head),1) || fronk_io(fronk_call.e,b*sizeof(fronk_call.e[0]),1) ||
        fronk_io(fronk_call.g,b*sizeof(fronk_call.g[0]),1) || fronk_io(fronk_call.x,(size_t)b*sizeof(fronk_call.x[0]),1)) {
        fronk_drop(); return -1;
    }
    return 0;
}
static int fronk_submit(unsigned layer,unsigned b,const float *x,const int32_t *ids,
                        const float *weights,const float *shared_logit) {
    if (!b || b>FRONK_BURST || fronk_pending_tokens || !x || !ids || !weights) return fronk_fail();
    fronk_call.head[0]=0x4b4e5246u; fronk_call.head[1]=layer; fronk_call.head[2]=b;
    for (unsigned t=0;t<b;++t) {
        for (unsigned j=0;j<FRONK_ROUTED;++j) {
            if (ids[t*FRONK_ROUTED+j]<0 || ids[t*FRONK_ROUTED+j]>65535) return fronk_fail();
            fronk_call.e[t][j]=(uint16_t)ids[t*FRONK_ROUTED+j]; fronk_call.g[t][j]=weights[t*FRONK_ROUTED+j];
        }
        fronk_call.g[t][FRONK_ROUTED]=shared_logit?1.0f/(1.0f+expf(-shared_logit[t])):0.0f;
    }
    memcpy(fronk_call.x,x,(size_t)b*sizeof(fronk_call.x[0]));
    (void)fronk_send_call();  /* a failed send is retried by collect */
    fronk_pending_layer=layer; fronk_pending_tokens=b; return 0;
}
static int fronk_collect(unsigned layer,unsigned b,float *y,uint32_t *missing) {
    if (!b || b!=fronk_pending_tokens || layer!=fronk_pending_layer || !y || !missing) return fronk_fail();
    unsigned waited=0,unreachable=0;
    for (;;) {
        int32_t status; uint32_t miss;
        if (fronk_fd>=0 && !fronk_io(&status,4,0) && !fronk_io(&miss,4,0) &&
            !fronk_io(y,(size_t)b*FRONK_HIDDEN*sizeof(float),0) && !status && !miss) break;
        fronk_drop();
        if (!waited) fprintf(stderr,"llama-fronk: cluster paused at layer %u: waiting for the head and its nodes\n",layer);
        sleep(1); ++waited;
        if (fronk_send_call()) {
            if (++unreachable>=FRONK_GIVE_UP_S) { fprintf(stderr,"llama-fronk: head unreachable for %u s\n",unreachable); return fronk_fail(); }
        } else unreachable=0;
    }
    if (waited) fprintf(stderr,"llama-fronk: cluster resumed after %u s at layer %u\n",waited,layer);
    fronk_pending_tokens=0; *missing=0; return 0;
}
static int fronk_moe(unsigned layer,unsigned T,const float *x,const int32_t *ids,const float *weights,
                     const float *shared_logit,float *y,uint32_t *missing) {
    *missing=0;
    for (unsigned base=0;base<T;base+=FRONK_BURST) {
        unsigned b=T-base<FRONK_BURST?T-base:FRONK_BURST;uint32_t miss;
        if (fronk_submit(layer,b,x+(size_t)base*FRONK_HIDDEN,ids+base*FRONK_ROUTED,weights+base*FRONK_ROUTED,
                         shared_logit?shared_logit+base:NULL) ||
            fronk_collect(layer,b,y+(size_t)base*FRONK_HIDDEN,&miss)) return -1;
        *missing|=miss;
    }
    return 0;
}
#endif

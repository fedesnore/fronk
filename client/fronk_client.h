/* fronk_client.h - header-only client for the head's expert service (like
 * hiredis, but one call). The engine passes the routed-expert inputs of one
 * layer and receives the gated sum of routed + shared experts. It knows
 * nothing about ds4. Enabled when FRONK_MOE_SOCKET names the head's socket.
 * Wire format: see src/serve.h. Bursts above 8 tokens are sent in chunks.
 */
#ifndef FRONK_CLIENT_H
#define FRONK_CLIENT_H
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#define FRONK_HIDDEN 2560
#define FRONK_ROUTED 10
#define FRONK_BURST 8
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
/* x[T][2560] expert input, ids/weights[T][10] router choice, shared_logit[T]
 * raw shared-expert gate (sigmoid applied here; NULL leaves the shared expert to
 * the caller), y[T][2560] gated sum. */
static int fronk_moe(unsigned layer,unsigned T,const float *x,const int32_t *ids,const float *weights,
                     const float *shared_logit,float *y,uint32_t *missing) {
    if (fronk_connect()) return -1;
    *missing=0;
    for (unsigned base=0;base<T;base+=FRONK_BURST) {
        uint32_t b=T-base<FRONK_BURST?T-base:FRONK_BURST,head[3]={0x4b4e5246u,layer,b},miss; int32_t status;
        uint16_t e[FRONK_BURST][FRONK_ROUTED]; float g[FRONK_BURST][FRONK_ROUTED+1];
        for (unsigned t=0;t<b;++t) {
            for (unsigned j=0;j<FRONK_ROUTED;++j) { e[t][j]=(uint16_t)ids[(base+t)*FRONK_ROUTED+j]; g[t][j]=weights[(base+t)*FRONK_ROUTED+j]; }
            g[t][FRONK_ROUTED]=shared_logit?1.0f/(1.0f+expf(-shared_logit[base+t])):0.0f; /* NULL: caller adds the shared expert */
        }
        if (fronk_io(head,sizeof(head),1) || fronk_io(e,b*sizeof(e[0]),1) || fronk_io(g,b*sizeof(g[0]),1) ||
            fronk_io((void *)(x+(size_t)base*FRONK_HIDDEN),(size_t)b*FRONK_HIDDEN*sizeof(float),1) ||
            fronk_io(&status,4,0) || fronk_io(&miss,4,0) || fronk_io(y+(size_t)base*FRONK_HIDDEN,(size_t)b*FRONK_HIDDEN*sizeof(float),0)) {
            close(fronk_fd); fronk_fd=-1; return -1;
        }
        if (status) return -1;
        *missing|=miss;
    }
    return 0;
}
#endif

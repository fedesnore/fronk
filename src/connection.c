/* connection.c - CONNECTION phase on the head, with bounded waits and storage.
 * Discovery advertisements are untrusted until TCP authentication and LINK
 * identity confirmation. UDP probes require that live authenticated link.
 * Loopback uses multicast HELLO because Darwin lo0 does not deliver broadcast.
 */
#include "wire.h"
#include "connection.h"
#include "control.h"
#include "auth.h"
#include "identity.h"
#include "net.h"
#include <arpa/inet.h>
#include <errno.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

static int wait_until(int fd, int writing, int64_t deadline) {
    int64_t now=net_now_ns();
    if (now<0) return -1;
    if (now>=deadline) return 0;
    return net_wait(fd,writing,(int)((deadline-now+999999)/1000000));
}
static int send_packet(int fd, const void *bytes, size_t len, const struct sockaddr_in *peer, int64_t deadline) {
    for (;;) {
        ssize_t n=net_udp_send(fd,bytes,len,peer);
        if (n==(ssize_t)len) return 0;
        if (n>=0 || (errno!=EAGAIN && errno!=EWOULDBLOCK) || wait_until(fd,1,deadline)!=1) return -1;
    }
}
/* Stable node ids make text output and repeated plans easy to compare. */
static void sort_links(fr_connection *r) {
    for (unsigned i=1; i<r->count; ++i) {
        fr_link item=r->links[i]; unsigned j=i;
        while (j && r->links[j-1].identity.id>item.identity.id) { r->links[j]=r->links[j-1]; --j; }
        r->links[j]=item;
    }
}
static int discover(const fr_config *cfg, fr_connection *r, const unsigned char cookie[16],const fr_interface *via) {
    struct sockaddr_in target;
    if (net_address(r->local?FR_DISCOVERY_GROUP:cfg->broadcast,cfg->discovery_port,&target)) return -1;
    unsigned char bytes[FR_ANNOUNCE_BYTES], hello[FR_HELLO_BYTES];
    probe_hello(hello,cookie,NULL);
    int64_t start=net_now_ns(), next=start, deadline=start+500000000;
    if (start<0) return -1;
    unsigned sent=0;
    for (;;) {
        int64_t now=net_now_ns(); if (now<0) return -1;
        if (now>=deadline) break;
        if (sent<3 && now>=next) {
            if (send_packet(r->udp,hello,sizeof(hello),&target,deadline)) return -1;
            for (unsigned i=0;i<cfg->peer_count;++i) if (interface_contains(via,cfg->peers[i].address)) {
                struct sockaddr_in peer;
                if (net_address(cfg->peers[i].address,cfg->discovery_port,&peer) || send_packet(r->udp,hello,sizeof(hello),&peer,deadline)) return -1;
            }
            ++sent; next=start+(int64_t)sent*100000000;
        }
        int ready=wait_until(r->udp,0,sent<3?next:deadline);
        if (ready<0) return -1;
        if (!ready) continue;
        struct sockaddr_in peer; unsigned char received_cookie[16]; fr_identity id;
        ssize_t n=net_udp_recv(r->udp,bytes,sizeof(bytes),&peer);
        if (n<0) {
            if (errno==EAGAIN || errno==EWOULDBLOCK || errno==EMSGSIZE) { ++r->ignored; continue; }
            return -1;
        }
        int parsed=probe_discovery_read(bytes,(size_t)n,received_cookie,&id);
        if (parsed!=2 || memcmp(cookie,received_cookie,16) ||
            ntohs(peer.sin_port)!=cfg->discovery_port || (r->local && (ntohl(peer.sin_addr.s_addr)>>24)!=127)) {
#ifdef FRONK_DEBUG
            if (getenv("FRONK_TRACE_DISCOVERY")) fprintf(stderr,"discovery reject interface=%s parsed=%d bytes=%ld cookie_match=%d port=%u local=%d\n",
                via->name,parsed,(long)n,parsed==2 && !memcmp(cookie,received_cookie,16),ntohs(peer.sin_port),r->local);
#endif
            ++r->ignored; continue;
        }
        char ip[16]; if (!inet_ntop(AF_INET,&peer.sin_addr,ip,sizeof(ip))) return -1;
        if (cfg->peer_count) {
            unsigned i;
            for (i=0;i<cfg->peer_count;++i) if (!strcmp(cfg->peers[i].address,ip) && !strcmp(cfg->peers[i].name,id.name)) break;
            if (i==cfg->peer_count) {
#ifdef FRONK_DEBUG
                if (getenv("FRONK_TRACE_DISCOVERY")) fprintf(stderr,"discovery reject roster name=%s address=%s\n",id.name,ip);
#endif
                ++r->ignored; continue;
            }
        }
        unsigned i;
        for (i=0; i<r->count; ++i) {
            fr_link *l=&r->links[i];
            if (l->identity.id==id.id) {
                if (strcmp(l->ip,ip) || l->identity.control_port!=id.control_port || strcmp(l->identity.name,id.name) || l->identity.data_port!=id.data_port) return -1;
                ++r->duplicates; break;
            }
            if (!strcmp(l->ip,ip) && l->identity.control_port==id.control_port) return -1;
        }
        if (i<r->count) continue;
        if (r->count==FR_CONNECTION_NODES) return -1;
        fr_link *l=&r->links[r->count++]; l->identity=id; strcpy(l->ip,ip);
    }
    sort_links(r);
    return r->count?0:1;
}
static int authenticate(const fr_config *cfg, fr_connection *r, unsigned index) {
    fr_link *l=&r->links[index]; char response[RESP_MAX_MESSAGE], request[128]; resp_value v[RESP_MAX_VALUES];
    l->fd=net_tcp_connect_from(l->ip,l->identity.control_port,3000,l->via.address,l->via.index);
    if (l->fd<0) return -1;
    int64_t now=net_now_ns(); if (now<0) return -1;
    int64_t deadline=now/1000000+3000;
    const char *error=control_authenticate(l->fd,cfg->password,response,v,deadline);
    if (error) { fprintf(stderr,"connection: %s: %s\n",l->identity.name,error); return -1; }
    l->authenticated=1; int port=net_port(r->udp); if (port<0) return -1;
    int n=snprintf(request,sizeof(request),"LINK %" PRIu32 " %d %u\r\n",r->session,port,index);
    if (n<0 || (size_t)n>=sizeof(request) || control_exchange(l->fd,request,(size_t)n,response,v,deadline)) return -1;
    if (v[0].type!=RESP_ARRAY || v[0].len!=4) return -1;
    char id[17]; uint64_t confirmed=0;
    if (v[4].type!=RESP_BULK || v[4].is_null || v[4].len!=16) return -1;
    memcpy(id,v[4].data,16); id[16]=0;
    if (identity_parse(id,&confirmed) || confirmed!=l->identity.id) return -1;
    return v[0].type==RESP_ARRAY && v[0].len==4 && v[1].type==RESP_BULK && !v[1].is_null &&
        v[1].len==strlen(l->identity.name) && !memcmp(v[1].data,l->identity.name,v[1].len) &&
        v[2].type==RESP_INTEGER && v[2].integer==l->identity.control_port &&
        v[3].type==RESP_INTEGER && v[3].integer==l->identity.data_port ? 0:-1;
}
static int source(const fr_link *l, const struct sockaddr_in *peer) {
    struct sockaddr_in expected;
    return !net_address(l->ip,l->identity.data_port,&expected) &&
        peer->sin_addr.s_addr==expected.sin_addr.s_addr && peer->sin_port==expected.sin_port;
}
static int multicast_check(const fr_config *cfg, fr_connection *r) {
    struct sockaddr_in group; unsigned char bytes[FR_PROBE_BYTES];
    if (net_address(cfg->multicast,cfg->discovery_port,&group)) return -1;
    /* Up to five probes, 200 ms each: one lost or late datagram (a node busy
     * paging) must not fail CONNECTION. A reply to any of these probes counts. */
    unsigned mask=0, all=(1u<<r->count)-1; uint32_t first=r->step+1;
    for (unsigned attempt=0; attempt<5 && mask!=all; ++attempt) {
        fr_probe p={.session=r->session,.step=++r->step,.node=255};
        size_t n=probe_make(bytes,&p); int64_t now=net_now_ns(); if (now<0) return -1;
        int64_t deadline=now+200000000;
        if (send_packet(r->udp,bytes,n,&group,deadline)) return -1;
        while (mask!=all) {
            int ready=wait_until(r->udp,0,deadline); if (ready!=1) break;
            struct sockaddr_in peer; fr_probe reply;
            ssize_t got=net_udp_recv(r->udp,bytes,sizeof(bytes),&peer);
            if (got<0) { if (errno==EAGAIN || errno==EWOULDBLOCK || errno==EMSGSIZE) continue; return -1; }
            if (probe_read(bytes,(size_t)got,&reply) || !reply.reply || reply.kind || reply.session!=r->session ||
                reply.step<first || reply.step>r->step || reply.node>=r->count || !source(&r->links[reply.node],&peer)) { ++r->ignored; continue; }
            unsigned bit=1u<<reply.node;
            if (mask&bit) ++r->duplicates;
            mask|=bit; r->links[reply.node].multicast=1;
        }
    }
    for (unsigned i=0; i<r->count; ++i) if (!r->links[i].multicast)
        fprintf(stderr,"connection: no multicast reply from %s on %s; unicast on this interface\n",r->links[i].identity.name,r->links[i].via.name);
    return mask==all?0:-1;
}
static int measure(fr_connection *r, unsigned index, unsigned kind) {
    fr_link *l=&r->links[index]; struct sockaddr_in peer;
    if (net_address(l->ip,l->identity.data_port,&peer)) return -1;
    unsigned count=probe_count(kind), all=(1u<<count)-1;
    static unsigned char packets[16*FR_PROBE_BYTES], received[16*FR_PROBE_BYTES]; size_t sizes[16];
    for (unsigned attempt=0; attempt<FR_RTT_SAMPLES+10 && l->rtt[kind].count<FR_RTT_SAMPLES; ++attempt) {
        fr_probe p={.session=r->session,.step=++r->step,.kind=kind,.node=255};
        /* The train is one datagram, as REQ/PART frames are on the data plane. */
        size_t train=0;
        for (unsigned f=0; f<count; ++f) { p.frag=f; train+=sizes[f]=probe_make(packets+train,&p); }
        int64_t start=net_now_ns(); if (start<0) return -1;
        int64_t deadline=start+100000000; unsigned mask=0;
        if (send_packet(r->udp,packets,train,&peer,deadline)) return -1;
        while (mask!=all) {
            int ready=wait_until(r->udp,0,deadline); if (ready<0) return -1; if (!ready) break;
            struct sockaddr_in from; fr_probe reply;
            ssize_t n=net_udp_recv(r->udp,received,sizeof(received),&from);
            if (n<0) { if (errno==EAGAIN || errno==EWOULDBLOCK || errno==EMSGSIZE) continue; return -1; }
            if (!source(l,&from)) { ++r->ignored; continue; }
            for (size_t at=0,len;at<(size_t)n;at+=len) {
                if (!(len=fr_frame_next(received+at,(size_t)n-at))) { ++r->ignored; break; }
                if (probe_read(received+at,len,&reply) || !reply.reply || reply.node!=index ||
                    reply.session!=r->session || reply.step!=r->step || reply.kind!=kind) { ++r->ignored; continue; }
                unsigned bit=1u<<reply.frag;
                if (mask&bit) ++r->duplicates;
                mask|=bit;
            }
        }
        if (mask!=all) { ++l->losses[kind]; if (l->losses[kind]>10) return -1; continue; }
        int64_t end=net_now_ns(); if (end<start || stats_add(&l->rtt[kind],(uint64_t)(end-start))) return -1;
    }
    return l->rtt[kind].count==FR_RTT_SAMPLES?0:-1;
}
static int report(fr_link *l, unsigned kind, FILE *output) {
    uint64_t p[3]; if (stats_percentiles(&l->rtt[kind],p)) return -1;
    char hist[192]; size_t used=0;
    for (unsigned i=0; i<FR_HIST_BINS; ++i) {
        int n=snprintf(hist+used,sizeof(hist)-used,"%s%u",i?",":"",l->rtt[kind].bins[i]);
        if (n<0 || (size_t)n>=sizeof(hist)-used) return -1;
        used+=(size_t)n;
    }
    char request[512], response[RESP_MAX_MESSAGE]; resp_value v[RESP_MAX_VALUES];
    int n=snprintf(request,sizeof(request),"LINK REPORT %u %u %u %" PRIu64 " %" PRIu64 " %" PRIu64 " %s\r\n",
        kind,l->rtt[kind].count,l->losses[kind],p[0],p[1],p[2],hist);
    int64_t now=net_now_ns(); if (now<0) return -1;
    if (n<0 || (size_t)n>=sizeof(request) || control_exchange(l->fd,request,(size_t)n,response,v,now/1000000+3000) ||
        v[0].type!=RESP_STRING || v[0].len!=2 || memcmp(v[0].data,"OK",2)) return -1;
    if (output && fprintf(output,"hist node=%s bytes=%u samples=%u losses=%u mean_us=%.3f p50_us=%.3f p99_us=%.3f p999_us=%.3f bins=%s\n",
        l->identity.name,probe_train_bytes(kind),l->rtt[kind].count,l->losses[kind],
        (double)l->rtt[kind].sum/l->rtt[kind].count/1000,p[0]/1000.0,p[1]/1000.0,p[2]/1000.0,hist)<0) return -1;
    return 0;
}
int connection_best_path(const fr_path *paths,unsigned count,uint64_t id,unsigned kind) {
    if (!paths || count>FR_CONNECTION_PATHS || kind>2) return -1;
    int best=-1;
    for (unsigned i=0;i<count;++i) {
        const fr_path *p=&paths[i];
        if (!p->ready || p->identity.id!=id || !p->percentile[kind][1]) continue;
        if (best<0 || p->percentile[kind][1]<paths[best].percentile[kind][1] ||
            (p->percentile[kind][1]==paths[best].percentile[kind][1] &&
             (strcmp(p->via.name,paths[best].via.name)<0 ||
              (!strcmp(p->via.name,paths[best].via.name) && strcmp(p->ip,paths[best].ip)<0)))) best=(int)i;
    }
    return best;
}
static int release_links(fr_connection *r) {
    int rc=0;
    for (unsigned i=0;i<r->count;++i) if (r->links[i].fd>=0) {
        fr_link *l=&r->links[i];
        if (l->authenticated) {
            char reply[RESP_MAX_MESSAGE]; resp_value v[RESP_MAX_VALUES];
            /* Revoke the previous probe owner before another interface tries
             * the same node. A closed TCP fd alone can race its read event. */
            if (control_exchange(l->fd,"LINK RELEASE\r\n",14,reply,v,net_now_ns()/1000000+3000)) rc=-1;
        }
        if (close(l->fd)) rc=-1;
        l->fd=-1;
    }
    return rc;
}
static int report_path(fr_link *l) {
    char request[256],response[RESP_MAX_MESSAGE]; resp_value v[RESP_MAX_VALUES];
    int n=snprintf(request,sizeof(request),"LINK PATH %s %s %" PRIu64 " %s\r\n",l->via.name,
                   interface_type(l->via.type),l->via.speed_bps,l->via.address);
    if (n<0 || (size_t)n>=sizeof(request) || control_exchange(l->fd,request,(size_t)n,response,v,net_now_ns()/1000000+3000) ||
        v[0].type!=RESP_STRING || v[0].len!=2 || memcmp(v[0].data,"OK",2)) return -1;
    return 0;
}
int connection_run(const fr_config *cfg,fr_connection *r,FILE *output) {
    memset(r,0,sizeof(*r)); r->udp=-1;
    for (unsigned i=0;i<FR_CONNECTION_NODES;++i) r->links[i].fd=-1;
    for (unsigned i=0;i<FR_INTERFACES;++i) r->sockets[i]=-1;
    if (!cfg->password[0] || !cfg->discovery_port ||
        interfaces_read(cfg->interfaces,cfg->interface,r->interfaces,&r->interface_count)) {
        fprintf(stderr,"connection: password, discovery-port and existing local interfaces are required\n"); return -1;
    }
    r->local=r->interface_count==1 && r->interfaces[0].loopback;
    auth_state random={0}; char hex[33];
    if (auth_nonce(&random,hex)) return -1;
    r->session=(uint32_t)random.nonce[0]<<24|(uint32_t)random.nonce[1]<<16|(uint32_t)random.nonce[2]<<8|random.nonce[3];
    if (!r->session) r->session=1;
    fr_connection *scan=calloc(1,sizeof(*scan)); if (!scan) { auth_reset(&random); return -1; }
    int rc=-1;
    /* Wired interfaces first: a node with a measured Ethernet or Thunderbolt
     * path is not probed again over Wi-Fi or link-local. Repeating the full
     * probe train on those fallback paths can add seconds to every reconnect. */
    unsigned order[FR_INTERFACES],ordered=0;
    for (unsigned pass=0;pass<2;++pass) for (unsigned i=0;i<r->interface_count;++i)
        if ((r->interfaces[i].type==FR_LINK_ETHERNET || r->interfaces[i].type==FR_LINK_THUNDERBOLT)==!pass) order[ordered++]=i;
    for (unsigned o=0;o<ordered;++o) {
        unsigned iface=order[o];
        memset(scan,0,sizeof(*scan)); scan->udp=-1; scan->session=r->session; scan->step=r->step;
        for (unsigned i=0;i<FR_CONNECTION_NODES;++i) scan->links[i].fd=-1;
        fr_interface *via=&r->interfaces[iface]; fr_config local=*cfg;
        strcpy(local.interface,via->address);
        if (r->interface_count>1 || !strcmp(cfg->broadcast,"255.255.255.255")) strcpy(local.broadcast,via->broadcast);
        scan->local=(int)via->loopback;
        scan->udp=net_udp_bind(via->address,0,0); r->sockets[iface]=scan->udp;
        if (scan->udp<0 || net_bind_interface(scan->udp,via->index) ||
            (via->loopback && net_multicast(scan->udp,FR_DISCOVERY_GROUP,via->address,0))) goto done;
        int discovered=discover(&local,scan,random.nonce,via);
        if (discovered<0) goto done; /* socket error or conflicting advertisements */
        if (!scan->count) {
            if (output && fprintf(output,"interface=%s address=%s paths=0 ignored=%u\n",via->name,via->address,scan->ignored)<0) goto done;
            continue;
        }
        for (unsigned i=0;i<scan->count;++i) {
            fr_link *l=&scan->links[i]; l->via=*via; l->interface_slot=iface;
            if (authenticate(&local,scan,i)) l->authenticated=0;
        }
        if (interface_multicast_wanted(via->type,scan->count) && !net_multicast(scan->udp,cfg->multicast,via->address,0)) (void)multicast_check(&local,scan);
        int wired=via->type==FR_LINK_ETHERNET || via->type==FR_LINK_THUNDERBOLT;
        for (unsigned i=0;i<scan->count;++i) {
            fr_link *l=&scan->links[i]; unsigned measured=0,skip=0;
            for (unsigned j=0;!wired && j<r->path_count;++j) if (r->paths[j].identity.id==l->identity.id && r->paths[j].ready &&
                (r->paths[j].via.type==FR_LINK_ETHERNET || r->paths[j].via.type==FR_LINK_THUNDERBOLT)) skip=1;
            if (l->authenticated && !skip) for (unsigned kind=0;kind<3;++kind) {
                if (measure(scan,i,kind)) break;
                ++measured;
            }
            if (r->path_count==FR_CONNECTION_PATHS) goto done;
            unsigned pi=r->path_count++; fr_path *path=&r->paths[pi];
            path->identity=l->identity; strcpy(path->ip,l->ip); path->via=*via; path->interface_slot=iface;
            path->ready=l->authenticated && measured==3; path->skipped=skip;
            for (unsigned k=0;k<3;++k) {
                path->samples[k]=l->rtt[k].count; path->losses[k]=l->losses[k];
                if (path->samples[k]) (void)stats_percentiles(&l->rtt[k],path->percentile[k]);
            }
            unsigned slot;
            for (slot=0;slot<r->count && r->links[slot].identity.id!=l->identity.id;++slot) {}
            if (slot==r->count) {
                if (r->count==FR_CONNECTION_NODES) goto done;
                ++r->count; r->links[slot]=*l; r->links[slot].path=pi; r->links[slot].fd=-1;
            } else if (strcmp(r->links[slot].identity.name,l->identity.name)) goto done;
            int best=connection_best_path(r->paths,r->path_count,l->identity.id,FR_PATH_RTT_KIND);
            if (best==(int)pi) { r->links[slot]=*l; r->links[slot].path=pi; r->links[slot].fd=-1; }
        }
        r->ignored+=scan->ignored; r->duplicates+=scan->duplicates; r->step=scan->step;
        if (release_links(scan)) goto done;
    }
    if (!r->count) { fprintf(stderr,"connection: no nodes found\n"); goto done; }
    if (cfg->peer_count && r->count!=cfg->peer_count) { fprintf(stderr,"connection: expected %u configured nodes, found %u\n",cfg->peer_count,r->count); goto done; }
    sort_links(r);
    unsigned valid=0;
    for (unsigned i=0;i<r->count;++i) {
        fr_link *l=&r->links[i];
        if (r->paths[l->path].ready) { r->paths[l->path].selected=1; ++valid; }
    }
    if (output) for (unsigned i=0;i<r->path_count;++i) {
        const fr_path *p=&r->paths[i];
        if (fprintf(output,"path node=%s id=%016" PRIx64 " address=%s via=%s local=%s type=%s speed_bps=%" PRIu64
            " barrier_bytes=21248 p50_us=%.3f p99_us=%.3f chosen=%s status=%s\n",p->identity.name,p->identity.id,p->ip,
            p->via.name,p->via.address,interface_type(p->via.type),p->via.speed_bps,
            p->percentile[FR_PATH_RTT_KIND][0]/1000.0,p->percentile[FR_PATH_RTT_KIND][1]/1000.0,
            p->selected?"yes":"no",p->ready?"measured":p->skipped?"skipped_wired_path_measured":"incomplete")<0) goto done;
    }
    /* Automatic discovery: a node without a complete path is left out of this
     * plan (it rejoins at the next CONNECTION); a fixed roster stays strict. */
    if (valid!=r->count) {
        if (!valid && !cfg->peer_count) {
            fputs("connection: nodes answered, but none completed a path with this head: if they report \"authentication failed\","
                  " they belong to another cluster; install them with this head's secret (its state-dir/cluster.secret)\n",stderr);
            goto done;
        }
        if (cfg->peer_count) { fprintf(stderr,"connection: a node has no complete authenticated path\n"); goto done; }
        unsigned kept=0;
        for (unsigned i=0;i<r->count;++i) {
            if (r->paths[r->links[i].path].ready) r->links[kept++]=r->links[i];
            else fprintf(stderr,"connection: %s has no complete path; left out until the next CONNECTION\n",r->links[i].identity.name);
        }
        r->count=kept;
    }
    for (unsigned i=0;i<r->count;++i) {
        fr_link *l=&r->links[i]; r->udp=r->sockets[l->interface_slot];
        if (authenticate(cfg,r,i) || report_path(l)) goto done;
        for (unsigned kind=0;kind<3;++kind) if (report(l,kind,output)) goto done;
    }
    if (output && (fprintf(output,"CONNECTION complete nodes=%u paths=%u interfaces=%u ignored=%u duplicates=%u\n",
        r->count,r->path_count,r->interface_count,r->ignored,r->duplicates)<0 || fflush(output))) goto done;
    rc=0;
done:
    (void)release_links(scan); free(scan); auth_reset(&random); return rc;
}
int connection_close(fr_connection *r) {
    int rc=0;
    for (unsigned i=0; i<FR_CONNECTION_NODES; ++i) {
        if (r->links[i].fd>=0 && close(r->links[i].fd)<0) rc=-1;
        r->links[i].fd=-1;
    }
    if (r->interface_count) {
        for (unsigned i=0;i<r->interface_count;++i) {
            if (r->sockets[i]>=0 && close(r->sockets[i])) rc=-1;
            r->sockets[i]=-1;
        }
    } else if (r->udp>=0 && close(r->udp)<0) rc=-1;
    r->udp=-1; return rc;
}

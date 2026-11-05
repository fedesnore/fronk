/* discovery.c - authenticated hardware collection and atomic profile output.
 * Compute fields enter only when actual backend measurements exist.
 */
#include "discovery.h"
#include "control.h"
#include "identity.h"
#include "net.h"
#include "planner.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int discovery_fetch(int fd, fr_hwinfo *out) {
    char reply[RESP_MAX_MESSAGE]; resp_value values[RESP_MAX_VALUES];
    int64_t now=net_now_ns(); if (now<0) return -1;
    if (control_exchange(fd,"HW\r\n",4,reply,values,now/1000000+3000) ||
        values[0].type!=RESP_BULK || values[0].is_null) return -1;
    return hw_parse(values[0].data,values[0].len,out);
}
static int profile(FILE *out, const fr_hwinfo *h, const fr_link *link,const fr_bench_report *bench) {
    char text[FR_HW_TEXT]; int n=hw_format(h,text,sizeof(text));
    if (n<0 || fprintf(out,"# %s %016" PRIx64 " %s\n",link?"node":"head",h->id,h->name)<0) return -1;
    /* On-disk text uses LF; HW's strict wire representation uses CRLF. */
    for (int i=0;i<n;++i) if (text[i]!='\r' && fputc(text[i],out)==EOF) return -1;
    if (link) {
        if (fprintf(out,"address:%s\ncontrol_port:%u\ndata_port:%u\nmulticast:%s\n",
            link->ip,link->identity.control_port,link->identity.data_port,
            link->via.type==FR_LINK_WIFI?"not_required_wifi":link->multicast?"ok":"not_required_singleton")<0) return -1;
        if (fprintf(out,"head_interface:%s\nhead_interface_address:%s\nhead_link_type:%s\nhead_link_speed_bps:%" PRIu64 "\n",
            link->via.name,link->via.address,interface_type(link->via.type),link->via.speed_bps)<0) return -1;
        const char *sizes[]={"64","5k","20k"};
        for (unsigned k=0;k<3;++k) {
            uint64_t p[3]; if (stats_percentiles(&link->rtt[k],p)) return -1;
            if (fprintf(out,"rtt_us_%s:p50=%.3f,p99=%.3f,p999=%.3f\nrtt_%s_samples:%u\nrtt_%s_losses:%u\n",
                sizes[k],p[0]/1000.0,p[1]/1000.0,p[2]/1000.0,sizes[k],link->rtt[k].count,sizes[k],link->losses[k])<0) return -1;
        }
    }
    if (bench) {
        char computed[FR_BENCH_TEXT]; int count=fr_bench_format(bench,computed,sizeof(computed));
        if (count<0) return -1;
        for (int i=0;i<count;++i) if (computed[i]!='\r' && fputc(computed[i],out)==EOF) return -1;
    }
    return fputc('\n',out)==EOF?-1:0;
}
int discovery_run(const fr_connection *links, fr_discovery *out, FILE *output) {
    memset(out,0,sizeof(*out)); out->burst=1;
    if (!links->count || links->count>FR_CONNECTION_NODES) return -1;
    for (unsigned i=0; i<links->count; ++i) {
        const fr_link *l=&links->links[i];
        if (!l->authenticated || l->fd<0 || !l->identity.id) return -1;
        for (unsigned k=0; k<3; ++k) if (l->rtt[k].count!=FR_RTT_SAMPLES) return -1;
        for (unsigned j=0;j<i;++j) if (links->links[j].identity.id==l->identity.id) return -1;
    }
    for (unsigned i=0; i<links->count; ++i) {
        const fr_link *l=&links->links[i]; fr_hwinfo h;
        if (discovery_fetch(l->fd,&h) || h.id!=l->identity.id || strcmp(h.name,l->identity.name)) {
            fprintf(stderr,"discovery: invalid or unavailable HW from %s\n",l->identity.name); return -1;
        }
        if (h.translated) {
            ++out->excluded;
            if (output && fprintf(output,"excluded=%016" PRIx64 " name=%s reason=\"%s\"\n",h.id,h.name,hw_reason(&h))<0) return -1;
            continue;
        }
        out->nodes[out->count++]=h;
        if (output && profile(output,&h,l,NULL)) return -1;
    }
    if (output && (fprintf(output,"DISCOVERY complete nodes=%u excluded=%u\n",out->count,out->excluded)<0 || fflush(output))) return -1;
    return out->count?0:-1;
}
static const fr_link *node_link(const fr_connection *links,uint64_t id) {
    for (unsigned j=0;j<links->count;++j) if (links->links[j].identity.id==id) return &links->links[j];
    return NULL;
}
static int choice_profile(FILE *out,const fr_selection *s) {
    if (!s->count) return 0;
    char text[FR_SELECTION_TEXT]; int n=selection_format(s,text,sizeof(text));
    if (n<0 || fprintf(out,"# backend_cost_curves\n%s",text)<0) return -1;
    if (s->assigned) {
        if (selection_reason(s,s->assigned,text,sizeof(text))<0 || fprintf(out,"backend_choice:%s\n",text)<0) return -1;
        int best=selection_pick(s,s->assigned);
        if (fprintf(out,"backend_selected:%s\nbackend_assigned_blocks:%u\n",s->candidate[best].name,s->assigned)<0) return -1;
    }
    /* Persist the winner at every feasible integer share for the planner. */
    for (unsigned b=1;b<=FR_BLOCKS;++b) {
        int best=selection_pick(s,b); if (best<0) continue;
        if (fprintf(out,"backend_for_%u_blocks:%s cost_ns=%.17g%s\n",b,s->candidate[best].name,
            selection_cost(&s->candidate[best],b),b>FR_SWEEP?" estimated":" measured")<0) return -1;
    }
    return 0;
}
/* BENCH in two halves, so the head can measure itself while the nodes sweep:
 * start sends every node its BENCH SWEEP, collect waits for the answers. */
int discovery_bench_start(const fr_connection *links,fr_discovery *out,FILE *output,int refresh) {
    if (!links || !out || !out->count || out->count>FR_CONNECTION_NODES || links->count>FR_CONNECTION_NODES ||
        !out->burst || out->burst>FR_BENCH_BURST_MAX) return -1;
    out->measured=0;
    /* Every node sweeps (or answers from its cache) concurrently. */
    char command[64]; int n=snprintf(command,sizeof(command),"BENCH SWEEP %u%s\r\n",out->burst,refresh?" refresh":"");
    for (unsigned i=0;i<out->count;++i) {
        const fr_link *link=node_link(links,out->nodes[i].id);
        if (!link || !link->authenticated || link->fd<0) return -1;
        for (size_t sent=0;sent<(size_t)n;) {
            if (net_wait(link->fd,1,10000)!=1) return -1;
            ssize_t k=net_send(link->fd,command+sent,(size_t)n-sent); if (k<=0) return -1; sent+=(size_t)k;
        }
        if (output) { fprintf(output,"BENCH node=%s sweep burst=%u%s\n",out->nodes[i].name,out->burst,refresh?" refresh":""); fflush(output); }
    }
    return 0;
}
int discovery_bench_collect(const fr_connection *links,fr_discovery *out,FILE *output) {
    if (!links || !out || !out->count || out->count>FR_CONNECTION_NODES) return -1;
    for (unsigned i=0;i<out->count;++i) {
        const fr_link *link=node_link(links,out->nodes[i].id); char reply[RESP_MAX_MESSAGE]; resp_value values[RESP_MAX_VALUES];
        int64_t now=net_now_ns(), start=now; fr_selection *s=&out->selection[i];
        /* A first sweep takes minutes: say every 15 s that the node is still on it. */
        for (int ready=0; !ready && now>=0 && (now-start)/1000000000<3600; now=net_now_ns()) {
            ready=net_wait(link->fd,0,15000);
            if (ready<0) break;
            if (!ready && output) { fprintf(output,"BENCH waiting node=%s %.0f s (sweeping its backends)\n",out->nodes[i].name,(net_now_ns()-start)/1e9); fflush(output); }
        }
        const char *error=now<0?"clock unavailable":control_exchange(link->fd,"",0,reply,values,now/1000000+3600000);
        if (error || values[0].type!=RESP_BULK || values[0].is_null || selection_parse(values[0].data,values[0].len,s) ||
            s->burst!=out->burst || !selection_complete(s)) {
            int said=!error && values[0].type==RESP_ERROR && values[0].len<200;
            fprintf(stderr,"discovery: BENCH SWEEP failed for %s%s%s%.*s\n",out->nodes[i].name,error||said?": ":"",error?error:"",
                    said?(int)values[0].len:0,said?values[0].data:""); return -1;
        }
        if (output) { fprintf(output,"BENCH node=%s done %.1f s\n",out->nodes[i].name,(net_now_ns()-start)/1e9); fflush(output); }
        strcpy(out->last_backend[i],s->candidate[0].name);
        if (output && choice_profile(output,s)) return -1;
    }
    return 0;
}
int discovery_bench(const fr_connection *links,fr_discovery *out,FILE *output,int refresh) {
    return discovery_bench_start(links,out,output,refresh) ? -1 : discovery_bench_collect(links,out,output);
}
int discovery_configure(const fr_config *cfg,const fr_connection *links,fr_discovery *out,
                        unsigned head_blocks,const unsigned *shares,unsigned ssd,FILE *output) {
    if (!cfg || !links || !out || !shares || !output || out->count>FR_CONNECTION_NODES || head_blocks>FR_BLOCKS) return -1;
    fr_capacity_node caps[FR_PLAN_NODES]; unsigned count=0,total=head_blocks+ssd;
    fr_hwinfo head; if (hw_probe(&head) || hw_node_available(&head) || identity_load(cfg->state_dir,&head.id)) return -1; /* as at PLAN (head.c) */
    head.margin=cfg->mem_margin; head.head_reserve=cfg->head_reserve; head.head_fixed=cfg->head_blocks; if (hw_finalize(&head)) return -1;
    if (head_blocks) {
        int best=selection_pick(&out->head_selection,head_blocks); if (best<0) return -1;
        uint64_t cap=selection_capacity(&head,out->head_selection.candidate[best].name,1);
        if (!selection_fits(&out->head_selection.candidate[best],&head,1,head_blocks)) { fprintf(stderr,"discovery: this Mac now has room for %llu blocks, the plan gave it %u (free memory dropped); run CONNECTION again\n",(unsigned long long)cap,head_blocks); return -1; }
        caps[count++]=(fr_capacity_node){head.id,cap};
    }
    for (unsigned i=0;i<out->count;++i) if (shares[i]) {
        if (shares[i]>FR_BLOCKS || total>FR_BLOCKS-shares[i]) return -1;
        int best=selection_pick(&out->selection[i],shares[i]); const fr_link *link=node_link(links,out->nodes[i].id); fr_hwinfo fresh;
        if (best<0 || !link || discovery_fetch(link->fd,&fresh) || fresh.id!=out->nodes[i].id) {
            fprintf(stderr,"discovery: %s stopped answering after DISCOVERY; check it still runs fronk node, then run CONNECTION again\n",out->nodes[i].name); return -1; }
        uint64_t cap=selection_capacity(&fresh,out->selection[i].candidate[best].name,0);
        if (!selection_fits(&out->selection[i].candidate[best],&fresh,0,shares[i])) { fprintf(stderr,"discovery: %s now has room for %llu blocks, the plan gave it %u (free memory dropped); run CONNECTION again\n",
            out->nodes[i].name,(unsigned long long)cap,shares[i]); return -1; }
        caps[count++]=(fr_capacity_node){fresh.id,cap}; total+=shares[i];
    }
    fr_capacity capacity; char text[FR_SELECTION_TEXT];
    /* head-ssd blocks are counted as covered; capacity shows only resident ones. */
    if (total!=FR_BLOCKS || planner_capacity(caps,count,&capacity)<0) return -1;
    if (!ssd && (planner_capacity_format(&capacity,text,sizeof(text))<0 || fputs(text,output)==EOF)) return -1;
    /* No node is configured until every share and the selected-capacity gate pass. */
    for (unsigned i=0;i<out->count;++i) if (shares[i]) {
        const fr_link *link=node_link(links,out->nodes[i].id); char command[40],reply[RESP_MAX_MESSAGE]; resp_value values[RESP_MAX_VALUES];
        int n=snprintf(command,sizeof(command),"BACKEND %u\r\n",shares[i]); int64_t now=net_now_ns();
        if (now<0 || control_exchange(link->fd,command,(size_t)n,reply,values,now/1000000+10000) || values[0].type!=RESP_BULK || values[0].is_null ||
            selection_reason(&out->selection[i],shares[i],text,sizeof(text))<0 || strlen(text)!=values[0].len || memcmp(text,values[0].data,values[0].len)) {
            fprintf(stderr,"discovery: %s did not accept its %u blocks (BACKEND)\n",out->nodes[i].name,shares[i]); return -1; }
        out->selection[i].assigned=shares[i];
        if (fprintf(output,"CONFIGURATION %s %s\n",out->nodes[i].name,text)<0) return -1;
    }
    if (head_blocks) {
        out->head_selection.assigned=head_blocks;
        if (selection_reason(&out->head_selection,head_blocks,text,sizeof(text))<0 || fprintf(output,"CONFIGURATION head %s\n",text)<0) return -1;
    }
    return fflush(output)?-1:0;
}
int discovery_capacity(const fr_config *cfg,const fr_discovery *hw,FILE *output) {
    if (!cfg || !hw || !output || hw->count>FR_CONNECTION_NODES) return -1;
    fr_hwinfo head;
    if (hw_probe(&head) || hw_node_available(&head) || identity_load(cfg->state_dir,&head.id)) return -1;
    head.margin=cfg->mem_margin; head.head_reserve=cfg->head_reserve; head.head_fixed=cfg->head_blocks;
    if (hw_finalize(&head)) return -1;
    fr_capacity_node nodes[FR_PLAN_NODES]={{head.id,head.translated?0:head.head_blocks}};
    unsigned count=1;
    for (unsigned i=0;i<hw->count;++i) {
        const fr_hwinfo *h=&hw->nodes[i];
        if (h->id==head.id) continue; /* same machine cannot supply both caps */
        uint64_t cap=h->translated?0:h->blocks;
        const fr_selection *s=&hw->selection[i];
        if (s->count) {
            cap=0;
            if (!selection_complete(s)) return -1;
            if (s->assigned) { int best=selection_pick(s,s->assigned); if (best<0) return -1; cap=s->candidate[best].capacity; }
            else for (unsigned c=0;c<s->count;++c) if (s->candidate[c].capacity>cap) cap=s->candidate[c].capacity;
        }
        nodes[count++]=(fr_capacity_node){h->id,cap};
    }
    fr_capacity capacity; char text[512]; int rc=planner_capacity(nodes,count,&capacity);
    if (rc<0 || planner_capacity_format(&capacity,text,sizeof(text))<0 || fputs(text,output)==EOF || fflush(output)) return -1;
    return rc;
}
int discovery_save(const fr_config *cfg, const fr_connection *links, const fr_discovery *hw) {
    if (!hw->count || hw->count>FR_CONNECTION_NODES || links->count>FR_CONNECTION_NODES || !cfg->profile_file[0]) return -1;
    fr_hwinfo head;
    if (hw_probe(&head) || hw_node_available(&head) || identity_load(cfg->state_dir,&head.id)) return -1;
    strcpy(head.name,cfg->name); head.margin=cfg->mem_margin; head.head_reserve=cfg->head_reserve; head.head_fixed=cfg->head_blocks;
    if (hw_finalize(&head)) return -1;
    char temp[sizeof(cfg->profile_file)+16];
    int n=snprintf(temp,sizeof(temp),"%s.XXXXXX",cfg->profile_file);
    if (n<0 || (size_t)n>=sizeof(temp)) return -1;
    int fd=mkstemp(temp); if (fd<0) return -1;
    FILE *out=fdopen(fd,"w");
    if (!out) { close(fd); unlink(temp); return -1; }
    const char *who=cfg->name;
    int rc=profile(out,&head,NULL,NULL);
    if (!rc && hw->head_selection.count) rc=choice_profile(out,&hw->head_selection);
    for (unsigned i=0; !rc && i<hw->count; ++i) {
        const fr_hwinfo *h=&hw->nodes[i]; const fr_link *link=NULL; who=h->name;
        for (unsigned j=0;j<links->count;++j) if (links->links[j].identity.id==h->id) link=&links->links[j];
        /* CONNECTION already enforced multicast for wired groups of 2+ nodes. */
        if (!link || h->translated || !link->authenticated || strcmp(link->identity.name,h->name)) rc=-1;
        else {
            rc=profile(out,h,link,(hw->measured&(1u<<i))?&hw->compute[i]:NULL);
            if (!rc && hw->last_backend[i][0] && fprintf(out,"bench_backend:%s\n",hw->last_backend[i])<0) rc=-1;
            if (!rc) rc=choice_profile(out,&hw->selection[i]); }
    }
    if (rc) fprintf(stderr,"discovery: %s's hardware or measurements could not be saved\n",who);
    if (fflush(out) || fsync(fd)) rc=-1;
    if (fclose(out)) rc=-1;
    if (!rc && rename(temp,cfg->profile_file)) { rc=-1; fprintf(stderr,"discovery: %s could not be written\n",cfg->profile_file); }
    if (rc) unlink(temp);
    return rc;
}

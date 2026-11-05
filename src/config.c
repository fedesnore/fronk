/* config.c - strict, small key/value config with transactional loading.
 *
 * Unknown/duplicate keys and overlong lines are errors, so typos cannot
 * silently change a cluster. Diagnostics never include configuration values.
 */
#include "config.h"
#include "model.h"
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void config_defaults(fr_config *out) {
    memset(out,0,sizeof(*out));
    char hostname[256]={0};
    if (!gethostname(hostname,sizeof(hostname)) && memchr(hostname,0,sizeof(hostname))) {
        char *dot=strchr(hostname,'.'); if (dot) *dot=0;
        if (strlen(hostname)<sizeof(out->name)) strcpy(out->name,hostname);
    }
    strcpy(out->bind,"127.0.0.1");
    out->mem_margin=FR_DEFAULT_MARGIN; out->head_reserve=UINT64_C(10000000000); /* ds4 pages ~7.5 GB in after its first request (BENCH run 35); 3 blocks with 8.6 GB left still swapped (run 38) */
    strcpy(out->profile_file,"fronk-profile.txt");
    strcpy(out->multicast,"239.7.7.7");
    strcpy(out->interface,"127.0.0.1"); strcpy(out->broadcast,"255.255.255.255");
    out->control_port=7700; out->data_port=7701; out->discovery_port=7702;
    out->burst=1;
}
static int gb_value(const char *s, uint64_t *out) {
    uint64_t whole=0, fraction=0, scale=100000000; unsigned digits=0;
    if (*s<'0' || *s>'9') return -1;
    while (*s>='0' && *s<='9') {
        unsigned d=(unsigned)(*s++-'0');
        if (whole>(UINT64_MAX-d)/10) return -1;
        whole=whole*10+d;
    }
    if (*s=='.') {
        ++s;
        while (*s>='0' && *s<='9') {
            if (++digits>9) return -1;
            fraction+=(unsigned)(*s++-'0')*scale; scale/=10;
        }
        if (!digits) return -1;
    }
    if (*s || whole>(UINT64_MAX-fraction)/1000000000) return -1;
    *out=whole*1000000000+fraction;
    return *out==FR_DEFAULT_MARGIN?-1:0;
}
static int port_value(const char *s, unsigned *out) {
    unsigned n=0;
    for (; *s; ++s) {
        if (*s<'0' || *s>'9') return -1;
        n=n*10+(unsigned)(*s-'0');
        if (n>65535) return -1;
    }
    *out=n; return 0;
}
static const char *parse_line(char *line, fr_config *cfg, unsigned *seen) {
    char *comment=strchr(line,'#');
    if (comment) *comment='\0';
    char *save=NULL, *key=strtok_r(line," \t\r",&save);
    if (!key) return NULL;
    char *value=strtok_r(NULL," \t\r",&save);
    if (!value || strtok_r(NULL," \t\r",&save)) return "expected one key and one value";
    const char *keys[]={"name","bind","control-port","data-port","multicast","password","discovery-port","interface","broadcast",
        "mem-margin-gb","head-reserve-gb","state-dir","profile-file","interfaces","burst","password-env","node","model","ds4","head-blocks","backend"};
    unsigned field;
    for (field=0; field<21 && strcmp(key,keys[field]); ++field) {}
    if (field==21) return "unknown directive";
    if (field!=16 && (*seen & (1u<<field))) return "duplicate directive";
    *seen |= 1u<<field;
    if (field==15) {
        for (const char *p=value;*p;++p)
            if (!((*p>='a' && *p<='z') || (*p>='A' && *p<='Z') || *p=='_' || (p!=value && *p>='0' && *p<='9'))) return "invalid environment variable name";
        /* Unset: the stable cluster secret saved in state-dir is used (config_load). */
        const char *secret=getenv(value);
        if (secret && (!*secret || strlen(secret)>=sizeof(cfg->password))) return "password environment variable invalid";
        if (secret) strcpy(cfg->password,secret);
        return NULL;
    }
    if (field==17 || field==18) {
        char *target=field==17?cfg->model:cfg->ds4; size_t cap=field==17?sizeof(cfg->model):sizeof(cfg->ds4);
        if (strlen(value)>=cap) return "value too long";
        strcpy(target,value); return NULL;
    }
    if (field==16) {
        char *address=strchr(value,'@'); struct in_addr ip;
        if (!address || address==value || cfg->peer_count==FR_CONFIG_NODES) return "expected node name@IPv4";
        *address++=0;
        if (strlen(value)>=sizeof(cfg->peers[0].name) || strlen(address)>=sizeof(cfg->peers[0].address) ||
            inet_pton(AF_INET,address,&ip)!=1 || !ip.s_addr || ip.s_addr==INADDR_BROADCAST || IN_MULTICAST(ntohl(ip.s_addr))) return "invalid node entry";
        for (const char *p=value;*p;++p)
            if (!((*p>='a' && *p<='z') || (*p>='A' && *p<='Z') || (*p>='0' && *p<='9') || *p=='_' || *p=='-' || *p=='.')) return "invalid node name";
        for (unsigned i=0;i<cfg->peer_count;++i)
            if (!strcmp(cfg->peers[i].name,value) || !strcmp(cfg->peers[i].address,address)) return "duplicate node entry";
        strcpy(cfg->peers[cfg->peer_count].name,value); strcpy(cfg->peers[cfg->peer_count++].address,address); return NULL;
    }
    if (field==14) {
        if (port_value(value,&cfg->burst) || cfg->burst<1 || cfg->burst>4) return "invalid BENCH burst (1..4)";
        return NULL;
    }
    if (field==13) {
        if (strlen(value)>=sizeof(cfg->interfaces)) return "interface list too long";
        if (strcmp(value,"auto")) {
            const char *p=value; unsigned count=0;
            while (*p) {
                const char *end=strchr(p,','); size_t len=end?(size_t)(end-p):strlen(p); char ip[16]; struct in_addr addr;
                if (!len || len>=sizeof(ip) || ++count>8 || (end && !end[1])) return "invalid interface list";
                memcpy(ip,p,len); ip[len]=0;
                if (inet_pton(AF_INET,ip,&addr)!=1 || !addr.s_addr || addr.s_addr==INADDR_BROADCAST || IN_MULTICAST(ntohl(addr.s_addr))) return "invalid interface list";
                p=end?end+1:p+len;
            }
        }
        strcpy(cfg->interfaces,value); return NULL;
    }
    if (field==20) {
        /* Checked against the compiled backends where it is used (node, head). */
        if (strlen(value)>=sizeof(cfg->backend)) return "backend name too long";
        for (const char *p=value;*p;++p)
            if (!((*p>='a' && *p<='z') || (*p>='0' && *p<='9') || *p=='-' || *p=='_')) return "invalid backend name";
        strcpy(cfg->backend,value);
        return NULL;
    }
    if (field==19) {
        unsigned blocks;
        if (port_value(value,&blocks) || blocks>FR_BLOCKS) return "invalid head-blocks (0..20)";
        cfg->head_blocks=blocks; return NULL;
    }
    if (field==9 || field==10) return gb_value(value,field==9?&cfg->mem_margin:&cfg->head_reserve)?"invalid GB value":NULL;
    if (field==2 || field==3 || field==6) {
        unsigned port;
        if (port_value(value,&port)<0) return "invalid port";
        if (field==2) cfg->control_port=port;
        else if (field==3) cfg->data_port=port; else cfg->discovery_port=port;
        return NULL;
    }
    char *target; size_t cap;
    switch (field) {
        case 0: target=cfg->name; cap=sizeof(cfg->name); break;
        case 1: target=cfg->bind; cap=sizeof(cfg->bind); break;
        case 4: target=cfg->multicast; cap=sizeof(cfg->multicast); break;
        case 7: target=cfg->interface; cap=sizeof(cfg->interface); break;
        case 8: target=cfg->broadcast; cap=sizeof(cfg->broadcast); break;
        case 11: target=cfg->state_dir; cap=sizeof(cfg->state_dir); break;
        case 12: target=cfg->profile_file; cap=sizeof(cfg->profile_file); break;
        default: target=cfg->password; cap=sizeof(cfg->password); break;
    }
    size_t len=strlen(value);
    if (len>=cap) return "value too long";
    if (field==0) {
        for (size_t i=0; i<len; ++i) {
            unsigned char c=(unsigned char)value[i];
            if (!((c>='a' && c<='z') || (c>='A' && c<='Z') ||
                  (c>='0' && c<='9') || c=='_' || c=='-' || c=='.')) return "invalid node name";
        }
    }
    if (field==1 || field==4 || field==7 || field==8) {
        struct in_addr addr;
        if (inet_pton(AF_INET,value,&addr)!=1) return "invalid IPv4 address";
        if (field==4 && !IN_MULTICAST(ntohl(addr.s_addr))) return "invalid multicast address";
        if (field==7 && (!addr.s_addr || addr.s_addr==INADDR_BROADCAST || IN_MULTICAST(ntohl(addr.s_addr)))) return "invalid interface address";
        if (field==8 && (!addr.s_addr || IN_MULTICAST(ntohl(addr.s_addr)))) return "invalid broadcast address";
    }
    memcpy(target,value,len+1);
    return NULL;
}
/* The cluster secret is stable across runs so a running node rejoins any later
 * CONNECTION: a value given in the environment is saved (0600) to
 * state-dir/cluster.secret, and an unset environment reads it back. */
static const char *cluster_secret(fr_config *cfg) {
    if (!cfg->state_dir[0]) return cfg->password[0]?NULL:"password environment variable missing";
    char path[300]; snprintf(path,sizeof(path),"%s/cluster.secret",cfg->state_dir);
    if (cfg->password[0]) {
        char have[sizeof(cfg->password)]={0}; FILE *f=fopen(path,"r");
        if (f) { if (!fgets(have,sizeof(have),f)) have[0]='\0'; fclose(f); }
        if (!strcmp(have,cfg->password)) return NULL;
        mkdir(cfg->state_dir,0700);
        int fd=open(path,O_WRONLY|O_CREAT|O_TRUNC,0600);
        if (fd<0) return "cannot save cluster secret in state-dir";
        size_t n=strlen(cfg->password); int ok=write(fd,cfg->password,n)==(ssize_t)n;
        if (close(fd) || !ok) return "cannot save cluster secret in state-dir";
        return NULL;
    }
    FILE *f=fopen(path,"r");
    if (!f) return "cluster secret missing: set the password environment variable once";
    if (!fgets(cfg->password,sizeof(cfg->password),f)) cfg->password[0]='\0';
    fclose(f); cfg->password[strcspn(cfg->password,"\r\n")]='\0';
    return cfg->password[0]?NULL:"cluster secret file is empty";
}
int config_load(const char *path, fr_config *out, char *error, size_t error_cap) {
    if (!path || !out || (!error && error_cap)) return -1;
    FILE *file=fopen(path,"r");
    if (!file) { if (error_cap) snprintf(error,error_cap,"%s: cannot open config",path); return -1; }
    fr_config cfg; config_defaults(&cfg);
    char line[512]; size_t used=0; unsigned line_no=1, seen=0;
    const char *reason=NULL; int c;
    while ((c=fgetc(file))!=EOF) {
        if (!c) { reason="NUL in configuration"; break; }
        if (c=='\n') {
            line[used]='\0'; reason=parse_line(line,&cfg,&seen);
            if (reason) break;
            used=0; ++line_no;
        } else {
            if (used==sizeof(line)-1) { reason="line too long"; break; }
            line[used++]=(char)c;
        }
    }
    if (!reason && ferror(file)) reason="cannot read config";
    if (!reason && used) { line[used]='\0'; reason=parse_line(line,&cfg,&seen); }
    if (fclose(file)!=0 && !reason) reason="cannot close config";
    if (!reason && cfg.control_port && cfg.control_port==cfg.data_port) reason="control and data ports must differ";
    if (!reason && cfg.data_port && cfg.data_port==cfg.discovery_port) reason="data and discovery ports must differ";
    if (!reason && !cfg.name[0]) reason="cannot determine short hostname; configure name";
    if (!reason && (seen&(1u<<5)) && (seen&(1u<<15))) reason="choose password or password-env, not both";
    if (!reason && (seen&(1u<<15))) reason=cluster_secret(&cfg);
    if (reason) {
        if (error_cap) snprintf(error,error_cap,"%s:%u: %s",path,line_no,reason);
        return -1;
    }
    *out=cfg;
    if (error_cap) error[0]='\0';
    return 0;
}

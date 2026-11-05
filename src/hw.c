/* hw.c - read-only platform discovery, conservative capacity and text snapshots.
 * Memory is sampled on request, never allocated or wired here. Linux parsers
 * and x86 feature decoding also build on arm64 so fixtures test foreign paths.
 */
#include "hw.h"
#include "identity.h"
#include "model.h"
#include "config.h"
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif
#ifdef __x86_64__
#include <cpuid.h>
#endif

static int clean(const char *s, size_t cap) {
    size_t n=strnlen(s,cap);
    if (!n || n==cap) return 0;
    for (size_t i=0; i<n; ++i) if ((unsigned char)s[i]<32 || (unsigned char)s[i]>126) return 0;
    return 1;
}
static int number(const char *s, uint64_t *out) {
    uint64_t n=0;
    if (!*s) return -1;
    for (; *s; ++s) {
        unsigned d=(unsigned char)*s-'0';
        if (d>9 || n>(UINT64_MAX-d)/10) return -1;
        n=n*10+d;
    }
    *out=n; return 0;
}
unsigned hw_x86_isa(uint32_t a, uint32_t b, uint64_t xcr0) {
    return ((b&(1u<<5))?FR_ISA_AVX2:0)|((a&(1u<<12))?FR_ISA_FMA:0)|((a&(1u<<29))?FR_ISA_F16C:0)|
        ((a&(1u<<28))?FR_ISA_AVX:0)|((a&(1u<<27))?FR_ISA_OSXSAVE:0)|((a&(1u<<26))?FR_ISA_XSAVE:0)|
        ((xcr0&6)==6?FR_ISA_YMM:0);
}
const char *hw_backend(const fr_hwinfo *h) {
    if (h->translated) return "none";
    return "unmeasured";
}
const char *hw_reason(const fr_hwinfo *h) {
    const char *backend=hw_backend(h);
    if (!strcmp(backend,"none")) return "none: translated process excluded";
    return "unmeasured: awaiting backend BENCH sweep and assigned share";
}
int hw_finalize(fr_hwinfo *h) {
    if (!clean(h->os,sizeof(h->os)) || !clean(h->arch,sizeof(h->arch)) || !clean(h->version,sizeof(h->version)) ||
        (strcmp(h->os,"macos") && strcmp(h->os,"linux")) ||
        (strcmp(h->arch,"arm64") && strcmp(h->arch,"x86_64")) ||
        !clean(h->cpu,sizeof(h->cpu)) || !clean(h->gpu,sizeof(h->gpu)) ||
        !h->logical_cores || h->logical_cores>4096 || h->cores>h->logical_cores || h->isa>FR_ISA_USABLE ||
        h->metal>1 || h->translated>1 || !h->total || h->available>h->total ||
        (h->isa && strcmp(h->arch,"x86_64")) || (h->translated && strcmp(h->os,"macos")) ||
        (h->metal && (strcmp(h->os,"macos") || !h->metal_limit)) ||
        (!h->metal && (h->metal_limit || strcmp(h->gpu,"none")))) return -1;
    if (h->margin==FR_DEFAULT_MARGIN) {
        h->margin=!strcmp(h->os,"linux")?UINT64_C(1500000000):!strcmp(h->arch,"arm64")?UINT64_C(1000000000):UINT64_C(3000000000);
        /* A flat default leaves a large-memory machine paging: 14 slices on a 48 GB
         * iMac pushed its compute from 4.2 to 6.1 ms (BENCH.md). Keep a tenth of RAM. */
        if (strcmp(h->arch,"arm64") && h->margin<h->total/10) h->margin=h->total/10;
    }
    uint64_t available=h->available;
    h->cap=available>h->margin?available-h->margin:0;
    h->blocks=h->cap/FR_WEIGHT_BLOCK_BYTES;
    h->head_cap=h->available>h->head_reserve?h->available-h->head_reserve:0;
    h->head_blocks=h->head_cap/FR_WEIGHT_BLOCK_BYTES;
    /* head-blocks asks for a count; free memory with ds4 warm still caps it,
     * since 3 blocks next to ds4 on a 24 GB Mac paged 1.7 GB per test pass (BENCH run 37). */
    if (h->head_fixed && h->head_fixed<h->head_blocks) { h->head_blocks=h->head_fixed; h->head_cap=h->head_fixed*FR_WEIGHT_BLOCK_BYTES; }
    return 0;
}
int hw_pages(fr_hwinfo *h, uint64_t page, uint64_t free_pages, uint64_t inactive, uint64_t purgeable) {
    if (!page || inactive>UINT64_MAX-free_pages || purgeable>UINT64_MAX-free_pages-inactive) return -1;
    uint64_t pages=free_pages+inactive+purgeable;
    if (pages>UINT64_MAX/page) return -1;
    h->available=pages*page;
    if (h->available>h->total) h->available=h->total;
    return 0;
}
int hw_unpinned(fr_hwinfo *h, uint64_t page, uint64_t wired, uint64_t anonymous, uint64_t compressed) {
    if (!page || anonymous>UINT64_MAX-wired || compressed>UINT64_MAX-wired-anonymous) return -1;
    uint64_t pages=wired+anonymous+compressed;
    if (pages>UINT64_MAX/page) return -1;
    h->available=pages*page>=h->total?0:h->total-pages*page;
    return 0;
}
/* free+inactive+purgeable plus the pages of kept units counted inactive units
 * twice: the iMac got 13 blocks right after a session and 10 after a restart,
 * and at 13 it paged (8.3 ms per layer, BENCH run 48). Linux MemAvailable
 * already counts page cache, so it stands as hw_probe read it. */
int hw_node_available(fr_hwinfo *h) {
#ifdef __APPLE__
    mach_port_t host=mach_host_self(); vm_size_t page=0; vm_statistics64_data_t vm={0};
    mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
    int ok=host_page_size(host,&page)==KERN_SUCCESS && page &&
        host_statistics64(host,HOST_VM_INFO64,(host_info64_t)&vm,&count)==KERN_SUCCESS &&
        count*sizeof(integer_t)>=offsetof(vm_statistics64_data_t,internal_page_count)+sizeof(vm.internal_page_count);
    if (mach_port_deallocate(mach_task_self(),host)!=KERN_SUCCESS) ok=0;
    if (!ok) return -1;
    return hw_unpinned(h,page,vm.wire_count,vm.internal_page_count,vm.compressor_page_count);
#else
    (void)h; return 0;
#endif
}
int hw_format(const fr_hwinfo *h, char *out, size_t cap) {
    fr_hwinfo c=*h;
    if (!h->id || !clean(h->name,sizeof(h->name)) || hw_finalize(&c) || c.margin!=h->margin || c.cap!=h->cap ||
        c.blocks!=h->blocks || c.head_cap!=h->head_cap || c.head_blocks!=h->head_blocks) return -1;
    char frequency[32]="n/a", wired[32]="n/a";
    if (h->frequency) snprintf(frequency,sizeof(frequency),"%" PRIu64,h->frequency);
    if (h->wired_mb!=UINT64_MAX) snprintf(wired,sizeof(wired),"%" PRIu64,h->wired_mb);
    int n=snprintf(out,cap,
        "id:%016" PRIx64 "\r\nname:%s\r\nos:%s\r\nos_version:%s\r\narch:%s\r\ncpu_model:%s\r\ngpu_name:%s\r\n"
        "cores:%u\r\nlogical_cores:%u\r\nisa_avx2:%u\r\nisa_fma:%u\r\nisa_f16c:%u\r\nisa_avx:%u\r\nisa_osxsave:%u\r\nisa_xsave:%u\r\nisa_ymm:%u\r\n"
        "translated:%s\r\nmetal:%u\r\nmem_total_bytes:%" PRIu64 "\r\nmem_available_bytes:%" PRIu64 "\r\nmem_margin_bytes:%" PRIu64
        "\r\nmem_cap_bytes:%" PRIu64 "\r\nmem_cap_blocks:%" PRIu64 "\r\nmetal_working_set_bytes:%" PRIu64
        "\r\ncpu_frequency_hz:%s\r\niogpu_wired_limit_mb:%s\r\nhead_reserve_bytes:%" PRIu64 "\r\nhead_cap_bytes:%" PRIu64
        "\r\nhead_cap_blocks:%" PRIu64 "\r\nbackend:%s\r\nbackend_reason:%s\r\n",
        h->id,h->name,h->os,h->version,h->arch,h->cpu,h->gpu,h->cores,h->logical_cores,
        !!(h->isa&1),!!(h->isa&2),!!(h->isa&4),!!(h->isa&8),!!(h->isa&16),!!(h->isa&32),!!(h->isa&64),
        h->translated?"yes":"no",h->metal,h->total,h->available,h->margin,h->cap,h->blocks,h->metal_limit,
        frequency,wired,h->head_reserve,h->head_cap,h->head_blocks,hw_backend(h),hw_reason(h));
    return n<0 || (size_t)n>=cap?-1:n;
}
int hw_parse(const char *text, size_t len, fr_hwinfo *out) {
    if (!len || len>=FR_HW_TEXT || memchr(text,0,len)) return -1;
    char copy[FR_HW_TEXT]; memcpy(copy,text,len); copy[len]=0;
    const char *keys[]={"id","name","os","os_version","arch","cpu_model","gpu_name","cores","logical_cores",
        "isa_avx2","isa_fma","isa_f16c","isa_avx","isa_osxsave","isa_xsave","isa_ymm","translated","metal",
        "mem_total_bytes","mem_available_bytes","mem_margin_bytes","mem_cap_bytes","mem_cap_blocks","metal_working_set_bytes",
        "cpu_frequency_hz","iogpu_wired_limit_mb","head_reserve_bytes","head_cap_bytes","head_cap_blocks","backend","backend_reason"};
    char *v[31]={0}, *line=copy; uint32_t seen=0;
    while (*line) {
        char *end=strstr(line,"\r\n"); if (!end) return -1; *end=0;
        char *colon=strchr(line,':'); if (!colon) return -1; *colon++=0;
        unsigned i; for (i=0;i<31 && strcmp(line,keys[i]);++i) {}
        if (i==31 || (seen&(1u<<i))) return -1;
        seen|=1u<<i; v[i]=colon; line=end+2;
    }
    if (seen!=UINT32_C(0x7fffffff)) return -1;
    fr_hwinfo h={0}; char *strings[]={h.name,h.os,h.version,h.arch,h.cpu,h.gpu}; size_t caps[]={64,16,64,16,128,128};
    for (unsigned i=0;i<6;++i) { if (!clean(v[i+1],caps[i])) return -1; strcpy(strings[i],v[i+1]); }
    if (identity_parse(v[0],&h.id)) return -1;
    uint64_t n[29]={0};
    for (unsigned i=7;i<29;++i) {
        if (i==16) { if (strcmp(v[i],"yes") && strcmp(v[i],"no")) return -1; n[i]=!strcmp(v[i],"yes"); }
        else if ((i==24 || i==25) && !strcmp(v[i],"n/a")) n[i]=i==25?UINT64_MAX:0;
        else if (number(v[i],&n[i])) return -1;
    }
    if (n[7]>4096 || n[8]>4096 || n[17]>1) return -1;
    h.cores=(unsigned)n[7]; h.logical_cores=(unsigned)n[8]; h.translated=(unsigned)n[16]; h.metal=(unsigned)n[17];
    for (unsigned i=9;i<16;++i) { if(n[i]>1) return -1; h.isa|=(unsigned)n[i]<<(i-9); }
    h.total=n[18]; h.available=n[19]; h.margin=n[20]; h.metal_limit=n[23]; h.frequency=n[24]; h.wired_mb=n[25]; h.head_reserve=n[26];
    if (h.margin==FR_DEFAULT_MARGIN || hw_finalize(&h) || h.cap!=n[21] || h.blocks!=n[22] || h.head_cap!=n[27] || h.head_blocks!=n[28] ||
        strcmp(hw_backend(&h),v[29]) || strcmp(hw_reason(&h),v[30])) return -1;
    *out=h; return 0;
}
static char *trim(char *s) {
    while (*s==' ' || *s=='\t') ++s;
    size_t n=strlen(s);
    while (n && (s[n-1]==' ' || s[n-1]=='\t' || s[n-1]=='\r' || s[n-1]=='\n')) s[--n]=0;
    return s;
}
/* Missing package/core IDs are common in VMs: cores=0 means unknown. Logical
 * CPUs still come from the processor records; never infer physical topology. */
int hw_linux_read(FILE *cpu, FILE *mem, fr_hwinfo *out) {
    fr_hwinfo h={.os="linux",.arch="x86_64",.version="n/a",.gpu="none",.margin=FR_DEFAULT_MARGIN,.head_reserve=UINT64_C(14000000000),.wired_mb=UINT64_MAX};
    struct { unsigned id, package, core, fields; } records[4096];
    unsigned count=0; char line[4096];
    while (fgets(line,sizeof(line),cpu)) {
        if (!strchr(line,'\n') && !feof(cpu)) return -1;
        char *key=trim(line), *colon=strchr(key,':'); if (!colon) continue;
        *colon++=0; key=trim(key); char *value=trim(colon); uint64_t n;
        if (!strcmp(key,"processor")) {
            if (number(value,&n) || n>UINT_MAX || count==4096) return -1;
            for (unsigned i=0; i<count; ++i) if (records[i].id==n) return -1;
            records[count].id=(unsigned)n; records[count++].fields=0;
        } else if (!strcmp(key,"model name")) {
            if (!count || !clean(value,sizeof(h.cpu))) return -1;
            if (h.cpu[0] && strcmp(h.cpu,value)) return -1;
            strcpy(h.cpu,value);
        } else if (!strcmp(key,"physical id") || !strcmp(key,"core id")) {
            if (!count || number(value,&n) || n>UINT_MAX) return -1;
            unsigned bit=!strcmp(key,"physical id")?1:2;
            if (records[count-1].fields&bit) return -1;
            records[count-1].fields|=bit;
            if (bit==1) records[count-1].package=(unsigned)n; else records[count-1].core=(unsigned)n;
        }
    }
    if (ferror(cpu) || !count) return -1;
    h.logical_cores=count; int topology=1;
    for (unsigned i=0; i<count; ++i) if (records[i].fields!=3) topology=0;
    if (topology) for (unsigned i=0; i<count; ++i) {
        unsigned j;
        for (j=0; j<i; ++j) if (records[j].package==records[i].package && records[j].core==records[i].core) break;
        if (j==i) ++h.cores;
    }
    unsigned seen=0;
    while (fgets(line,sizeof(line),mem)) {
        if (!strchr(line,'\n') && !feof(mem)) return -1;
        char *colon=strchr(line,':'); if (!colon) continue; *colon++=0;
        unsigned bit=!strcmp(line,"MemTotal")?1:!strcmp(line,"MemAvailable")?2:0;
        if (!bit) continue;
        if (seen&bit) return -1;
        seen|=bit; char *value=trim(colon), *space=strpbrk(value," \t");
        if (!space) return -1;
        *space++=0; uint64_t n;
        if (strcmp(trim(space),"kB") || number(value,&n) || n>UINT64_MAX/1024) return -1;
        if (bit==1) h.total=n*1024; else h.available=n*1024;
    }
    if (ferror(mem) || seen!=3 || hw_finalize(&h)) return -1;
    *out=h; return 0;
}
/* Recorded sysctl/vm_stat/sw_vers inputs exercise the same capacity arithmetic
 * as live Mach queries. CPU instruction flags never come from these strings. */
int hw_macos_read(FILE *sysctl, FILE *vm, FILE *version, fr_hwinfo *out) {
    fr_hwinfo h={.os="macos",.gpu="none",.margin=FR_DEFAULT_MARGIN,.head_reserve=UINT64_C(14000000000),.wired_mb=UINT64_MAX};
    char line[512]; unsigned seen=0;
    while (fgets(line,sizeof(line),sysctl)) {
        char *colon=strchr(line,':'); if (!colon) return -1; *colon++=0;
        char *v=trim(colon); uint64_t n; unsigned bit=0;
        if (!strcmp(line,"machdep.cpu.brand_string")) { if (!clean(v,sizeof(h.cpu))) return -1; strcpy(h.cpu,v); bit=1; }
        else if (!strcmp(line,"hw.machine")) { if (!clean(v,sizeof(h.arch))) return -1; strcpy(h.arch,v); bit=2; }
        else {
            if (number(v,&n)) return -1;
            if (!strcmp(line,"hw.memsize")) { h.total=n; bit=4; }
            else if (!strcmp(line,"hw.physicalcpu") && n<=4096) { h.cores=(unsigned)n; bit=8; }
            else if (!strcmp(line,"hw.logicalcpu") && n<=4096) { h.logical_cores=(unsigned)n; bit=16; }
            else if (!strcmp(line,"hw.cpufrequency_max")) { h.frequency=n; bit=32; }
            else return -1;
        }
        if (seen&bit) return -1;
        seen|=bit;
    }
    if (ferror(sysctl) || (seen&31)!=31 || !fgets(line,sizeof(line),vm)) return -1;
    char *start=strstr(line,"page size of "), *end=start?strstr(start," bytes)"):NULL; uint64_t page, pages[3]={0};
    if (!end) return -1;
    *end=0; if (number(start+13,&page)) return -1;
    seen=0;
    while (fgets(line,sizeof(line),vm)) {
        char *colon=strchr(line,':'); if (!colon) continue; *colon++=0;
        int k=!strcmp(line,"Pages free")?0:!strcmp(line,"Pages inactive")?1:!strcmp(line,"Pages purgeable")?2:-1;
        if (k<0) continue;
        char *v=trim(colon); size_t len=strlen(v);
        if (!len || v[len-1]!='.' || (seen&(1u<<k))) return -1;
        v[len-1]=0; if (number(v,&pages[k])) return -1; seen|=1u<<k;
    }
    if (ferror(vm) || seen!=7 || hw_pages(&h,page,pages[0],pages[1],pages[2])) return -1;
    while (fgets(line,sizeof(line),version)) {
        char *colon=strchr(line,':'); if (!colon) return -1; *colon++=0;
        if (!strcmp(line,"ProductVersion")) {
            char *v=trim(colon); if (h.version[0] || !clean(v,sizeof(h.version))) return -1; strcpy(h.version,v);
        }
    }
    if (ferror(version) || hw_finalize(&h)) return -1;
    *out=h; return 0;
}
#ifdef __APPLE__
static int sysvalue(const char *key, void *out, size_t size) {
    size_t len=size;
    return sysctlbyname(key,out,&len,NULL,0) || len!=size?-1:0;
}
static int mac_probe(fr_hwinfo *h) {
    if (sysvalue("hw.memsize",&h->total,sizeof(h->total)) ||
        sysvalue("hw.physicalcpu",&h->cores,sizeof(h->cores)) ||
        sysvalue("hw.logicalcpu",&h->logical_cores,sizeof(h->logical_cores))) return -1;
    size_t len=sizeof(h->cpu);
    if (sysctlbyname("machdep.cpu.brand_string",h->cpu,&len,NULL,0) || !len || len>sizeof(h->cpu) || h->cpu[len-1]) return -1;
    int translated=0;
    if (sysvalue("sysctl.proc_translated",&translated,sizeof(translated)) && errno!=ENOENT) return -1;
    h->translated=translated!=0;
    len=sizeof(h->version);
    if (sysctlbyname("kern.osproductversion",h->version,&len,NULL,0) || !len || len>sizeof(h->version) || h->version[len-1]) return -1;
#ifdef __x86_64__
    if (sysvalue("hw.cpufrequency_max",&h->frequency,sizeof(h->frequency)) && errno!=ENOENT) return -1;
#endif
    uint64_t wired=0; len=sizeof(wired);
    if (!sysctlbyname("iogpu.wired_limit_mb",&wired,&len,NULL,0)) {
        if (len!=4 && len!=8) return -1;
        h->wired_mb=wired;
    } else if (errno!=ENOENT) return -1;
    mach_port_t host=mach_host_self(); vm_size_t page=0; vm_statistics64_data_t vm={0};
    mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
    int ok=host_page_size(host,&page)==KERN_SUCCESS && page &&
        host_statistics64(host,HOST_VM_INFO64,(host_info64_t)&vm,&count)==KERN_SUCCESS && count*sizeof(integer_t)>=offsetof(vm_statistics64_data_t,purgeable_count)+sizeof(vm.purgeable_count);
    if (mach_port_deallocate(mach_task_self(),host)!=KERN_SUCCESS) ok=0;
    if (!ok) return -1;
    if (hw_pages(h,page,vm.free_count,vm.inactive_count,vm.purgeable_count) || hw_metal_probe(h)) return -1;
    return 0;
}
#endif
int hw_probe(fr_hwinfo *out) {
    fr_hwinfo h={.gpu="none",.margin=FR_DEFAULT_MARGIN,.head_reserve=UINT64_C(14000000000),.wired_mb=UINT64_MAX}; struct utsname u;
    if (uname(&u)) return -1;
#ifdef __APPLE__
    if (strcmp(u.sysname,"Darwin") || (strcmp(u.machine,"arm64") && strcmp(u.machine,"x86_64"))) return -1;
    strcpy(h.os,"macos"); strcpy(h.arch,u.machine);
    if (mac_probe(&h)) return -1;
#else
    if (strcmp(u.sysname,"Linux") || strcmp(u.machine,"x86_64")) return -1;
    FILE *cpu=fopen("/proc/cpuinfo","r"), *mem=fopen("/proc/meminfo","r"); int rc=-1;
    if (cpu && mem) rc=hw_linux_read(cpu,mem,&h);
    if (cpu && fclose(cpu)) rc=-1;
    if (mem && fclose(mem)) rc=-1;
    if (rc || strlen(u.release)>=sizeof(h.version)) return -1;
    strcpy(h.version,u.release);
    FILE *frequency=fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq","r");
    if (frequency) {
        char line[64]; uint64_t khz;
        int ok=fgets(line,sizeof(line),frequency) && !number(trim(line),&khz) && khz<=UINT64_MAX/1000;
        if (fclose(frequency)) ok=0;
        if (!ok) return -1;
        h.frequency=khz*1000;
    } else if (errno!=ENOENT) return -1;
#endif
#ifdef __x86_64__
    unsigned a,b,c,d, leaf1=0, leaf7=0; uint64_t xcr0=0;
    if (__get_cpuid(1,&a,&b,&c,&d)) {
        leaf1=c;
        if ((c&((1u<<26)|(1u<<27)))==((1u<<26)|(1u<<27))) {
            unsigned low,high;
            __asm__ volatile("xgetbv" : "=a"(low),"=d"(high) : "c"(0));
            xcr0=(uint64_t)high<<32|low;
        }
    }
    if (__get_cpuid_count(7,0,&a,&b,&c,&d)) leaf7=b;
    h.isa=hw_x86_isa(leaf1,leaf7,xcr0);
#endif
    if (hw_finalize(&h)) return -1;
    *out=h; return 0;
}

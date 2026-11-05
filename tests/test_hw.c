/* test_hw.c - hardware policy, Linux input fixtures and live native probing.
 * Fixtures are synthetic unless their provenance explicitly says recorded.
 */
#include "test.h"
#include "../src/hw.h"
#include "../src/model.h"
#include "../src/config.h"
#include <string.h>
#include <inttypes.h>
static const uint64_t gib=UINT64_C(1073741824);
static int linux_text(const char *cpu, const char *mem, fr_hwinfo *h) {
    FILE *c=tmpfile(), *m=tmpfile(); int rc=-1;
    if (c && m && fputs(cpu,c)>=0 && fputs(mem,m)>=0 && !fseek(c,0,SEEK_SET) && !fseek(m,0,SEEK_SET)) rc=hw_linux_read(c,m,h);
    if (c) fclose(c);
    if (m) fclose(m);
    return rc;
}
int main(int argc, char **argv) {
    test_begin("hw");
    uint32_t leaf1=(1u<<26)|(1u<<27)|(1u<<28)|(1u<<12)|(1u<<29);
    check(hw_x86_isa(leaf1,1u<<5,6)==FR_ISA_USABLE,"all CPU and OS vector-state bits reported");
    fr_hwinfo h={.os="macos",.arch="arm64",.version="fixture",.cpu="Fixture Apple CPU",.gpu="Fixture GPU",.name="fixture",.id=1,
        .cores=10,.logical_cores=10,.metal=1,.total=16*gib,.available=12*gib,.metal_limit=10*gib,
        .margin=FR_DEFAULT_MARGIN,.head_reserve=UINT64_C(14000000000),.wired_mb=UINT64_MAX};
    check(!hw_finalize(&h) && h.margin==1000000000 && h.cap==12*gib-1000000000 && !strcmp(hw_backend(&h),"unmeasured"),"hardware reports CPU-available capacity before backend selection");
    check(h.blocks==h.cap/FR_WEIGHT_BLOCK_BYTES && !h.head_cap && !h.head_blocks,"whole blocks and head reserve saturate");
    h.available=1; check(!hw_finalize(&h) && !h.cap,"low memory saturates");
    strcpy(h.arch,"x86_64"); h.available=12*gib; h.margin=FR_DEFAULT_MARGIN; h.isa=FR_ISA_USABLE;
    check(!hw_finalize(&h) && h.margin==3000000000 && h.cap==12*gib-3000000000 && !strcmp(hw_backend(&h),"unmeasured"),"Intel Mac ignores Radeon working-set limit");
    unsigned bits[]={FR_ISA_AVX2,FR_ISA_FMA,FR_ISA_F16C,FR_ISA_AVX,FR_ISA_OSXSAVE,FR_ISA_XSAVE,FR_ISA_YMM}; int gated=1;
    for (unsigned i=0;i<sizeof(bits)/sizeof(bits[0]);++i) { h.isa=FR_ISA_USABLE&~bits[i]; if (strcmp(hw_backend(&h),"unmeasured")) gated=0; }
    check(gated,"CPU flags alone never select an unmeasured backend");
    h.isa=FR_ISA_USABLE; h.translated=1;
    check(!strcmp(hw_backend(&h),"none") && strstr(hw_reason(&h),"translated"),"Rosetta excluded even with every flag present");
    h.translated=0; h.available=h.total+1;
    check(hw_finalize(&h)<0,"available above total rejected");
    check(!(hw_x86_isa(leaf1,1u<<5,2)&FR_ISA_YMM) && (hw_x86_isa(leaf1,1u<<5,2)&FR_ISA_AVX2),"CPU flags retained when OS YMM state is unavailable");
    check(!(hw_x86_isa(0x0298220f,0,0)&FR_ISA_AVX2),"recorded Rosetta CPUID has no AVX2 flag");
    const char *cpu="processor : 0\nmodel name : Fixture Intel CPU\nphysical id : 0\ncore id : 0\n\nprocessor : 1\nmodel name : Fixture Intel CPU\nphysical id : 0\ncore id : 0\n\nprocessor : 2\nmodel name : Fixture Intel CPU\nphysical id : 0\ncore id : 1\n\n";
    const char *mem="MemTotal: 16777216 kB\nMemFree: 123 kB\nMemAvailable: 12582912 kB\n";
    check(!linux_text(cpu,mem,&h) && h.cores==2 && h.logical_cores==3 && h.total==16*gib && h.available==12*gib,"Linux proc fixture counts physical cores and SMT threads, converts KiB");
    check(linux_text(cpu,"MemTotal: 1024 kB\n",&h)<0,"missing MemAvailable fails instead of inventing capacity");
    check(linux_text(cpu,"MemTotal: 1024 MB\nMemAvailable: 1 kB\n",&h)<0,"unexpected memory unit rejected");
    check(linux_text(cpu,"MemTotal: 18446744073709551615 kB\nMemAvailable: 1 kB\n",&h)<0,"memory byte conversion overflow rejected");
    check(linux_text(cpu,"MemTotal: 1024 kB\nMemTotal: 1024 kB\nMemAvailable: 1 kB\n",&h)<0,"duplicate memory keys rejected");
    check(linux_text("processor : 0\nmodel name : Virtual CPU\n",mem,&h)==0 && h.cores==0 && h.logical_cores==1,"missing physical topology is explicitly unknown, never guessed");
    check(linux_text("processor : 0\n",mem,&h)<0,"missing CPU model rejected");
    check(linux_text("processor : 0\nmodel name : CPU\n\nprocessor : 0\nmodel name : CPU\n",mem,&h)<0,"duplicate processor ids rejected");
    check(!linux_text(cpu,mem,&h),"restore valid Linux snapshot");
    h.isa=FR_ISA_USABLE; h.id=1; strcpy(h.name,"fixture"); check(!hw_finalize(&h),"finalize Linux ISA selection");
    char text[FR_HW_TEXT], copy[FR_HW_TEXT+16]; int n=hw_format(&h,text,sizeof(text)); fr_hwinfo parsed={0};
    check(n>0 && !hw_parse(text,(size_t)n,&parsed) && hw_format(&parsed,copy,sizeof(copy))==n && !strcmp(text,copy),"HW text round trip preserves complete snapshot");
    check(hw_format(&h,copy,8)<0,"small output buffer fails");
    check(hw_parse(text,(size_t)n-1,&parsed)<0,"truncated HW text rejected");
    memcpy(copy,text,(size_t)n); copy[3]='\0';
    check(hw_parse(copy,(size_t)n,&parsed)<0,"embedded NUL rejected");
    snprintf(copy,sizeof(copy),"%sos:linux\r\n",text);
    check(hw_parse(copy,strlen(copy),&parsed)<0,"duplicate HW fields rejected");
    snprintf(copy,sizeof(copy),"%snew:1\r\n",text);
    check(hw_parse(copy,strlen(copy),&parsed)<0,"unknown HW fields rejected");
    char *cap=strstr(text,"mem_cap_bytes:");
    if (cap) cap[14]=cap[14]=='9'?'8':'9';
    check(cap && hw_parse(text,(size_t)n,&parsed)<0,"inconsistent advertised capacity rejected");
    h.cpu[0]='\n'; check(hw_format(&h,text,sizeof(text))<0,"hardware strings cannot inject text lines");
    memset(h.os,'x',sizeof(h.os)); check(hw_finalize(&h)<0,"unterminated platform field rejected");
    FILE *shuttle_cpu=fopen("tests/fixtures/shuttle/cpuinfo.txt","r");
    FILE *shuttle_mem=fopen("tests/fixtures/shuttle/meminfo.txt","r");
    check(shuttle_cpu && shuttle_mem && !hw_linux_read(shuttle_cpu,shuttle_mem,&parsed) &&
          parsed.cores==4 && parsed.logical_cores==8 && !strcmp(parsed.cpu,"Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz"),"actual Shuttle cpuinfo fixture matches lscpu topology and brand");
    check(shuttle_cpu && shuttle_mem && parsed.total==UINT64_C(16512028672) && parsed.available==UINT64_C(15838007296),"actual Shuttle meminfo fixture converts total and available bytes exactly");
    if (shuttle_cpu) fclose(shuttle_cpu);
    if (shuttle_mem) fclose(shuttle_mem);
    FILE *mac=fopen("tests/fixtures/imac/sysctl.txt","r"), *vm=fopen("tests/fixtures/imac/vm_stat.txt","r"), *ver=fopen("tests/fixtures/imac/sw_vers.txt","r");
    check(mac && vm && ver && !hw_macos_read(mac,vm,ver,&parsed) && parsed.cores==4 && parsed.logical_cores==4 && parsed.total==UINT64_C(51539607552) && parsed.frequency==3800000000 && !strcmp(parsed.version,"13.7.8"),"actual iMac sysctl and version parsing");
    check(parsed.available==(UINT64_C(4143925)+3117437+143853)*4096 && parsed.margin==UINT64_C(51539607552)/10,"actual iMac free + inactive + purgeable policy; 48 GB keeps a tenth as margin");
    if(mac) fclose(mac);
    if(vm) fclose(vm);
    if(ver) fclose(ver);
    check(hw_pages(&parsed,4096,UINT64_MAX,1,1)<0,"page count overflow rejected");
    fr_hwinfo unpinned={.total=UINT64_C(51539607552)};
    check(!hw_unpinned(&unpinned,16384,205000,512700,5000) && unpinned.available==UINT64_C(51539607552)-UINT64_C(722700)*16384,
          "node memory: total minus wired, anonymous and compressed pages");
    check(!hw_unpinned(&unpinned,16384,4000000,0,0) && unpinned.available==0,"pinned beyond total leaves nothing");
    check(hw_unpinned(&unpinned,0,1,1,1)<0 && hw_unpinned(&unpinned,4096,UINT64_MAX,1,0)<0,"unpinned rejects zero page size and overflow");
    const char *machines[]={"shuttle","imac"};
    for (unsigned i=0;i<2;++i) {
        char path[128]; snprintf(path,sizeof(path),"tests/fixtures/%s/cpuid.txt",machines[i]);
        FILE *f=fopen(path,"r"); uint32_t one=0,seven=0; uint64_t xcr0=0;
        int ok=f && fscanf(f,"leaf1_ecx=0x%" SCNx32 " leaf7_ebx=0x%" SCNx32 " xcr0=0x%" SCNx64,&one,&seven,&xcr0)==3;
        if(f) fclose(f);
        check(ok && hw_x86_isa(one,seven,xcr0)==FR_ISA_USABLE,"recorded native x86 CPUID and XCR0 permit AVX2 FMA F16C");
    }
    mac=fopen("tests/fixtures/m1pro/sysctl.txt","r"); vm=fopen("tests/fixtures/m1pro/vm_stat.txt","r"); ver=fopen("tests/fixtures/m1pro/sw_vers.txt","r");
    check(mac && vm && ver && !hw_macos_read(mac,vm,ver,&parsed) && parsed.cores==8 && parsed.logical_cores==8 &&
        !strcmp(parsed.arch,"arm64") && !strcmp(parsed.cpu,"Apple M1 Pro") && parsed.available==(UINT64_C(16644)+391503+7627)*16384,
        "actual M1 Pro fixture includes purgeable pages");
    if(mac) fclose(mac);
    if(vm) fclose(vm);
    if(ver) fclose(ver);
    parsed.total=3*FR_WEIGHT_BLOCK_BYTES; parsed.available=2*FR_WEIGHT_BLOCK_BYTES-1; parsed.margin=0; parsed.head_reserve=1;
    check(!hw_finalize(&parsed) && parsed.blocks==1 && parsed.head_blocks==1,"configurable reserves round ownership down at a block boundary");
    int rc=hw_probe(&h);
    check(!rc && h.logical_cores && h.total && h.available<=h.total && h.cap<=h.available,"native hardware probe returns bounded actual capacity");
    if (!rc) {
        h.id=1; strcpy(h.name,"native-test"); n=hw_format(&h,text,sizeof(text));
        check(n>0 && !hw_parse(text,(size_t)n,&parsed),"native report passes same wire validation as fixtures");
        if (n>0) fputs(text,stdout);
        if (argc==2) { FILE *f=fopen(argv[1],"w"); check(f && fwrite(text,1,(size_t)n,f)==(size_t)n && !fclose(f),"save explicitly requested native fixture"); }
    }
    return test_end();
}

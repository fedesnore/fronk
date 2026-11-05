/* Cache-resident kernel qualification, not an end-to-end throughput test.
 * A/B/A shares weights, quantized inputs, arithmetic and complete outputs. */
#define FR_AVX2_MTP_ROWS 1
#define FR_AVX2_UNION 1
#define FR_AVX2_UNION_PHASES 1
#include "../backends/avx2.c"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
static uint32_t rng=76123;
static uint32_t random_u32(void) {rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static unsigned char data[2*FR_NATIVE_GATE_BYTES] __attribute__((aligned(64)));
static q8k inputs;
static unsigned positions[8]={5,0,7,2,3,6,1,4};
static float result[2][32][8];
static volatile float consumed;
static void evaluate(unsigned T,int candidate,float out[2][32][8]) {
    for(unsigned n=0;n<32;n++) {
        const unsigned char *g=data+n*144*(FR_HIDDEN/256),*u=g+FR_NATIVE_GATE_BYTES;
        if(candidate) {
            void (*fn)(const unsigned char *,const q8k *,const unsigned *,float *)=T==3?q4k_row3:T==4?q4k_row4:q4k_row8;
            fn(g,&inputs,positions,out[0][n]);fn(u,&inputs,positions,out[1][n]);
        } else for(unsigned c=0;c<T;c+=2) {
            if(T-c>1) q4k_pair2(g,u,&inputs,positions+c,out[0][n]+c,out[1][n]+c);
            else q4k_pair1(g,u,&inputs,positions+c,out[0][n]+c,out[1][n]+c);
        }
    }
    consumed=out[1][31][T-1];
}
static void request_for(fr_req *r,unsigned T,unsigned pattern,unsigned layer) {
    memset(r,0,sizeof(*r));r->burst=T;r->layer=layer;
    for(unsigned t=0;t<T;t++) {
        for(unsigned d=0;d<FR_HIDDEN;d++) r->x[t][d]=(float)((int)((d*17+t*31+layer*7)%257)-128)/128;
        for(unsigned j=0;j<FR_ACTIVE_ROUTED;j++) {
            unsigned id=pattern==0?j:pattern==1?t*10+j:pattern==2?(j<5?j:10+t*5+j):((j+t)%10);
            r->expert[t][j]=(uint16_t)((layer*37+id)%FR_ROUTED_EXPERTS);
            r->gate[t][j]=(1+t+j)*.005f;
        }
    }
}
static int replay(const char *file,unsigned policy) {
    struct {char magic[8];uint32_t size,endian;} header;
    FILE *f=fopen(file,"rb");
    struct stat st;
    if(!policy || (policy>8 && policy!=99 && policy!=100) || !f || fstat(fileno(f),&st) || st.st_size<=(off_t)sizeof(header) ||
        ((uint64_t)st.st_size-sizeof(header))%sizeof(fr_req) ||
        fread(&header,sizeof(header),1,f)!=1 || memcmp(header.magic,"FRNCAP01",8) ||
        header.size!=sizeof(fr_req) || header.endian!=0x01020304) return 1;
    fr_req *rows=calloc(4096,sizeof(*rows));if(!rows) return 1;
    size_t n=fread(rows,sizeof(*rows),4096,f);
    int error=ferror(f) || !feof(f);fclose(f);if(error || !n) return 1;
    static fr_part ref,got;
    unsigned groups[9]={0},tokens=0;double raw_s=0,reduce_s=0;
    for(size_t i=0;i<n;i++) {
        state->union_profile=0;state->union_enabled=0;state->mtp_rows=0;if(fr_avx2.run(rows+i,&ref)) return 1;
        state->union_enabled=policy>=99;
        state->union_profile=policy>=99;
#ifdef __clang__
        const unsigned combined=8;
#else
        const unsigned combined=3;
#endif
        state->mtp_rows=policy==100?combined:policy==99?0:policy;
        if(fr_avx2.run(rows+i,&got) || memcmp(&ref,&got,sizeof(ref))) {
            fprintf(stderr,"FAIL replay row=%zu\n",i);return 1;
        }
        tokens+=rows[i].burst;
        if(state->union_profile) {raw_s+=state->union_compute_s;reduce_s+=state->union_reduce_s;}
        for(unsigned j=0;j<state->experts;j++) ++groups[__builtin_popcount(state->expert[j].mask)];
    }
    printf("REPLAY calls=%zu token_rows=%u minimum_shared_rows=%u groups_by_shared_rows=",n,tokens,policy);
    for(unsigned i=1;i<=8;i++) printf("%s%u:%u",i>1?",":"",i,groups[i]);
    puts("");
    if(policy>=99) printf("REPLAY diagnostic_only raw_compute_us=%.3f ordered_reduce_us=%.3f\n",raw_s*1e6/n,reduce_s*1e6/n);
    state->union_profile=0;
    for(unsigned repeat=0;repeat<5;repeat++) {
        double us[3];
        for(unsigned phase=0;phase<3;phase++) {
            state->union_enabled=phase==1 && policy>=99;
#ifdef __clang__
            const unsigned combined=8;
#else
            const unsigned combined=3;
#endif
            state->mtp_rows=phase!=1 || policy==99?0:policy==100?combined:policy;
            double begin=now();
            for(size_t i=0;i<n;i++) if(fr_avx2.run(rows+i,&got)) return 1;
            us[phase]=(now()-begin)*1e6/n;
        }
        printf("REPLAY repeat=%u before_us=%.3f candidate_us=%.3f after_us=%.3f\n",repeat,us[0],us[1],us[2]);
    }
    free(rows);puts("PASS all captured requests bit-identical");return 0;
}
static int backend(const char *dir,const char *list,unsigned workers,const char *trace,unsigned policy) {
    static fr_req requests[48];static fr_part ref,got;
    void *maps[FR_BLOCKS]={0};char ids[128];snprintf(ids,sizeof(ids),"%s",list);
    configured_workers=workers;
    for(char *s=strtok(ids,",");s;s=strtok(NULL,",")) {
        unsigned id=(unsigned)strtoul(s,NULL,10);if(id>=FR_BLOCKS || maps[id]) return 1;
        char path[1024];snprintf(path,sizeof(path),"%s/unit-%02u.bin",dir,id);
        int fd=open(path,O_RDONLY);struct stat st;
        if(fd<0 || fstat(fd,&st) || (uint64_t)st.st_size!=FR_NATIVE_BLOCK_BYTES) {perror(path);return 1;}
        maps[id]=mmap(NULL,FR_NATIVE_BLOCK_BYTES,PROT_READ,MAP_SHARED,fd,0);close(fd);
        if(maps[id]==MAP_FAILED || fr_avx2.load(id,maps[id],FR_NATIVE_BLOCK_BYTES)) return 1;
    }
    if(!state) return 1;
    printf("BACKEND workers=%u blocks=%u real_units=%s\n",state->threads,state->blocks,list);
    if(trace) return replay(trace,policy);
    const unsigned widths[]={1,2,3,4,6,8};
    for(unsigned k=0;k<6;k++) for(unsigned pattern=0;pattern<4;pattern++) {
        unsigned T=widths[k];
        for(unsigned i=0;i<48;i++) request_for(requests+i,T,pattern,i);
        for(unsigned i=0;i<48;i++) {
            state->mtp_rows=0;if(fr_avx2.run(requests+i,&ref)) return 1;
            state->mtp_rows=1;if(fr_avx2.run(requests+i,&got) || memcmp(&ref,&got,sizeof(ref))) {
                fprintf(stderr,"FAIL backend bit identity T=%u pattern=%u layer=%u\n",T,pattern,i);return 1;
            }
        }
        unsigned groups=state->experts,reused=0;
        for(unsigned i=0;i<groups;i++) reused+=__builtin_popcount(state->expert[i].mask)>=3;
        for(unsigned repeat=0;repeat<3;repeat++) {
            double us[3];
            for(unsigned phase=0;phase<3;phase++) {
                state->mtp_rows=phase==1;
                double begin=now();
                for(unsigned i=0;i<96;i++) if(fr_avx2.run(requests+i%48,&got)) return 1;
                us[phase]=(now()-begin)*1e6/96;
            }
            printf("BACKEND T=%u pattern=%u groups=%u eligible=%u repeat=%u before_us=%.3f candidate_us=%.3f after_us=%.3f\n",
                T,pattern,groups,reused,repeat,us[0],us[1],us[2]);
        }
    }
    for(unsigned b=0;b<FR_BLOCKS;b++) if(maps[b]) {fr_avx2.unload(b);munmap(maps[b],FR_NATIVE_BLOCK_BYTES);}
    fr_avx2_close();puts("PASS complete backend: exact outputs including unused rows, 48 layers, T=1/2/3/4/6/8, four routing patterns");
    return 0;
}
int main(int argc,char **argv) {
    fr_hwinfo hw; if(probe(&hw)) return 2;
    setbuf(stdout,NULL);
    if(argc==5 && !strcmp(argv[1],"--backend")) return backend(argv[2],argv[3],(unsigned)atoi(argv[4]),NULL,1);
    if(argc==7 && !strcmp(argv[1],"--replay")) return backend(argv[2],argv[3],(unsigned)atoi(argv[4]),argv[5],(unsigned)atoi(argv[6]));
    for(unsigned i=0;i<sizeof(data);i++) data[i]=(unsigned char)random_u32();
    for(unsigned i=0;i<sizeof(data);i+=144) {
        uint16_t d=0x1400+(random_u32()&0x3ff),m=0x1000+(random_u32()&0x3ff);
        memcpy(data+i,&d,2);memcpy(data+i+2,&m,2);
    }
    for(unsigned t=0;t<8;t++) {
        float row[FR_HIDDEN];
        for(unsigned i=0;i<FR_HIDDEN;i++) row[i]=(int)(random_u32()%2047)/1024.f-1.f;
        quantize8(row,FR_HIDDEN,256,inputs.q[t],inputs.d[t],inputs.s[t]);
    }
    const unsigned widths[]={3,4,8};
    for(unsigned k=0;k<3;k++) {
        const unsigned T=widths[k];float ref[2][32][8]={0};
        memset(result,0,sizeof(result));evaluate(T,0,ref);evaluate(T,1,result);
        if(memcmp(ref,result,sizeof(ref))) {fprintf(stderr,"FAIL bit identity T=%u\n",T);return 1;}
        printf("PASS row kernel T=%u, distinct inputs, permuted positions and untouched rows\n",T);
        for(unsigned repeat=0;repeat<5;repeat++) {
            double us[3];
            for(unsigned phase=0;phase<3;phase++) {
                for(unsigned warm=0;warm<20;warm++) evaluate(T,phase==1,result);
                double begin=now();
                for(unsigned i=0;i<400;i++) evaluate(T,phase==1,result);
                us[phase]=(now()-begin)*1e6/400;
            }
            printf("Q4K T=%u repeat=%u before_us=%.3f candidate_us=%.3f after_us=%.3f\n",T,repeat,us[0],us[1],us[2]);
        }
    }
    return consumed==123456.f;
}

/* Explicit real-GPU benchmark and long-session RSS validation driver. */
#include "../src/bench.h"
#include <stdlib.h>
#include <errno.h>
int main(int argc,char **argv) {
    if (argc!=5) { fputs("Usage: fronk-metal-bench BLOCKS SECONDS SAMPLES CSV\n",stderr); return 2; }
    char *end; errno=0; unsigned long blocks=strtoul(argv[1],&end,10);
    if (errno || *end || blocks<1 || blocks>FR_BLOCKS) return 2;
    double duration=strtod(argv[2],&end); if (*end || duration<0 || duration>86400) return 2;
    unsigned long samples=strtoul(argv[3],&end,10); if (*end || !samples || samples>FR_BENCH_SAMPLES) return 2;
    FILE *csv=fopen(argv[4],"wx"); if (!csv) { perror("CSV"); return 1; }
    fr_hwinfo hw;
    if (!hw_probe(&hw)) printf("hardware os=%s version=%s available=%llu node_blocks=%llu head_blocks=%llu metal_limit=%llu\n",hw.os,hw.version,
        (unsigned long long)hw.available,(unsigned long long)hw.blocks,
        (unsigned long long)hw.head_blocks,(unsigned long long)hw.metal_limit);
    fr_bench_report report;
    int rc=fr_bench_measure((unsigned)blocks,(unsigned)samples,4,duration,stdout,csv,NULL,&report);
    if (fclose(csv)) rc=FR_IO;
    if (!rc) {
        printf("bw_eff_gbs=%.9g t0_us=%.9g burst_slope=%.9g sustained_ratio=%.9g\n",report.cost.bw_eff/1e9,
               report.cost.t0_ns/1000,report.burst_slope,report.sustained_ratio);
        for (unsigned mode=0;mode<2;++mode) for (unsigned b=0;b<4;++b) {
            printf("%s burst=%u",mode?"reused":"distinct",b+1);
            for (unsigned p=0;p<FR_METAL_PHASES;++p) printf(" phase%u_p50_ns=%.0f phase%u_p99_ns=%.0f",p,report.percentile[mode][b][p][0],p,report.percentile[mode][b][p][1]);
            putchar('\n');
        }
    }
    return rc?1:0;
}

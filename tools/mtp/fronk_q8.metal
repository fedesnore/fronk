/* EXPERIMENT ONLY: M4 Pro measurements reject these kernels as performance
 * defaults. See docs/PLAN-mtp-native.md. FRONK_NATIVE_Q8_REUSE is opt-in;
 * the test harness retains this negative result for comparison.
 * Q8_0 projection for Fronk MTP. Keep the scalar decoder's K walk and
 * two-stage SIMD reduction, but load/decode each weight for all token rows.
 * All token accumulators reduce across the threadgroup in two barriers. */
template <short NT>
kernel void kernel_fronk_q8_rows(
        constant ds4_metal_args_mul_mv &args,
        device const char *weights,device const char *inputs,device char *outputs,
        threadgroup char *scratch [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR=2,NQ=8;
    const short NSG=FC_mul_mv_nsg;
    const int first=group.y*NT,row0=group.x*NR,blocks=args.ne00/32;
    const int ib0=sg*NQ+lane/4,k=lane%4*NQ;
    device const block_q8_0 *w[NR];
    FOR_UNROLL(short r=0;r<NR;r++) w[r]=(device const block_q8_0 *)(weights+(uint64_t)(row0+r)*args.nb01);
    float sum[NT][NR];
    FOR_UNROLL(short t=0;t<NT;t++) FOR_UNROLL(short r=0;r<NR;r++) sum[t][r]=0;
    for(int ib=ib0;ib<blocks;ib+=NSG*NQ) {
        FOR_UNROLL(short r=0;r<NR;r++) {
            if(row0+r<args.ne01) {
                float dot[NT];
                FOR_UNROLL(short t=0;t<NT;t++) dot[t]=0;
                FOR_UNROLL(short i=0;i<NQ;i++) {
                    const float q=w[r][ib].qs[k+i];
                    FOR_UNROLL(short t=0;t<NT;t++) {
                        device const float *v=(device const float *)(inputs+(uint64_t)(first+t)*args.nb11);
                        dot[t]+=q*(first+t<args.ne1?v[ib*32+k+i]:0);
                    }
                }
                const float scale=w[r][ib].d;
                FOR_UNROLL(short t=0;t<NT;t++) sum[t][r]+=dot[t]*scale;
            }
        }
    }
    threadgroup float *shared=(threadgroup float *)scratch;
    FOR_UNROLL(short t=0;t<NT;t++) FOR_UNROLL(short r=0;r<NR;r++) {
        if(sg==0) shared[(t*NR+r)*32+lane]=0;
        sum[t][r]=simd_sum(sum[t][r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) FOR_UNROLL(short t=0;t<NT;t++) FOR_UNROLL(short r=0;r<NR;r++) shared[(t*NR+r)*32+sg]=sum[t][r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL(short t=0;t<NT;t++) FOR_UNROLL(short r=0;r<NR;r++) {
        float value=simd_sum(shared[(t*NR+r)*32+lane]);
        if(lane==0 && sg==0 && first+t<args.ne1 && row0+r<args.ne01)
            ((device float *)outputs)[(uint64_t)(first+t)*args.ne0+row0+r]=value;
    }
}
#define FRONK_Q8_INSTANCE(N) \
template [[host_name("kernel_fronk_q8_rows" #N)]] \
kernel void kernel_fronk_q8_rows<N>(constant ds4_metal_args_mul_mv &,device const char *, \
    device const char *,device char *,threadgroup char *,uint3,ushort,ushort);
FRONK_Q8_INSTANCE(2)
FRONK_Q8_INSTANCE(4)
FRONK_Q8_INSTANCE(8)
#undef FRONK_Q8_INSTANCE

/* Alternative on-chip residency policy: keep the whole two-output weight
 * tile in threadgroup memory. Two token teams walk it at a time, preserving
 * scalar register pressure and retaining the tile for the remaining tokens. */
kernel void kernel_fronk_q8_shared(
        constant ds4_metal_args_mul_mv &args,
        device const char *weights,device const char *inputs,device char *outputs,
        threadgroup char *scratch [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort group_sg [[simdgroup_index_in_threadgroup]]) {
    const short NSG=FC_mul_mv_nsg;
    const uint tid=group_sg*32+lane,nth=NSG*64;
    const int row0=group.x*2,nb=args.ne00/32;
    const uint weight_bytes=(uint)(args.nb01*2),reduce_offset=(weight_bytes+15u)&~15u;
    const uint valid_bytes=(uint)(args.nb01*min(2,args.ne01-row0));
    for(uint i=tid;i<valid_bytes;i+=nth) scratch[i]=weights[(uint64_t)row0*args.nb01+i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const block_q8_0 *w[2]={
        (threadgroup const block_q8_0 *)scratch,
        (threadgroup const block_q8_0 *)(scratch+args.nb01)};
    threadgroup float *red=(threadgroup float *)(scratch+reduce_offset);
    const ushort team=group_sg/NSG,sg=group_sg%NSG;
    const int ib0=sg*8+lane/4,k=lane%4*8;
    for(int first=0;first<args.ne1;first+=2) {
        const int token=first+team;
        device const float *x=(device const float *)(inputs+(uint64_t)token*args.nb11);
        float sum[2]={0,0};
        if(token<args.ne1) for(int ib=ib0;ib<nb;ib+=NSG*8) {
            float xv[8];
            FOR_UNROLL(short i=0;i<8;i++) xv[i]=x[ib*32+k+i];
            FOR_UNROLL(short r=0;r<2;r++) if(row0+r<args.ne01) {
                float dot=0;
                FOR_UNROLL(short i=0;i<8;i++) dot+=w[r][ib].qs[k+i]*xv[i];
                sum[r]+=dot*w[r][ib].d;
            }
        }
        FOR_UNROLL(short r=0;r<2;r++) {
            if(sg==0) red[(team*2+r)*32+lane]=0;
            sum[r]=simd_sum(sum[r]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(lane==0) FOR_UNROLL(short r=0;r<2;r++) red[(team*2+r)*32+sg]=sum[r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL(short r=0;r<2;r++) {
            const float v=simd_sum(red[(team*2+r)*32+lane]);
            if(lane==0 && sg==0 && token<args.ne1 && row0+r<args.ne01)
                ((device float *)outputs)[(uint64_t)token*args.ne0+row0+r]=v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

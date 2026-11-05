/* Fronk causal-batch mixer. A SIMD group owns one output coordinate for
 * up to eight token rows. Each F16 weight is loaded once for those rows.
 * Low-rank values and sigmoid factors are staged separately: precomputing
 * silu would change the scalar kernel's multiply order under fast math. */
template <ushort N>
kernel void kernel_fronk_hc_mix_f16(
        constant ds4_metal_args_qwen4_hc_gate_mix &args,
        device const float *xn, device const float *lo,
        device const char *weights, device float *mixed,
        threadgroup float2 *prepared [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 threads [[threads_per_threadgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint E=args.n_embd, rank=args.n_rank, hc=args.n_hc;
    const uint first=group.y*N, d=group.x*(threads.x/32)+simd;
    const float scale=1.0f/(float)hc;
    for(uint i=tid;i<N*rank;i+=threads.x) {
        const uint token=first+i/rank, r=i%rank;
        const float x=token<args.n_tokens ? lo[(uint64_t)token*rank+r]*scale : 0.0f;
        prepared[i]=float2(x,qwen4_sigmoid(x));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(d>=E) return;
    const uint stream=lane/8, k=lane%8;
    device const half *w=(device const half *)weights+(uint64_t)(stream*E+d)*rank;
    float acc[N];
    for(ushort t=0;t<N;++t) acc[t]=0.0f;
    for(uint r=k;r<rank;r+=8) {
        const float weight=(float)w[r];
        for(ushort t=0;t<N;++t) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
            const float2 a=prepared[t*rank+r];
            const float product=a.x*weight;
            const float term=product*a.y;
            acc[t]=acc[t]+term;
        }
    }
    for(ushort t=0;t<N;++t) {
        if(first+t>=args.n_tokens) break;
        float a=acc[t];
        a+=simd_shuffle_xor(a,1);
        a+=simd_shuffle_xor(a,2);
        a+=simd_shuffle_xor(a,4);
        float v=qwen4_sigmoid(a)*xn[(uint64_t)(first+t)*E*hc+stream*E+d];
        v+=simd_shuffle_xor(v,8);
        v+=simd_shuffle_xor(v,16);
        if(lane==0) mixed[(uint64_t)(first+t)*E+d]=v/(float)hc;
    }
}
#define FRONK_HC_INSTANCE(N) \
template [[host_name("kernel_fronk_hc_mix_f16_" #N)]] \
kernel void kernel_fronk_hc_mix_f16<N>(constant ds4_metal_args_qwen4_hc_gate_mix &, \
    device const float *,device const float *,device const char *,device float *, \
    threadgroup float2 *,uint3,ushort,ushort3,ushort,ushort);
FRONK_HC_INSTANCE(1)
FRONK_HC_INSTANCE(2)
FRONK_HC_INSTANCE(4)
FRONK_HC_INSTANCE(8)
#undef FRONK_HC_INSTANCE

struct fronk_history_args { uint channels,history,first,count; };
kernel void kernel_fronk_history_prefix(
        constant fronk_history_args &a,device float *history,
        device const float *rows,uint c [[thread_position_in_grid]]) {
    if(c>=a.channels) return;
    float h[9];
    for(uint i=0;i<a.history;i++) h[i]=history[i*a.channels+c];
    for(uint t=0;t<a.count;t++) {
        for(uint i=0;i+1<a.history;i++) h[i]=h[i+1];
        h[a.history-1]=rows[(a.first+t)*a.channels+c];
    }
    for(uint i=0;i<a.history;i++) history[i*a.channels+c]=h[i];
}

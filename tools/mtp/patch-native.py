"""Apply the native-chain prototype only inside an explicitly staged engine."""
import pathlib,sys
root=pathlib.Path(sys.argv[1])
p=root/'ds4.c';s=p.read_text()
def replace(old,new):
 global s
 if s.count(old)!=1:raise RuntimeError('Unexpected upstream native patch anchor: '+old[:70])
 s=s.replace(old,new)
replace('static bool qwen4_gemv_rows(ds4_gpu_tensor *out,', 'static bool fronk_native_rows, fronk_native_matrix8;\nstatic bool qwen4_gemv_rows(ds4_gpu_tensor *out,')
replace('const bool legacy = getenv("DS4_QWEN4_DENSE_MM_LEGACY") != NULL;', 'const bool legacy = fronk_native_rows || getenv("DS4_QWEN4_DENSE_MM_LEGACY") != NULL;')
replace('    const bool legacy = fronk_native_rows || getenv("DS4_QWEN4_DENSE_MM_LEGACY") != NULL;', '''    /* Fixed eight-column FP32 matrix policy, including single-token oracle
     * execution. Spare rows are allocated graph scratch, computed and ignored,
     * as in upstream's padded small-batch path. No copy, allocation or wait.
     * This is opt-in: its reduction order differs from the scalar target. */
    if (fronk_native_rows && fronk_native_matrix8 && n_tok>=1u && n_tok<=8u &&
        w->type==DS4_TENSOR_Q8_0 && in_dim && out_dim &&
        in_dim<=UINT32_MAX && out_dim<=UINT32_MAX &&
        in_dim%32u==0 && out_dim%32u==0 &&
        ds4_gpu_tensor_bytes(x)>=8u*in_dim*sizeof(float) &&
        ds4_gpu_tensor_bytes(out)>=8u*out_dim*sizeof(float))
        return ds4_gpu_qwen4_batch_mm_q8_tensor(out,x,m->map,m->size,w->abs_offset,
            8u,(uint32_t)in_dim,(uint32_t)out_dim)!=0;
    const bool legacy = fronk_native_rows || getenv("DS4_QWEN4_DENSE_MM_LEGACY") != NULL;''')
replace('struct fronk_tree;','struct fronk_native;\nstatic void fronk_native_free(struct fronk_native *n);\nstruct fronk_tree;')
replace('    struct fronk_tree *fronk_tree;','    struct fronk_tree *fronk_tree;\n    struct fronk_native *fronk_native;')
replace('    if (g && g->fronk_tree) { fronk_tree_free(g->fronk_tree); g->fronk_tree = NULL; }','    if (g && g->fronk_native) { fronk_native_free(g->fronk_native); g->fronk_native = NULL; }\n    if (g && g->fronk_tree) { fronk_tree_free(g->fronk_tree); g->fronk_tree = NULL; }')
replace('#include "fronk_mtp_worker.inc"','static void (*fronk_native_observe)(unsigned,unsigned,const float *,const int32_t *,const float *,const float *);\n#include "fronk_mtp_worker.inc"')
replace('#include "fronk_mtp_tree.inc"','#include "fronk_mtp_tree.inc"\n#include "fronk_mtp_native.inc"')
replace('static void fronk_tree_free(struct fronk_tree *tree) { (void)tree; }','static void fronk_tree_free(struct fronk_tree *tree) { (void)tree; }\nstatic void fronk_native_free(struct fronk_native *n) { (void)n; }')
replace('        int n = fronk_tree_cycle(s, first_token,', '        int native=fronk_native_cycle(s,first_token,temperature,accepted,accepted_cap,err,errlen);\n        if (native) return native;\n        int n = fronk_tree_cycle(s, first_token,')
replace('return T == 3u && g->verify_rows_exact;', 'return T <= 8u && g->verify_rows_exact;')
# Generalize exact HC gate/mix and causal attention sub-batches; changing only
# the GEMV selector would still use differently rounded multi-row gates.
a=s.index('    if (T == 3u && g->verify_rows_exact) {',s.index('static bool qwen4_graph_hc_mix'))
b=s.index('    return ok && ds4_gpu_qwen4_hc_gate_mix_tensor(g->mixed',a)
s=s[:a]+'''    if (fronk_native_rows && up->type==DS4_TENSOR_F16)
        return ok && ds4_gpu_qwen4_hc_gate_mix_tensor(g->mixed,g->xn,g->lo,m->map,m->size,up->abs_offset,up->type,T,DS4_N_EMBD,DS4_N_HC,DS4_N_HC_LOWRANK);
    if (T > 2u && T <= 8u && g->verify_rows_exact) {
        const uint64_t dim=(uint64_t)DS4_N_EMBD*DS4_N_HC;
        for (uint32_t base=0;ok && base<T;base+=2) {
            uint32_t count=T-base>2?2:T-base;
            ds4_gpu_tensor *xn=ds4_gpu_tensor_view(g->xn,base*dim*sizeof(float),count*dim*sizeof(float));
            ds4_gpu_tensor *lo=ds4_gpu_tensor_view(g->lo,(uint64_t)base*DS4_N_HC_LOWRANK*sizeof(float),(uint64_t)count*DS4_N_HC_LOWRANK*sizeof(float));
            ds4_gpu_tensor *mixed=ds4_gpu_tensor_view(g->mixed,(uint64_t)base*DS4_N_EMBD*sizeof(float),(uint64_t)count*DS4_N_EMBD*sizeof(float));
            ok=xn && lo && mixed && ds4_gpu_qwen4_hc_gate_mix_tensor(mixed,xn,lo,m->map,m->size,up->abs_offset,up->type,count,DS4_N_EMBD,DS4_N_HC,DS4_N_HC_LOWRANK);
            ds4_gpu_tensor_free(mixed);ds4_gpu_tensor_free(lo);ds4_gpu_tensor_free(xn);
        }
        return ok;
    }
'''+s[b:]
replace('if (T == 3u && g->verify_rows_exact) {','if (T > 2u && T <= 8u && g->verify_rows_exact) {')
# Attention's split-key reduction also depends on the batch's last position.
# Scalar row views preserve each causal row's own split boundaries; all QKV
# projections stay batched and all dispatches share the command submission.
replace('if (T > 2u && T <= 8u && g->verify_rows_exact) {\n        /* 3-row speculative verify:', 'if ((T > 2u || (fronk_native_rows && T > 1u)) && T <= 8u && g->verify_rows_exact) {\n        /* 3-row speculative verify:')
replace('for (uint32_t sub0 = 0; sub0 < T; sub0 += 2u) {\n            const uint32_t subT = T - sub0 > 2u ? 2u : T - sub0;', 'const uint32_t stride=fronk_native_rows?1u:2u;\n        for (uint32_t sub0 = 0; sub0 < T; sub0 += stride) {\n            const uint32_t subT = T - sub0 > stride ? stride : T - sub0;')

replace('static bool qwen4_graph_linear(ds4_qwen4_gpu_graph *g,', 'static bool (*fronk_native_tape_capture)(ds4_qwen4_gpu_graph *,unsigned,unsigned,unsigned);\nstatic bool qwen4_graph_linear(ds4_qwen4_gpu_graph *g,')
a=s.index('static bool qwen4_graph_linear(ds4_qwen4_gpu_graph *g,');b=s.index('/* T tokens at positions',a)
part=s[a:b].replace('    bool ok;','    bool ok;\n    if (fronk_native_tape_capture && !fronk_native_tape_capture(g,il,0,T)) return false;')
part=part.replace('    if (ok) {\n        ok = ds4_gpu_qwen4_gdn_scan_tensor', '    if (ok && fronk_native_tape_capture) ok=fronk_native_tape_capture(g,il,1,T);\n    if (ok) {\n        ok = ds4_gpu_qwen4_gdn_scan_tensor')
s=s[:a]+part+s[b:]
replace('        ok = fronk_ple_layer(g, m, l, il, T, hc_dim);', '        ok = fronk_ple_layer(g, m, l, il, T, hc_dim);\n        if (ok && fronk_native_tape_capture && ds4_qwen4_layer_is_ple(il)) ok=fronk_native_tape_capture(g,il,2,T);')
replace('static bool qwen4_graph_hc_mix(ds4_qwen4_gpu_graph *g, const ds4_model *m,', 'static bool (*fronk_native_dense_observe)(unsigned,ds4_gpu_tensor *,unsigned,unsigned);\nstatic bool qwen4_graph_hc_mix(ds4_qwen4_gpu_graph *g, const ds4_model *m,')
for marker,stage,tensor,dim in [('        QWEN4_PROF(0);',0,'R','hc_dim'),('        QWEN4_PROF(1);',1,'mixed','DS4_N_EMBD'),('        QWEN4_PROF(ds4_qwen4_layer_is_linear(il) ? 2 : 3);',2,'blk','DS4_N_EMBD'),('        QWEN4_PROF(4);',3,'mixed','DS4_N_EMBD')]:
 replace(marker,marker+'\n        if (ok && il==0 && fronk_native_dense_observe) ok=fronk_native_dense_observe('+str(stage)+',g->'+tensor+',T,'+dim+');')
p.write_text(s)
p=root/'ds4_metal.m' ;s=p.read_text()
for old,new in [('(n_tok == 3 && g_qwen4_verify_rows_exact)','(n_tok <= 8 && g_qwen4_verify_rows_exact)'),('(n_tokens == 3u && g_qwen4_verify_rows_exact)','(n_tokens <= 8u && g_qwen4_verify_rows_exact)')]:
 if old not in s:raise RuntimeError('Missing exact-row kernel selector')
 s=s.replace(old,new)
# Native verification keeps each row's single-token reduction order. The
# existing few-row kernels change it even when the launch geometry matches.
replace('static bool g_qwen4_verify_rows_exact;', 'static bool g_qwen4_verify_rows_exact;\nstatic bool g_fronk_native_rows;\nvoid ds4_gpu_fronk_native_rows(bool on) { g_fronk_native_rows=on; }')
replace('return ds4_gpu_matmul_f16_tensor_impl(out, model_map, model_size,\n        weight_offset, in_dim, out_dim, x, n_tok, false);', 'return ds4_gpu_matmul_f16_tensor_impl(out, model_map, model_size,\n        weight_offset, in_dim, out_dim, x, n_tok, g_fronk_native_rows);')
replace('return ds4_gpu_matmul_q8_0_tensor_impl(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok,\n        ds4_gpu_device_name_contains("M3 Ultra"));', 'if (g_fronk_native_rows && n_tok>1 && n_tok<=8) return ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out,model_map,model_size,weight_offset,in_dim,out_dim,x,n_tok);\n    return ds4_gpu_matmul_q8_0_tensor_impl(out, model_map, model_size, weight_offset, in_dim, out_dim, x, n_tok,\n        ds4_gpu_device_name_contains("M3 Ultra"));')
replace('    QWEN4_K_HC_GATE_MIX_PAIR_Q8,', '    QWEN4_K_HC_GATE_MIX_PAIR_Q8,\n    FRONK_K_HC_2, FRONK_K_HC_4, FRONK_K_HC_8, FRONK_K_HISTORY,')
replace('    "kernel_qwen4_hc_gate_mix_pair_q8",', '    "kernel_qwen4_hc_gate_mix_pair_q8",\n    "kernel_fronk_hc_mix_f16_2", "kernel_fronk_hc_mix_f16_4", "kernel_fronk_hc_mix_f16_8", "kernel_fronk_history_prefix",')
replace('    const bool pair = n_tokens == 2u && getenv("DS4_QWEN4_NO_HC_PAIR") == NULL;', '''    if (g_fronk_native_rows && weight_type==1u && n_tokens>1u && n_tokens<=8u) {
        const uint32_t tile=n_tokens<=2u?2u:n_tokens<=4u?4u:8u;
        return qwen4_dispatch(tile==2u?FRONK_K_HC_2:tile==4u?FRONK_K_HC_4:FRONK_K_HC_8,&args,sizeof(args),b,4,
            MTLSizeMake((n_embd+3u)/4u,(n_tokens+tile-1u)/tile,1),MTLSizeMake(128,1,1),
            (NSUInteger)tile*n_rank*2u*sizeof(float));
    }
    const bool pair = n_tokens == 2u && !g_fronk_native_rows && getenv("DS4_QWEN4_NO_HC_PAIR") == NULL;''')
# The F32 scalar kernel already indexes the token dimension. Dispatch it for
# all rows in one encoder instead of selecting the differently rounded ext.
a=s.index('int ds4_gpu_matmul_f32_tensor(');b=s.index('        const bool router_batch',a)
part=s[a:b].replace('if (n_tok == 1) {','if (n_tok == 1 || g_fronk_native_rows) {').replace('ds4_gpu_make_f32_mv_args(in_dim, out_dim, 1);','ds4_gpu_make_f32_mv_args(in_dim, out_dim, n_tok);')
part=part.replace('                                                  1,\n                                                  1)', '                                                  (NSUInteger)n_tok,\n                                                  1)')
s=s[:a]+part+s[b:]
# Keep this specialization local to the native engine's exact Q8 path.
a=s.index('int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(');b=s.index('int ds4_gpu_matmul_q8_0_decode_mpp_tensor(',a)
part=s[a:b]
part=part.replace('        if (out_dim > 65536u) dispatch.nsg = 8;', '        if (out_dim > 65536u) dispatch.nsg = 8;\n        const bool fronk_batch=g_fronk_native_rows && n_rows>1u && n_rows<=8u && row_bytes*2u<=24576u && getenv("FRONK_NATIVE_Q8_REUSE")!=NULL;\n        if (fronk_batch) {dispatch.function_name="kernel_fronk_q8_shared";dispatch.smem=((row_bytes*2u+15u)&~15u)+512u;}')
part=part.replace('(NSUInteger)n_rows,\n                            1)', '(NSUInteger)(fronk_batch?1u:n_rows),\n                            1)')
part=part.replace('MTLSizeMake(32, (NSUInteger)dispatch.nsg, 1)', 'MTLSizeMake(32, (NSUInteger)dispatch.nsg*(fronk_batch?2u:1u), 1)')
s=s[:a]+part+s[b:]

s += """
int ds4_gpu_fronk_history_prefix(ds4_gpu_tensor *history,const ds4_gpu_tensor *rows,
        uint32_t channels,uint32_t length,uint32_t first,uint32_t count) {
    struct {uint32_t channels,history,first,count;} args={channels,length,first,count};
    qwen4_bind b[2];
    if (!channels || !length || length>9 || first>8 || count>8-first ||
        !qwen4_bind_tensor(&b[0],history,(uint64_t)length*channels*4,"Fronk history") ||
        !qwen4_bind_tensor(&b[1],rows,(uint64_t)(first+count)*channels*4,"Fronk history rows")) return 0;
    return qwen4_dispatch(FRONK_K_HISTORY,&args,sizeof(args),b,2,
        MTLSizeMake((channels+255u)/256u,1,1),MTLSizeMake(256,1,1),0);
}
"""
p.write_text(s)
p=root/'Makefile';p.write_text(p.read_text()+'\nds4.o: fronk_mtp_native.inc fronk_mtp_wave.inc\n')

p=root/'fronk_mtp_worker.inc';s=p.read_text()
anchor='        const double t2 = prof ? qwen4_fronk_now() : 0;'
if s.count(anchor)!=1:raise RuntimeError('Missing observer anchor')
s=s.replace(anchor,'        if (ok && fronk_native_observe) fronk_native_observe(jobs[0].layer,rows,x,ids,w,y);\n'+anchor)
p.write_text(s)

p=root/"ds4_gpu.h";s=p.read_text();s += "\nvoid ds4_gpu_fronk_native_rows(bool on);\n";p.write_text(s)

p=root/"metal/qwen4.metal"
if "FRONK_HC_INSTANCE(1)" not in p.read_text():
 p.write_text(p.read_text()+"\n"+(pathlib.Path(__file__).parent/"fronk_dense.metal").read_text())

p=root/"metal/dense.metal";p.write_text(p.read_text()+"\n"+(pathlib.Path(__file__).parent/"fronk_q8.metal").read_text())

p=root/"ds4_gpu.h";p.write_text(p.read_text()+"\nint ds4_gpu_fronk_history_prefix(ds4_gpu_tensor *,const ds4_gpu_tensor *,uint32_t,uint32_t,uint32_t,uint32_t);\n")

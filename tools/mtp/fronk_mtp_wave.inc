/* Two groups of the SAME causal chain, not independent branches. Each group
 * owns transient activations/mailboxes. Persistent layer state is borrowed
 * from the main graph and advanced in A,B order on one Metal command queue.
 * Never call qwen4_graph_free on this shallow graph: only scratch is owned. */
static void fronk_wave_free(ds4_qwen4_gpu_graph *b) {
    if(!b) return;
#define FRONK_WAVE_FREE(field) ds4_gpu_tensor_free(b->field);
    DS4_QWEN4_SCRATCH_FIELDS(FRONK_WAVE_FREE)
#undef FRONK_WAVE_FREE
    free(b->host_row); free(b->host_pos3); free(b);
}
static bool fronk_wave_reserve(struct fronk_native *p,const ds4_qwen4_gpu_graph *g) {
    if(p->wave) return true;
    ds4_qwen4_gpu_graph *b=calloc(1,sizeof(*b));
    if(!b) return false;
    bool ok=true; uint64_t bytes=0;
#define FRONK_WAVE_ALLOC(field) do { \
    uint64_t n=ds4_gpu_tensor_bytes(g->field); bytes+=n; \
    b->field=ds4_gpu_tensor_alloc(n); ok=ok && b->field && ds4_gpu_tensor_fill_f32(b->field,0,n/sizeof(float)); \
} while(0);
    DS4_QWEN4_SCRATCH_FIELDS(FRONK_WAVE_ALLOC)
#undef FRONK_WAVE_ALLOC
    b->host_row=malloc((size_t)g->cap_tokens*DS4_N_HC*DS4_N_EMBD*sizeof(float));
    b->host_pos3=malloc((size_t)g->cap_tokens*4*sizeof(int32_t));
    if(!ok || !b->host_row || !b->host_pos3) {fronk_wave_free(b);return false;}
    p->wave=b;
    fprintf(stderr,"FRONK_WAVE workspace_bytes=%llu groups=2 state_copies=0\n",(unsigned long long)bytes);
    return true;
}
static bool fronk_wave_layer(ds4_qwen4_gpu_graph *g,const ds4_model *m,
                             const ds4_weights *w,unsigned l,unsigned T) {
    const ds4_layer_weights *layer=&w->layer[l];
    bool ok=fronk_ple_layer(g,m,layer,l,T,DS4_N_HC*DS4_N_EMBD);
    if(ok && fronk_native_tape_capture && ds4_qwen4_layer_is_ple(l))
        ok=fronk_native_tape_capture(g,l,2,T);
    return ok &&
        qwen4_graph_hc_mix(g,m,layer->hc_attn_norm,layer->hc_attn_down,layer->hc_attn_up,layer->hc_attn_inject,T) &&
        (ds4_qwen4_layer_is_linear(l)?qwen4_graph_linear(g,m,layer,l,T):qwen4_graph_attention(g,m,layer,l,g->pos,T)) &&
        fronk_ffn_mix(g,m,layer,T) && qwen4_graph_moe(g,m,layer,T);
}
static bool fronk_wave_forward(ds4_qwen4_gpu_graph *g,const ds4_model *m,const ds4_weights *w,
                              const int *tokens,unsigned T,float *rows) {
    struct fronk_native *p=g->fronk_native;
    if(T<4 || T>8 || !p || !fronk_wave_reserve(p,g)) return false;
    const unsigned first=T/2,tail=T-first,position=g->pos;
    const unsigned layers=DS4_N_LAYER-DS4_N_NEXTN_PREDICT;
    const uint64_t rb=(uint64_t)DS4_N_HC*DS4_N_EMBD*sizeof(float);
    if(T>g->cap_tokens || T>g->n_logit_rows || position>g->ctx_cap || T>g->ctx_cap-position) return false;
    for(unsigned i=0;i<T;i++) if(tokens[i]<0 || tokens[i]>=(int)DS4_N_VOCAB) return false;
    ds4_qwen4_gpu_graph *b=p->wave;
    /* Rebind borrowed state every call: prefix rollback swaps the main
     * graph's state pointers. Keep only the privately allocated scratch. */
    ds4_qwen4_gpu_graph scratch=*b;
    *b=*g;
#define FRONK_WAVE_KEEP(field) b->field=scratch.field;
    DS4_QWEN4_SCRATCH_FIELDS(FRONK_WAVE_KEEP)
#undef FRONK_WAVE_KEEP
    b->host_row=scratch.host_row; b->host_pos3=scratch.host_pos3;
    b->snap_after_first=b->snap_after_second=false;
    b->fronk_resume_value=g->fronk_resume_value=0;
    b->fronk_defer_reduce=g->fronk_defer_reduce=true;
    b->verify_rows_exact=g->verify_rows_exact=true;
    p->wave_base=first;
    bool ok=qwen4_graph_stage_inputs(g,m,w,tokens,first);
    b->pos=position+first; b->mrope_delta=g->mrope_delta;
    memcpy(b->ple_prev,g->ple_prev,sizeof(b->ple_prev));
    if(ok) ok=qwen4_graph_stage_inputs(b,m,w,tokens+first,tail);
    fronk_native_rows=true; ds4_gpu_fronk_native_rows(true);
    ds4_gpu_qwen4_set_verify_rows_exact(true);
    if(ok) ok=glm_graph_begin_commands_if_needed();
    for(unsigned l=0;ok && l<layers;l++) {
        if(l) ok=fronk_finish_experts(g,first,g->fronk_resume_value,g->fronk_shared_local);
        if(ok) ok=fronk_wave_layer(g,m,w,l,first);
        /* The tail stage runs before waiting for this layer's first group.
         * It depends on the preceding layer's tail reply, already queued
         * before this layer's first-group request on the worker timeline. */
        if(ok && l) ok=fronk_finish_experts(b,tail,b->fronk_resume_value,b->fronk_shared_local);
        if(ok) ok=fronk_wave_layer(b,m,w,l,tail);
        /* Submit the first pair promptly, without CPU/GPU completion wait. */
        if(ok && l==0) ok=ds4_gpu_flush_commands()!=0;
    }
    if(ok) ok=fronk_finish_experts(g,first,g->fronk_resume_value,g->fronk_shared_local) &&
              fronk_finish_experts(b,tail,b->fronk_resume_value,b->fronk_shared_local) &&
              ds4_gpu_tensor_copy(g->R,first*rb,b->R,0,tail*rb) &&
              qwen4_graph_hc_mix(g,m,w->output_hc_norm,w->output_hc_down,w->output_hc_up,NULL,T) &&
              qwen4_gemv(g->logits,m,w->output,g->mixed,T);
    /* Drain even a failed partial submission before either mailbox can be
     * reused or freed. Worker signals completion on its error path too. */
    uint64_t last=g->fronk_resume_value>b->fronk_resume_value?g->fronk_resume_value:b->fronk_resume_value;
    if(last && !ds4_gpu_fronk_resume(last)) ok=false;
    if(!ds4_gpu_end_commands() || __atomic_load_n(&qwen4_fronk.error,__ATOMIC_ACQUIRE)) ok=false;
    if(ok) ok=ds4_gpu_tensor_read(g->logits,0,rows,(uint64_t)T*DS4_N_VOCAB*sizeof(float))!=0;
    if(ok) {g->pos=position+T;g->mrope_delta=b->mrope_delta;memcpy(g->ple_prev,b->ple_prev,sizeof(g->ple_prev));}
    if(g->snap_after_first) g->snap_valid=ok;
    if(g->snap_after_second) g->snap2_valid=ok;
    g->snap_after_first=g->snap_after_second=false;
    g->fronk_defer_reduce=b->fronk_defer_reduce=false;
    g->verify_rows_exact=b->verify_rows_exact=false;
    ds4_gpu_qwen4_set_verify_rows_exact(false);
    fronk_native_rows=false;ds4_gpu_fronk_native_rows(false);
    return ok;
}

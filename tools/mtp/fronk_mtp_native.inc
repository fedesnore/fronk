/* A single causal verifier graph, one distributed burst per layer. Greedy
 * proposal chains are speculative; only the target-matching prefix commits.
 * Snapshots after rows 0/1 give constant-time rollback for early rejection.
 * Later rejection restores recurrent state from saved activations, without
 * re-reading weights or invoking distributed experts. */
#define FRONK_NATIVE_MAX 8
struct fronk_native_layer_tape { ds4_gpu_tensor *raw,*qkv,*a,*b; };
struct fronk_native {
    unsigned depth, pending, position;
    int token[FRONK_NATIVE_MAX];
    uint64_t prefix, cycles, verified, accepted, replayed;
    float *rows, *check;
    struct fronk_native_layer_tape *tape;
    ds4_gpu_tensor *ple_rows;
    unsigned tape_layers;
    ds4_qwen4_gpu_graph *oracle,*wave;
    unsigned wave_base;
    unsigned char *state_a,*state_b; size_t state_cap;
    bool valid;
};
static void fronk_wave_free(ds4_qwen4_gpu_graph *b);
static void fronk_native_free(struct fronk_native *n) {
    if (!n) return;
    if (n->oracle) { qwen4_graph_free(n->oracle); free(n->oracle); }
    fronk_wave_free(n->wave);
    free(n->state_a);free(n->state_b);
    for(unsigned l=0;l<n->tape_layers;l++) {
        ds4_gpu_tensor_free(n->tape[l].raw);ds4_gpu_tensor_free(n->tape[l].qkv);
        ds4_gpu_tensor_free(n->tape[l].a);ds4_gpu_tensor_free(n->tape[l].b);
    }
    free(n->tape);ds4_gpu_tensor_free(n->ple_rows);
    free(n->rows); free(n->check); free(n);
}
static unsigned fronk_native_depth(void) {
    const char *v=getenv("FRONK_NATIVE_MTP");
    return v && v[0]>='3' && v[0]<='8' && !v[1] ? (unsigned)(v[0]-'0') : 0;
}
static bool fronk_native_forward(ds4_qwen4_gpu_graph *g,const ds4_model *m,const ds4_weights *w,
                                  const int *tokens,unsigned count,float *rows,bool all) {
    fronk_native_rows=true;ds4_gpu_fronk_native_rows(true);
    g->verify_rows_exact=true; ds4_gpu_qwen4_set_verify_rows_exact(true);
    bool ok=qwen4_graph_forward_tokens(g,m,w,tokens,count,rows,all);
    ds4_gpu_qwen4_set_verify_rows_exact(false); g->verify_rows_exact=false;
    fronk_native_rows=false;ds4_gpu_fronk_native_rows(false);
    return ok;
}
static bool fronk_native_equal(struct fronk_native *p,ds4_gpu_tensor *a,ds4_gpu_tensor *b,uint64_t offset,uint64_t bytes) {
    if (bytes>p->state_cap) {
        unsigned char *x=malloc(bytes),*y=malloc(bytes);
        if (!x || !y) { free(x);free(y);return false; }
        free(p->state_a);free(p->state_b);p->state_a=x;p->state_b=y;p->state_cap=bytes;
    }
    return ds4_gpu_tensor_read(a,offset,p->state_a,bytes) && ds4_gpu_tensor_read(b,offset,p->state_b,bytes) && !memcmp(p->state_a,p->state_b,bytes);
}
static bool fronk_native_tape_reserve(struct fronk_native *p) {
    if(!p->tape) {
        p->tape=calloc(DS4_N_LAYER,sizeof(*p->tape));
        if(!p->tape) return false;
        p->tape_layers=DS4_N_LAYER;
    }
    const uint64_t qbytes=8ull*DS4_N_LIN_CONV_DIM*sizeof(float),gbytes=8ull*DS4_N_LIN_V_HEAD*sizeof(float);
    for(unsigned l=0;l<p->tape_layers;l++) if(ds4_qwen4_layer_is_linear(l)) {
        struct fronk_native_layer_tape *t=&p->tape[l];
        if(!t->raw) t->raw=ds4_gpu_tensor_alloc(qbytes);
        if(!t->qkv) t->qkv=ds4_gpu_tensor_alloc(qbytes);
        if(!t->a) t->a=ds4_gpu_tensor_alloc(gbytes);
        if(!t->b) t->b=ds4_gpu_tensor_alloc(gbytes);
        if(!t->raw || !t->qkv || !t->a || !t->b) return false;
    }
    if(!p->ple_rows) p->ple_rows=ds4_gpu_tensor_alloc(8ull*DS4_N_HC*DS4_N_EMBD*sizeof(float));
    return p->ple_rows!=NULL;
}
static bool fronk_native_capture(ds4_qwen4_gpu_graph *g,unsigned layer,unsigned stage,unsigned T) {
    struct fronk_native *p=g->fronk_native;
    if(!p || !p->tape || layer>=p->tape_layers || T>8) return false;
    const uint64_t qb=(uint64_t)T*DS4_N_LIN_CONV_DIM*sizeof(float),gb=(uint64_t)T*DS4_N_LIN_V_HEAD*sizeof(float);
    const uint64_t base=g==p->wave?p->wave_base:0;
    if(base+T>8) return false;
    struct fronk_native_layer_tape *t=&p->tape[layer];
    if(stage==2) return ds4_gpu_tensor_copy(p->ple_rows,base*DS4_N_HC*DS4_N_EMBD*sizeof(float),g->ple_normed,0,(uint64_t)T*DS4_N_HC*DS4_N_EMBD*sizeof(float));
    if(stage==0) return ds4_gpu_tensor_copy(t->raw,base*DS4_N_LIN_CONV_DIM*sizeof(float),g->qkv,0,qb);
    return ds4_gpu_tensor_copy(t->qkv,base*DS4_N_LIN_CONV_DIM*sizeof(float),g->qkv,0,qb) &&
           ds4_gpu_tensor_copy(t->a,base*DS4_N_LIN_V_HEAD*sizeof(float),g->ga,0,gb) &&
           ds4_gpu_tensor_copy(t->b,base*DS4_N_LIN_V_HEAD*sizeof(float),g->gb,0,gb);
}
/* The accepted rows already have correct residuals, KV rows and logits. Only
 * recurrence and convolution history must be advanced from snapshot row 1.
 * Reuse normalized QKV/gates, not the dense projections that produced them. */
static bool fronk_native_recover(struct fronk_native *p,ds4_qwen4_gpu_graph *g,const int *tokens,unsigned n) {
    const unsigned count=n-2;
    const uint64_t qb=(uint64_t)DS4_N_LIN_CONV_DIM*sizeof(float),gb=(uint64_t)DS4_N_LIN_V_HEAD*sizeof(float);
    if(!p->tape || !glm_graph_begin_commands_if_needed()) return false;
    bool ok=true;
    ds4_gpu_qwen4_set_verify_rows_exact(true);
    for(unsigned l=0;ok && l<p->tape_layers;l++) if(ds4_qwen4_layer_is_linear(l)) {
        struct fronk_native_layer_tape *t=&p->tape[l];
        ds4_gpu_tensor *q=ds4_gpu_tensor_view(t->qkv,2*qb,count*qb);
        ds4_gpu_tensor *a=ds4_gpu_tensor_view(t->a,2*gb,count*gb),*b=ds4_gpu_tensor_view(t->b,2*gb,count*gb);
        ok=q && a && b && ds4_gpu_qwen4_gdn_scan_tensor(g->lin_o,g->layer_lin_state[l],q,a,b,count,
            DS4_N_LIN_K_HEAD,DS4_N_LIN_V_HEAD,DS4_N_LIN_HEAD_DIM,NULL,0,NULL,0) &&
            ds4_gpu_fronk_history_prefix(g->layer_lin_hist[l],t->raw,DS4_N_LIN_CONV_DIM,DS4_N_LIN_CONV-1,2,count);
        ds4_gpu_tensor_free(b);ds4_gpu_tensor_free(a);ds4_gpu_tensor_free(q);
    }
    if(ok) ok=ds4_gpu_fronk_history_prefix(g->ple_hist,p->ple_rows,DS4_N_HC*DS4_N_EMBD,(DS4_N_PLE_CONV-1)*DS4_N_PLE_NGRAM,2,count);
    if(!ds4_gpu_end_commands()) ok=false;
    ds4_gpu_qwen4_set_verify_rows_exact(false);
    if(ok) {
        uint32_t ids[DS4_MAX_PLE_HEADS];
        for(unsigned i=2;i<n;i++) qwen4_ple_step(tokens[i],g->ple_prev,ids);
        g->pos+=count;
    }
    return ok;
}
/* Diagnostic-only capture: the worker observes already materialized request
 * arrays; no extra device synchronization or network messages are introduced. */
static struct { unsigned mode,row; float *x,*y,*w; int32_t *ids; } fronk_native_trace;
static bool fronk_native_trace_init(void) {
    size_t rows=(size_t)DS4_N_LAYER*FRONK_NATIVE_MAX;
    if (!fronk_native_trace.x) fronk_native_trace.x=malloc(rows*DS4_N_EMBD*sizeof(float));
    if (!fronk_native_trace.y) fronk_native_trace.y=malloc(rows*DS4_N_EMBD*sizeof(float));
    if (!fronk_native_trace.w) fronk_native_trace.w=malloc(rows*DS4_N_EXPERT_USED*sizeof(float));
    if (!fronk_native_trace.ids) fronk_native_trace.ids=malloc(rows*DS4_N_EXPERT_USED*sizeof(int32_t));
    return fronk_native_trace.x && fronk_native_trace.y && fronk_native_trace.w && fronk_native_trace.ids;
}
static double fronk_native_relative(const float *a,const float *b,unsigned n) {
    double d=0,v=0;for(unsigned i=0;i<n;++i) {double z=(double)a[i]-b[i];d+=z*z;v+=(double)b[i]*b[i];}return sqrt(d/fmax(v,1e-30));
}
static bool fronk_native_dense_observer(unsigned stage,ds4_gpu_tensor *tensor,unsigned rows,unsigned dim) {
    static float *saved[4],*current; static size_t size[4],current_size;
    size_t bytes=(size_t)dim*rows*sizeof(float);
    if(!ds4_gpu_end_commands()) return false;
    if (fronk_native_trace.mode==1) {
        if(size[stage]<bytes) {free(saved[stage]);saved[stage]=malloc(bytes);size[stage]=bytes;}
        if(!saved[stage] || !ds4_gpu_tensor_read(tensor,0,saved[stage],bytes)) return false;
    } else if(fronk_native_trace.mode==2 && rows==1) {
        if(current_size<bytes) {free(current);current=malloc(bytes);current_size=bytes;}
        if(!current || !ds4_gpu_tensor_read(tensor,0,current,bytes)) return false;
        fprintf(stderr,"FRONK_NATIVE_DENSE stage=%u row=%u relative_l2=%.9g\n",stage,fronk_native_trace.row,
            fronk_native_relative(saved[stage]+(size_t)fronk_native_trace.row*dim,current,dim));
    }
    return glm_graph_begin_commands_if_needed();
}
static void fronk_native_observer(unsigned layer,unsigned rows,const float *x,const int32_t *ids,const float *w,const float *y) {
    if(layer>=DS4_N_LAYER || rows>FRONK_NATIVE_MAX) return;
    const size_t base=(size_t)layer*FRONK_NATIVE_MAX;
    if(fronk_native_trace.mode==1) {
        memcpy(fronk_native_trace.x+base*DS4_N_EMBD,x,rows*DS4_N_EMBD*sizeof(float));
        memcpy(fronk_native_trace.y+base*DS4_N_EMBD,y,rows*DS4_N_EMBD*sizeof(float));
        memcpy(fronk_native_trace.w+base*DS4_N_EXPERT_USED,w,rows*DS4_N_EXPERT_USED*sizeof(float));
        memcpy(fronk_native_trace.ids+base*DS4_N_EXPERT_USED,ids,rows*DS4_N_EXPERT_USED*sizeof(int32_t));
    } else if(fronk_native_trace.mode==2 && rows==1) {
        unsigned row=fronk_native_trace.row;
        double dx=fronk_native_relative(fronk_native_trace.x+(base+row)*DS4_N_EMBD,x,DS4_N_EMBD),dy=fronk_native_relative(fronk_native_trace.y+(base+row)*DS4_N_EMBD,y,DS4_N_EMBD);
        bool same=!memcmp(fronk_native_trace.ids+(base+row)*DS4_N_EXPERT_USED,ids,DS4_N_EXPERT_USED*sizeof(int32_t));
        if(dx>1e-7 || dy>1e-7 || !same) fprintf(stderr,"FRONK_NATIVE_DIFF layer=%u row=%u input_l2=%.9g output_l2=%.9g same_experts=%u\n",layer,row,dx,dy,same);
    }
}
#include "fronk_mtp_wave.inc"
static int fronk_native_cycle(ds4_session *s,int first,float temperature,int *accepted,int cap,char *err,size_t errlen) {
    ds4_qwen4_gpu_graph *g=&s->qwen4_graph;
    unsigned depth=fronk_native_depth();
    static bool policy_initialized, skip_catchup_heads, wave_enabled;
    if (!policy_initialized) {
        const char *policy=getenv("FRONK_NATIVE_MATRIX8");
        fronk_native_matrix8=policy && !strcmp(policy,"1");
        const char *skip=getenv("FRONK_NATIVE_SKIP_CATCHUP_HEADS");
        skip_catchup_heads=skip && !strcmp(skip,"1");
        const char *wave=getenv("FRONK_NATIVE_WAVE");
        wave_enabled=wave && !strcmp(wave,"1");
        policy_initialized=true;
    }
    if (!depth || temperature!=0 || cap<1 || !g->mtp_R || !g->snap_ple_hist ||
        g->vis_span_count || g->steer_dirs || !qwen4_fronk_env("FRONK_SHARED_LOCAL",1)) {
        if (g->fronk_native) g->fronk_native->valid=false;
        return 0;
    }
    const ds4_model *m=&s->engine->model; const ds4_weights *w=&s->engine->weights;
    const unsigned pos=g->pos, V=DS4_N_VOCAB;
    const bool check=getenv("FRONK_NATIVE_VERIFY")!=NULL, prof=getenv("FRONK_PROFILE")!=NULL;
    if (depth>g->cap_tokens || !qwen4_graph_ensure_snap2(g)) return 0;
    if (!g->fronk_native) {
        struct fronk_native *p=calloc(1,sizeof(*p));
        if (!p) return 0;
        p->rows=malloc((size_t)FRONK_NATIVE_MAX*V*sizeof(float));
        p->check=malloc((size_t)V*sizeof(float));
        if (!p->rows || !p->check) { fronk_native_free(p); return 0; }
        g->fronk_native=p;
        if (prof) fprintf(stderr,"FRONK_NATIVE_POLICY q8=%s oracle=independent_single_token_same_policy\n",
            fronk_native_matrix8?"matrix8_fp32":"scalar_order");
    }
    struct fronk_native *p=g->fronk_native;
    unsigned T=1;
    if (p->valid && p->depth==depth && p->position==pos && p->token[0]==first && p->prefix==fronk_prefix_id(s)) T=p->pending;
    if (T>(unsigned)cap) T=(unsigned)cap;
    if (T>g->ctx_cap-pos) T=g->ctx_cap-pos;
    if (!T) return 0;
    const bool wave=wave_enabled && T>=4;
    int tokens[FRONK_NATIVE_MAX]; tokens[0]=first;
    for (unsigned i=1;i<T;++i) tokens[i]=p->token[i];
    p->valid=false;
    const double started=prof?now_sec():0;
    if (check) {
        unsigned capacity=(pos+T+511u)&~511u;
        if (capacity>2048u) goto failed;
        if (!p->oracle) p->oracle=calloc(1,sizeof(*p->oracle));
        if (!p->oracle) goto failed;
        if (p->oracle->ctx_cap<capacity) {
            qwen4_graph_free(p->oracle);
            if (!qwen4_graph_alloc(p->oracle,w,capacity,fronk_native_matrix8?8:1,false,NULL,NULL,NULL,0)) goto failed;
            if (fronk_native_matrix8) {
                ds4_gpu_tensor_free(p->oracle->logits);
                p->oracle->logits=qwen4_graph_alloc_f32(8ull*V);
                free(p->oracle->host_logits);
                p->oracle->host_logits=xmalloc(8ull*V*sizeof(float));
                p->oracle->n_logit_rows=8;
                if(!p->oracle->logits) goto failed;
            }
        }
        if (!glm_graph_begin_commands_if_needed()) goto failed;
        bool copied=fronk_fork_state(p->oracle,g,0,pos);
        if (!ds4_gpu_end_commands() || !copied) goto failed;
    }
    g->snap_after_first=T>1;g->snap_after_second=T>2;
    g->snap_valid=g->snap2_valid=false;
    ds4_gpu_qwen4_stream_verify_begin(T);
    if(T>3) {
        if(!fronk_native_tape_reserve(p)) goto failed;
        fronk_native_tape_capture=fronk_native_capture;
    }
    if(check && !wave) {if(!fronk_native_trace_init()) goto failed;fronk_native_trace.mode=1;fronk_native_observe=fronk_native_observer;fronk_native_dense_observe=fronk_native_dense_observer;}
    if (!(wave?fronk_wave_forward(g,m,w,tokens,T,p->rows):fronk_native_forward(g,m,w,tokens,T,p->rows,true))) goto failed;
    fronk_native_tape_capture=NULL;
    fronk_native_observe=NULL;fronk_native_dense_observe=NULL;
    unsigned n=1;
    while (n<T && sample_argmax(p->rows+(size_t)(n-1)*V,V)==tokens[n]) ++n;
    ds4_gpu_qwen4_stream_verify_end(n);
    const double verified=prof?now_sec():0;
    unsigned replay=0;
    if (n<T) {
        if (n==1) { if (!g->snap_valid || !qwen4_graph_state_swap(g)) goto failed; }
        else {
            if (!g->snap2_valid || !qwen4_graph_state_swap2(g)) goto failed;
            replay=n-2;
            if (replay && !fronk_native_recover(p,g,tokens,n)) goto failed;
        }
    }
    if (g->pos!=pos+n) goto failed;
    if (check) {
        for (unsigned i=0;i<n;++i) {
            if(!wave) {fronk_native_trace.mode=2;fronk_native_trace.row=i;fronk_native_observe=fronk_native_observer;fronk_native_dense_observe=fronk_native_dense_observer;}
            bool oracle_ok=fronk_native_matrix8 ?
                fronk_native_forward(p->oracle,m,w,tokens+i,1,p->check,false) :
                qwen4_graph_forward_tokens(p->oracle,m,w,tokens+i,1,p->check,false);
            if (!oracle_ok) goto failed;
            fronk_native_observe=NULL;fronk_native_dense_observe=NULL;
            double diff=0,norm=0;
            for (unsigned j=0;j<V;++j) { double a=p->rows[(size_t)i*V+j],b=p->check[j];diff+=(a-b)*(a-b);norm+=b*b; }
            double relative=sqrt(diff/fmax(norm,1e-30));
            uint64_t bits;memcpy(&bits,&relative,sizeof(bits));
            bool ok=(bits&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000) && relative<=1e-5;
            if(wave && sample_argmax(p->rows+(size_t)i*V,V)!=sample_argmax(p->check,V)) ok=false;
            fprintf(stderr,"FRONK_NATIVE_CHECK pos=%u row=%u T=%u relative_l2=%.9g %s\n",pos,i,T,relative,ok?"pass":"FAIL");
            if (!ok) goto failed;
        }
        bool state=g->pos==p->oracle->pos && g->mrope_delta==p->oracle->mrope_delta && !memcmp(g->ple_prev,p->oracle->ple_prev,sizeof(g->ple_prev));
        if (state) state=fronk_native_equal(p,g->ple_hist,p->oracle->ple_hist,0,ds4_gpu_tensor_bytes(g->ple_hist));
        for (unsigned l=0;state && l<DS4_N_LAYER-DS4_N_NEXTN_PREDICT;++l) {
            if (ds4_qwen4_layer_is_linear(l)) {
                state=fronk_native_equal(p,g->layer_lin_state[l],p->oracle->layer_lin_state[l],0,ds4_gpu_tensor_bytes(g->layer_lin_state[l])) &&
                      fronk_native_equal(p,g->layer_lin_hist[l],p->oracle->layer_lin_hist[l],0,ds4_gpu_tensor_bytes(g->layer_lin_hist[l]));
            } else {
                uint64_t kv=2ull*DS4_N_HEAD_KV*DS4_N_HEAD_DIM,idx=sizeof(float)*DS4_N_INDEXER_HEAD_DIM;
                state=fronk_native_equal(p,g->layer_k_cache[l],p->oracle->layer_k_cache[l],kv*pos,kv*n) &&
                      fronk_native_equal(p,g->layer_v_cache[l],p->oracle->layer_v_cache[l],kv*pos,kv*n) &&
                      fronk_native_equal(p,g->layer_ik_cache[l],p->oracle->layer_ik_cache[l],idx*pos,idx*n);
            }
        }
        fprintf(stderr,"FRONK_NATIVE_STATE pos=%u T=%u accepted=%u replay=%u %s\n",pos,T,n,replay,state?"pass":"FAIL");
        if (!state) goto failed;
    }
    memcpy(s->logits,p->rows+(size_t)(n-1)*V,(size_t)V*sizeof(float));
    for (unsigned i=0;i<n;++i) { accepted[i]=tokens[i];token_vec_push(&s->checkpoint,tokens[i]); }
    s->checkpoint_valid=true;s->qwen4_rewound=false;s->mtp_draft_valid=false;
    s->glm_mtp_have=0;s->glm_mtp_have2=false;
    g->snap_valid=g->snap2_valid=g->snap0_valid=false;
    const double committed=prof?now_sec():0;
    const int parent=sample_argmax(s->logits,V);
    int next[FRONK_NATIVE_MAX],draft=-1;
    for (unsigned i=0;i<n;++i) next[i]=i+1<n?tokens[i+1]:parent;
    bool proposed=true;
    unsigned skipped_heads=0;
    /* Catch up the existing predictor over accepted residuals, in its native
     * <=3-row blocks. Its speculative KV frontier is overwritten, not trusted. */
    for (unsigned base=0;base<n && proposed;base+=3) {
        unsigned count=n-base<3?n-base:3;
        const bool want_logits=!skip_catchup_heads || base+count==n;
        proposed=qwen4_graph_mtp_steps(g,m,w,base,next+base,count,pos+base,want_logits,NULL,&draft);
        if(!want_logits) {
            /* Intermediate argmax is discarded. Preserve predictor KV and
             * residual updates, omit only the head projection/readback.
             * The no-logits API leaves commands open: drain before the next
             * call overwrites the shared embedding buffer on the CPU. This
             * is the same submission boundary as the original logits path. */
            if(!ds4_gpu_end_commands()) proposed=false;
            ++skipped_heads;
        }
    }
    const double caught_up=prof?now_sec():0;
    unsigned pending=1; p->token[0]=parent;
    if (proposed && g->pos+1<g->ctx_cap) {
        p->token[pending++]=draft;
        while (pending<depth && g->pos+pending<g->ctx_cap) {
            if (!qwen4_graph_mtp_chain_step(g,m,w,p->token[pending-1],g->pos+pending-2,&draft)) break;
            p->token[pending++]=draft;
        }
    }
    p->pending=pending;p->depth=depth;p->position=g->pos;p->prefix=fronk_prefix_id(s);p->valid=true;
    ++p->cycles;p->verified+=T;p->accepted+=n;p->replayed+=replay;
    if (prof) {
      uint64_t draft_hash=UINT64_C(14695981039346656037);
      for(unsigned i=0;i<pending;i++) draft_hash=(draft_hash^(uint32_t)p->token[i])*UINT64_C(1099511628211);
      const double done=now_sec();
      fprintf(stderr,"FRONK_NATIVE depth=%u T=%u accepted=%u replay=%u cycles=%llu verified=%llu emitted=%llu replayed=%llu verify_ms=%.3f commit_ms=%.3f draft_ms=%.3f catchup_ms=%.3f chain_ms=%.3f skipped_heads=%u draft_hash=%016llx wave=%u\n",
        depth,T,n,replay,(unsigned long long)p->cycles,(unsigned long long)p->verified,(unsigned long long)p->accepted,
        (unsigned long long)p->replayed,1000*(verified-started),1000*(committed-verified),1000*(done-committed),
        1000*(caught_up-committed),1000*(done-caught_up),skipped_heads,(unsigned long long)draft_hash,wave);
    }
    return (int)n;
failed:
    fronk_native_tape_capture=NULL;
    fronk_native_observe=NULL;fronk_native_dense_observe=NULL;
    ds4_gpu_qwen4_stream_verify_end(0);
    g->snap_after_first=g->snap_after_second=g->verify_rows_exact=false;
    ds4_gpu_qwen4_set_verify_rows_exact(false);p->valid=false;s->checkpoint_valid=false;
    if (errlen) snprintf(err,errlen,"Fronk native MTP verification failed at position %u",pos);
    return -1;
}

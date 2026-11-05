/* A causal token tree. Every token owns its recurrent state and MoE mailbox.
 * At each layer, a child inherits the parent's attention/GDN state after the
 * parent has submitted its experts. Its dense stage can then run while those
 * experts are outstanding. Only the verified prefix is committed. */
#define FRONK_TREE_MAX 4
#define FRONK_TREE_ROWS 3
#define FRONK_TREE_NODES (1 + 2 * FRONK_TREE_MAX)
struct fronk_tree {
    ds4_qwen4_gpu_graph lane[FRONK_TREE_NODES];
    int path[FRONK_TREE_MAX][FRONK_TREE_ROWS];
    unsigned width, active, capacity, position;
    uint64_t prefix, cycles, hits, seconds, emitted;
    bool valid, disabled;
};

static void fronk_tree_free(struct fronk_tree *tree) {
    if (!tree) return;
    for (unsigned i = 0; i < FRONK_TREE_NODES; ++i) qwen4_graph_free(&tree->lane[i]);
    free(tree);
}

static unsigned fronk_tree_width(void) {
    const char *v = getenv("FRONK_MTP_BRANCHES");
    return v && v[0] >= '2' && v[0] <= '4' && !v[1] ? (unsigned)(v[0] - '0') : 0;
}

static uint64_t fronk_prefix_id(const ds4_session *s) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (int i = 0; i < s->checkpoint.len; ++i)
        h = (h ^ (uint32_t)s->checkpoint.v[i]) * UINT64_C(1099511628211);
    return h;
}

/* Only this layer's state is copied. Copying the whole graph here would
 * overwrite states already advanced by the child at earlier layers. */
static bool fronk_fork_layer(ds4_qwen4_gpu_graph *dst, const ds4_qwen4_gpu_graph *src,
                              unsigned l, unsigned begin, unsigned pos) {
    if (ds4_qwen4_layer_is_linear(l))
        return ds4_gpu_tensor_copy(dst->layer_lin_state[l], 0, src->layer_lin_state[l], 0,
                                   ds4_gpu_tensor_bytes(src->layer_lin_state[l])) &&
               ds4_gpu_tensor_copy(dst->layer_lin_hist[l], 0, src->layer_lin_hist[l], 0,
                                   ds4_gpu_tensor_bytes(src->layer_lin_hist[l]));
    if (pos == begin) return true;
    const uint64_t kv = 2ull * DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint64_t idx = sizeof(float) * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t key = 2ull * DS4_N_INDEXER_HEAD_DIM;
    bool ok = ds4_gpu_tensor_copy(dst->layer_k_cache[l], kv*begin, src->layer_k_cache[l], kv*begin, kv*(pos-begin)) &&
              ds4_gpu_tensor_copy(dst->layer_v_cache[l], kv*begin, src->layer_v_cache[l], kv*begin, kv*(pos-begin)) &&
              ds4_gpu_tensor_copy(dst->layer_ik_cache[l], idx*begin, src->layer_ik_cache[l], idx*begin, idx*(pos-begin));
    if (ok && pos/4 > begin/4)
        ok = ds4_gpu_tensor_copy(dst->layer_block_key[l], key*(begin/4), src->layer_block_key[l], key*(begin/4), key*(pos/4-begin/4));
    return ok;
}

static bool fronk_fork_state(ds4_qwen4_gpu_graph *dst, const ds4_qwen4_gpu_graph *src,
                             unsigned begin, unsigned pos) {
    if (begin > pos || pos > dst->ctx_cap || pos > src->ctx_cap) return false;
    bool ok = ds4_gpu_tensor_copy(dst->ple_hist, 0, src->ple_hist, 0, ds4_gpu_tensor_bytes(src->ple_hist));
    if (ok && pos > begin)
        ok = ds4_gpu_tensor_copy(dst->pos3, 16ull*begin, src->pos3, 16ull*begin, 16ull*(pos-begin));
    for (unsigned l = 0; ok && l < DS4_N_LAYER - DS4_N_NEXTN_PREDICT; ++l)
        ok = fronk_fork_layer(dst, src, l, begin, pos);
    dst->pos = pos;
    dst->mrope_delta = src->mrope_delta;
    memcpy(dst->ple_prev, src->ple_prev, sizeof(dst->ple_prev));
    return ok;
}

static bool fronk_tree_reserve(struct fronk_tree *t, const ds4_qwen4_gpu_graph *g, const ds4_weights *w) {
    if (t->capacity >= g->pos + FRONK_TREE_ROWS) return true;
    unsigned cap = (g->pos + FRONK_TREE_ROWS + 511u) & ~511u;
    if (cap > g->ctx_cap) cap = g->ctx_cap;
    /* Limit the experimental workspace to short/medium contexts. Falling
     * back leaves the ordinary MTP path responsible for longer conversations. */
    if (cap > 2048u) return false;
    for (unsigned i = 0; i < 1 + 2*t->width; ++i) {
        qwen4_graph_free(&t->lane[i]);
        if (!qwen4_graph_alloc(&t->lane[i], w, cap, 1, false, NULL, NULL, NULL, 0)) return false;
    }
    t->capacity = cap;
    return true;
}

static unsigned fronk_tree_parent(unsigned i, unsigned width) {
    return i <= width ? 0 : i-width;
}

static bool fronk_tree_layer(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w, unsigned l) {
    const ds4_layer_weights *layer = &w->layer[l];
    return fronk_ple_layer(g, m, layer, l, 1, DS4_N_HC * DS4_N_EMBD) &&
        qwen4_graph_hc_mix(g, m, layer->hc_attn_norm, layer->hc_attn_down, layer->hc_attn_up, layer->hc_attn_inject, 1) &&
        (ds4_qwen4_layer_is_linear(l) ? qwen4_graph_linear(g, m, layer, l, 1) : qwen4_graph_attention(g, m, layer, l, g->pos, 1)) &&
        fronk_ffn_mix(g, m, layer, 1) && qwen4_graph_moe(g, m, layer, 1);
}

static bool fronk_tree_verify(struct fronk_tree *t, ds4_qwen4_gpu_graph *g, const ds4_model *m,
                              const ds4_weights *w, int token, unsigned count) {
    if (!glm_graph_begin_commands_if_needed()) return false;
    bool ok = fronk_fork_state(&t->lane[0], g, 0, g->pos);
    if (!ds4_gpu_end_commands()) ok = false;
    if (!ok || !glm_graph_begin_commands_if_needed()) return false;
    for (unsigned i = 0; ok && i < count; ++i) {
        ds4_qwen4_gpu_graph *lane = &t->lane[i];
        int input = token;
        if (i) {
            ds4_qwen4_gpu_graph *parent = &t->lane[fronk_tree_parent(i, t->active)];
            lane->pos = parent->pos + 1;
            lane->mrope_delta = parent->mrope_delta;
            memcpy(lane->ple_prev, parent->ple_prev, sizeof(lane->ple_prev));
            ok = ds4_gpu_tensor_copy(lane->pos3, 0, parent->pos3, 0, 16ull*lane->pos);
            input = i <= t->active ? t->path[i-1][1] : t->path[i-t->active-1][2];
        }
        lane->fronk_defer_reduce = true;
        if (ok) ok = qwen4_graph_stage_inputs(lane, m, w, &input, 1);
    }
    const unsigned layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    for (unsigned l = 0; ok && l <= layers; ++l) {
        for (unsigned i = 0; ok && i < count; ++i) {
            ds4_qwen4_gpu_graph *lane = &t->lane[i];
            if (l) ok = fronk_finish_experts(lane, 1, lane->fronk_resume_value, lane->fronk_shared_local);
            if (ok && i && l < layers) {
                ds4_qwen4_gpu_graph *parent = &t->lane[fronk_tree_parent(i, t->active)];
                ok = fronk_fork_layer(lane, parent, l, 0, lane->pos);
                if (ok && ds4_qwen4_layer_is_ple(l))
                    ok = ds4_gpu_tensor_copy(lane->ple_hist, 0, parent->ple_hist, 0, ds4_gpu_tensor_bytes(parent->ple_hist));
            }
            if (ok && l < layers) ok = fronk_tree_layer(lane, m, w, l);
            else if (ok) ok = qwen4_graph_hc_mix(lane, m, w->output_hc_norm, w->output_hc_down, w->output_hc_up, NULL, 1) &&
                qwen4_gemv(lane->logits, m, w->output, lane->mixed, 1);
        }
    }
    // Drain submitted jobs on both success and failure before reusing/freeing
    // their graph mailboxes. Ready signals alone do not wait for the CPU.
    uint64_t last_resume = 0;
    for (unsigned i = 0; i < count; ++i)
        if (t->lane[i].fronk_resume_value > last_resume) last_resume = t->lane[i].fronk_resume_value;
    if (last_resume && !ds4_gpu_fronk_resume(last_resume)) ok = false;
    if (!ds4_gpu_end_commands() || __atomic_load_n(&qwen4_fronk.error, __ATOMIC_ACQUIRE)) ok = false;
    for (unsigned i = 0; ok && i < count; ++i) {
        ++t->lane[i].pos;
        ok = ds4_gpu_tensor_read(t->lane[i].logits, 0, t->lane[i].host_logits, DS4_N_VOCAB * sizeof(float));
    }
    return ok;
}

/* Generate width siblings from a single MTP distribution. Each sibling's
 * continuation starts from the same predictor residual and KV frontier. */
static bool fronk_tree_propose(struct fronk_tree *t, ds4_qwen4_gpu_graph *g, const ds4_model *m,
                               const ds4_weights *w, const int *accepted, unsigned n, unsigned pos, int parent) {
    int next[FRONK_TREE_ROWS];
    for (unsigned i = 0; i < n; ++i) next[i] = i+1 < n ? accepted[i+1] : parent;
    float *logits = g->host_logits;
    if (!qwen4_graph_mtp_steps(g,m,w,0,next,n,pos,true,logits,NULL)) return false;
    const size_t seed_bytes = (size_t)DS4_N_HC * DS4_N_EMBD * sizeof(float);
    const size_t seed_offset = (n-1) * seed_bytes;
    // host_row is scratch used inside the predictor, so the seed is private.
    float *seed = malloc(seed_bytes);
    if (!seed) return false;
    bool ok = ds4_gpu_tensor_read(g->mtp_R, seed_offset, seed, seed_bytes);
    int siblings[FRONK_TREE_MAX] = {-1, -1, -1, -1};
    for (int id = 0; id < (int)DS4_N_VOCAB; ++id) {
        for (unsigned rank = 0; rank < t->width; ++rank) {
            if (siblings[rank] >= 0 && logits[id] <= logits[siblings[rank]]) continue;
            for (unsigned j=t->width-1; j>rank; --j) siblings[j]=siblings[j-1];
            siblings[rank]=id;
            break;
        }
    }
    // Spend extra branches on ambiguous predictions. On confident tokens a
    // single continuation already covers most of the predictor distribution.
    double total = 0, mass = 0;
    for (unsigned id = 0; id < DS4_N_VOCAB; ++id) total += exp((double)logits[id]-logits[siblings[0]]);
    t->active = 0;
    do {
        mass += exp((double)logits[siblings[t->active]]-logits[siblings[0]]);
        ++t->active;
    } while (t->active < t->width && mass < 0.8*total);
    for (unsigned i = 0; ok && i < t->active; ++i) {
        int token = siblings[i];
        t->path[i][0] = parent; t->path[i][1] = token;
        g->mtp_last_rows = n; g->mtp_pos = pos+n;
        ok = ds4_gpu_tensor_write(g->mtp_R,seed_offset,seed,seed_bytes) &&
             qwen4_graph_mtp_chain_step(g,m,w,token,pos+n,&t->path[i][2]);
    }
    if (ok) ok = ds4_gpu_tensor_write(g->mtp_R,seed_offset,seed,seed_bytes);
    free(seed);
    g->mtp_last_rows = n; g->mtp_pos = pos+n;
    return ok;
}

/* Optional expensive validation against the normal single-token entry point.
 * This checks every committed row, including partial/rejected proposals, on
 * real weights. It is never enabled by the normal installer or chat. */
static bool fronk_tree_check(struct fronk_tree *t, ds4_qwen4_gpu_graph *g, const ds4_model *m,
                             const ds4_weights *w, const int *tokens, const unsigned *chosen, unsigned n) {
    if (!getenv("FRONK_MTP_VERIFY")) return true;
    ds4_gpu_tensor *mtp = g->mtp_R;
    g->mtp_R = NULL;
    bool ok = true;
    for (unsigned row = 0; ok && row < n; ++row) {
        ok = qwen4_graph_forward_tokens(g, m, w, tokens+row, 1, g->host_logits, false);
        double err = 0, norm = 0;
        for (unsigned v = 0; ok && v < DS4_N_VOCAB; ++v) {
            double a = g->host_logits[v], b = t->lane[chosen[row]].host_logits[v];
            err += (a-b)*(a-b); norm += a*a;
        }
        double relative = sqrt(err / (norm + 1e-30));
        uint64_t bits;
        memcpy(&bits, &relative, sizeof(bits));
        // The engine uses fast-math, so inspect the exponent for NaN/Inf.
        if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000) || relative > 1e-5) ok = false;
        fprintf(stderr, "FRONK_TREE_CHECK pos=%u row=%u relative_l2=%.9g %s\n", g->pos, row, relative, ok ? "pass" : "FAIL");
    }
    g->mtp_R = mtp;
    return ok;
}

static int fronk_tree_cycle(ds4_session *s, int token, float temperature, int top_k,
                            float top_p, float min_p, uint64_t *rng,
                            int *accepted, int cap, char *err, size_t errlen) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    unsigned width = fronk_tree_width();
    if (!width || cap < FRONK_TREE_ROWS || !g->mtp_R || g->pos+FRONK_TREE_ROWS > g->ctx_cap ||
        g->vis_span_count || g->steer_dirs || !qwen4_fronk_env("FRONK_PIPELINE",1) ||
        !qwen4_fronk_env("FRONK_SHARED_LOCAL",1) || qwen4_fronk_prefill_local(1)) return 0;
    const ds4_model *m = &s->engine->model;
    const ds4_weights *w = &s->engine->weights;
    if (!g->fronk_tree) {
        g->fronk_tree = calloc(1,sizeof(*g->fronk_tree));
        if (!g->fronk_tree) return 0;
        g->fronk_tree->width = width;
    }
    struct fronk_tree *t = g->fronk_tree;
    if (t->disabled) return 0;
    const unsigned pos = g->pos;
    if (!fronk_tree_reserve(t,g,w)) {
        for (unsigned i=0; i<FRONK_TREE_NODES; ++i) qwen4_graph_free(&t->lane[i]);
        t->disabled = true;
        fprintf(stderr,"FRONK_TREE workspace limit reached, resuming linear verification\n");
        return 0;
    }
    const unsigned active = t->active;
    bool pending = t->valid && s->glm_mtp_have && t->position == pos &&
                   t->path[0][0] == token && t->prefix == fronk_prefix_id(s);
    t->valid = false;
    if (!fronk_tree_verify(t,g,m,w,token,pending ? 1+2*t->active : 1)) goto failed;
    uint64_t sampled_rng = rng ? *rng : 0;
    int winner = -1;
    unsigned chosen[FRONK_TREE_ROWS] = {0}, n = 1;
    accepted[0] = token;
    if (pending) {
        ++t->cycles;
        int target = sample_top_p_min_p(t->lane[0].host_logits, DS4_N_VOCAB, temperature,
                                        top_k, top_p, min_p, &sampled_rng, s->sample_probs);
        for (unsigned i=0; i<t->active; ++i) if (t->path[i][1] == target) { winner = (int)i; break; }
        if (winner >= 0) {
            chosen[n] = 1+(unsigned)winner;
            accepted[n++] = target;
            ++t->hits;
            if (rng) *rng = sampled_rng;
            int child = sample_top_p_min_p(t->lane[chosen[1]].host_logits, DS4_N_VOCAB, temperature,
                                           top_k, top_p, min_p, &sampled_rng, s->sample_probs);
            if (t->path[winner][2] == child) {
                chosen[n] = 1+t->active+(unsigned)winner;
                accepted[n++] = child;
                ++t->seconds;
                if (rng) *rng = sampled_rng;
            }
        }
    }
    if (!fronk_tree_check(t,g,m,w,accepted,chosen,n) || !glm_graph_begin_commands_if_needed()) goto failed;
    ds4_qwen4_gpu_graph *leaf = &t->lane[chosen[n-1]];
    bool ok = fronk_fork_state(g,leaf,pos,pos+n);
    const uint64_t row_bytes = (uint64_t)DS4_N_HC*DS4_N_EMBD*sizeof(float);
    for (unsigned i=0; ok && i<n; ++i)
        ok = ds4_gpu_tensor_copy(g->R,i*row_bytes,t->lane[chosen[i]].R,0,row_bytes);
    if (!ds4_gpu_end_commands() || !ok) goto failed;
    memcpy(s->logits,leaf->host_logits,DS4_N_VOCAB*sizeof(float));
    for (unsigned i=0; i<n; ++i) token_vec_push(&s->checkpoint,accepted[i]);
    s->checkpoint_valid=true; s->qwen4_rewound=false; s->mtp_draft_valid=false;
    s->glm_mtp_have=0; s->glm_mtp_have2=false;
    g->snap_valid=g->snap2_valid=g->snap0_valid=false;
    // Peek at the next target draw without consuming it. A miss also leaves
    // the caller's RNG untouched, so the ordinary sampler draws that token.
    sampled_rng = rng ? *rng : 0;
    int parent = sample_top_p_min_p(s->logits, DS4_N_VOCAB, temperature,
                                    top_k, top_p, min_p, &sampled_rng, s->sample_probs);
    if (g->pos+FRONK_TREE_ROWS <= g->ctx_cap && fronk_tree_propose(t,g,m,w,accepted,n,pos,parent)) {
        t->position=g->pos; t->prefix=fronk_prefix_id(s); t->valid=true;
        s->glm_mtp_draft=t->path[0][1]; s->glm_mtp_parent=parent; s->glm_mtp_have=1;
    }
    t->emitted+=n;
    fprintf(stderr,"FRONK_TREE width=%u active=%u cycles=%llu sibling_hits=%llu continuation_hits=%llu emitted=%llu selected=%d accepted=%u overlapped_calls=%llu coalesced_jobs=%llu\n",
            width,pending ? active : 0,(unsigned long long)t->cycles,(unsigned long long)t->hits,(unsigned long long)t->seconds,
            (unsigned long long)t->emitted,winner,n,(unsigned long long)__atomic_load_n(&fronk_tree_overlaps,__ATOMIC_RELAXED),
            (unsigned long long)__atomic_load_n(&fronk_tree_coalesced,__ATOMIC_RELAXED));
    return (int)n;
failed:
    t->valid=false; s->checkpoint_valid=false;
    if (errlen) snprintf(err,errlen,"Fronk branch verification failed at position %u",pos);
    return -1;
}

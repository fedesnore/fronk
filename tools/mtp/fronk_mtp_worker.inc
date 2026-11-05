/* Coalesce adjacent, already-ready tree jobs. Mode 1 requires identical
 * ordered routing; mode 2 requires ranked kernels on every participant and
 * at least two same-rank reused experts. Legacy nodes cannot safely process
 * partial overlap: it changes their floating-point reduction order. Never reorder
 * jobs: the GPU completion timeline is monotonic, not an independent fence
 * per mailbox. No sleep, lookahead scan, or extra GPU wait to form a batch. */
static uint64_t fronk_tree_coalesced;
static void *qwen4_fronk_worker(void *arg) {
    (void)arg;
    float *x = NULL, *w = NULL, *sh = NULL, *y = NULL, *ones = NULL;
    int32_t *ids = NULL;
    uint32_t capacity = 0;
    const int prof = getenv("FRONK_PROFILE") != NULL;
    /* Match the native depth parser once at worker startup. Native chains
     * use ordinary mailboxes, so fronk_defer_reduce alone does not identify
     * every speculative verification. Missing shards must never be committed. */
    const char *native_depth = getenv("FRONK_NATIVE_MTP");
    const bool native = native_depth && native_depth[0]>='3' &&
                        native_depth[0]<='8' && !native_depth[1];
    int coalesce = qwen4_fronk_env("FRONK_MTP_COALESCE", 0);
    const char *batch_mode = getenv("FRONK_MTP_COALESCE");
    if (batch_mode && !strcmp(batch_mode,"2")) coalesce=2;
#ifndef FR_EXPERT_RANKED
    if (coalesce > 1) {
        fprintf(stderr,"ds4: partial reuse requires ranked expert kernels on every participant; disabled\n");
        coalesce = 0;
    }
#endif
    uint64_t merged = 0, candidates = 0, different_route = 0;
    double scheduler_us = 0;
    double acc[6] = {0}, last = 0;
    unsigned calls = 0;
#ifdef FRONK_PATH_PROFILE
    struct { unsigned calls; double ready,read,moe,write,resume; } path[9] = {0};
    unsigned path_calls=0;
#endif
    for (;;) {
        qwen4_fronk_job jobs[8];
        pthread_mutex_lock(&qwen4_fronk.mu);
        while (qwen4_fronk.head == qwen4_fronk.tail)
            pthread_cond_wait(&qwen4_fronk.cv, &qwen4_fronk.mu);
        jobs[0] = qwen4_fronk.q[qwen4_fronk.head++ % 512];
        pthread_mutex_unlock(&qwen4_fronk.mu);
        unsigned count = 1;
        uint32_t rows = jobs[0].T;
        const double t0 = prof ? qwen4_fronk_now() : 0;
        bool ok = ds4_gpu_fronk_wait(jobs[0].ready, 120000.0) != 0 &&
                  !__atomic_load_n(&qwen4_fronk.error, __ATOMIC_ACQUIRE);
        const bool tree = jobs[0].g->fronk_defer_reduce;
        uint64_t ready_before = tree ? ds4_gpu_fronk_ready_value() : 0;
        const double schedule_start = prof && coalesce && tree ? qwen4_fronk_now() : 0;
        /* Only single-row tree lanes qualify. Read 40 bytes of routing, not
         * activations, from an immutable mailbox after its ready event. */
        if (ok && coalesce && tree && jobs[0].T == 1 && jobs[0].shared_local) {
            int32_t route[8][DS4_N_EXPERT_USED], next_route[DS4_N_EXPERT_USED];
            ok = ds4_gpu_tensor_read(jobs[0].g->selected, 0, route[0], sizeof(route[0]));
            pthread_mutex_lock(&qwen4_fronk.mu);
            while (ok && count < 8 && qwen4_fronk.head != qwen4_fronk.tail) {
                qwen4_fronk_job next = qwen4_fronk.q[qwen4_fronk.head % 512];
                if (!next.g->fronk_defer_reduce || next.layer != jobs[0].layer ||
                    !next.shared_local || next.T != 1 || next.ready > ready_before) break;
                ++candidates;
                if (!ds4_gpu_tensor_read(next.g->selected, 0, next_route, sizeof(next_route))) {
                    ok = false; break;
                }
                bool compatible = !memcmp(route[0], next_route, sizeof(next_route));
#ifdef FR_EXPERT_RANKED
                if (coalesce > 1) {
                    /* Bounded comparison, no sorting or allocation. Reuse at
                     * least two weights at the same rank with an admitted row. */
                    compatible = false;
                    for (unsigned i=0;i<count && !compatible;++i) {
                        unsigned shared=0;
                        for (unsigned e=0;e<DS4_N_EXPERT_USED;++e) shared+=route[i][e]==next_route[e];
                        compatible=shared>=2;
                    }
                }
#endif
                if (!compatible) { ++different_route; break; }
                memcpy(route[count],next_route,sizeof(next_route));
                jobs[count++] = next;
                ++rows;
                ++qwen4_fronk.head;
            }
            pthread_mutex_unlock(&qwen4_fronk.mu);
            merged += count-1;
            if (count > 1) __atomic_add_fetch(&fronk_tree_coalesced, count-1, __ATOMIC_RELAXED);
        }
        if (schedule_start) scheduler_us += qwen4_fronk_now()-schedule_start;
        const double t1 = prof ? qwen4_fronk_now() : 0;
        if (rows > capacity) {
            x = realloc(x, (size_t)rows * DS4_N_EMBD * sizeof(float));
            y = realloc(y, (size_t)rows * DS4_N_EMBD * sizeof(float));
            w = realloc(w, (size_t)rows * DS4_N_EXPERT_USED * sizeof(float));
            ids = realloc(ids, (size_t)rows * DS4_N_EXPERT_USED * sizeof(int32_t));
            sh = realloc(sh, rows * sizeof(float));
            ones = realloc(ones, rows * sizeof(float));
            ok = ok && x && y && w && ids && sh && ones;
            capacity = ok ? rows : 0;
        }
        uint32_t offset = 0;
        for (unsigned i = 0; ok && i < count; ++i) {
            qwen4_fronk_job *job = &jobs[i];
            ok = ds4_gpu_tensor_read(job->g->mixed, 0, x + (size_t)offset*DS4_N_EMBD,
                                      (uint64_t)job->T*DS4_N_EMBD*sizeof(float)) &&
                 ds4_gpu_tensor_read(job->g->selected, 0, ids + (size_t)offset*DS4_N_EXPERT_USED,
                                      (uint64_t)job->T*DS4_N_EXPERT_USED*sizeof(int32_t)) &&
                 ds4_gpu_tensor_read(job->g->weights, 0, w + (size_t)offset*DS4_N_EXPERT_USED,
                                      (uint64_t)job->T*DS4_N_EXPERT_USED*sizeof(float)) &&
                 (job->shared_local || ds4_gpu_tensor_read(job->g->sh_gate_logit, 0, sh+offset, job->T*sizeof(float)));
            offset += job->T;
        }
        uint32_t missing = 0;
#ifdef FRONK_PATH_PROFILE
        const double read_done=qwen4_fronk_now();
#endif
        if (ok) {
            ok = fronk_moe(jobs[0].layer, rows, x, ids, w, jobs[0].shared_local ? NULL : sh, y, &missing) == 0;
            /* Speculation cannot commit a prefix computed with missing shards. */
            if ((tree || native) && missing) ok = false;
        }
        const double t2 = prof ? qwen4_fronk_now() : 0;
        if (jobs[0].g->fronk_defer_reduce && ds4_gpu_fronk_ready_value() > ready_before)
            __atomic_add_fetch(&fronk_tree_overlaps, 1, __ATOMIC_RELAXED);
        if (ok) for (uint32_t row = 0; row < rows; ++row) ones[row] = 1.0f;
        offset = 0;
        for (unsigned i = 0; ok && i < count; ++i) {
            qwen4_fronk_job *job = &jobs[i];
            ok = ds4_gpu_tensor_write(job->g->part, 0, y + (size_t)offset*DS4_N_EMBD,
                                       (uint64_t)job->T*DS4_N_EMBD*sizeof(float)) &&
                 ds4_gpu_tensor_write(job->g->weights, 0, ones, job->T*sizeof(float));
            offset += job->T;
        }
        if (!ok && !__atomic_exchange_n(&qwen4_fronk.error, 1, __ATOMIC_ACQ_REL))
            fprintf(stderr, "ds4: fronk expert call failed for layer %u\n", jobs[0].layer);
#ifdef FRONK_PATH_PROFILE
        const double write_done=qwen4_fronk_now();
#endif
        /* Complete even on failure so the GPU can drain, report the error and
         * release every mailbox. Poll mode is used only for non-tree jobs. */
        for (unsigned i = 0; i < count; ++i) if (!ds4_gpu_fronk_signal(jobs[i].resume)) {
            if (!__atomic_exchange_n(&qwen4_fronk.error, 1, __ATOMIC_ACQ_REL))
                fprintf(stderr, "ds4: fronk GPU resume poll timed out\n");
        }
#ifdef FRONK_PATH_PROFILE
        /* This build requires FRONK_PROFILE=1 for the existing timestamps.
         * CPU ready wait includes GPU execution and scheduling, not just compute. */
        if(prof && rows<=8) {
            ++path[rows].calls;
            path[rows].ready+=t1-t0;path[rows].read+=read_done-t1;
            path[rows].moe+=t2-read_done;path[rows].write+=write_done-t2;
            path[rows].resume+=qwen4_fronk_now()-write_done;
            if(++path_calls==480) {
                for(unsigned t=1;t<=8;t++) if(path[t].calls) fprintf(stderr,
                    "FRONK_PATH_WORKER T=%u calls=%u ready_us=%.3f read_us=%.3f moe_us=%.3f write_us=%.3f resume_us=%.3f\n",
                    t,path[t].calls,path[t].ready/path[t].calls,path[t].read/path[t].calls,
                    path[t].moe/path[t].calls,path[t].write/path[t].calls,path[t].resume/path[t].calls);
                memset(path,0,sizeof(path));path_calls=0;
            }
        }
#endif
        if (prof) {
            if (last > 0 && t0-last < 1e6) {
                if (t0-last > 2000) { acc[4] += t0-last; ++acc[5]; }
                else acc[0] += t0-last;
            }
            acc[1] += t1-t0; acc[2] += t2-t1; acc[3] += rows;
            last = qwen4_fronk_now();
            if (++calls % 480 == 0) {
                fprintf(stderr, "FRONK_PROFILE pipeline calls=%u avg_T=%.2f us/call: idle=%.0f gpu_until_ready=%.0f cluster=%.0f; boundaries=%.0f avg=%.0f us\n",
                        calls, acc[3]/480, acc[0]/480, acc[1]/480, acc[2]/480, acc[5], acc[5] ? acc[4]/acc[5] : 0);
                if (coalesce) fprintf(stderr,
                    "FRONK_SCHED calls=%u merged_jobs=%llu candidates=%llu different_route=%llu scheduler_us_per_call=%.3f\n",
                    calls, (unsigned long long)merged, (unsigned long long)candidates,
                    (unsigned long long)different_route, scheduler_us/480);
                scheduler_us = 0;
                memset(acc, 0, sizeof(acc));
            }
        }
    }
    return NULL;
}

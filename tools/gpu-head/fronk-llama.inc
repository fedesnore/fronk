// Called by llama.cpp after its Qwen router has normalized the top-ten gates.
// ggml schedules this custom op on the CPU and transfers its inputs from the GPU.
// The shared expert stays in the dense graph. Fronk supplies the routed sum.
#include "fronk_client.h"
static void fronk_llama_experts(ggml_tensor *dst,const ggml_tensor *x,const ggml_tensor *ids,
                                const ggml_tensor *gates,int ith,int,void *layer) {
    if (ith) return;
    GGML_ASSERT(x->type==GGML_TYPE_F32 && ids->type==GGML_TYPE_I32 && gates->type==GGML_TYPE_F32);
    GGML_ASSERT(x->ne[0]==FRONK_HIDDEN && ids->ne[0]==FRONK_ROUTED && ids->ne[1]==x->ne[1]);
    GGML_ASSERT(gates->ne[0]==1 && gates->ne[1]==FRONK_ROUTED && gates->ne[2]==x->ne[1]);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(ids) && ggml_is_contiguous(gates) && ggml_is_contiguous(dst));
    uint32_t missing=0;
    if (fronk_moe((unsigned)(uintptr_t)layer,(unsigned)x->ne[1],(const float *)x->data,
                  (const int32_t *)ids->data,(const float *)gates->data,nullptr,(float *)dst->data,&missing))
        GGML_ABORT("Fronk expert connection failed on layer %u",(unsigned)(uintptr_t)layer);
}

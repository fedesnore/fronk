// Exercise the same Vulkan Q8 matmul used for the head's dense projections.
// Device enumeration alone cannot establish that a driver runs GGML kernels.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool measure(ggml_backend_dev_t device,double &ns_per_byte) {
    constexpr int cols=2560, rows=8192;
    auto backend=ggml_backend_dev_init(device,nullptr);
    if (!backend) return false;
    auto ctx=ggml_init({2*1024*1024,nullptr,true});
    auto weights=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,cols,rows);
    auto input=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,cols);
    auto output=ggml_mul_mat(ctx,weights,input);
    auto graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,output);
    auto buf=ggml_backend_alloc_ctx_tensors(ctx,backend);
    bool ok=buf && ggml_backend_dev_supports_op(device,output);
    std::vector<unsigned char> packed(ggml_nbytes(weights));
    std::vector<float> x(cols,1.f), y(rows);
    for (int r=0;r<rows;++r) for (int b=0;b<cols/32;++b) {
        unsigned char *q=packed.data()+34*(r*(cols/32)+b);
        q[0]=0; q[1]=0x28; // fp16 scale of 1/32
        memset(q+2,r%2?255:1,32);
    }
    if (ok) {
        ggml_backend_tensor_set(weights,packed.data(),0,packed.size());
        ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float));
        std::vector<double> times;
        for (int i=0;i<8 && ok;++i) {
            auto start=std::chrono::steady_clock::now();
            ok=ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS;
            ggml_backend_synchronize(backend);
            double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count();
            if (i) times.push_back(ns);
        }
        if (ok) {
            ggml_backend_tensor_get(output,y.data(),0,y.size()*sizeof(float));
            for (int r=0;r<rows;++r) if (!(std::fabs(y[r]-(r%2?-80.f:80.f))<0.1f)) { ok=false; break; }
            std::sort(times.begin(),times.end()); ns_per_byte=times[times.size()/2]/packed.size();
        }
    }
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx); ggml_backend_free(backend);
    return ok && ns_per_byte>0;
}
int main(int argc,char **argv) {
    if (argc!=2 || (strcmp(argv[1],"--device") && strcmp(argv[1],"--profile") && strcmp(argv[1],"--timing"))) return 2;
    ggml_backend_load_all();
    for (size_t i=0;i<ggml_backend_dev_count();++i) {
        auto d=ggml_backend_dev_get(i);
        auto type=ggml_backend_dev_type(d);
        const char *name=ggml_backend_dev_name(d), *desc=ggml_backend_dev_description(d);
        if ((type!=GGML_BACKEND_DEVICE_TYPE_GPU && type!=GGML_BACKEND_DEVICE_TYPE_IGPU) ||
            strncmp(name,"Vulkan",6) || strstr(desc,"llvmpipe") || strstr(desc,"lavapipe")) continue;
        double ns=0;
        if (measure(d,ns)) {
            fprintf(stderr,"Fronk GPU head: %s (%s) passed the Q8_0 dense projection check.\n",desc,name);
            if (!strcmp(argv[1],"--profile")) {
                size_t free_bytes=0,total_bytes=0;
                ggml_backend_dev_memory(d,&free_bytes,&total_bytes);
                printf("%s %s %zu %zu\n",name,type==GGML_BACKEND_DEVICE_TYPE_IGPU?"shared":"dedicated",total_bytes,free_bytes);
            } else if (!strcmp(argv[1],"--device")) puts(name);
            else printf("%.9g\n",ns);
            return 0;
        }
    }
    fputs("Fronk GPU head: no hardware Vulkan device passed the dense projection check.\n",stderr);
    return 1;
}

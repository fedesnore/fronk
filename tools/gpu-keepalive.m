// gpu-keepalive [--watch PID] [ITERATIONS] - dispatch a tiny compute kernel in a
// loop so the head's GPU never idles while llama.cpp waits for the nodes. Idle,
// macOS lowers the GPU clock: with a 3 ms wait per layer the head took 1.88 ms
// per layer instead of 0.72 (PLAN-gsq-rco.md). --watch: exit when PID is gone.
#import <Metal/Metal.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
static const char *src =
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void spin(device float *o [[buffer(0)]], constant uint &n [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
"  float a = i; for (uint k = 0; k < n; ++k) a = fma(a, 0.999f, 0.001f); o[i] = a; }\n";
int main(int argc, char **argv) { @autoreleasepool {
    pid_t watch = 0; int a = 1;
    if (argc > 2 && !strcmp(argv[1], "--watch")) { watch = (pid_t)atoi(argv[2]); a = 3; }
    uint32_t n = argc > a ? (uint32_t)atoi(argv[a]) : 20000;
    unsigned long round = 0;
    id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError *e = nil;
    id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:src] options:nil error:&e];
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"spin"] error:&e];
    id<MTLCommandQueue> q = [d newCommandQueue]; id<MTLBuffer> b = [d newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    for (;;) { @autoreleasepool {
        if (watch && (++round & 1023) == 0 && kill(watch, 0) && errno == ESRCH) return 0;
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        [en setComputePipelineState:p]; [en setBuffer:b offset:0 atIndex:0]; [en setBytes:&n length:4 atIndex:1];
        [en dispatchThreads:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [en endEncoding];
        [cb commit]; [cb waitUntilCompleted];
    } }
} }

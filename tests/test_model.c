/* test_model.c - protect the fixed model's distribution and storage geometry.
 *
 * Expected totals are independent design values, including counts too large
 * for a 32-bit integer. No model weights are needed for these checks.
 */
#include "test.h"
#include "../src/model.h"

int main(void) {
    test_begin("model");
    check(FR_LAYERS == 48 && FR_DELTANET_LAYERS == 36 && FR_QSA_LAYERS == 12,
          "48 layers with the 3:1 attention split");
    check(FR_HIDDEN == 2560 && FR_INTERMEDIATE == 640, "fixed matrix dimensions");
    check(FR_EXPERTS == 513 && FR_ACTIVE_EXPERTS == 11, "routed plus shared experts");
    check(FR_SHARED_EXPERT == 512, "shared ID follows all routed IDs");
    check(FR_BLOCKS == 20 && FR_BLOCK_NEURONS == 32, "twenty ownership blocks");
    int covered[FR_INTERMEDIATE] = {0};
    for (int block = 0; block < FR_BLOCKS; ++block)
        for (int neuron = 0; neuron < FR_BLOCK_NEURONS; ++neuron)
            ++covered[block * FR_BLOCK_NEURONS + neuron];
    int exact = 1;
    for (int neuron = 0; neuron < FR_INTERMEDIATE; ++neuron)
        if (covered[neuron] != 1) exact = 0;
    check(exact, "first through last neurons are covered once without overlap");
    check(FR_EXPERT_PARAMS == UINT64_C(121031884800), "all expert parameter count");
    check(FR_EXPERT_BYTES == UINT64_C(68080435200), "Q4_0 bytes include scales");
    check(FR_WEIGHT_BLOCK_BYTES == UINT64_C(3404021760), "one block across all experts/layers");
    check(FR_WEIGHT_BLOCK_BYTES * FR_BLOCKS == FR_EXPERT_BYTES, "partition loses no bytes");
    check(FR_MAX_BURST == 8 && FR_BENCH_BURST_MAX == 4, "binding burst limits");
    return test_end();
}

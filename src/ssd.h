#ifndef FR_SSD_H
#define FR_SSD_H
#include "native.h"
#include <stdatomic.h>
#define FR_SSD_WORKERS 8
#define FR_SSD_DEFAULT_CACHE (UINT64_C(256)<<20)
#define FR_SSD_MAX_BLOCKS 2
typedef struct fr_ssd fr_ssd;
typedef struct {
    /* read_ns is the parallel I/O window, including ready callbacks submitted
     * while other reads run; total_ns includes cache lookup and staging. */
    uint64_t hits,misses,bytes,read_ns,total_ns;
} fr_ssd_sample;
/* Owned, bounded expert cache. Reads never expose incomplete entries. Sources
 * are immutable verified native units; callers serialize load/attach/detach. */
fr_ssd *fr_ssd_open(uint64_t bytes,unsigned workers);
int fr_ssd_attach(fr_ssd *s,unsigned block,int fd);
/* Calibration only: a physically written native layer repeated logically. */
int fr_ssd_attach_fixture(fr_ssd *s,unsigned block,int fd);
int fr_ssd_detach(fr_ssd *s,unsigned block);
int fr_ssd_load(fr_ssd *s,const fr_req *req,uint32_t mask,void *layers[FR_BLOCKS]);
/* Called on the load caller, never an I/O thread, once each block's staging
 * is complete. Masks are disjoint and staging stays immutable until return.
 * On callback/read failure all reads are drained; callers must drain any GPU
 * commands they submitted before reusing staging or publishing an output. */
typedef int (*fr_ssd_ready_fn)(void *context,uint32_t ready_mask);
int fr_ssd_load_ready(fr_ssd *s,const fr_req *req,uint32_t mask,void *layers[FR_BLOCKS],fr_ssd_ready_fn ready,void *context);
void fr_ssd_sample_get(const fr_ssd *s,fr_ssd_sample *out);
void fr_ssd_clear(fr_ssd *s); /* cold-cache calibration; never purges OS caches */
void fr_ssd_close(fr_ssd *s);
int fr_ssd_fixture_file(const char *dir); /* unlinked, fully written synthetic layer */
#endif

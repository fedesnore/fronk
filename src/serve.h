/* serve.h - the head's expert service for ds4 and the REPORT phase.
 * ds4 sends each layer's routed+shared expert request: linked into this
 * program (`ds4 builtin`) through fronk_moe_hook, or as a child process over a
 * unix socket (client/fronk_client.h); requests wait while a re-plan runs.
 * Wire: u32 magic 'FRNK', u32 layer, u32 burst, u16 expert[burst][10],
 * f32 gate[burst][11], f32 x[burst][2560] -> i32 status, u32 missing, f32 y[burst][2560].
 */
#ifndef FR_SERVE_H
#define FR_SERVE_H
#include "config.h"
#include "plane.h"
#include "head.h"
#define FR_SERVE_MAGIC 0x4b4e5246u
int fr_serve_start(const fr_config *cfg);
int fr_serve_attach(fr_head_plane *h,const fr_connection *links,const fr_plan *plan);
void fr_serve_detach(void);
/* Test generation, REPORT, then chat until [m] (returns 1) or [q] (returns 0). */
int fr_serve_session(const fr_config *cfg,const fr_connection *links,const fr_plan *plan,fr_head_plane *h);
void fr_serve_stop(void);
/* fronk demo: decode and prompt reading on synthetic units; prints the estimate. */
int fr_serve_demo(const fr_config *cfg,const fr_connection *links,const fr_plan *plan,fr_head_plane *h,unsigned passes);
/* Until ds4 answers HTTP, sends its first expert request, or 10 minutes pass. */
int fr_serve_wait_ds4(int allow_pending_expert);
#endif

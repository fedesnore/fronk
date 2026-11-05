/* Startup policy for the ordinary head. Explicit environment overrides are
 * retained for reproducible diagnostics; no policy lookup runs per token. */
#ifndef FR_RUNTIME_H
#define FR_RUNTIME_H
int fr_head_defaults(void);
#endif

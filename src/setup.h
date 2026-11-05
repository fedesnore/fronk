/* setup.h - "fronk backends": choose this machine's compute backend. */
#ifndef FR_SETUP_H
#define FR_SETUP_H
#include <stddef.h>
int setup_main(int argc,char **argv);
/* BENCH cache identity: the SHA-256 of the binary at ARGV0 plus the hardware. */
void setup_cache_identity(const char *argv0);
/* The choice saved by "fronk backends" in DIR/backend; "" when none (auto). */
int setup_saved_backend(const char *dir,char *out,size_t size);
#endif

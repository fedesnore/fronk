/* identity.c - publish a random 64-bit ID once using an atomic hard link.
 * Readers never see a partially written first-start file. A corrupt existing
 * identity is an error, not a reason to silently become a different machine.
 */
#include "identity.h"
#include "auth.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static int read_id(int dir, uint64_t *out) {
    int fd=openat(dir,"node.id",O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC);
    if (fd<0) return -1;
    struct stat st; char text[18]; ssize_t n;
    int ok=!fstat(fd,&st) && S_ISREG(st.st_mode) && st.st_size==17;
    do { n=read(fd,text,sizeof(text)); } while (n<0 && errno==EINTR);
    if (close(fd)) ok=0;
    if (!ok || n!=17 || text[16]!='\n') { errno=EINVAL; return -1; }
    text[16]=0;
    if (identity_parse(text,out)) { errno=EINVAL; return -1; }
    return 0;
}
int identity_load(const char *directory, uint64_t *out) {
    char path[1024]; const char *dir=directory;
    if (!dir[0]) {
        const char *home=getenv("HOME");
        if (!home || !home[0]) return -1;
        int n=snprintf(path,sizeof(path),"%s/.fronk",home);
        if (n<0 || (size_t)n>=sizeof(path)) return -1;
        dir=path;
    }
    if (mkdir(dir,0700) && errno!=EEXIST) return -1;
    int fd=open(dir,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC); if (fd<0) return -1;
    int rc=read_id(fd,out);
    if (!rc || errno!=ENOENT) { close(fd); return rc; }
    auth_state random={0}; char hex[33], temp[64], text[18]; uint64_t id;
    if (auth_nonce(&random,hex)) { close(fd); return -1; }
    hex[16]=0;
    if (identity_parse(hex,&id)) { auth_reset(&random); close(fd); return -1; }
    snprintf(temp,sizeof(temp),".node-%ld-%s",(long)getpid(),hex);
    snprintf(text,sizeof(text),"%016" PRIx64 "\n",id); auth_reset(&random);
    int output=openat(fd,temp,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if (output<0) { close(fd); return -1; }
    size_t used=0; rc=0;
    while (used<17) {
        ssize_t n=write(output,text+used,17-used);
        if (n<0 && errno==EINTR) continue;
        if (n<=0) { rc=-1; break; }
        used+=(size_t)n;
    }
    if (!rc && fsync(output)) rc=-1;
    if (close(output)) rc=-1;
    if (!rc && linkat(fd,temp,fd,"node.id",0) && errno!=EEXIST) rc=-1;
    if (unlinkat(fd,temp,0)) rc=-1;
    if (!rc) rc=read_id(fd,out);
    if (close(fd)) rc=-1;
    return rc;
}

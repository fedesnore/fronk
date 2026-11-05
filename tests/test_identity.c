/* test_identity.c - persistence, isolation, invalid storage and concurrent start. */
#include "test.h"
#include "../src/identity.h"
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>
int main(void) {
    test_begin("identity"); uint64_t a=0,b=0;
    check(!identity_parse("0123456789abcdef",&a) && a==UINT64_C(0x0123456789abcdef),"exact sixteen-digit ID");
    check(identity_parse("0000000000000000",&a)<0 && identity_parse("123",&a)<0 && identity_parse("0123456789abcdeg",&a)<0,"zero, short and malformed IDs rejected");
    char dir[]="/tmp/fronk-id.XXXXXX", path[256], other[128];
    check(mkdtemp(dir)!=NULL,"private test state directory");
    snprintf(path,sizeof(path),"%s/node.id",dir); snprintf(other,sizeof(other),"%s/other",dir);
    check(!identity_load(dir,&a) && a,"generate persistent random ID");
    check(!identity_load(dir,&b) && a==b,"restart retains identity");
    struct stat st;
    check(!stat(path,&st) && (st.st_mode&0777)==0600 && st.st_size==17,"private canonical ID file");
    check(!identity_load(other,&b) && a!=b,"isolated test node has different identity");
    check(!unlink(path),"remove owned test ID before race");
    int p[2]; check(!pipe(p),"race result pipe");
    fflush(NULL); pid_t child=fork();
    if (!child) { close(p[0]); uint64_t id; int ok=!identity_load(dir,&id) && write(p[1],&id,sizeof(id))==sizeof(id); close(p[1]); _exit(!ok); }
    close(p[1]); int rc=identity_load(dir,&a), status=0;
    ssize_t n=read(p[0],&b,sizeof(b)); close(p[0]);
    check(child>0 && waitpid(child,&status,0)==child && !status && !rc && n==sizeof(b) && a==b,"concurrent first starts converge on one ID");
    FILE *f=fopen(path,"w"); int written=f && fputs("corrupt\n",f)>=0 && !fclose(f);
    check(written && identity_load(dir,&a)<0,"corrupt existing ID fails instead of replacing identity");
    check(!unlink(path) && !symlink("/dev/null",path) && identity_load(dir,&a)<0,"ID symlinks rejected");
    check(!unlink(path),"remove owned test symlink");
    snprintf(path,sizeof(path),"%s/node.id",other);
    check(!unlink(path) && !rmdir(other) && !rmdir(dir),"test storage cleaned");
    return test_end();
}

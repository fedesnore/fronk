#include "test.h"
#include "../src/vectors.h"
#include "../src/sha256.h"
#include <string.h>
#include <unistd.h>

static int wrong_run(const fr_req *req, fr_part *out) {
    memset(out,0,sizeof(*out)); out->burst=req->burst; out->layer=req->layer; return 0;
}
int main(void) {
    test_begin("vectors");
    check(fr_ref.self_test(NULL)==FR_INVALID,"null directory rejected");
    check(fr_ref.self_test("tests/vectors/absent")==FR_IO,"missing golden corpus fails");
    check(!fr_ref.self_test("tests/vectors"),"ref passes every committed golden partial");
    check(!fr_ref_q8.self_test("tests/vectors"),"Q8 variant passes fixed backend tolerance");
    fr_backend broken=fr_ref; broken.run=wrong_run;
    check(fr_vectors_check(&broken,"tests/vectors",&fr_backend_tolerance,NULL)==FR_MISMATCH,"zero-output backend cannot pass nonzero corpus");
    char dir[]="/tmp/fronk-vectors-XXXXXX"; char *created=mkdtemp(dir);
    check(created!=NULL,"create corruption fixture directory");
    if (created) {
        char path[256]; snprintf(path,sizeof(path),"%s/SHA256SUMS",dir);
        FILE *f=fopen(path,"w");
        if (f) { fputs("bad zero-input.txt\n",f); fclose(f); }
        check(f && fr_ref.self_test(dir)==FR_IO,"missing listed vector fails");
        char vector_path[256]; snprintf(vector_path,sizeof(vector_path),"%s/zero-input.txt",dir);
        f=fopen(vector_path,"w"); if (f) { fputs("corrupt\n",f); fclose(f); }
        check(f && fr_ref.self_test(dir)==FR_MISMATCH,"corrupt checksum fails");
        /* A matching checksum must not hide a malformed/truncated vector. */
        unsigned char hash[32]; sha256("corrupt\n",8,hash);
        f=fopen(path,"w");
        if (f) { for (int i=0; i<32; ++i) fprintf(f,"%02x",hash[i]); fputs("  zero-input.txt\n",f); fclose(f); }
        check(f && fr_ref.self_test(dir)==FR_MISMATCH,"matching hash cannot bypass format validation");
        check(fr_vectors_generate(dir)==FR_IO,"generator refuses to overwrite frozen files");
        unlink(vector_path); unlink(path); rmdir(dir);
    }
    /* All failure paths above must leave no loaded fixtures behind. */
    check(!fr_ref.self_test("tests/vectors"),"self-test remains usable after failures");
    return test_end();
}

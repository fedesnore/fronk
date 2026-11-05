/* Read-only release audit: compare every emitted unit byte with an independent
 * reverse mapping into GGUF tensor rows/columns, then check the pinned SHA-256.
 * No unit files are written and no saved manifest can skip the computation.
 * This intentionally does expensive work; it is not an installation/reconnect
 * step. The installer already hashes the full source once and checks unit
 * hashes during preparation/transfer.
 */
#include "../src/prep.h"
#include "../src/sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const fr_gguf *model;
    unsigned unit;
    uint64_t position;
    sha256_ctx hash;
} audit;

/* Reverse mapping is expressed in tensor coordinates, independently of the
 * extractor's copy loops. Each returned extent stays inside one contiguous row
 * or column block in the source tensor. */
static const unsigned char *source_at(const audit *a,size_t *extent) {
    const fr_gguf *g=a->model;
    uint64_t layer=a->position/FR_NATIVE_LAYER_BYTES;
    uint64_t local=a->position%FR_NATIVE_LAYER_BYTES;
    unsigned shared=local>=(uint64_t)FR_ROUTED_EXPERTS*FR_NATIVE_SLICE_BYTES;
    uint64_t expert=shared?0:local/FR_NATIVE_SLICE_BYTES;
    uint64_t within=shared?local-(uint64_t)FR_ROUTED_EXPERTS*FR_NATIVE_SLICE_BYTES:local%FR_NATIVE_SLICE_BYTES;
    uint64_t row_bytes=shared?2720:1440, column_bytes=shared?34:17;
    uint64_t projection=32*row_bytes, source;
    if (within<2*projection) {
        unsigned up=within>=projection;
        uint64_t row_part=within%projection;
        source=(shared?(up?g->sup[layer]:g->sgate[layer]):(up?g->up[layer]:g->gate[layer]))+
            (expert*640+a->unit*32)*row_bytes+row_part;
        *extent=(size_t)(projection-row_part);
    } else {
        uint64_t column_part=within-2*projection;
        uint64_t row=column_part/column_bytes, byte=column_part%column_bytes;
        source=(shared?g->sdown[layer]:g->down[layer])+
            (expert*2560+row)*20*column_bytes+a->unit*column_bytes+byte;
        *extent=(size_t)(column_bytes-byte);
    }
    return g->map+source;
}
static int inspect(void *context,const void *data,size_t bytes) {
    audit *a=context;
    const unsigned char *p=data;
    if (bytes>FR_NATIVE_BLOCK_BYTES-a->position || sha256_update(&a->hash,data,bytes)) return -1;
    while (bytes) {
        size_t extent;
        const unsigned char *source=source_at(a,&extent);
        if (extent>bytes) extent=bytes;
        if (memcmp(source,p,extent)) {
            fprintf(stderr,"unit %u differs at native byte %llu\n",a->unit,(unsigned long long)a->position);
            return -1;
        }
        a->position+=extent; p+=extent; bytes-=extent;
    }
    return 0;
}
int main(int argc,char **argv) {
    if (argc!=3) { fprintf(stderr,"usage: %s MODEL.gguf model-units.sha256\n",argv[0]); return 2; }
    FILE *manifest=fopen(argv[2],"r");
    char line[160],expected[FR_BLOCKS][65]; unsigned unit;
    if (!manifest || !fgets(line,sizeof(line),manifest) || strncmp(line,"gguf-sha256 ",12)) return 1;
    for (unsigned b=0;b<FR_BLOCKS;++b) {
        if (!fgets(line,sizeof(line),manifest) || sscanf(line,"unit-%u %64s",&unit,expected[b])!=2 || unit!=b || strlen(expected[b])!=64) return 1;
    }
    if (fgets(line,sizeof(line),manifest) || fclose(manifest)) return 1;
    fr_gguf model;
    if (fr_gguf_open(argv[1],&model)) { fputs("Invalid/unsupported GGUF tensor layout\n",stderr); return 1; }
    int result=0;
    audit units[FR_BLOCKS];
    unsigned char *slice=malloc(FR_NATIVE_SHARED_BYTES);
    if (!slice) { fr_gguf_close(&model); return 1; }
    for (unsigned b=0;b<FR_BLOCKS;++b) {
        units[b]=(audit){.model=&model,.unit=b}; sha256_init(&units[b].hash);
    }
    /* Visit every tensor once, keeping one expert's source rows hot while all
     * twenty units gather from it. Each hash still sees native unit order.
     * Unit-major traversal would reread strided down-matrix pages 20 times. */
    for (unsigned layer=0;!result && layer<FR_LAYERS;++layer) {
        for (unsigned expert=0;!result && expert<FR_EXPERTS;++expert) {
            for (unsigned b=0;b<FR_BLOCKS;++b) {
                size_t size=fr_slice_copy(&model,b,layer,expert,slice,FR_NATIVE_SHARED_BYTES);
                if (!size || inspect(&units[b],slice,size)) { result=1; break; }
            }
        }
        printf("layer %u/%u checked for all 20 units\n",layer+1,FR_LAYERS); fflush(stdout);
    }
    for (unsigned b=0;!result && b<FR_BLOCKS;++b) {
        unsigned char digest[32]; char hex[65];
        if (units[b].position!=FR_NATIVE_BLOCK_BYTES || sha256_final(&units[b].hash,digest)) { result=1; break; }
        for (unsigned i=0;i<32;++i) snprintf(hex+2*i,3,"%02x",digest[i]);
        if (strcmp(hex,expected[b])) { fprintf(stderr,"unit %u digest does not match package\n",b); result=1; break; }
        printf("unit-%02u neurons=%u..%u bytes=%llu sha256=%s exact-source=yes\n",b,b*32,b*32+31,(unsigned long long)units[b].position,hex);
        fflush(stdout);
    }
    free(slice);
    fr_gguf_close(&model);
    if (!result) puts("PASS: all 20 units match source tensor coordinates and the pinned unit manifest; source GGUF digest must be verified separately.");
    return result;
}

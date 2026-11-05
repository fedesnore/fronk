/* Qualification candidate: one Q4_K output row, several token columns.
 * Same superblock FMA/minimum order as q4k_pair. No repacking or weight copy.
 * Opt-in backend build; qualify full requests as well as the kernel probe.
 * Explicit unrolling prevents GCC from walking accumulator arrays dynamically. */
AVX2 static inline __attribute__((always_inline)) void q4k_row_batch(
    const unsigned char *p,const q8k *x,const unsigned *pos,unsigned count,float *out) {
    const __m256i mask=_mm256_set1_epi8(15);
    __m256 acc[8]; __m128 mins[8];
    #pragma GCC unroll 8
    for(unsigned i=0;i<count;i++) {acc[i]=_mm256_setzero_ps();mins[i]=_mm_setzero_ps();}
    for(unsigned sb=0;sb<FR_HIDDEN/256;sb++) {
        const unsigned char *b=p+sb*144;
#if FR_Q4K_PREFETCH
        _mm_prefetch((const char *)(b+FR_Q4K_PREFETCH*144),_MM_HINT_T0);
        _mm_prefetch((const char *)(b+FR_Q4K_PREFETCH*144+64),_MM_HINT_T0);
#endif
        uint32_t half;memcpy(&half,b,4);
        const __m128 f=_mm_cvtph_ps(_mm_cvtsi32_si128((int)half));
        const __m128i packed=q4k_scales(b+4);
        const __m256i scales=_mm256_broadcastsi128_si256(_mm_cvtepu8_epi16(packed));
        const __m128i minimum=_mm_cvtepu8_epi16(_mm_srli_si128(packed,8));
        __m256i dot[8];
        #pragma GCC unroll 8
        for(unsigned i=0;i<count;i++) dot[i]=_mm256_setzero_si256();
        for(unsigned gp=0;gp<4;gp++) {
            const __m256i q=_mm256_loadu_si256((const __m256i *)(b+16+gp*32));
            const __m256i lo=_mm256_and_si256(q,mask),hi=_mm256_and_si256(_mm256_srli_epi16(q,4),mask);
            const __m256i sl=_mm256_shuffle_epi8(scales,_mm256_set1_epi16((short)(4*gp+(4*gp+1)*256)));
            const __m256i sh=_mm256_shuffle_epi8(scales,_mm256_set1_epi16((short)(4*gp+2+(4*gp+3)*256)));
            #pragma GCC unroll 8
        for(unsigned i=0;i<count;i++) {
                const int8_t *v=x->q[pos[i]]+sb*256+gp*64;
                dot[i]=_mm256_add_epi32(dot[i],_mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_maddubs_epi16(lo,_mm256_loadu_si256((const __m256i *)v)),sl),
                    _mm256_madd_epi16(_mm256_maddubs_epi16(hi,_mm256_loadu_si256((const __m256i *)(v+32))),sh)));
            }
        }
        #pragma GCC unroll 8
        for(unsigned i=0;i<count;i++) {
            const float xd=x->d[pos[i]][sb];
            const __m128i m=_mm_madd_epi16(minimum,_mm_loadu_si128((const __m128i *)(x->s[pos[i]]+sb*8)));
            acc[i]=_mm256_fmadd_ps(_mm256_cvtepi32_ps(dot[i]),_mm256_set1_ps(_mm_cvtss_f32(f)*xd),acc[i]);
            mins[i]=_mm_fmadd_ps(_mm_cvtepi32_ps(m),_mm_set1_ps(_mm_cvtss_f32(_mm_movehdup_ps(f))*xd),mins[i]);
        }
    }
    #pragma GCC unroll 8
    for(unsigned i=0;i<count;i++) {
        __m128 m=_mm_hadd_ps(mins[i],mins[i]);m=_mm_hadd_ps(m,m);
        out[i]=native_sum(acc[i])-_mm_cvtss_f32(m);
    }
}
AVX2 static void q4k_row3(const unsigned char *p,const q8k *x,const unsigned *pos,float *out) {q4k_row_batch(p,x,pos,3,out);}
AVX2 static void q4k_row4(const unsigned char *p,const q8k *x,const unsigned *pos,float *out) {q4k_row_batch(p,x,pos,4,out);}
AVX2 static void q4k_row8(const unsigned char *p,const q8k *x,const unsigned *pos,float *out) {q4k_row_batch(p,x,pos,8,out);}

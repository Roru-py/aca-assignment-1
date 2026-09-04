// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"
//ezsum helper function for simd operations (same as before)
static inline float ezsum(__m256 x) {
        __m128 lo = _mm256_castps256_ps128(x);
        lo = _mm_add_ps(lo,_mm256_extractf128_ps(x,1));

        __m128 mid = _mm_movehdup_ps(lo);
        lo = _mm_add_ps(lo,mid);
        mid = _mm_movehl_ps(mid,lo);
        lo = _mm_add_ss(lo,mid);
        return _mm_cvtss_f32(lo);
    }

//most are same as prefetch.cpp because it already implements prefetching + simd + tiling + unrolling
//we start by pasting prefetch and then modifying it to get a 25x speedup form 20x
void matmul_optimized(const float* __restrict A, const float* __restrict B, float* __restrict C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
        const int tile_len=256; //256 chosen cos its best for this computer
        const int pf_dist=64;
        
        for(long j0=0;j0<N;j0+=tile_len) {
            long jend= j0+tile_len<(long)N ? j0+tile_len : (long)N;
            long i=0;
            for(;i<=M-6;i+=6) {
                const float* __restrict a0 = A + i*lda;
                const float* __restrict a1 = A + (i+1)*lda;
                const float* __restrict a2 = A + (i+2)*lda;
                const float* __restrict a3 = A + (i+3)*lda;
                const float* __restrict a4 = A + (i+4)*lda;
                const float* __restrict a5 = A + (i+5)*lda;

                long j=j0;
                for(;j<=jend-2;j+=2) {
                    
                    __m256 acc00 = _mm256_setzero_ps();
                    __m256 acc01 = _mm256_setzero_ps();
                    __m256 acc10 = _mm256_setzero_ps();
                    __m256 acc11 = _mm256_setzero_ps();
                    __m256 acc20 = _mm256_setzero_ps();
                    __m256 acc21 = _mm256_setzero_ps();
                    __m256 acc30 = _mm256_setzero_ps();
                    __m256 acc31 = _mm256_setzero_ps();
                    __m256 acc40 = _mm256_setzero_ps();
                    __m256 acc41 = _mm256_setzero_ps();
                    __m256 acc50 = _mm256_setzero_ps();
                    __m256 acc51 = _mm256_setzero_ps();

                    
                    const float* __restrict b0 = B + j*ldb;
                    const float* __restrict b1 = B + (j+1)*ldb;

                    long k=0;
                    #pragma GCC unroll 8
                    for(;k<=K-16;k+=16) {
                    //prefetch no matter the bound (if out of bound then cpu discards)
                        _mm_prefetch((const char*)(b0+k+pf_dist),_MM_HINT_T0);
                        _mm_prefetch((const char*)(b1+k+pf_dist),_MM_HINT_T0);
                        //first 8 remains same as before
                        __m256 bv0 = _mm256_loadu_ps(b0+k);
                        __m256 bv1 = _mm256_loadu_ps(b1+k);

                        __m256 av = _mm256_loadu_ps(a0+k);
                        acc00 = _mm256_fmadd_ps(av, bv0, acc00);
                        acc01 = _mm256_fmadd_ps(av, bv1, acc01);
                        av = _mm256_loadu_ps(a1+k);
                        acc10 = _mm256_fmadd_ps(av, bv0, acc10);
                        acc11 = _mm256_fmadd_ps(av, bv1, acc11);
                        av = _mm256_loadu_ps(a2+k);
                        acc20 = _mm256_fmadd_ps(av, bv0, acc20);
                        acc21 = _mm256_fmadd_ps(av, bv1, acc21);
                        av = _mm256_loadu_ps(a3+k);
                        acc30 = _mm256_fmadd_ps(av, bv0, acc30);
                        acc31 = _mm256_fmadd_ps(av, bv1, acc31);
                        av = _mm256_loadu_ps(a4+k);
                        acc40 = _mm256_fmadd_ps(av, bv0, acc40);
                        acc41 = _mm256_fmadd_ps(av, bv1, acc41);
                        av = _mm256_loadu_ps(a5+k);
                        acc50 = _mm256_fmadd_ps(av, bv0, acc50);
                        acc51 = _mm256_fmadd_ps(av, bv1, acc51);

                        //we also add the next 8 operations cos we changed unroll to 16
                        bv0 = _mm256_loadu_ps(b0+k+8);
                        bv1 = _mm256_loadu_ps(b1+k+8);

                        av = _mm256_loadu_ps(a0+k+8);
                        acc00 = _mm256_fmadd_ps(av, bv0, acc00);
                        acc01 = _mm256_fmadd_ps(av, bv1, acc01);
                        av = _mm256_loadu_ps(a1+k+8);
                        acc10 = _mm256_fmadd_ps(av, bv0, acc10);
                        acc11 = _mm256_fmadd_ps(av, bv1, acc11);
                        av = _mm256_loadu_ps(a2+k+8);
                        acc20 = _mm256_fmadd_ps(av, bv0, acc20);
                        acc21 = _mm256_fmadd_ps(av, bv1, acc21);
                        av = _mm256_loadu_ps(a3+k+8);
                        acc30 = _mm256_fmadd_ps(av, bv0, acc30);
                        acc31 = _mm256_fmadd_ps(av, bv1, acc31);
                        av = _mm256_loadu_ps(a4+k+8);
                        acc40 = _mm256_fmadd_ps(av, bv0, acc40);
                        acc41 = _mm256_fmadd_ps(av, bv1, acc41);
                        av = _mm256_loadu_ps(a5+k+8);
                        acc50 = _mm256_fmadd_ps(av, bv0, acc50);
                        acc51 = _mm256_fmadd_ps(av, bv1, acc51);
                    }
                    

                    float s00 = ezsum(acc00); 
                    float s01 = ezsum(acc01);
                    float s10 = ezsum(acc10);
                    float s11 = ezsum(acc11);
                    float s20 = ezsum(acc20); 
                    float s21 = ezsum(acc21);
                    float s30 = ezsum(acc30); 
                    float s31 = ezsum(acc31);
                    float s40 = ezsum(acc40); 
                    float s41 = ezsum(acc41);
                    float s50 = ezsum(acc50); 
                    float s51 = ezsum(acc51);

                    for(;k<K;k++) {
                        float bb0=b0[k]; float bb1=b1[k];
                        s00+=a0[k]*bb0; s01+=a0[k]*bb1;
                        s10+=a1[k]*bb0; s11+=a1[k]*bb1;
                        s20+=a2[k]*bb0; s21+=a2[k]*bb1;
                        s30+=a3[k]*bb0; s31+=a3[k]*bb1;
                        s40+=a4[k]*bb0; s41+=a4[k]*bb1;
                        s50+=a5[k]*bb0; s51+=a5[k]*bb1;
                    }
                    C[i*ldc+j]=s00;
                    C[i*ldc+j+1]=s01;
                    C[(i + 1)*ldc+j]=s10;
                    C[(i + 1)*ldc+j+1]=s11;
                    C[(i + 2)*ldc+j]=s20;
                    C[(i + 2)*ldc+j+1]=s21;
                    C[(i + 3)*ldc+j]=s30;
                    C[(i + 3)*ldc+j+1]=s31;
                    C[(i + 4)*ldc+j]=s40;
                    C[(i + 4)*ldc+j+1]=s41;
                    C[(i + 5)*ldc+j]=s50;
                    C[(i + 5)*ldc+j+1]=s51;
                }
                for(;j<jend;j++) {
                    const float* __restrict b=B+j*ldb;

                    __m256 acc0=_mm256_setzero_ps(); __m256 acc1=_mm256_setzero_ps();
                    __m256 acc2=_mm256_setzero_ps(); __m256 acc3=_mm256_setzero_ps();
                    __m256 acc4=_mm256_setzero_ps(); __m256 acc5=_mm256_setzero_ps();

                    long k=0;
                    //prefetch again without bounds to avoid if branch fault
                    for(;k<=K-16;k+=16) {
                        _mm_prefetch((const char*)(b+k+pf_dist),_MM_HINT_T0);
                        __m256 bv=_mm256_loadu_ps(b+k);
                        acc0=_mm256_fmadd_ps(_mm256_loadu_ps(a0+k),bv,acc0);
                        acc1=_mm256_fmadd_ps(_mm256_loadu_ps(a1+k),bv,acc1);
                        acc2=_mm256_fmadd_ps(_mm256_loadu_ps(a2+k),bv,acc2);
                        acc3=_mm256_fmadd_ps(_mm256_loadu_ps(a3+k),bv,acc3);
                        acc4=_mm256_fmadd_ps(_mm256_loadu_ps(a4+k),bv,acc4);
                        acc5=_mm256_fmadd_ps(_mm256_loadu_ps(a5+k),bv,acc5);

                        bv=_mm256_loadu_ps(b+k+8);
                        acc0=_mm256_fmadd_ps(_mm256_loadu_ps(a0+k+8),bv,acc0);
                        acc1=_mm256_fmadd_ps(_mm256_loadu_ps(a1+k+8),bv,acc1);
                        acc2=_mm256_fmadd_ps(_mm256_loadu_ps(a2+k+8),bv,acc2);
                        acc3=_mm256_fmadd_ps(_mm256_loadu_ps(a3+k+8),bv,acc3);
                        acc4=_mm256_fmadd_ps(_mm256_loadu_ps(a4+k+8),bv,acc4);
                        acc5=_mm256_fmadd_ps(_mm256_loadu_ps(a5+k+8),bv,acc5);
                    }
                    float s0=ezsum(acc0); float s1=ezsum(acc1);
                    float s2=ezsum(acc2); float s3=ezsum(acc3);
                    float s4=ezsum(acc4); float s5=ezsum(acc5);

                    for (;k<K;k++) {
                        float bb=b[k];
                        s0+=a0[k]*bb; s1+=a1[k]*bb;
                        s2+=a2[k]*bb; s3+=a3[k]*bb; 
                        s4+=a4[k]*bb; s5+=a5[k]*bb;
                    }

                    C[i*ldc+j]=s0;
                    C[(i+1)*ldc+j]=s1;
                    C[(i+2)*ldc+j]=s2;
                    C[(i+3)*ldc+j]=s3;
                    C[(i+4)*ldc+j]=s4;
                    C[(i+5)*ldc+j]=s5;
                }
            }
            for(;i<M;i++) {
                for(long j=j0;j<jend;j++) {
                    long row_A=i*lda;
                    long row_B=j*ldb;
                    
                    __m256 acc=_mm256_setzero_ps();
                    long k=0;
                    //prefetch again (same as above - without bounds like before)
                    for(;k<=K-16;k+=16) {
                        _mm_prefetch((const char*)(B+row_B+k+pf_dist),_MM_HINT_T0);
                        __m256 a_vec=_mm256_loadu_ps(A+row_A+k);
                        __m256 b_vec=_mm256_loadu_ps(B+row_B+k);
                        acc=_mm256_fmadd_ps(a_vec,b_vec,acc);

                        a_vec=_mm256_loadu_ps(A+row_A+k+8);
                        b_vec=_mm256_loadu_ps(B+row_B+k+8);
                        acc=_mm256_fmadd_ps(a_vec,b_vec,acc);
                    }
                    
                    float sum=ezsum(acc);
                    for (;k<K;k++) {
                        sum+=A[row_A+k]*B[row_B+k];
                    }
                    C[i*ldc+j]=sum;
                }    
            }
        }
}

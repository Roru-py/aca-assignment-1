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
void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
        const int tile_len=128; //128 chosen cos its best for this computer
        const int pf_dist=64;
        
        for(long j0=0;j0<N;j0+=tile_len) {
            long jend= j0+tile_len<(long)N ? j0+tile_len : (long)N;
            long i=0;
            for(;i<=M-3;i+=3) {
                const float* a0 = A + i*lda;
                const float* a1 = A + (i+1)*lda;
                const float* a2 = A + (i+2)*lda;
                // const float* a3 = A + (i+3)*lda;
                // const float* a4 = A + (i+4)*lda;
                // const float* a5 = A + (i+5)*lda;

                long j=j0;
                for(;j<=jend-4;j+=4) {
                    
                    __m256 acc00 = _mm256_setzero_ps();
                    __m256 acc01 = _mm256_setzero_ps();
                    __m256 acc02 = _mm256_setzero_ps();
                    __m256 acc03 = _mm256_setzero_ps();
    
                    __m256 acc10 = _mm256_setzero_ps();
                    __m256 acc11 = _mm256_setzero_ps();
                    __m256 acc12 = _mm256_setzero_ps();
                    __m256 acc13 = _mm256_setzero_ps();
                    
                    __m256 acc20 = _mm256_setzero_ps();
                    __m256 acc21 = _mm256_setzero_ps();
                    __m256 acc22 = _mm256_setzero_ps();
                    __m256 acc23 = _mm256_setzero_ps();

                    
                    const float* b0 = B + j*ldb;
                    const float* b1 = B + (j+1)*ldb;
                    const float* b2 = B + (j+2)*ldb;
                    const float* b3 = B + (j+3)*ldb;

                    long k=0;
                    for(;k<=K-16;k+=16) {
                    //prefetch no matter the bound (if out of bound then cpu discards)
                        _mm_prefetch((const char*)(b0+k+pf_dist),_MM_HINT_T0);
                        _mm_prefetch((const char*)(b1+k+pf_dist),_MM_HINT_T0);
                        _mm_prefetch((const char*)(b2+k+pf_dist),_MM_HINT_T0);
                        _mm_prefetch((const char*)(b3+k+pf_dist),_MM_HINT_T0);
                        //first 8 remains same as before
                        __m256 av0 = _mm256_loadu_ps(a0 + k);
                        __m256 av1 = _mm256_loadu_ps(a1 + k);
                        __m256 av2 = _mm256_loadu_ps(a2 + k);

                        __m256 bv = _mm256_loadu_ps(b0+k);
                        acc00 = _mm256_fmadd_ps(av0, bv, acc00);
                        acc10 = _mm256_fmadd_ps(av1, bv, acc10);
                        acc20 = _mm256_fmadd_ps(av2, bv, acc20);

                        bv = _mm256_loadu_ps(b1+k);
                        acc01 = _mm256_fmadd_ps(av0, bv, acc01);
                        acc11 = _mm256_fmadd_ps(av1, bv, acc11);
                        acc21 = _mm256_fmadd_ps(av2, bv, acc21);

                        bv = _mm256_loadu_ps(b2+k);
                        acc02 = _mm256_fmadd_ps(av0, bv, acc02);
                        acc12 = _mm256_fmadd_ps(av1, bv, acc12);
                        acc22 = _mm256_fmadd_ps(av2, bv, acc22);

                        bv = _mm256_loadu_ps(b3+k);
                        acc03 = _mm256_fmadd_ps(av0, bv, acc03);
                        acc13 = _mm256_fmadd_ps(av1, bv, acc13);
                        acc23 = _mm256_fmadd_ps(av2, bv, acc23);

                        //we also add the next 8 operations cos we changed unroll to 16
                        av0 = _mm256_loadu_ps(a0+k+8);
                        av1 = _mm256_loadu_ps(a1+k+8);
                        av2 = _mm256_loadu_ps(a2+k+8);

                        bv = _mm256_loadu_ps(b0+k+8);
                        acc00 = _mm256_fmadd_ps(av0, bv, acc00);
                        acc10 = _mm256_fmadd_ps(av1, bv, acc10);
                        acc20 = _mm256_fmadd_ps(av2, bv, acc20);

                        bv = _mm256_loadu_ps(b1+k+8);
                        acc01 = _mm256_fmadd_ps(av0, bv, acc01);
                        acc11 = _mm256_fmadd_ps(av1, bv, acc11);
                        acc21 = _mm256_fmadd_ps(av2, bv, acc21);

                        bv = _mm256_loadu_ps(b2+k+8);
                        acc02 = _mm256_fmadd_ps(av0, bv, acc02);
                        acc12 = _mm256_fmadd_ps(av1, bv, acc12);
                        acc22 = _mm256_fmadd_ps(av2, bv, acc22);

                        bv = _mm256_loadu_ps(b3+k+8);
                        acc03 = _mm256_fmadd_ps(av0, bv, acc03);
                        acc13 = _mm256_fmadd_ps(av1, bv, acc13);
                        acc23 = _mm256_fmadd_ps(av2, bv, acc23);
                    }
                        
                    float s00 = ezsum(acc00);
                    float s01 = ezsum(acc01); 
                    float s02 = ezsum(acc02); 
                    float s03 = ezsum(acc03);
                    float s10 = ezsum(acc10); 
                    float s11 = ezsum(acc11); 
                    float s12 = ezsum(acc12); 
                    float s13 = ezsum(acc13);
                    float s20 = ezsum(acc20); 
                    float s21 = ezsum(acc21); 
                    float s22 = ezsum(acc22); 
                    float s23 = ezsum(acc23);

                    for(;k<K;k++) {
                        float bb0 = b0[k]; 
                        float bb1 = b1[k]; 
                        float bb2 = b2[k]; 
                        float bb3 = b3[k];
                        s00 += a0[k]*bb0; 
                        s01 += a0[k]*bb1; 
                        s02 += a0[k]*bb2; 
                        s03 += a0[k]*bb3;
                        s10 += a1[k]*bb0; 
                        s11 += a1[k]*bb1; 
                        s12 += a1[k]*bb2; 
                        s13 += a1[k]*bb3;
                        s20 += a2[k]*bb0; 
                        s21 += a2[k]*bb1; 
                        s22 += a2[k]*bb2; 
                        s23 += a2[k]*bb3;
                    }
                    C[i*ldc+j] = s00;
                    C[i*ldc+j+1] = s01;
                    C[i*ldc+j+2] = s02;
                    C[i*ldc+j+3] = s03;
                    
                    C[(i+1)*ldc+j] = s10;
                    C[(i+1)*ldc+j+1] = s11;
                    C[(i+1)*ldc+j+2] = s12;
                    C[(i+1)*ldc+j+3] = s13;
                    
                    C[(i+2)*ldc+j] = s20;
                    C[(i+2)*ldc+j+1] = s21;
                    C[(i+2)*ldc+j+2] = s22;
                    C[(i+2)*ldc+j+3] = s23;
                }
                for(;j<jend;j++) {
                    const float* b=B+j*ldb;

                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();

                    long k=0;
                    //prefetch again without bounds to avoid if branch fault
                    for(;k<=K-16;k+=16) {
                        _mm_prefetch((const char*)(b+k+pf_dist),_MM_HINT_T0);
                        __m256 bv=_mm256_loadu_ps(b+k);
                        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + k), bv, acc0);
                        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + k), bv, acc1);
                        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + k), bv, acc2);

                        bv = _mm256_loadu_ps(b + k + 8);
                        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + k + 8), bv, acc0);
                        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + k + 8), bv, acc1);
                        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + k + 8), bv, acc2);
                    }
                    float s0=ezsum(acc0);
                    float s1=ezsum(acc1);
                    float s2=ezsum(acc2);

                    for (;k<K;k++) {
                        float bb=b[k];
                        s0+=a0[k]*bb;
                        s1+=a1[k]*bb;
                        s2+=a2[k]*bb; 
                    }

                    C[i*ldc+j]=s0;
                    C[(i+1)*ldc+j]=s1;
                    C[(i+2)*ldc+j]=s2;
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

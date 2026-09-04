// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

static inline float ezsum(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    lo = _mm_add_ps(lo,_mm256_extractf128_ps(x,1));

    __m128 mid = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo,mid);
    mid = _mm_movehl_ps(mid,lo);
    lo = _mm_add_ss(lo,mid);
    return _mm_cvtss_f32(lo);
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    long i=0;
    for(;i<=M-6;i+=6) {
        long j=0;
        for(;j<=N-2;j+=2) {
            
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

            const float* a0 = A + i*lda;
            const float* a1 = A + (i+1)*lda;
            const float* a2 = A + (i+2)*lda;
            const float* a3 = A + (i+3)*lda;
            const float* a4 = A + (i+4)*lda;
            const float* a5 = A + (i+5)*lda;
            const float* b0 = B + j*ldb;
            const float* b1 = B + (j+1)*ldb;

            long k=0;
            for(;k<=K-8;k+=8) {
                __m256 bv0 = _mm256_loadu_ps(b0+k);
                __m256 bv1 = _mm256_loadu_ps(b1+k);

                __m256 av0 = _mm256_loadu_ps(a0+k);
                acc00 = _mm256_fmadd_ps(av0, bv0, acc00);
                acc01 = _mm256_fmadd_ps(av0, bv1, acc01);
                __m256 av1 = _mm256_loadu_ps(a1+k);
                acc10 = _mm256_fmadd_ps(av1, bv0, acc10);
                acc11 = _mm256_fmadd_ps(av1, bv1, acc11);
                __m256 av2 = _mm256_loadu_ps(a2+k);
                acc20 = _mm256_fmadd_ps(av2, bv0, acc20);
                acc21 = _mm256_fmadd_ps(av2, bv1, acc21);
                __m256 av3 = _mm256_loadu_ps(a3+k);
                acc30 = _mm256_fmadd_ps(av3, bv0, acc30);
                acc31 = _mm256_fmadd_ps(av3, bv1, acc31);
                __m256 av4 = _mm256_loadu_ps(a4+k);
                acc40 = _mm256_fmadd_ps(av4, bv0, acc40);
                acc41 = _mm256_fmadd_ps(av4, bv1, acc41);
                __m256 av5 = _mm256_loadu_ps(a5+k);
                acc50 = _mm256_fmadd_ps(av5, bv0, acc50);
                acc51 = _mm256_fmadd_ps(av5, bv1, acc51);
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
                s00 += a0[k] * b0[k]; s01 += a0[k] * b1[k];
                s10 += a1[k] * b0[k]; s11 += a1[k] * b1[k];
                s20 += a2[k] * b0[k]; s21 += a2[k] * b1[k];
                s30 += a3[k] * b0[k]; s31 += a3[k] * b1[k];
                s40 += a4[k] * b0[k]; s41 += a4[k] * b1[k];
                s50 += a5[k] * b0[k]; s51 += a5[k] * b1[k];
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
        for(;j<N;j++) {
            const float* a0=A+i*lda;
            const float* a1=A+(i+1)*lda;
            const float* a2=A+(i+2)*lda;
            const float* a3=A+(i+3)*lda;
            const float* a4=A+(i+4)*lda;
            const float* a5=A+(i+5)*lda;
            const float* b=B+j*ldb;

            __m256 acc0=_mm256_setzero_ps(); __m256 acc1=_mm256_setzero_ps();
            __m256 acc2=_mm256_setzero_ps(); __m256 acc3=_mm256_setzero_ps();
            __m256 acc4=_mm256_setzero_ps(); __m256 acc5=_mm256_setzero_ps();

            long k=0;
            for(;k<=K-8;k++) {
                __m256 bv=_mm256_loadu_ps(b+k);
                acc0=_mm256_fmadd_ps(_mm256_loadu_ps(a0+k),bv,acc0);
                acc1=_mm256_fmadd_ps(_mm256_loadu_ps(a1+k),bv,acc1);
                acc2=_mm256_fmadd_ps(_mm256_loadu_ps(a2+k),bv,acc2);
                acc2=_mm256_fmadd_ps(_mm256_loadu_ps(a2+k),bv,acc2);
                acc4=_mm256_fmadd_ps(_mm256_loadu_ps(a4+k),bv,acc4);
                acc5=_mm256_fmadd_ps(_mm256_loadu_ps(a5+k),bv,acc5);
            }
            float s0=ezsum(acc0); float s1=ezsum(acc1);
            float s2=ezsum(acc2); float s3=ezsum(acc3);
            float s4=ezsum(acc4); float s5=ezsum(acc5);

            for (; k<K; ++k) {
                s0+=a0[k]*b[k]; s1+=a1[k]*b[k];
                s2+=a2[k]*b[k]; s3+=a3[k]*b[k];
                s4+=a4[k]*b[k]; s5+=a5[k]*b[k];
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
        for(long j=0;j<N;j++) {
            long row_A=i*lda;
            long row_B=j*ldb;
            
            __m256 acc=_mm256_setzero_ps();
            long k=0;
            for (;k<=K-8;k+=8) {
                __m256 a_vec=_mm256_loadu_ps(A+row_A+k);
                __m256 b_vec=_mm256_loadu_ps(B+row_B+k);
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

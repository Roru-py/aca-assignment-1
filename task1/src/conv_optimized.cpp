// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
        const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    for (int oy = 0; oy < H; ++oy){
        int ox = 0;

        // unrolled by 8 , 64 columns needed
        for (; ox <= W - 64; ox += 64){

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps();
            __m256 acc7 = _mm256_setzero_ps();
            
            for (int ky = 0; ky < K; ++ky){
                for (int kx = 0; kx < K; ++kx){

                    __m256 weight  = _mm256_set1_ps(ker[ky * K + kx]);

                    __m256 in0 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    __m256 in1 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 8)]);
                    __m256 in2 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 16)]);
                    __m256 in3 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 24)]);
                    __m256 in4  = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 32)]);
                    __m256 in5  = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 40)]);
                    __m256 in6  = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 48)]);
                    __m256 in7  = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx + 56)]);
                    

                    acc0 = _mm256_fmadd_ps(in0, weight, acc0);
                    acc1 = _mm256_fmadd_ps(in1, weight, acc1);
                    acc2 = _mm256_fmadd_ps(in2, weight, acc2);
                    acc3 = _mm256_fmadd_ps(in3, weight, acc3);
                    acc4 = _mm256_fmadd_ps(in4, weight, acc4);
                    acc5 = _mm256_fmadd_ps(in5, weight, acc5);
                    acc6 = _mm256_fmadd_ps(in6, weight, acc6);
                    acc7 = _mm256_fmadd_ps(in7, weight, acc7);
   
                }
            }

            _mm256_storeu_ps(&out[oy * W + ox], acc0);
            _mm256_storeu_ps(&out[oy * W + ox + 8], acc1);
            _mm256_storeu_ps(&out[oy * W + ox + 16], acc2);
            _mm256_storeu_ps(&out[oy * W + ox + 24], acc3);
            _mm256_storeu_ps(&out[oy * W + ox + 32], acc4);
            _mm256_storeu_ps(&out[oy * W + ox + 40], acc5);
            _mm256_storeu_ps(&out[oy * W + ox + 48], acc6);
            _mm256_storeu_ps(&out[oy * W + ox + 56], acc7);

            


        }

        // cleaning up stuff
        for (; ox < W; ox += 8){
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky){
                for (int kx = 0; kx < K; ++kx){
                    __m256 weight = _mm256_set1_ps(ker[ky * K + kx]);
                    __m256 in_vec = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    acc = _mm256_fmadd_ps(in_vec, weight, acc);
                }
            }

             _mm256_storeu_ps(&out[oy * W + ox], acc);
        }



    }
}   

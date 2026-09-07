// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        for (; ox <= W - 4; ox += 4) {
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float wt = ker[ky * K + kx];
                    acc1 += in[(oy + ky) * in_stride + (ox + kx)] * wt;
                    acc2 += in[(oy + ky) * in_stride + (ox + 1 + kx)] * wt;
                    acc3 += in[(oy + ky) * in_stride + (ox + 2 + kx)] * wt;
                    acc4 += in[(oy + ky) * in_stride + (ox + 3 + kx)] * wt;
                }
            }
            out[oy * W + ox] = acc1;
            out[oy * W + ox + 1] = acc2;
            out[oy * W + ox + 2] = acc3;
            out[oy * W + ox + 3] = acc4;
        }

        
    }
}

// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int TILE = 32;    

    

    for(int ty = 0; ty < H; ty += TILE)
    {
        for(int tx = 0; tx < W; tx += TILE)
        {



            int y_end = H > ty + TILE ? ty + TILE : H;
            int x_end = W > tx + TILE? tx + TILE : W;
            
            for (int oy = ty; oy < y_end; ++oy) {
                for (int ox = tx; ox < x_end; ++ox) {
                    

                    float  acc = 0.0;

                    for(int ky = 0 ; ky < K ; ++ky ){
                        for(int kx = 0 ; kx < K ; kx++){

                            acc+= in[(oy + ky)* in_stride + (ox+ kx)] * ker[ky*K + kx];

                        }
                    }
                    out[oy* W + ox] = acc;

                }
            }

           
        }
    }
}
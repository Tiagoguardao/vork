// Host instantiation of the canvas integer spec used by conformance tests.
#define NV12_PLANES const unsigned char *luma, const unsigned char *chroma, int pitch
#define NV12_READ_Y(x, y) (4 * luma[(y) * pitch + (x)])
#define NV12_READ_UV(x, y) ((Nv12UV){4 * chroma[(y) * pitch + 2 * (x)], 4 * chroma[(y) * pitch + 2 * (x) + 1]})
#include "nv12.h"

void vork_nv12_reference(const unsigned char *nv12, int width, int height, int matrix, Nv12RGBA *out) {
    const unsigned char *uv = nv12 + width * height;
    for (int y = 0; y < height; ++y)
        for (int col = 0; col < width / 2; ++col) {
            Nv12Pair pair = nv12_round_pair(nv12_pair16(nv12, uv, width, col, y, height, matrix));
            *out++ = pair.first;
            *out++ = pair.second;
        }
}

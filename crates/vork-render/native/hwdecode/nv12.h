// Executable canvas conversion for opaque, progressive, limited-range 4:2:0.
// The host reference and both device compilers instantiate these same integer
// tables and operations. Plane reads return samples in 10-bit units; an
// 8-bit sample is shifted left twice. Both formats retain all source bits.
#ifndef NV12_INLINE
#define NV12_INLINE static inline
#endif

// ISO/IEC 23091-2 matrix_coefficients, also used by AVFrame.colorspace.
enum { NV12_BT709 = 1, NV12_UNSPECIFIED = 2, NV12_BT470BG = 5, NV12_SMPTE170M = 6 };
static inline int nv12_supported_size(int width, int height) {
    return width >= 32 && height >= 32 && !(width & 1) && !(height & 1);
}
static inline int nv12_supported_matrix(int matrix) {
    return matrix == NV12_UNSPECIFIED || matrix == NV12_BT709 ||
           matrix == NV12_BT470BG || matrix == NV12_SMPTE170M;
}

#ifndef NV12_METADATA_ONLY
typedef struct Nv12UV { int u, v; } Nv12UV;
typedef struct { unsigned char r, g, b, a; } Nv12RGBA;

typedef struct { Nv12RGBA first, second; } Nv12Pair;
typedef struct { unsigned short r, g, b, a; } Nv12RGBA16;
typedef struct { Nv12RGBA16 first, second; } Nv12Pair16;

// The deep result is (luma + chroma + 8192) >> 14. Luma is a multiple
// of 128, so shift chroma by seven before adding without losing a carry.
// Each chroma product fits int32: the largest absolute tap sum is 5018,
// giving |u|, |v| <= 80288. Even the sum of both green products fits.
NV12_INLINE unsigned short nv12_component(int luma, int chroma) {
    int deep = (luma + chroma + 64) >> 7;
    return deep < 0 ? 0 : deep > 65535 ? 65535 : deep;
}

NV12_INLINE Nv12RGBA16 nv12_pixel(int y_sample, int red, int green, int blue) {
    int luma = (y_sample - 64) * 9539;
    Nv12RGBA16 pixel = {nv12_component(luma, red), nv12_component(luma, green), nv12_component(luma, blue), 65535};
    return pixel;
}

// A left-sited chroma column supplies both horizontal luma samples.
NV12_INLINE Nv12Pair16 nv12_pair16(NV12_PLANES, int column, int y, int height, int matrix) {
    // Limited or unspecified range, left or unspecified siting. Horizontal
    // neighbours share UV. The vertical bicubic phase reverses on even rows.
    const int interior[4] = {-346, 3572, 985, -115};
    const int top[3][4] = {
        {4432, -336, 0, 0}, {3226, 985, -115, 0}, {959, 3473, -336, 0}};
    const int bottom[3][4] = {
        {0, -346, 3572, 870}, {0, -115, 985, 3226}, {0, 0, -346, 4442}};
    int u = 0, v = 0;
    for (int tap = 0; tap < 4; ++tap) {
        int row = y / 2 + ((y & 1) ? tap - 1 : 1 - tap);
        int weight = interior[tap];
        if (y < 3) {
            row = tap;
            weight = top[y][tap];
        } else if (y >= height - 3) {
            row = height / 2 - 4 + tap;
            weight = bottom[y - height + 3][tap];
        }
        Nv12UV uv = NV12_READ_UV(column, row);
        u += weight * (uv.u - 512);
        v += weight * (uv.v - 512);
    }
    u >>= 5;
    v >>= 5;
    // BT.470BG, SMPTE 170M and unspecified all select the 601 matrix.
    const int coefficients[2][4] = {{13075, -3209, -6660, 16525}, {14686, -1747, -4366, 17305}};
    int m = matrix == NV12_BT709;
    int red = (coefficients[m][0] * v) >> 7;
    int green = (coefficients[m][1] * u + coefficients[m][2] * v) >> 7;
    int blue = (coefficients[m][3] * u) >> 7;
    Nv12Pair16 pair = {nv12_pixel(NV12_READ_Y(2 * column, y), red, green, blue),
                     nv12_pixel(NV12_READ_Y(2 * column + 1, y), red, green, blue)};
    return pair;
}

NV12_INLINE Nv12RGBA nv12_round(Nv12RGBA16 p) {
    Nv12RGBA rounded = {(unsigned char)((p.r + 128) / 257), (unsigned char)((p.g + 128) / 257),
                        (unsigned char)((p.b + 128) / 257), 255};
    return rounded;
}
NV12_INLINE Nv12Pair nv12_round_pair(Nv12Pair16 deep) {
    Nv12Pair pair = {nv12_round(deep.first), nv12_round(deep.second)};
    return pair;
}
#endif

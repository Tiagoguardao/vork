// Decoder pitches are in bytes. Both planes outlive stream completion.
#define NV12_INLINE static inline __attribute__((device))
#define NV12_PLANES const unsigned char *luma, const unsigned char *chroma, int luma_pitch, int chroma_pitch, int ten_bit
// P016 stores Main10 in the high ten bits of each little-endian word.
NV12_INLINE int read_sample(const unsigned char *row, int index, int ten_bit) {
    return ten_bit ? reinterpret_cast<const unsigned short *>(row)[index] >> 6 : 4 * row[index];
}
#define NV12_READ_Y(x, y) read_sample(luma + (long long)(y) * luma_pitch, x, ten_bit)
#define NV12_READ_UV(x, y) Nv12UV{read_sample(chroma + (long long)(y) * chroma_pitch, 2 * (x), ten_bit), \
                                  read_sample(chroma + (long long)(y) * chroma_pitch, 2 * (x) + 1, ten_bit)}
#include "nv12.h"

extern "C" __attribute__((global)) void
convert(NV12_PLANES, int width, int height, unsigned char *rgba, int rgba_pitch, int matrix) {
    int col = __nvvm_read_ptx_sreg_ctaid_x() * 16 + __nvvm_read_ptx_sreg_tid_x();
    int y = __nvvm_read_ptx_sreg_ctaid_y() * 16 + __nvvm_read_ptx_sreg_tid_y();
    if (col < width / 2 && y < height) {
        // Vulkan's buffer base and row pitch are both aligned to eight bytes.
        auto row = reinterpret_cast<unsigned long long *>(rgba + (long long)y * rgba_pitch);
        auto pair = nv12_round_pair(nv12_pair16(luma, chroma, luma_pitch, chroma_pitch, ten_bit, col, y, height, matrix));
        row[col] = __builtin_bit_cast(unsigned long long, pair);
    }
}

extern "C" __attribute__((global)) void
convert16(NV12_PLANES, int width, int height, unsigned char *rgba, int rgba_pitch, int matrix) {
    int col = __nvvm_read_ptx_sreg_ctaid_x() * 16 + __nvvm_read_ptx_sreg_tid_x();
    int y = __nvvm_read_ptx_sreg_ctaid_y() * 16 + __nvvm_read_ptx_sreg_tid_y();
    if (col < width / 2 && y < height) {
        auto row = reinterpret_cast<unsigned long long *>(rgba + (long long)y * rgba_pitch);
        auto pair = nv12_pair16(luma, chroma, luma_pitch, chroma_pitch, ten_bit, col, y, height, matrix);
        row[2 * col] = __builtin_bit_cast(unsigned long long, pair.first);
        row[2 * col + 1] = __builtin_bit_cast(unsigned long long, pair.second);
    }
}

#include <metal_stdlib>
using namespace metal;
#define NV12_PLANES texture2d<float, access::read> luma, texture2d<float, access::read> chroma
#define NV12_READ_Y(x, y) (4 * int(round(luma.read(uint2(x, y)).r * 255.0f)))
#define NV12_READ_UV(x, y) read_uv(chroma, uint2(x, y))
struct Nv12UV;
inline Nv12UV read_uv(texture2d<float, access::read> plane, uint2 pos);
#include "nv12.h"

inline Nv12UV read_uv(texture2d<float, access::read> plane, uint2 pos) {
    int2 uv = int2(round(plane.read(pos).rg * 255.0f));
    return {4 * uv.x, 4 * uv.y};
}

kernel void convert(texture2d<float, access::read> luma [[texture(0)]],
                    texture2d<float, access::read> chroma [[texture(1)]],
                    texture2d<float, access::write> rgba [[texture(2)]],
                    constant int &matrix [[buffer(0)]], uint2 pos [[thread_position_in_grid]]) {
    if (pos.x >= rgba.get_width() / 2 || pos.y >= rgba.get_height()) return;
    Nv12Pair pair = nv12_round_pair(nv12_pair16(luma, chroma, int(pos.x), int(pos.y), int(rgba.get_height()), matrix));
    uint2 xy = uint2(2 * pos.x, pos.y);
    rgba.write(float4(pair.first.r, pair.first.g, pair.first.b, pair.first.a) / 255.0f, xy);
    rgba.write(float4(pair.second.r, pair.second.g, pair.second.b, pair.second.a) / 255.0f, xy + uint2(1, 0));
}

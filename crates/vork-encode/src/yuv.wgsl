@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> planes: array<u32>;
override x86_rounding: bool = false;

// Each invocation handles eight columns. Share the horizontal chroma with
// neighbouring rows so vertical filtering doesn't reload their RGB pixels.
var<workgroup> horizontal_u: array<vec4<i32>, 176>;
var<workgroup> horizontal_v: array<vec4<i32>, 176>;

fn rgb(x: u32, y: u32) -> vec3<i32> {
    return vec3<i32>(textureLoad(source, vec2<u32>(x, y), 0).rgb * 255.0 + 0.5);
}

fn luma(p: vec3<i32>) -> u32 {
    let scaled = (dot(p, vec3(5983, 20127, 2032)) + 256) >> 9;
    return u32(16 + ((scaled + 32) >> 6));
}

fn write_row(x: u32, y: u32, width: u32, stride: u32, shared_index: u32, write_y: bool) {
    var u: vec4<i32>;
    var v: vec4<i32>;
    var packed_y: vec2<u32>;
    for (var pair = 0u; pair < 4u; pair++) {
        let left = x + pair * 2u;
        let a = rgb(min(left, width - 2u), y);
        let b = rgb(min(left, width - 2u) + 1u, y);
        let sum = a + b;
        // Average adjacent RGB pixels, retaining six chroma fraction bits.
        u[pair] = 16384 + 2 * ((dot(sum, vec3(-3298, -11094, 14392)) + 512) >> 10);
        v[pair] = 16384 + 2 * ((dot(sum, vec3(14392, -13073, -1320)) + 512) >> 10);
        if write_y {
            let pair_y = luma(a) | (luma(b) << 8u);
            packed_y[pair / 2u] |= pair_y << ((pair % 2u) * 16u);
        }
    }
    horizontal_u[shared_index] = u;
    horizontal_v[shared_index] = v;
    if write_y {
        let index = y * stride * 2u + x / 4u;
        planes[index] = packed_y.x;
        planes[index + 1u] = packed_y.y;
    }
}

fn pack(v: vec4<i32>) -> u32 {
    let p = vec4<u32>(clamp(v, vec4(0), vec4(255)));
    return p.x | (p.y << 8u) | (p.z << 16u) | (p.w << 24u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let size = textureDimensions(source);
    let stride = (size.x + 7u) / 8u;
    let x = id.x * 8u;
    let y = id.y * 2u;
    let shared_index = local.x + local.y * 16u;
    if id.x < stride {
        write_row(x, min(y, size.y - 1u), size.x, stride, shared_index + 24u, y < size.y);
        write_row(x, min(y + 1u, size.y - 1u), size.x, stride, shared_index + 32u, y < size.y);
        if local.y < 3u {
            let block_y = y - local.y * 2u;
            let halo = local.x + local.y * 8u;
            write_row(x, u32(max(i32(block_y + local.y) - 3, 0)), size.x, stride, halo, false);
            write_row(x, min(block_y + 16u + local.y, size.y - 1u), size.x, stride, halo + 152u, false);
        }
    }
    workgroupBarrier();
    if id.x >= stride || y >= size.y {
        return;
    }

    // With unchanged geometry the default swscale bicubic filter applies
    // vertically only. Fold the out-of-image weights onto the edge rows.
    var weights = array<i32, 8>(-58, -172, 492, 1786, 1786, 492, -172, -58);
    if y == 0u {
        weights = array<i32, 8>(0, 0, 0, 2048, 1786, 492, -172, -58);
    } else if y == 2u {
        weights = array<i32, 8>(0, -230, 492, 1786, 1786, 492, -172, -58);
    } else if y == size.y - 4u {
        weights = array<i32, 8>(-58, -172, 492, 1786, 1786, 492, -230, 0);
    } else if y == size.y - 2u {
        weights = array<i32, 8>(-58, -172, 492, 1786, 2048, 0, 0, 0);
    }
    // FFmpeg 6.1's default x86-64 path reduces each product separately.
    // Its final chroma row and scalar path accumulate full products.
    let product_shift = select(0u, 16u, x86_rounding && y < size.y - 2u);
    let round_bias = select(262144, 7, product_shift != 0u);
    var u = vec4(round_bias);
    var v = vec4(round_bias);
    for (var tap = 0u; tap < 8u; tap++) {
        let index = shared_index + tap * 8u;
        u += (weights[tap] * horizontal_u[index]) >> vec4(product_shift);
        v += (weights[tap] * horizontal_v[index]) >> vec4(product_shift);
    }
    let chroma_size = stride * (size.y / 2u);
    let index = 4u * chroma_size + id.y * stride + id.x;
    planes[index] = pack(u >> vec4(19u - product_shift));
    planes[index + chroma_size] = pack(v >> vec4(19u - product_shift));
}

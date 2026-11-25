
        struct BlurUniforms {
          // x: direction (0 = horizontal, 1 = vertical)
          // y: kernel size
          // z: texture width
          // w: texture height
          params: vec4<f32>,
          // Packed weights and offsets: xy = weight and offset for sample i
          weightsAndOffsets: array<vec4<f32>, 16>,
        };

        @group(0) @binding(0) var<uniform> blur: BlurUniforms;
        @group(0) @binding(1) var inputSampler: sampler;
        @group(0) @binding(2) var inputTexture: texture_2d<f32>;

        struct VSOut {
          @builtin(position) position: vec4<f32>,
          @location(0) uv: vec2<f32>,
        };

        @vertex
        fn vs(@builtin(vertex_index) vid: u32) -> VSOut {
          var pos = array<vec2<f32>, 3>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>( 3.0, -1.0),
            vec2<f32>(-1.0,  3.0)
          );
          var uv = array<vec2<f32>, 3>(
            vec2<f32>(0.0, 0.0),
            vec2<f32>(2.0, 0.0),
            vec2<f32>(0.0, 2.0)
          );
          var out: VSOut;
          out.position = vec4<f32>(pos[vid], 0.0, 1.0);
          out.uv = uv[vid];
          return out;
        }

        fn gaussianBlurSample(uv: vec2<f32>, size: vec2<f32>, blur: BlurUniforms, inputSampler: sampler, inputTexture: texture_2d<f32>) -> vec4<f32> {
          let direction = blur.params.x;
          let kernelSize = i32(blur.params.y);
          let texelSize = vec2<f32>(1.0) / size;

          var color = vec4<f32>(0.0);

          var sampleDir = select(
            vec2<f32>(texelSize.x, 0.0),
            vec2<f32>(0.0, texelSize.y),
            direction > 0.5
          );

          if (direction == 2.0) { sampleDir = blur.weightsAndOffsets[0].zw * texelSize; }

          let centerWeight = blur.weightsAndOffsets[0].x;
          color += textureSample(inputTexture, inputSampler, uv) * centerWeight;

          for (var i = 1; i < min(kernelSize, 16); i++) {
            let weightOffset = blur.weightsAndOffsets[i];
            let weight = weightOffset.x;
            let offset = weightOffset.y;

            let coord1 = uv + sampleDir * offset;
            let coord2 = uv - sampleDir * offset;

            color += textureSample(inputTexture, inputSampler, coord1) * weight;
            color += textureSample(inputTexture, inputSampler, coord2) * weight;
          }

          return color;
        }

        fn shade_fs(in: VSOut, blur: BlurUniforms, inputSampler: sampler, inputTexture: texture_2d<f32>) -> vec4<f32> {
          // The centre tap has no sample offset. Its spare lanes carry the
          // cropped-source flag and texel origin on the first filter pass.
          let center = blur.weightsAndOffsets[0];
          if (center.y == 1.0) {
            let size = vec2<f32>(textureDimensions(inputTexture));
            let pixel = select(in.uv * blur.params.zw, in.position.xy * 2.0, blur.params.x == 2.0);
            return gaussianBlurSample((pixel - center.zw) / size, size, blur, inputSampler, inputTexture);
          }
          let uv = select(in.uv, in.position.xy / blur.params.zw, blur.params.x == 2.0);
          return gaussianBlurSample(uv, blur.params.zw, blur, inputSampler, inputTexture);
        }

        fn shade_fs_border(in: VSOut, blur: BlurUniforms, inputSampler: sampler, inputTexture: texture_2d<f32>) -> vec4<f32> {
          let p = vec2<i32>(in.position.xy) + vec2<i32>(i32(blur.params.y));
          if (any(p < vec2<i32>(0)) || any(p >= vec2<i32>(textureDimensions(inputTexture)))) {
            return vec4<f32>(0.0);
          }
          return textureLoad(inputTexture, p, 0);
        }

@fragment
fn fs_border(in: VSOut) -> @location(0) vec4<f32> {
  return shade_fs_border(in, blur, inputSampler, inputTexture);
}
@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  return shade_fs(in, blur, inputSampler, inputTexture);
}

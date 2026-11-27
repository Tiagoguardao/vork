struct Inputs { float cut = 0.68; };
float4 paint(PaintContext c, Inputs i) {
    float light = c.uv.x + 0.35 * c.uv.y;
    float aa = max(fwidth(light), 0.0001);
    float band = smoothstep(i.cut - aa, i.cut + aa, light);
    return float4(lerp(float3(0.29, 0.13, 0.16), float3(0.89, 0.54, 0.40), band), 1);
}

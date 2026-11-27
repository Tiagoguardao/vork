struct Inputs { float breadth = 0.34; };
float4 paint(PaintContext c, Inputs i) {
    float taper = sin(3.14159265 * c.stroke_uv.x);
    float d = abs(c.stroke_uv.y - 0.5) - (0.12 + i.breadth * taper * taper);
    float aa = max(fwidth(d), 0.0001);
    float alpha = 1 - smoothstep(-aa, aa, d);
    return float4(float3(0.09, 0.05, 0.08) * alpha, alpha);
}

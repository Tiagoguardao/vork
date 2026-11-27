IMAGES
image.color_space overrides embedded colour tags. Untagged Radiance HDR (.hdr)
and OpenEXR (.exr) use Linear Rec.709. Untagged PNG and TIFF use sRGB at both
8 and 16 bits, and untagged JPEG uses sRGB. IES stores linear intensity.
Use Non-Color for data.
Canvas PQ/HLG footage maps to SDR (100-nit nominal white) via FFmpeg’s clip tone map
in linear light before working-space conversion; requires zscale and tonemap. Other tagged HDR images and unsupported
wide-gamut footage need explicit color_space.
Camera textures inherit shot profile/timing. Worlds and canvases can nest; bind
a canvas picture via image.render.root and image.render.resolution to a mesh
material. Resolution is fixed.
A canvas inside a canvas cannot set viewBox or preserveAspectRatio.
Missing sampled images (including environments) warn with the filename and use one magenta texel;
corrupt images, IES and colour-transform data refuse. 16-bit integer and float
inputs skip block compression and use half-float precision.
Sound: vork help render.
<!-- all -->
image.file supports .mov/.mp4/.mkv/.webm (install ffmpeg and ffprobe), or %d/%04d
frame patterns. VP8/VP9 alpha uses FFmpeg libvpx.
Animate image.time (seconds into clip) at ["images",name,"time"]; default node
clock or scene time without one. Canvas fills and strokes use ordinary image binding.
Sequence time zero is the first file. Sequences use render.fps unless the first EXR supplies
framesPerSecond (must be finite and positive). Interior gaps refuse and name the missing frames. Video uses
displayed timestamps relative to its first displayed frame, honours edit lists
and rotation, and holds the last frame at or before time. Out of range holds
end frames.
Surface/environment footage uses the transport sample clock; final compositor
footage uses nominal frame time. Captures use parent sample pose/time with a
closed shutter and retain dependencies when the compositor replaces beauty.
Compositor sampling: vork help compositor.

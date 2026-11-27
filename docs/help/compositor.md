COMPOSITOR
composition.compositor is ordered material-name|{material,name?,resolution?};
material may be inline. The last image is output. Use images.<name>.composition.compositor
for a capture. Worlds and canvases run their own graphs before their views; a
canvas painted inline inside another canvas cannot take a compositor.
resolution: [width,height] or positive output scale (floor, min 1); default output
size. Last image is resampled to output size.
Bindings: {step:earlier-name}, {image:name}, {channel:color|depth|view_normal|object_index|motion|shadow_catcher}, {aov:name}.
Step names are graph-local. No temporal history.
Canvas accepts only channel color, no AOV or rendered-image input. Camera fields
are identity/zero and ctx.samples is 1. Canvas images are premultiplied in
composition.color_space, including file inputs. Default sRGB uses code values;
decode for linear maths.
Channels read raw values. First visible hit: depth .x in view-axis metres
(background 1e10), view_normal .xyz signed camera-space unit vector with authored
smoothing (background 0), object_index .x numeric ID (background 0).
motion .xy is current minus previous in pixels, +Y down.
<!-- all -->
float4 composite(CompositeContext, Inputs) returns premultiplied RGBA in working
encoding (scene-linear for worlds). A non-finite result refuses the frame.
File/video samples are premultiplied before filtering. Runs after shutter
accumulation/denoising or canvas drawing, before the view.
compositor_sample(uv) reads the previous result (first: beauty or canvas picture).
compositor_sample(image,uv): clamped bilinear; compositor_load(image,int2):
clamped pixel; compositor_size(image): pixels; compositor_size(): previous size.
UV origin is top-left; ctx.size is working extent.

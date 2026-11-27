LIMITS
Run vork setup to check the adapter. Supported: Apple Silicon Metal (macOS 15+),
recent NVIDIA Vulkan (Windows/Linux) and Linux AMD Vulkan with ray queries and storage buffer arrays
(tested: Mesa RADV, RDNA3). Intel GPUs are unsupported. CPU mode is opt-in below.
Install scripts: Windows x64 install.ps1, macOS/Linux install.sh (release assets).
See README.md (Install).

Device-limit errors name the failing resource, limit and remedy.

Canvas nodes support transform, bounds and screen. Canvas text and paths support size and length.
A canvas inside a canvas cannot set viewBox or preserveAspectRatio.
Canvas frames with more than 335,544 ordered items use additional depth passes.

Volume carriers, world media and fog transport limits: vork help fog.
Use one environment light; combine contributions in one program or precomposite maps.

For indirect-light omissions and profile budgets see vork help profiles.
std/path still has finite depth, adaptive sampling and denoising. It clamps each
indirect contribution to mean absolute RGB 10 by uniform RGB scaling; direct
contributions are unclamped. The clamp matches Cycles’ brightness measure and preserves
hue, but bright indirect light can lose energy.
Path transport refuses `render.root` and reachable `image.render.root` subtree
renders on every device, including CPU Vulkan. Use `--profile std/final` with depth of
field and path-only effects disabled for subtree rendering, or render the
selected content in a separate scene without `root`. Computed attributes and their time/seek updates
do not have a separate CPU restriction.
Non-finite camera/light samples mark affected pixels magenta; output metadata lists their coordinates.
Final's rough glass shadows follow straight connections: roughness does not spread
transmitted shadows, light off-axis receivers, or test scattered-direction blockers.

On Apple Silicon, solid glass and heavy interiors can abort with code 4
("Impacting Interactivity"). Retry, use geometry_thin_walled:true for thin panes,
or try std/path or a supported NVIDIA/AMD GPU. Path is slower and can also fail; lower resolution is not a reliable fix.
Dense imported transmission is unqualified on Metal; test the full shot before
batching. Dense std/final transparency can lose the device even at low resolution.

A scene that draws meshes or imported models can bind a limited number of distinct
attribute names across all its materials. Apple Silicon Metal fits 17. On Vulkan the
device's vertex-input and inter-stage limits decide; the refusal states the device's
limit and how many names it fits. A Gaussian capture binds 1, 4, 9 or 16 names at SH
degree 0 to 3, so a degree-3 capture beside meshes leaves one more name on Metal. A
scene of Gaussian captures alone has no such ceiling. Bind fewer distinct names.

Large point scenes and working buffers can exceed a device's per-buffer binding
limit even when total free VRAM is sufficient. Follow the error's resolution/profile
advice or simplify scene buffers. Scene import support does not guarantee rendering
at any size.

vork setup builds the shared path engine (the same few pipelines across scenes) and warms the std/final starter. Without
setup, renders use scene kernels when needed without building the shared engine.
Without setup on an NVIDIA L4: first path render ~8 s with shipped shaders, ~40 s
when compiling; repeats 1–3 s.
With a warm engine, supported scenes render while scene kernels compile in a
detached background process; delivery and exit do not wait. A path-traced world inside a canvas starts no background compile. Authored programs,
media, AOVs, splats, points, hair and cameras other than one std/camera can need
their own scene kernels. Uncovered layouts compile on first use (can take minutes).
Required GPU/tool checks fail on error or timeout. Default GPU shader warming can defer at
10 minutes or on interruption (e.g. a driver crash) without failing setup. Reruns keep completed cache
entries. --skip-warmup checks only; --all warms bundled programs/profiles and
built-in descriptors with no shader time limit.

Hair displacement and lit stages are ignored with a warning. Curved propagation
selects path transport automatically; motion channels are zero with a warning,
while sampled shutter blur remains available.

Unsupported geometry effects refuse with the node and an authoring alternative.
Approximate effects can warn and render.
Spatial legacy environments evaluate at the origin. Unsupported light masks warn
and are ignored. Environments sampling render images use the light colour. Check
WARNING: REQUESTED EFFECT DROPPED and the structured warnings in meta.json and
SDK results; see vork help outputs.

## Opt-in CPU rendering

On Linux, install recent Mesa Vulkan drivers (`mesa-vulkan-drivers` on
Debian/Ubuntu; `mesa-vulkan-drivers` or `vulkan-swrast` elsewhere), then run
`vork setup --gpu cpu` (or `VORK_GPU=cpu vork setup`) and `VORK_GPU=cpu vork render scene.json --out frame.png`.
CPU Vulkan needs ray queries and storage buffer arrays. It is slow and opt-in.
Setup warms the default starter without a shader time limit. First compile can
take minutes. The driver pipeline cache is disabled to avoid a lavapipe compiler crash; portable shaders and Mesa
disk caching can help later runs.

macOS requires Apple Silicon; CPU mode is unavailable through Metal. Windows
requires a Vulkan loader and a recent Mesa lavapipe driver with its ICD manifest;
CPU rendering there has not been qualified.

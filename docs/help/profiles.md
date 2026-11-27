PROFILES
Choose with --profile; profile settings are separate from scene content.

Profile     Speed       Use                         Main omissions
std/previs  fastest     framing and direct light    low-budget emissive lighting
std/draft   quick       lighting previews           approximate GI, 4 samples
std/final   moderate    delivery (default)          approximate GI, 8 samples
std/path    slowest     interiors, glass, caustics  finite depth/budget, 64 samples

Resolution leaves the per-pixel budget unchanged.

Previs omits general indirect bounces and uses central-emitter hard shadows.
Broad-light highlights can narrow/change brightness and ray-camera effects can
be grainy. Judge lighting with final or path.
In std/previs, scalar std/openpbr geometry_opacity and alpha_cutoff use each
material's raster coverage, retaining direct lighting and visible emission. Sampled coverage can make colour and outline-driving depth, normals and AOVs noisy at low sample counts. Bindings
or animation of either input, and opacity combined with transmission, retain
whole-view path transport until qualified. Other effects can also require path
transport. The sample budget and denoise setting stay unchanged.
Raster previs uses analytic lights and approximate environment lighting. Emissive lighting
uses low-sample raster rays and warns of noisy or incomplete indirect light,
reflections and transmission. Proven unlit displays and null thin transmission
need no warning. Emissive previs uses fast GPU denoising when available; --set denoise=false
keeps the raw estimate.
Hair uses complete-camera path transport automatically, including render-image
captures. Promoted views keep the sample budget and default to denoising;
--set denoise=false turns it off. Glass and water retain final’s raster transport.
Draft and final use scene-ray indirect lighting.
Final's approximate GI can omit bounced paths such as camera → diffuse → glossy → light.
A custom surface material selects the older one-hit GI for the whole scene.
More samples cannot recover omitted paths; choose --profile std/path for full bounce lighting.
Path budget is independent of denoise.
Previs, draft and final use a cheaper implicit-field boundary test; exact zero
faces, edges and tangencies can differ. Use path to judge those shapes.
Views promoted to path transport retain the exact boundary test.
Draft, final and path denoise with bundled OIDN by default. --set denoise=false uses the profile
reconstruction (raw path samples for path); --set seed=42 makes sampling repeatable with
unchanged evaluated scene inputs, frame identity, render settings and backend.
Compare on the same machine and build. Reauthoring a pose can change its float
transform; other GPUs/drivers can change numeric results.
--set samples=N overrides the budget (1..4096); more samples reduce noise at a time cost.
Small text, thin geometry and subpixel bevel highlights can need extra camera samples;
see vork help geometry for the spatial sampling limits.
Without a fast GPU denoiser, draft and final announce it and use 16 and 32 undenoised samples;
--set denoise=true uses the slow CPU denoiser anyway.
See vork help limits for lighting approximations and scene-size limits.
<!-- all -->
PROFILE FILE REFERENCE
Profile files contain version (1), level and settings; fields below.
Levels: 0 = std/draft, 1 = std/final, 2 = std/path, 3 = std/previs.
Custom level-2 profiles without settings.samples use 64, independent of denoise.
Adaptive path sampling stops converged pixels early; AOVs and catchers use the full budget.
Camera optics and render.motion_blur belong to the scene.

MATERIALS
A 2D path can bind a material to its fill or stroke. See vork help paint for the fragment entry and coordinates.
Use std/openpbr (the default) for constant, image and named attribute inputs.
Instance attributes are equal-length columns in node.instances.attributes;
material attribute bindings select a column by name.
Custom logic needs a surface program. Data bindings share shipped code.
Custom surfaces can add first-render compilation.
OpenPBR defines emission_luminance in cd/m²; currently vork uses its value
directly as scene-linear radiance. emission_color is linear RGB.
Coat absorption follows the refracted path. coat_color tints beneath the coat, including emission; coat_darkening reduces
reflected base light. Emission alone does not make a surface unlit. Preview
emissive-lighting coverage: vork help profiles.
base_color is linear RGB; base_metalness=1
makes metal; specular_roughness controls highlight width. Inputs and defaults:
vork help field material.inputs
Image inputs bind {"image": name}; std/openpbr uses UV0, red for scalar inputs
and RGB for colour. Custom factors, channels and UV transforms need a program.
For glass panes use transmission_weight: 1 and geometry_thin_walled: true.
Use transmission_color for glass tint and subsurface_color for scattering colour.
Use std/path when refraction or indirect lighting is important.
Profile coverage, emissive lighting and transport selection: vork help profiles.
OpenPBR diffuse roughness uses Lambert diffuse. Transmission scatter and its
anisotropy use Beer-only transmission without interior scattering. Zero values
are neutral; nonzero values or varying bindings/tracks render with one
approximation warning per material. Use a volume material for interior scattering.

<!-- all -->
MATERIAL REFERENCE
Solid glass requires closed mesh boundaries; model hollow vessels with a shell.
Assign material.priority for overlapping solids (vork help field material.priority).
The owner supplies IOR/absorption; lower-priority boundaries pass through. Touching
faces cross directly between media; thin sheets have no interior.
With transmission_depth=0, a closed solid shows the authored transmission_color;
each thin sheet applies the full tint.
Thin-walled subsurface suits leaves, paper and curtains: geometry_thin_walled=true,
subsurface_weight>0. Set subsurface_color and subsurface_scatter_anisotropy;
positive anisotropy favours transmission. Radius controls apply to thick objects.
On thick meshes, transmission_weight mixes glass with the subsurface substrate.
Thick subsurface transport freezes homogeneous coefficients at entry and uses an
index-matched diffuse boundary; zero mean-free-path channels retain local response.
geometry_normal and geometry_coat_normal accept tangent-space XYZ vectors or
linear RGB textures. An omitted coat normal stays unperturbed by the base normal map.
Normal-map frames follow glTF/MikkTSpace (vork help geometry).
For transformed normal maps, custom programs use std.texture.
Raster ambient/GI reserves the same dielectric diffuse response as direct/path
shading, including mixed metals and thin subsurface. Raster transport remains an
approximation; use std/path to assess indirect lighting.
Use constant, texture or named attribute bindings for geometry_opacity and
alpha_cutoff; camera and ray hits share evaluation, preserving custom surface
programs and animation. Computed instance bindings: vork help buffers --all.
Numeric inputs animate with material tracks. Weighted/additive program tracks
start from the authored input, or zero when omitted. Untracked inputs retain
program initializers. Custom shading: vork help programs --all.

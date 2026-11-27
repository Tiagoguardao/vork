LIGHTS
Area lights emit along local -Z; target rotates the whole emitter. Check the whole transformed
emitter against walls/glazing, not just its centre. Check direct and reflected emitter visibility.
Use one environment for sky/HDR; render.background is camera-only. Without an
environment the world is dim grey (0.05). std/environment_image takes an Image.
vork std environment preprocesses HDR into environment + sun; vork std sky
declares atmosphere + sun. Helper inputs: vork help std --all.
vork std environment fps is positive numeric or rational; SDK options are numeric.
Use a separate output directory/stem; EXRs cannot alias sources, even by symlink.
Invalid rates and aliases refuse before frame writes.
Sun angle is full diameter in radians; 0 gives sharp shadows.
Energy: W for local lights, W/m² for sun, multiplier for environment. OpenPBR
emission_luminance is cd/m²; emission_color is linear RGB. AgX exposure defaults
to 0 stops (multiplier 2^exposure); no auto-exposure. See vork help render.
<!-- all -->
std/sky sun direction is [cos(el)*sin(rot),sin(el),-cos(el)*cos(rot)] in radians;
Rotation matches Blender: rot 0 = -Z, pi/2 = +X. Link with illuminates and shadowed_by; use shadowed_by: []
for no shadows. Use std/path for refracted caustics; see vork help limits.

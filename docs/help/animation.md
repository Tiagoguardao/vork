ANIMATION
scene.animation is an ordered list of layers containing tracks or a glTF clip.
Track paths are arrays of scene path segments. Keys increase in seconds and hold
endpoints. Defaults: LINEAR for numeric and CSS/SVG, STEP for bools and resources.
CUBICSPLINE uses [time,value] (zero tangents) or [time,value,in_tangent,out_tangent];
tangents are value units per second, scaled by segment duration. Euler keys interpolate angles; quaternions
take the short arc. A loop repeats the layer span; combine composes rotations.
Bools and camera names switch at effective layer weight >=0.5; zero-weight cuts
are inactive. A render.camera cut can supply the initial camera.
Settings and topology are fixed. Nonstructural numeric scene fields and material/program inputs
use the same track format.

Camera target is structural, including [x,y,z]. Aim at a named node and animate
its translation. Aim follows the evaluated translation track. Aiming and focus use evaluated poses, including
install and shutter samples; coincident evaluated aiming origins refuse with node and sample time. Unused base poses may coincide.

Clock fields (duration, image.time, simulation t0/dt, key times) accept numeric
seconds or rational strings; FPS accepts numeric or rational rates. Transforms,
materials, speeds, offsets and weights remain numeric. Output frame and simulation
tick identities stay integer with FPS and t0/dt respectively. Numeric weights use the ordinary evaluator. Media clocks retain exact STEP/LINEAR/cubic results
through layer controls, loops and blends until source-frame selection. For a
numeric clock field, vork uses the fraction with the smallest denominator that
rounds to the same binary64 number. Ties go to the smallest absolute numerator.
Fraction strings give exact values.
Buffer clocks use their stored IEEE values. A left-side cut stays on that side
through retiming and previous-pose subtraction. Continuous animation uses the
endpoint value.

STEP changes at the key time and holds until the next key. At FPS p/q, frame n has numerator n*q and
denominator p in seconds. Use that fraction string for frame-aligned keys, cuts, activation and footage boundaries.

Canvas paths/text key filter, d, fill, stroke, stroke-* and text styles with
ordinary track paths. CSS/SVG interpolates per property. Incompatible values switch halfway in LINEAR.
Structured CSS/SVG keys use LINEAR or STEP with replace; numeric keys also support
CUBICSPLINE and combine. Resource allocation and composition topology stay fixed.

Fields: vork help animation --all. Recipes: vork help springs, sequences, captions.

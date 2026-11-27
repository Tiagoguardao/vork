SPRINGS
Bake easing or spring samples into [time,value] keys in the property's units.
At FPS p/q write frame f time as rational seconds "f*q/p" with evaluated integers
(e.g. "1001/30000"); values remain numeric. Use LINEAR between samples, including
shutter samples; STEP holds. A bake matches its source only at keys. Sample more densely for between-frame curves and
rebake when FPS or motion parameters change.
Keep overshoot for translation/scale; clamp bounded values such as opacity.
Pick an explicit end: tracks hold the last value. CUBICSPLINE tangents are
per second; convert per-frame derivatives.
examples/animation-recipes/spring.py uses Python math to bake a resting underdamped unit-step spring
and smoothstep. For exact library motion, bake that library's version, FPS and
parameters. Commands: examples/animation-recipes/README.md. See vork help animation.

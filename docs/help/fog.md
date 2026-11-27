FOG AND VOLUMES
Author a Slang program with MediumCoefficients medium(MediumContext, Inputs).
Bind its material to light.medium on the single environment light for world fog,
or to a closed triangle mesh or volume grid for finite fog, smoke or fire.
Carriers need no surface stage. Bindings: vork help programs --all.

MediumCoefficients fields all default to zero:
  absorption: float3, nonnegative finite extinction per world metre.
  scattering: float3, nonnegative finite scattering per world metre.
  emission: float3, nonnegative finite scene-linear Rec.709 radiance added per world metre;
    independent of extinction, not surface emission_luminance in cd/m².
  anisotropy: float, finite, strictly (-1,1); Henyey–Greenstein phase,
    positive forward, zero isotropic, negative backward.
Extinction = absorption + scattering. Invalid values refuse.
MediumContext supplies position (carrier-local; world for the environment),
world_position (metres), time (seconds), wavelength (nm; zero in RGB).
Node scaling changes carrier bounds, not coefficient units.

Finite carriers admit bounded spatial coefficient programs; world fog admits
constant or certified exponential-height absorption/scattering. Height extinction
has the form coefficient * exp(-falloff * (world_position.y - base_height)),
with base_height in metres and falloff in inverse metres; emission stays constant.
Keep zero channels zero even on exponential overflow. Nonfinite positive
coefficients refuse the image. Admission requires provable coefficient bounds and
bounded work. Arbitrary world density, image/geometry inputs, unprovable loops and
combined surface/medium programs refuse. FloatGrid and VectorGrid inputs (import std.grid; sample_grid) read a
dense buffer or a .vdb/.nvdb grid; extinction is bounded by the grid's stored range.
A geometry with topology: volume is a carrier bounded by its grids. Program Inputs initializers define authored input defaults.

World segments end at mathematical infinity. Positive constant extinction closes
the world; uniform nonemitting fog cannot receive light from infinity.
Height fog can have finite optical depth toward the sky. A zero-extinction
emitting channel at infinity is refused, as is an unrepresentable collision position.
Independent carriers and world fog add, including inside glass; a vacuum carrier
cannot subtract fog. Glass priority selects solid IOR/absorption, not independent fog.

Fog uses complete-camera path transport for shots and captures at the selected
sample budget. No qualified raster/real-time fallback. Builtin OpenPBR (including
glass and subsurface) can coexist with fog, implicit geometry and curved propagation.
Custom surfaces (including with displacement), deformed carriers and finite-fog
subtree renders refuse. Fog samples free flights with compensated Russian roulette.
Exhausting the transport work budget refuses the frame; simplify the medium.
Attachment field: vork help field light.medium.

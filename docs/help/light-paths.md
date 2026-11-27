# Light paths

Custom cameras and per-contribution sensors. Full reference: light-paths --all.
<!-- all -->

A camera material exports `CameraRay camera(CameraContext, Inputs)` and optionally
`float3 sensor(SensorContext, Inputs)`. Inputs use ordinary material tracks.
`std/camera` supplies Blender optics and projective acceleration; uncalibrated
programs trace beauty, depth, normals and IDs from their rays and export no K or distortion model; motion export requires projective calibration.
Export camera_calibration for projective acceleration.

`CameraContext`: film_size (pixels), filtered uv, lens_sample ([0,1)²), time,
world origin/right/up/forward, focus_distance (object depth or -1).
uv = (pixel + 0.5 + filter_offset) / film_size. Pixels are zero-based, x right,
y down, (0,0) top-left. film_size is [width,height]. Filter offsets can put uv
outside [0,1].
`CameraRay`: world origin/direction, weight (RGB; default 1), wavelength_scale
(default 1), t_min/t_max (metres along normalized direction; geometry visibility).
Lens samples are independent per path; weight includes any radiometric Jacobian.

Transport uses scene wavelength = detector wavelength × `wavelength_scale`.
Use radiance per wavelength: Doppler factor D gives scale D and weight D⁵. Animate
the camera with node tracks. Geometry uses scene time (no retarded-time evaluation).

`SensorContext` supplies `path_length` in optical metres, detector `wavelength` in
nanometres, filtered `uv`, and scene `time`. Return a linear Rec.709 multiplier. It applies per contribution after CIE 1931
matching (negative lobes retained), before accumulation; a sensor cannot replace the matching basis.

Finite emitters emit their impulse at optical length zero. Length includes the
camera-to-near-plane leg and sums phase index × segment length; numerical ray
continuation offsets add no length. Infinite emitters use the largest finite f32
as a non-finite-arrival sentinel. A half-open gate accepts `start <= path_length &&
path_length < end`. Animate gate inputs with material tracks. There is no render gate, speed,
spectral switch or wavelength-count setting.

Dispersion follows OpenPBR Surface 1.1 `transmission_dispersion_scale` and
`transmission_dispersion_abbe_number`. Dispersion changes transport independently of sensors and retains authored
roughness on indirect paths so glossy regularization does not erase wavelength dependence. Spectral transport is selected only
when reachable material/sensor outputs use wavelength or dispersion, or the
camera wavelength mapping cannot be proven one. Ordinary RGB paths report
wavelength zero. Surface and emitter contexts use scene wavelength;
sensors use detector wavelength.

Phase optical length is not group delay in a dispersive medium. RGB upsampling
uses positive smooth RGB basis functions and a white-normalized 6504 K Planck
emission illuminant. White reconstructs to white; saturated RGB primaries do not
reconstruct exactly. Extrapolated ultraviolet/infrared content is not uniquely
determined by RGB colours. The detector samples 380–780 nm; a camera mapping can
query scene spectra outside that interval.

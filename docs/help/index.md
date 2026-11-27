vork — 2D and 3D rendering from scene JSON.

FIRST IMAGE
vork init my-shot
cd my-shot
vork render scene.json --out shot.png
Use vork help init for scaffold languages and vork help sdk for SDK setup.
Edit SDK source, regenerate JSON, then render. Use SDK Session for repeated looks.
GPU and CPU requirements: vork help limits. Source builds: vork help build.
Authoring skill: vork skills add

RENDERING
vork render SCENE --out PATH [--at SECONDS | [--frame-start N] [--frame-end N]] [--profile P]
  [--set KEY=VALUE ...] [--channels LIST] [--exr-codec CODEC] [--png-depth 8|16] [--diagnostics]
For multiple destinations, repeat --out PATH. Use .png for a still (t=0 unless --at),
a directory for NNNNNN.png frames, PATH/%06d.exr for EXR, .mp4 for opaque video,
or .mov for alpha. Channels: color (default), depth, object_index, view_normal,
motion, camera, shadow_catcher, aov:NAME; all includes render.aovs. --channels applies to every --out.
Encoding and sidecars: vork help outputs.
Profiles: std/previs for framing, std/draft for lighting previews, std/final
for delivery (default), std/path for full bounce lighting. See vork help profiles.
<!-- generated-call-keys:start -->
--set KEY=VALUE accepts resolution, fps, seed, samples, denoise.
<!-- generated-call-keys:end -->
Values are JSON; resolution also accepts WxH or W,H.

COMMANDS
vork check SCENE|- [--profile P] [--shaders]    Validate on CPU
vork ask SCENE QUESTIONS|-                    Read values without rendering
vork export SCENE --out ASSET.glb [--root NODE] [--at SECONDS]
vork std FUNCTION JSON    JSON argument and result; '-' reads stdin
vork setup                Check GPU/tools and warm the std/final starter and path engine (10 min shader budget)
vork setup --all          Warm bundled programs/profiles and built-in descriptors without a shader time limit
vork setup --skip-warmup  Check GPU/tools only; add --verbose for raw encoder probes
vork help [TOPIC [--all] | field PATH]
2D canvas: vork help nodes, then vork help composition, vork help text, vork help path.

CONVENTIONS
Units: metres, seconds, radians; lens/sensor mm; linear Rec.709.
<!-- generated-standards:start -->
animation: follows glTF 2.0 interpolation
assets: follows glTF 2.0
cameras: follows Blender 5.2.1 physical camera, except the names fit, shift, projection
colour: follows AgX as Blender 5.2.1
geometry: follows glTF 2.0 topology and attribute semantics
lights: follows Blender 5.2.1 lights, except radius (Blender: shadow_soft_size)
materials: follows OpenPBR Surface 1.1.1
textures: follows glTF 2.0 samplers; KHR_texture_transform lowered on import
<!-- generated-standards:end -->
Use right-handed Y-up; camera local -Z. Names exclude '/'; paths are scene-relative.
Use names or inline resources; tables use {name: definition}. Node type fields
are flat siblings. Unknown fields refuse; defaults can be omitted.
A field path names one object, not its table: node.font-size (table: scene.nodes).
Normal maps: tangent-space +Y; TEXCOORD v=0 at image top.
Topology and program code are fixed during rendering.

COMMON FIELDS (advanced: vork help TOPIC --all; exact: vork help field PATH)
<!-- generated-api:start -->

Scene {
  $schema: string; editor schema URL
  animation: layer[] = []
  buffers: {name:buffer} = {}
  composition: composition|null; space 2d = canvas, 3d = world; null = 3D world
  geometry: {name:geometry} = {}
  geometry_stages: (deform|compute|displace)[] = [deform,compute,displace]; ordered, each at most once; default [deform,compute,displace]
  images: {name:image} = {}
  materials: {name:material} = {}
  nodes: node[]
  programs: {name:program} = {}
  render: render = {}
  version: 2
}

Node {
  active: true|false|[start,end); parent seconds; false removes picture and sound, children included; a window starts the clock at start; a boolean is keyable (step)
  animation: layer[]; tracks on this node's clock in content seconds; paths start here; a property has one owner; the clock cannot key itself
  audio: {file,gain}|null; file relative to scene; gain defaults to 1 and scales source plus children; active mutes, visible does not; node time controls playback and pitch
  audio.file: file-path|null; mono/stereo; relative to scene; omitted = movie soundtracks and children
  audio.gain: number = 1; linear amplitude over source plus children; default 1; keyable
  children: node[] = []
  clip-path: node-name|null; 2D; that node's shape, in this node's coordinates, clips this node; it is not painted itself
  composition: composition|null; 2d: a canvas, its children painted to one picture; in a 3D world seen through an image's render.root; 3d: in a canvas, a closed world drawn through its one camera to one picture the size of the canvas's box
  filter: string|null; 2D; blur(), brightness(), contrast(), drop-shadow(), grayscale(), hue-rotate(), invert(), opacity(), saturate(), sepia(); nonzero lengths require px; blur length is the Gaussian standard deviation before transforms
  isolation: auto|isolate = auto
  mask: node-name|null; 2D; that node's picture, in this node's coordinates, masks this node; it is not painted itself
  mask-type: luminance|alpha = luminance
  mix-blend-mode: normal|multiply|screen|overlay|darken|lighten|color-dodge|color-burn|hard-light|soft-light|difference|exclusion|hue|saturation|color|luminosity = normal
  name: string = ""; globally unique, including imported nodes
  opacity: number = 1; 2D; 0..1; the node's picture, children included, placed once
  rotation: [x,y,z,w] quaternion | {euler:[x,y,z]} (Blender XYZ) = [0,0,0,1]
  scale: [x,y,z] = [1,1,1]; any zero world-scale component hides geometry, including children
  target: node-name|[x,y,z]|null; aims -Z, world Y up; rotation is an offset on top; structural, not animatable: aim at a named node and animate its translation
  time: {start,offset,duration,speed,loop}|null; node-owned animation/media clock; time-reading programs, conflicting movie clocks and scene-time tracks inside retimed compositions refuse; null inherits parent/active-window clock
  time.duration: number|rational string|null; content seconds used from offset; > 0; null = the rest of the one source; only active ends the node
  time.loop: bool = false; wraps inside [offset, offset+duration); needs an end
  time.offset: number|rational string = 0; content seconds where the clip begins; >= 0
  time.pitch: follow|keep = follow; keep holds sound pitch when speed changes; keep is not built yet
  time.speed: number|rational string = 1; keyable >= 0 and integrated; a constant < 0 plays the trimmed section backwards from its end
  time.start: number|rational string|null; parent seconds where local = offset; null = active window start, else 0; shifts loop phase
  translation: [x,y,z] = [0,0,0]; parent frame
  visible: bool = true; false hides the subtree and disables its lights; animatable (step)
  visible_camera: bool = true
  type: node|mesh|asset|camera|light|text|path = "node"
}

Mesh (type: mesh) {
  geometry: geometry-name|inline geometry
  materials: (material-name|inline material)[] = []; empty = default std/openpbr with roughness 0.3
}

Asset (type: asset) {
  file: file-path
}

Camera (type: camera) {
  focus_object: node-name|null = null; camera-axis depth supplied to the program at sample time
  material: material-name|inline material|null = null; omitted = std/camera; optics are program inputs
}

Light (type: light) {
  color: RGB
  energy: number|null; sun W/m²; local W; environment multiplier
  material: material-name|inline material|null; environment program; omitted = uniform color × energy radiance
  medium: material-name|inline material|null; environment only; authored medium program for world fog
  shape: sun|point|spot|rectangle|ellipse|environment
  size: number = 1; only rectangle or ellipse lights; rectangle full side, ellipse diameter; size_y sets the second axis
}

Text (type: text, canvas) {
  content: string|run[]; position by the text box's top-left; switch content with STEP keys; runs set font and paint; white-space controls newlines
  fill: "none"|CSS colour|CSS gradient|{image, fit} = #000
  font: file-path|file-path[]; .ttf/.otf; a list offers faces font-weight and font-style choose from
  width: number|null; px; lines wrap and align in it; null = no wrapping
}

Path (type: path, canvas) {
  d: string = none; SVG path data, px; "none" or empty paints nothing
  fill: paint = #000; "none"|CSS colour|CSS linear-/radial-/conic-gradient()|{image, fit}|{material}
}

Material {
  double_sided: bool = true
  geometry.geometry: mesh-node name; read-only geometry at the program's canonical sample time
  inputs: {name: number|vector|bool|enum-token|{image}|{attribute}|{grid}|{aov}|{step}|{channel}|{geometry}} = {}
  opacity: coverage|transmittance|null = null; null = coverage, transmittance for transmitting surfaces; coverage = occupied area fraction; transmittance = continuous attenuation by 1 - geometry_opacity; renderer chooses integration
  priority: integer = 0; highest owns overlapping solid media; equal priorities: last entered wins, simultaneous entries: later solid order wins; lower-priority boundaries change membership only
  program: program-name|inline program|<library>/<module> = std/openpbr
}
std/openpbr inputs: base_weight base_color base_metalness specular_weight specular_color specular_roughness specular_roughness_anisotropy specular_ior transmission_weight transmission_color transmission_depth subsurface_weight subsurface_color subsurface_radius subsurface_radius_scale subsurface_scatter_anisotropy coat_weight coat_color coat_roughness coat_roughness_anisotropy coat_ior coat_darkening fuzz_weight fuzz_color fuzz_roughness emission_luminance emission_color thin_film_weight thin_film_thickness thin_film_ior geometry_opacity geometry_thin_walled geometry_normal geometry_tangent geometry_coat_normal geometry_coat_tangent transmission_dispersion_scale transmission_dispersion_abbe_number alpha_cutoff base_diffuse_roughness transmission_scatter transmission_scatter_anisotropy

Texture binding {
  image: image-name|inline image; the program owns sampling and UV transforms
}

Image {
  color_space: sRGB|Display P3|Linear Rec.709|Linear Rec.2020|ACEScg|Non-Color|null = null; omitted/null: embedded colour description, else hdr/exr/ies linear Rec.709 and other files sRGB
  file: file-path; XOR render; png jpg tif hdr exr ies mov mp4 mkv webm or %04d frame pattern; 8-bit may be block-compressed; 16-bit integer and floating-point inputs use uncompressed half-float precision, not exact source integer codes; ies: LM-63 Type C
}

Render {
  background: environment|transparent|RGB|CSS colour = environment; RGB and transparent apply to camera escapes and opacity passthrough; path glass transmission sees the World; a 2D canvas takes transparent or a CSS colour
  camera: camera-node-name|null = null; null = the only authored camera, else first imported; a camera cut track may select
  duration: number|rational string = 125/12; seconds
  fps: number|rational string = 24; positive output frames/second
  motion_blur: MotionBlur|null = null; non-null enables; shutter in frames
  motion_blur.shutter: number = 0.5; frames
  resolution: uvec2 = [1920,1080]
}
<!-- generated-api:end -->
Asset overrides and imported nodes: vork help nodes. Load SPZ and Gaussian PLY
as assets; replace their radiance material through materials overrides.

LIBRARIES
Std builders return {buffers,geometry}; name prefixes standalone CLI/SDK fragments.
Shapes are centred, in metres. See vork help std --all.
<!-- generated-shapes:start -->
Scene geometry: {std: NAME, ...options}; omit name: the table key or inline position owns identity.

<!-- generated-shapes:end -->
TOPICS
check: validate without rendering
upgrade: breaking changes
contract: validation and installation
nodes: assets, skins, morphs, object IDs
cameras: optics and focus
lights: lighting and sky
materials: surface inputs and media ownership
paint: 2D path fill and stroke programs
fog: medium programs and admission
geometry: mesh attributes and implicit fields
buffers: binary and computed data
images: textures and render-to-texture
animation: tracks, clips, layers
render: timing and render fields
profiles: quality, samples, denoising
compositor: image graph and sampling
light-paths: rays, sensors, spectral transport
programs: Slang stages and bindings
outputs: channels, metadata, codecs
camera-json: OpenCV sidecars
sdk: TypeScript and Python
std: builders and Slang modules
fluid: liquid simulation
smoke: bounded dense-grid smoke/fire simulation and editable example looks
cloth: XPBD fabric, native animated collider references and private geometry state
errors: exit codes
diagnostics: timings and environment variables
limits: hardware and rendering limits
build: source build prerequisites
init: scene scaffolds
ask: scene queries
parallel-mp4: join worker renders
export: glTF assets
springs, sequences, captions: animation recipes
blender: imported shader nodes

Editor schema: "$schema": "https://raw.githubusercontent.com/voyager-so/vork/v0.6.3/packages/ts/scene.schema.json"
vork init adds it. SDK writers preserve an explicit value. vork never fetches it; it does not affect rendering.

UPDATE NOTICES
vork checks npm for stable updates at most daily after commands and caches the result.
Notices go to stderr on later runs, including redirected logs. Set
VORK_NO_UPDATE_NOTIFIER (any value) or CI to disable checks and notices. Only a package-version request is sent; no identifiers or telemetry.
Network/cache failures are silent and affect neither output nor status.

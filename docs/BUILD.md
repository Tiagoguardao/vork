# Build from source

Run these commands from the checkout root. Source builds need Rust, a native
C/C++ toolchain, Python with safe tar extraction filters, and the pinned Slang
compiler. Cargo does not download Slang. The supported Unix toolchain archives
are Apple Silicon macOS and x86-64 Linux (see `dist/slang.json`); there is no
pinned Intel Mac or Linux ARM archive.

## Prerequisites

Use Apple Silicon with macOS 15 or newer.

On macOS, install Apple's command line tools, then install Homebrew from
https://brew.sh if it is not already available:

```sh
xcode-select --install
```

Finish that installer before continuing. Use Homebrew Python explicitly:
stock macOS Python 3.9 may lack the extraction filters required by the Slang
installer. No pip packages are needed.

```sh
brew install python@3.12
PYTHON="$(brew --prefix python@3.12)/bin/python3.12"
```

On Ubuntu 24.04 x86-64 (the Linux baseline), install:

```sh
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libclang-dev curl git python3 ca-certificates
PYTHON=python3
```

Basis texture bindings require libclang (`libclang-dev` on Ubuntu; supplied
by Apple's command line tools on macOS). If bindgen cannot find it, set
`LIBCLANG_PATH` to the directory containing libclang.

Other Linux distributions need equivalent C/C++ build tools, libclang and Python 3.12+
(or a Python release with backported `tarfile.data_filter`). Rendering also
needs a working Vulkan driver; building and GPU-free commands do not need a GPU.
For your own NVIDIA Linux machine, a [Docker runtime](REMOTE-GPU.md) is available.

Install Rust 1.96, matching CI, through rustup (https://rustup.rs):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/vork-rustup.sh
sh /tmp/vork-rustup.sh -y --default-toolchain 1.96
. "$HOME/.cargo/env"
```

## Install Slang and build

All pinned toolchain archives are public and need no authentication. Linux and
Windows use upstream archives; macOS uses the source-built archive described below.

Clone the repository if needed, then enter it.

Choose a new, empty destination outside the checkout. The installer downloads
and verifies the archive against `dist/slang.json`, and refuses to overwrite
an existing directory. Keep the entire toolchain, including its libraries,
modules and notices, together.

```sh
"$PYTHON" scripts/install-slang "$HOME/.local/share/vork/slang-2026.21"
export VORK_SLANGC="$HOME/.local/share/vork/slang-2026.21/bin/slangc"
"$VORK_SLANGC" -version
cargo +1.96 build --release --workspace --locked
./target/release/vork --help
./target/release/vork help
```

The build compiles shaders for both Vulkan and Metal on either host. It checks
the Slang version against the pin. `VORK_SLANGC` must be an absolute executable
path; alternatively the build can find the pinned `slangc` on `PATH`. Export
`VORK_SLANGC` again in each new shell, or add the export to your shell startup
file. A compiler-start failure includes the loader diagnostic; check the selected
toolchain and its OS requirement. A version mismatch means a different compiler
was selected.

The renderer and its `vork-slang-filesystem` companion must stay together when copied or packaged. Building the companion requires a C++17 compiler and the full pinned Slang headers; it loads the matching shared library at runtime.

The binary is `target/release/vork`, or `$CARGO_TARGET_DIR/release/vork` when
that environment variable is set. These commands do not install it globally.
Runtime program compilation also needs Slang; retain the export when running
the binary from the build directory.

## Rendering and test dependencies

Start with `vork help` and the scenes in `examples/`. macOS uses Metal; Linux
uses Vulkan. Install FFmpeg/ffprobe for video commands and encoding tests
(`brew install ffmpeg` on macOS, `sudo apt-get install ffmpeg` on Ubuntu).

The public workspace includes CPU unit tests and small fixtures. The text tests
use bundled Liberation Sans and Noto Sans Arabic fonts under SIL OFL 1.1; their
licences are in `crates/vork-std/tests/fonts/OFL.txt` and
`crates/vork-std/tests/fonts/OFL-Noto.txt`. Renderer fixtures are synthetic scene,
shader and glTF inputs shipped with the source. No asset downloads are needed to
run the default tests. Corpus-dependent tests that require unshipped assets are
excluded from the public source; they are not part of its test coverage.

Renderer GPU unit tests are marked ignored with a message naming their requirements.
When explicitly selected they require the runtime opt-in `VORK_GPU_TESTS=1` and
successful renderer creation; missing adapters fail.
CPU and numeric tests remain enabled. Tests marked ignored explain their additional requirements. GPU tests need a
supported Metal/Vulkan device, including ray-query support; a CPU-only test run
does not validate rendering. To run ignored tests after installing their stated
dependencies, set `VORK_GPU_TESTS=1` and use
`cargo test -p vork-render -p vork-cli -- --include-ignored --nocapture`.

OIDN is loaded at runtime, not required to compile. For denoised renders and
GPU tests that exercise denoising, install the pinned runtime with the same Python:

```sh
"$PYTHON" scripts/install-oidn "$HOME/.local/share/vork/oidn"
# macOS:
export VORK_OIDN="$HOME/.local/share/vork/oidn/lib/libOpenImageDenoise.2.dylib"
# Linux (use this instead):
export VORK_OIDN="$HOME/.local/share/vork/oidn/lib/libOpenImageDenoise.so.2"
```

Workspace tests additionally need Node.js/npm for generated SDK checks.
Install the TypeScript dependencies and fetch the locked crate sources for the
licence notice check, then check and test every public workspace target:

```sh
npm ci --prefix packages/ts
cargo +1.96 fetch --locked
cargo +1.96 check --workspace --lib --bins --examples --locked
cargo +1.96 check --workspace --tests --locked
cargo +1.96 test --workspace --locked
```

Use `-- --nocapture` to see test diagnostics. Missing fixtures, missing required
tools, missing adapters and failed assertions are failures in selected GPU tests.
An ignored GPU test supplies no GPU coverage.

See [README installation](../README.md#install) for binary installation and runtime
setup, and `vork help programs --all` for authored Slang programs. Node/npm and the
SDKs are not prerequisites for the Rust workspace build.



## Public workflows

Distribution archives are built privately and attached to public GitHub releases.
For source validation, run the workspace tests above and `npm ci && npm test`
in `packages/ts`. Rendering checks need the runtime dependencies above;
run the examples with a supported GPU and inspect the resulting frames.

## Release shader assets

Hosted distribution jobs run `scripts/prepare-shaders` with the final binary
(after macOS signing). It invokes the hidden `prepare-release-shaders` command:
the ordinary setup catalogue and a bounded set of starter/example scenes run
against a no-op wgpu device with Slang. The scene set in `scripts/prepare-shaders`
uses normal installation at `std/final` and `std/path`, capturing installed material
unions and linked cameras that isolated library-program jobs cannot cover.
A path kernel's key also names two facts that are not the scene's materials: its
light kinds and whether a camera can start inside a solid. Each prepared path
scene therefore compiles its kernel at every covered case of both
(`ProgramKernel::release_cases`): a single sun, area lights only and every kind
(the key of no analytic light or any mix), and, where solids are installed, the
camera proof held and not held. A scene with the same materials and one of those
cases starts without compiling its own kernel. Points or spots alone use a covering
kernel while their exact kernel compiles in the background.
The bundle holds each artifact as one XZ stream (`<entry>.xz`), expanded when a
render asks for it and checked against the entry's digest; a render reads only the
few it needs. Each prepared path scene also contributes covering kernels with its
material and resource family unchanged and refinement capabilities enabled.
Hair, subsurface and null-connection unions retain their proofs; every prepared
family supplies all eight combinations to preserve sampler arithmetic and speed.
Scenes differing only in those refinements can use the stored covering code while their exact
pipeline specializes in the background. Existing exact light/camera cases remain
in the bundle so their current hits retain the same kernel.
Shared portable artifacts deduplicate by their existing content keys; arbitrary
authored programs and uncommon configurations still compile on demand. Preparation runs
without opening a GPU driver, OIDN device or executing shaders. Denoiser
preparation constructs only the renderer's pack/unpack pipelines. macOS emits Metal source;
Linux and Windows emit SPIR-V. The bundle excludes all driver cache entries
and is bound to the packaged binary's SHA-256. `scripts/package-release` refuses
to package without this verified bundle. `vork setup` on the user's real GPU
then warms its own driver cache from those portable assets.

The shipped generic GI and path entries enable the same existing Slang function
boundaries as their specialized variants. Without these boundaries, Slang's
SPIR-V optimization expands the generic lighting/transport graph. The official
specialization fix reduces duplicate work; these boundaries further bound the
optimizer's memory use.

On Linux, measure individual compiler invocations with
`scripts/toolchains/measure-slang-memory.py` as `VORK_SLANGC`, setting
`VORK_SLANG_MEMORY_COMPILER` to the real pinned compiler and
`VORK_SLANG_MEMORY_REPORT` to a JSONL file outside the checkout. Use a fresh
Cargo target directory and `VORK_SHADER_TARGETS=spirv,metal`; disable compiler
artifact caching (`VORK_VM_SHADER_CACHE=0` on the shared VM) while retaining
admission guards. Each record contains the target, entry, command, exit status,
CPU time and peak process RSS in KiB. Wall time includes admission waits and is
not a compiler speed comparison. Compare both pins on the same renderer sources
and generated inputs; include failed invocations in the inventory.

`check-release --skip-setup --skip-render` verifies installation, compiler and
portable assets on GPU-free hosted runners. Installed rendering, video and
denoising still require the real-device acceptance check.

## Opt-in CPU rendering

On Linux, install recent Mesa Vulkan drivers (`mesa-vulkan-drivers` on
Debian/Ubuntu; `mesa-vulkan-drivers` or `vulkan-swrast` elsewhere), then run
`vork setup --gpu cpu` (or `VORK_GPU=cpu vork setup`) and `VORK_GPU=cpu vork render scene.json --out frame.png`.
The driver must expose ray queries and storage buffer arrays. CPU rendering is
slow and never selected automatically. Setup warms the default starter. CPU shader
warm-up has no time limit. First-run shader compilation can take several minutes. vork's driver
pipeline cache is disabled because it can trigger a lavapipe compiler crash.
Portable shader artifacts and Mesa's own shader disk cache can still speed later runs.

macOS requires Apple Silicon; CPU mode is unavailable through Metal. Windows
requires a Vulkan loader and a recent Mesa lavapipe driver with its ICD manifest;
CPU rendering there has not been qualified.

## Mac toolchain reproduction

`dist/macos-minimum.txt` owns the macOS 15.0 floor. The path-boundary
allocator uses `atomic_thread_fence`, available on Apple Silicon with Metal 3.2
(macOS 15). Lowering only Slang's deployment target cannot support that renderer
path on older Metal drivers. See Apple's
[Metal language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

`dist/slang.json` records the exact upstream source commit (which locks recursive
submodules), Xcode build, CMake/Ninja versions and CMake options for the Mac build.
The `slang-macos-build` workflow builds that source with an explicit deployment
target on macOS 15. LLVM host-code generation is disabled: vork uses Slang for
GPU code generation, and the LLVM payload is not part of vork's runtime. No
Slang shader compiler source is patched.

To reproduce the build, check out the recorded commit with its recursive
submodules, select the recorded Xcode, install the recorded CMake and Ninja
versions, then run `python3 scripts/toolchains/build-macos-slang.py --source
/path/to/slang-source --output /path/to/new-output`. The output contains the
archive, SHA-256 and `vork-build.json` recording compiler, SDK and dependency
identities. Archive timestamps and owners are normalized; this is a pinned
build recipe, not a claim of byte-identical compiler archives. Repeated builds can differ in
embedded serialized core-module data and Mach-O UUID/signature bytes even with
the same tool versions. CI compares exact SPIR-V and Metal shader bytes against the official
compiler, and separately proves startup and vork shader compilation on macOS
15 without a GPU. A render still requires Apple Silicon with Metal.

Archive assembly checks every bundled Mach-O slice against the floor and writes
`macos-runtime.json` with paths, hashes and deployment targets, and preserves
Slang's build provenance at `slang/vork-build.json`. npm and native
wheel assembly rechecks that inventory before copying the signed bytes. The
helper is compiled with the same explicit deployment target. Run
`python3 scripts/check-macos-runtime /path/to/extracted-runtime` to inspect it.

## Reusing release shader compilation

`python3 scripts/build-with-shader-cache --cache /path/to/fixed-cache -- cargo build --locked --release -p vork-cli`
uses the selected pinned `VORK_SLANGC` through a temporary build-only proxy. It
caches fixed-entry native code and reflection together. Its path is stable for an
unchanged compiler, so Cargo can still skip a no-op build. Module validation still
runs normally. Every lookup captures Slang's resolved inputs; source, compiler
and library bytes, target, entry, defines and flags identify the result. Unsupported
or path-sensitive requests compile normally. Cached output hashes are verified
before either output is restored. The original compiler is never replaced or
packaged with a proxy; use it for runtime preparation and release packaging.
The filesystem companion uses the original toolchain headers, preserving the
header paths in its debug information.

`prepare-shaders --cache /path/to/preparation-store` retains portable assets
between preparations. The store is partitioned by compiler bytes and platform.
Each invocation stages a temporary executable layout, allowing the renderer's
existing installed-asset reader to validate and materialize only the entries
used by that run. Export reads its fresh mutable store, so obsolete entries do
not accumulate in the shipped bundle. The manifest is regenerated for the actual
packaged binary. Neither store contains driver caches. Dist jobs restore/save
these stores with Actions cache; the postmerge service retains them across commit builds.
Each build-only store bounds completed shader payloads to 2 GiB, removing the
oldest entries after a successful run; eviction only causes recompilation.
Lock files and the fixed-compiler event log are separate metadata.

### VideoToolbox canvas footage on macOS

For a local source build with FFmpeg in a custom prefix, expose its
`libavformat.pc` on `PKG_CONFIG_PATH` before building. The native reader
looks for the matching ABI-qualified dylib in the reported library directory,
then in the Homebrew locations. Distribution builds use pinned headers and
search the Homebrew locations at runtime. The portable reader handles a missing
native runtime.

The macOS packaging workflow checks the Cargo build receipt for VideoToolbox.
An explicit historical source-SHA build may omit it when that source predates
the native reader.

### NVIDIA footage on Linux

To include the native CUDA footage reader, install the FFmpeg 6 development headers
(libavformat 60, libavcodec 60 and libavutil 58), pkg-config and clang++ with its
NVPTX backend before building. On Ubuntu 24.04 the packages are libavformat-dev,
libavcodec-dev, libavutil-dev, pkg-config and clang. The build embeds the
conversion kernel. At render time the reader loads the matching libav libraries
for demux and bitstream filtering, plus libcuda and libnvcuvid for NVIDIA decode.
Canvas and world movie textures share the decoder and the 4:2:0 conversion for
8-bit footage and limited-range HEVC Main10 in BT.601/709. HDR canvases use
their transfer-specific tone mapping. The world texture path retains
16-bit components. Unsupported streams and builds without the native reader use
FFmpeg software decoding. On opening an inactive CUDA primary context, the
reader requests blocking synchronization. An active context keeps its owner’s
scheduling mode, which the reader shares.

The NVIDIA driver exposes per-picture decode status for H.264, HEVC and JPEG.
Vork requires status for H.264. HEVC and JPEG can decode without status when
the runtime reports that the query is unavailable. AV1, VP8, VP9, MPEG-1/2/4
and VC-1 use the native decoder without a per-picture corruption check.

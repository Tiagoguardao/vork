# Build from source

Run from the checkout root. Full prerequisites, toolchain recipes and diagnostics:
docs/BUILD.md. GPU-free commands and builds need no GPU.

- Apple Silicon macOS 15+: finish xcode-select --install (includes libclang for Basis bindgen).
  Install Homebrew from https://brew.sh if absent, then brew install python@3.12.
  Use the Homebrew Python 3.12 executable; stock 3.9 may lack safe tar extraction.
- Linux baseline Ubuntu 24.04 x86-64: sudo apt-get update, then
  sudo apt-get install -y build-essential pkg-config libclang-dev curl git python3 ca-certificates
  Elsewhere use equivalent tools
  and Python 3.12+ or a backport of tarfile.data_filter.
- Windows x64: Visual Studio Build Tools (Desktop development with C++), LLVM
  (libclang), Python 3.12+ and Developer PowerShell. Install Rust 1.96 through rustup.
- All: Rust 1.96 matches CI. On Unix install via rustup:
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/vork-rustup.sh
      sh /tmp/vork-rustup.sh -y --default-toolchain 1.96
      . "$HOME/.cargo/env"
  If bindgen cannot find libclang, set LIBCLANG_PATH
  to its directory. Install Node/npm for workspace SDK tests.

Cargo does not download the pinned Slang. Use scripts/install-slang with the
Python above and a nonexistent destination outside the checkout. Keep libraries,
modules and notices with the executable. No pip packages needed. The installer verifies the checksum in
dist/slang.json. There is no pinned Intel Mac or Linux ARM archive.
See docs/BUILD.md for the macOS Slang build recipe.

On macOS set PYTHON="$(brew --prefix python@3.12)/bin/python3.12"; on Linux set PYTHON=python3.
Then run:
    "$PYTHON" scripts/install-slang "$HOME/.local/share/vork/slang-2026.21"
    export VORK_SLANGC="$HOME/.local/share/vork/slang-2026.21/bin/slangc"
    "$VORK_SLANGC" -version
    cargo +1.96 build --release --workspace --locked

    target/release/vork help

Developer PowerShell:
    python scripts/install-slang "$env:LOCALAPPDATA/vork/slang-2026.21"
    $env:VORK_SLANGC = "$env:LOCALAPPDATA/vork/slang-2026.21/bin/slangc.exe"
    cargo +1.96 build --release --workspace --locked

    .\target\release\vork.exe help

Use target/release/vork (vork.exe on Windows), or $CARGO_TARGET_DIR/release/vork.
The build does not install globally.
Help and scene JSON are identical across platforms. Builds check the pin and compile Vulkan and Metal shaders. Linux builds can set
VORK_SHADER_TARGETS=spirv to skip Metal. Keep absolute VORK_SLANGC set for runtime compilation or put
the pinned slangc on PATH. For startup failures check the loader diagnostic,
toolchain and OS requirements. A version mismatch means the wrong compiler.
Repeat the export in each new shell.

Rendering adapters and binary installation: vork help limits.
Linux/Windows video output and decoding/fallbacks need FFmpeg on PATH; inputs
also need ffprobe. Native Mac H.264 uses VideoToolbox. For denoising run
python scripts/install-oidn DESTINATION; set VORK_OIDN to the installed
lib/libOpenImageDenoise.so.2 (Linux), lib/libOpenImageDenoise.2.dylib (macOS),
or bin/OpenImageDenoise.dll (Windows).

Workspace checks (after installing Slang, FFmpeg/ffprobe and Node/npm):
    npm ci --prefix packages/ts
    cargo +1.96 fetch --locked
    cargo +1.96 check --workspace --all-targets --locked
    cargo +1.96 test --workspace --locked

Node/npm runs generated SDK checks; locked crate sources supply licence notices.
Public tests use synthetic fixtures and Liberation Sans and Noto Sans Arabic/Devanagari
(SIL OFL 1.1; crates/vork-{std,text}/tests/fonts/OFL.txt); no downloads or unshipped corpus coverage.
CPU/numeric tests stay enabled. Ignored tests name prerequisites and give no GPU coverage. Explicit GPU tests
need a working device and fail if it cannot open. Add the OIDN runtime for
denoising tests. On Unix:
    VORK_GPU_TESTS=1 cargo test -p vork-render -p vork-cli -- --include-ignored --nocapture
In PowerShell set $env:VORK_GPU_TESTS = "1" before the Cargo command.

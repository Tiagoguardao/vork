CHECK
vork check SCENE|- [--profile P] [--shaders]
CPU-only validation. Match the render profile (default std/final). Assets resolve
beside SCENE or from cwd for stdin.

Check collects independent errors per stage; untyped scenes stop after expansion.
Missing sampled images substitute magenta. File resources must be readable regular
files; later sequence frames can still fail to decode.

Check runs the render loader and install validation, including JSON/schema, expansion, resolution, bindings,
feature/fog admission, assets, shutter sanity, animation/timeline
checks, program reflection and effect proofs. Use the render Slang toolchain;
caches and error messages/categories are shared. --shaders also compiles portable shaders for the target
without opening a GPU driver; this can take minutes.

Success prints `Scene checked successfully` and exits 0; see vork help errors.
Validate GPU features/memory, encoder setup and frame-execution effects by rendering.

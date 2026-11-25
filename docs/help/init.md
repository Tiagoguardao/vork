INIT
vork init [dir] [--language json|typescript|python]
Create a starter project in dir (default: current directory), without network or GPU access.
Writes scene.json and README.md. The scene uses std sphere, box and grid builders.
Existing output files cause a refusal before writing;
unrelated files are preserved. scene.json renders immediately (vork help).

JSON is the default. TypeScript adds scene.ts and a package.json with type: module;
Python adds scene.py. README.md explains how to install the matching bundled SDK
and regenerate scene.json. See vork help sdk for SDK details.

To regenerate scene.json from scene.ts, use Node 22.18+ and install the SDK with npm.
On older Node versions that support type stripping, run `node --experimental-strip-types scene.ts`.

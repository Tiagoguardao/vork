INIT
vork init [dir] [--language json|typescript|python]
Writes scene.json and README.md in dir (default: cwd), without network or GPU.
The scene uses std sphere, box and grid and renders as written. Existing output
files refuse before writes; unrelated files survive.
Default json. typescript adds scene.ts and module package.json; python adds
scene.py. Install the matching SDK and regenerate JSON using README.md.
Runtime requirements: vork help sdk.

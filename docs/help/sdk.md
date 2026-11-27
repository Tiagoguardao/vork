SDK
vork init --language typescript|python creates scene.json and scene.ts or
scene.py. Use plain TS objects or Python dicts with JSON field names.
Use matching SDK and renderer versions.
Regenerate SDK-written scenes with the matching SDK after upgrading.

Python 3.11+: pip install vork in a venv (Ubuntu system Python: python3-venv).
Install the renderer separately with npm install -g @voyager-so/vork, the curl
installer or the platform archive (https://github.com/voyager-so/vork#install).
TypeScript: install @voyager-so/vork with npm. Use Node 22.18+ or an older
type-stripping Node with --experimental-strip-types.
Bundled SDKs: python/ and sdk/ under ~/.local/lib/vork on Linux/macOS or
%LOCALAPPDATA%\Programs\vork on Windows. For a custom install use its python/
and sdk/. See the starter README for platform-specific venv commands.

Python std(function, **options); TS named builders (e.g. roundedBox, extrude,
text). Merge returned buffers and geometry maps. See vork help std --all.
write_scene(scene, path, base_dir) / writeScene(scene, path, baseDir) rebase paths
and externalize large buffers. Base defaults to cwd; pass the original scene
directory when moving a loaded scene. Writers preserve explicit $schema; add it yourself.
Writers retire obsolete generated buffer siblings referenced by the previous JSON;
unrelated files and symlinks survive. Serialize writes and buffer reads at the same
destination. Copy JSON and all referenced files together for snapshots. New files
use 0666 subject to umask; replacement JSON keeps its mode.

render(scene_path, out, at=0) / render(path, options) return exposure and warnings;
they require current meta.json with warnings. SDK CLI phases forward stderr progress.
SDK render at and environmentImage/environment_image fps take numbers. Scene
clock fields, CLI --at and --set fps also accept rational strings.

For two looks at the generated starter in one Python process:
    import json
    from pathlib import Path
    from vork import Session

    scene = json.loads(Path("scene.json").read_text())
    with Session() as session:
        session.open(scene, base=".")
        session.render("before.png", profile="std/previs")
        scene["nodes"][0]["translation"][0] += 0.25
        session.open(scene, base=".")
        session.render("after.png", profile="std/previs")

Session.render results include revision, reuse, installs and provisional. Edit semantics:
vork help render.
VORK_BIN selects the executable; in Python an installed native wheel takes precedence.
Types: vork.types / TS Scene.
Schema: sdk/scene.schema.json in the archive; packages/ts/scene.schema.json in source.

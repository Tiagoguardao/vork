RENDER
<!-- generated-call-keys:start -->
--set KEY=VALUE accepts resolution, fps, seed, samples, denoise.
<!-- generated-call-keys:end -->
Values are JSON; resolution also accepts WxH or W,H. --frame-start/--frame-end select inclusive absolute frame indices; --at selects seconds (number or rational string).
--gpu cpu (or VORK_GPU=cpu) explicitly selects software Vulkan rendering; see limits.
--gpu N (or VORK_GPU=N) renders on hardware GPU N in PCI bus order. On multi-GPU NVIDIA Linux machines, verify the indices with nvidia-smi.
All destinations share call fields.
composition.view: local material binding; worlds default to std/agx, canvases to std/standard; never inherited.
A canvas takes a view too; a canvas inside a canvas paints inline in its parent, and a view on it is not built yet.
A canvas with a view, a compositor, a world in it or a linear working space keeps a floating-point picture, rounded once at output; any other canvas stays 8-bit. Only a world's picture is dithered.
composition.color_space selects working RGB; output.display selects each destination's encoding.
A canvas blends in any of sRGB (default), Display P3, Linear Rec.709, Linear Rec.2020, ACEScg: colours and images stay sRGB and are converted before blending; its view shows the result. A world's working space other than Linear Rec.709 is not built yet.
A Display P3 canvas keeps its own code values only for output.display Display P3; the default display, sRGB, clips it to sRGB.
std/agx and std/standard: exposure (stops), gamma; AgX also look and lut.
A white RGB background looks grey by default: AgX maps unit scene white below screen white; std/standard preserves it.
Whole-frame grades: composition.compositor, before the composition's view. Raw world EXR bypasses the view.
Output encoding owns dither; artistic grain is a compositor program.
Sampling noise uses samples/denoise render settings (vork help profiles).

SESSION
Use SDK Session to retain the GPU, pipelines and installed scene between renders.
open accepts a scene file or whole document; base resolves document-relative files.
Reopen the whole edited document: omitted fields reset to defaults; failed opens
keep the previous revision. Replies identify revision and changed track paths
(unnamed items by position). A retained install reports reused for an identical scene
or updated for eligible value edits, including light energy/colour and camera placement.
Moving a camera with a target requires a reinstall, even for a fixed target point.
Other edits can report reinstalled or, for a changed program body, recompiled.
--no-audio and --export-identity are one-shot vork render options only.
In a session, provisional=true requests the first available picture, labelled in
the reply and meta.json. Render again for final output.
close releases retained memory. The underlying vork session JSON-lines framing
is private; progress goes to stderr.

MOVIES
render.duration and render.fps define the clip; a directory destination writes
PNG frames, .mp4 opaque video, .mov video with alpha. A .png destination writes
one still. Frame ranges are inclusive. Encodings: vork help outputs.

PARALLEL MP4
Render consecutive MP4 ranges from frame 0 in separate one-shot calls. Use --no-audio,
inclusive --frame-start and --frame-end, and one output directory per worker.
For sound, render the full timeline once with --out mix.wav --export-identity.
--export-identity conflicts with --at. Keep each MP4 and WAV with its .meta.json sidecar.
All calls need the same scene, immutable assets, vork build and effective render settings
(including profile and --set overrides). Vork writes export_identity and payload_sha256
only when it can enumerate and hash all inputs. Sidecars from 0.6.1 lack both fields.
Re-render those parts before joining. Stateful cuts also need deterministic replay or
shared checkpoints to establish the state at each cut.
vork skills add installs chunk-export.py in .agents/skills/vork/scripts/ and
.claude/skills/vork/scripts/. Pass chunks in timeline order (Python 3.9+, FFmpeg 5.1+).
  python3 .agents/skills/vork/scripts/chunk-export.py --frames TOTAL --out movie.mp4 [--audio mix.wav] CHUNK.mp4 ...
Each chunk must contain only H.264 video, with matching stream settings, no B-frames,
a keyframe at its start and exact frame timestamps and durations at the render fps.
The joiner verifies each staged copy against its sidecar SHA-256, checks identities
and frame coverage, copies the video and muxes sound once. Allow temporary space
for the input files and joined MP4.
Use matching encoder settings for visual quality parity. Independent H.264 rate control can
change decoded pixels. Compare PSNR and inspect joins when qualifying an encoder setup.

Motion-blur shutter is in output frames: exposure time in seconds is shutter/fps.
Previous-pose and shutter sample times clamp to zero before animation-layer retiming.
RGB backgrounds affect camera escapes only, not scene illumination.
Named view inputs use material tracks; look/bool keys use STEP. White balance is a compositor.

The exposure summary measures display RGB weighted by alpha. Transparent renders
report clipping, near-black share and mean luma of covered pixels; an entirely
transparent image reports undefined exposure. Opaque black backgrounds count.
Clipped is a display channel at 99% or more, where a view ran out of range. A canvas
with no world, view or compositor shows authored colours as they are and reports no clipped share;
a linear canvas, or a Display P3 canvas shown as sRGB, passes its view and reports one.

Setup separates required checks from optional shader warming. Required stage
errors/timeouts fail; the default shader-warmup deadline or an interrupted warmup
(a driver crash, say) can end successfully with compilation deferred. Use
setup --skip-warmup for checks or setup --all for unbounded shader warming;
completed device-local cache entries survive reruns.

Progress on stderr reports frames delivered and sample passes recorded within a frame.
Recorded passes are CPU preparation. GPU completion counts advance at existing
submission boundaries, so updates can arrive in groups; once all passes
are recorded, the renderer may still be waiting for the GPU, denoising or output.
On Windows, Ctrl+C and Ctrl+Break finish submitted frames, close output files and
print the saved frame count and paths before exiting with code 130. A long GPU
frame or shader compilation must finish before cancellation reaches a safe boundary.

SOUND
Movies in supported node paints or material bindings play their first audio stream on the picture clock, including image.time remaps.
Frame sequences, compositor-only movies and movies without audio are silent. The first audio stream must be decodable mono or stereo. Remove a redundant audio.file to avoid doubling the soundtrack.
A node applies audio.gain (linear, keyable) to its own sound and its children's mix. Set it to 0 to mute the whole subtree.
Node clocks change pitch with speed. Held clocks are silent.
--no-audio omits the MP4 or MOV audio track. Audio-driven animation and WAV output still use scene audio.
MP4 carries AAC; MOV carries PCM. --out sound.wav writes 48 kHz stereo float PCM without a GPU; use --frame-start/--frame-end for a range.
WAV and picture destinations currently require separate calls. Read sound with vork help ask.

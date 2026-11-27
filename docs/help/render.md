RENDER
<!-- generated-call-keys:start -->
--set KEY=VALUE accepts resolution, fps, seed, samples, denoise.
<!-- generated-call-keys:end -->
Values are JSON; resolution also accepts WxH or W,H. --set applies to every
destination. --at takes seconds (number or rational string); --frame-start and
--frame-end take inclusive absolute frame indices. See vork help outputs.
--gpu cpu or VORK_GPU=cpu selects software Vulkan (see vork help limits).
--gpu N or VORK_GPU=N selects hardware GPU N in PCI bus order; check multi-GPU
NVIDIA Linux indices with nvidia-smi.

VIEW
composition.view is a local material binding, never inherited. Default:
std/agx for worlds, std/standard for canvases. A nested canvas painted inline
in its parent cannot take a view. Both programs take exposure (stops) and gamma;
AgX also takes look and lut. AgX maps unit white below screen white. For screen-white from unit RGB use std/standard.
Animate named view inputs with material tracks; use STEP for look and bools.
Grade before the view in composition.compositor; put white balance and grain
there. Raw world EXR bypasses the view. Only world pictures receive output dither. For sampling noise use samples/denoise (vork help profiles).
composition.color_space selects working RGB; output.display selects encoding.
Worlds use Linear Rec.709. Canvases blend in sRGB (default), Display P3, Linear
Rec.709, Linear Rec.2020 or ACEScg; sRGB paints and images convert before blending.
Output a Display P3 canvas to Display P3 to keep its gamut; default sRGB clips it.
A canvas with a view, compositor, world or linear working space uses floating
point until output; otherwise it uses 8-bit. RGB backgrounds affect camera
escapes and opacity passthrough; light the scene with an environment.

SESSION
CLI session framing is private. Use SDK Session to retain the GPU, pipelines and scene (vork help sdk).
Restart the session after changing the compiler, its runtime or helpers, or the media decoder installation.
Edits through a shared writable mapping become visible to a kept session when the OS updates the file timestamps, which can happen on sync or unmap.
open accepts a scene path or document; base resolves document-relative files.
Reopen the whole edit; omitted fields reset to defaults, failed opens keep the
previous revision. Replies identify revision and changed track paths (unnamed
items by position). Identical scenes report reused; eligible edits report updated
(e.g. light energy/colour, camera placement). A targeted camera move requires
reinstall even with a fixed point target. Other edits can report reinstalled or,
for changed program code, recompiled.
provisional=true requests the first available image, labelled in the reply and
meta.json. Render again for final output. close releases retained resources.
--no-audio and --export-identity require one-shot vork render.

MOVIES
Set render.duration and render.fps. Encodings: vork help outputs.
For separate workers and a joined MP4, use vork help parallel-mp4.
Motion-blur shutter is in output frames; exposure seconds = shutter/fps. Previous-pose and shutter sample
times clamp to zero before layer retiming.

EXPOSURE
The stdout summary measures display RGB weighted by alpha; fully transparent means
undefined, opaque black counts. It reports clipping (any display channel >=99%),
near-black share and mean luma.
A canvas with no world, view or compositor shows authored colours and skips the
clipped share; linear canvases and P3 shown as sRGB pass their view and report it.

SOUND
Supported node paints and material bindings play the first movie audio stream
on the picture clock, including image.time. It must be decodable mono or stereo.
Frame sequences, movies without audio, and compositor-only movies are silent.
Remove redundant audio.file to avoid doubling.
audio.gain is keyable linear gain over node and children; 0 mutes the subtree.
Clock speed changes pitch; held clocks are silent.
--no-audio omits the MP4/MOV audio track but keeps audio-driven animation and WAV.
MP4 carries AAC; MOV carries PCM. --out sound.wav writes 48 kHz stereo float PCM
without a GPU; use --frame-start/--frame-end for a range. Render WAV and picture
separately. Read sound with vork help ask.

Setup: vork help limits. Progress and interruption: vork help diagnostics.

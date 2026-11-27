PARALLEL MP4
Render consecutive ranges from frame 0 in separate one-shot calls. Use --no-audio,
inclusive --frame-start/--frame-end, and a separate output directory per worker.
For sound, render the full timeline with --out mix.wav --export-identity.
--export-identity conflicts with --at. Keep each MP4 and WAV with its .meta.json.

Use the same scene, immutable assets, vork build and effective settings (profile
and --set included). export_identity and payload_sha256 require enumerable,
hashable inputs. Re-render 0.6.1 parts; their sidecars lack these fields.
Stateful cuts require deterministic replay or shared checkpoints.

vork skills add installs chunk-export.py under .agents/skills/vork/scripts/
and .claude/skills/vork/scripts/. Use Python 3.9+ and FFmpeg 5.1+:
  python3 .agents/skills/vork/scripts/chunk-export.py --frames TOTAL --out movie.mp4 [--audio mix.wav] CHUNK.mp4 ...
List chunks in timeline order. Each needs matching H.264 video-only stream
settings, no B-frames, an initial keyframe and exact timestamps/durations at
render fps. The joiner checks staged-file SHA-256, identities and frame coverage,
copies video and muxes audio once. Allow temporary space for all inputs and output.
Match encoder settings and inspect joins; independent rate control can change
decoded pixels. Compare PSNR when qualifying an encoder setup.

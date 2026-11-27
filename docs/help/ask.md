ASK
vork ask SCENE QUESTIONS|-
JSON lists in/out (same order); '-' reads stdin. at: scene seconds or list (one answer per time).
Canvas nodes support transform, bounds and screen in output pixels.
node + read: transform|bounds|screen. Masks/clips/filters refuse dependent
canvas bounds/screen. 3D: instance picks copy, camera picks view. screen returns
projected bounds and inside (fraction in frame; ignores occlusion).
image + read: info.
{"text":"title","read":"size"}: canvas layout box, line_count, lines with
advance/baseline in local pixels.
{"path":"curve","read":"length"}: canvas length in local units, ignores pathLength.
3D: field names a path; buffer accepts optional range [first,count] or read count.
audio names a subtree; window [start,end) seconds required; at offsets it
(default 0). Omit read for 48 kHz stereo samples. file writes WAV or analysis
JSON; requires one at time.
read loudness: rms, peak, rms_dbfs, peak_dbfs (silence: null dBFS; no LUFS/true peak).
read spectrum: Hann-windowed mono average; magnitudes and frequency_step in Hz.
Spectrum window: 4..65536 power-of-two samples at 48000 Hz.

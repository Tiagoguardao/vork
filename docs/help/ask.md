ASK
vork ask SCENE QUESTIONS|-
Canvas nodes support transform, bounds and screen in output pixels.
Use {"text":"title","read":"size"} to get the local layout box, line_count and lines (advance and baseline in local pixels) of canvas text named title.
Mask, clip and filter coverage refuse dependent bounds and screen answers.
Use {"path":"curve","read":"length"} to get the geometric length of a canvas path named curve in local units, independent of pathLength.
QUESTIONS: a JSON list of addresses (- reads stdin); answers: a JSON list in the same order.
at: scene seconds, or a list of times for an answer at each.
Address keys: node with read transform|bounds|screen, image with read info,
text with read size (canvas), and path with read length (canvas).
For 3D scenes, use field with a path or buffer with an optional range ([first,count])
or read count. In 3D node queries, instance picks a copy and camera picks the view for screen.
screen gives projected bounds and inside, the fraction in frame; occlusion is not measured.
audio names a subtree; window: [start,end) is required in seconds; at offsets it (default 0).
Omit read for 48 kHz stereo samples; read loudness|spectrum for analysis.
file writes samples to WAV or analysis to JSON; requires one at time.
loudness: rms, peak, rms_dbfs, peak_dbfs (silence: null dBFS); not LUFS or true peak.
spectrum: Hann-windowed mono average, magnitudes and frequency_step in Hz; window contains 4..65536 power-of-two samples at 48000 Hz.

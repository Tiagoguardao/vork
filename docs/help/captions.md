CAPTIONS
Use STEP keys on node.content. Key cue text at its start and "" at its end.
At a shared boundary key only the incoming text. If the first cue starts after
zero, key "" at zero.

Write SRT milliseconds as "ms/1000" seconds. Cue times are independent of FPS; for local clocks see vork help sequences. A cue appears/disappears on the first
frame at or after its start/end. Use white-space: pre-line to keep explicit breaks;
width wraps the text. Fonts are explicit .ttf/.otf files. For repeatable emoji,
include a colour face in the font list, e.g. ["CaptionSans.ttf","NotoColorEmoji.ttf"]. Canvas tries the base character's font
list in order for the complete sequence, then a fallback colour face. The chosen
artwork supplies advance and visible glyph animation bounds.

examples/animation-recipes/captions.py converts UTF-8 SRT (BOM/CRLF accepted) into
one text node and STEP track. It does not transcribe, infer word times or paginate. It keeps multiline cues and markup literal, clears
gaps and rejects malformed, nonpositive, unordered or overlapping cues. For
overlapping speakers, use separate positioned nodes/tracks. Commands are in
examples/animation-recipes/README.md. Fields: vork help text.

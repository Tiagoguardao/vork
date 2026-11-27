SEQUENCES
At FPS p/q, frame f starts at f*q/p seconds. Keep integer frame counts in the generator; serialize rational strings.
Set node.active to [a*q/p,b*q/p] for frames [a,b): b-a frames, local time zero
at a. Node animation paths start at the node; scene tracks start at the scene.
node.time can remap the local clock. active gates picture and sound; opacity
changes picture only.

Clips of A and B frames with overlap O (0 <= O <= min(A,B)) occupy [0,A) and
[A-O,A-O+B); total A+B-O. Advance each start by the previous length minus overlap.
List siblings back to front. For a dissolve, keep the outgoing opaque sibling beneath the incoming one.
Key incoming opacity from 0 at local frame 0 to 1 at O; at O=0, start fully opaque and omit the fade.
Fading both exposes the background. For a slide key incoming translation.

examples/animation-recipes/sequences.py demonstrates a dissolve at 30000/1001 FPS.
Commands: examples/animation-recipes/README.md. See vork help field node.active,
vork help field node.time and vork help animation.

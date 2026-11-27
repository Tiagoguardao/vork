PATH PAINT
Set fill or stroke to {"material":"cloth"} to shade the covered part of a 2D path.
An inline material such as {"material":{"program":"bands","inputs":{"cut":0.6}}}
also works. A plain string selects a CSS paint. Named materials can share a program
and supply different values. Animate a field with ["materials","cloth","inputs","cut"].

Declare programs in scene.programs (vork help programs).

Return premultiplied RGBA in the canvas working encoding. The renderer applies
coverage, fill or stroke opacity, masks and node opacity. Inputs are numeric.
Text uses CSS or image paints.

c.position is the local SVG position in pixels before placement, with Y increasing down.
c.uv measures that position against the untransformed fill bounds. On a stroke it
can extend outside [0,1]. A zero-size axis gives 0.5. c.time is local seconds.
On strokes, c.stroke_uv.x runs from 0 to 1 along each subpath, continuing through
dash gaps. c.stroke_uv.y is 0 on the left edge and 1 on the right edge, walking in path direction.
On fills both components are zero. Use fwidth for antialiased bands inside the shape.

See examples/motion-graphics/path-paint/scene.json for a cel band on a jacket
and a tapered cheek line.

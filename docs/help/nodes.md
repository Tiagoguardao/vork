NODES
Look up a node field with `vork help field node.font-size`. Type-specific fields
are flat siblings of `type`; type: node has only common fields.
`vork help text`, `vork help path`, `vork help mesh`, `vork help camera`, `vork help light` and
`vork help asset` list one type's fields; `vork help composition` lists the canvas's.
Canvas fields (`vork help field PATH`): composition.space, composition.viewBox, node.d, node.fill, node.stroke,
node.stroke-width, node.content, node.font, node.font-size, node.opacity, node.clip-path, node.mask, node.filter.
A canvas inside a canvas cannot set viewBox or preserveAspectRatio.
Visibility per ray type is ANDed with ancestors. pass_index is inherited; otherwise IDs
are assigned stably in meta.json (0 = background).
Assets load .glb/.gltf. Imported names: asset/node-name or asset/#index.
Imported joints are not target, focus or attachment addresses.
Keep even unselected glTF nodes acyclic; repair child references at the reported node index
or re-export.

<!-- all -->
Painted backdrops: disable visible_shadow; disable other ray visibility when the card should appear only to the camera.
FROM BLENDER
  Y-up: Blender (x,y,z) -> vork (x,z,-y). Euler is Blender XYZ order about vork axes; a Blender Z
  turn is vork Y. Lights and cameras aim local -Z, so identity is horizontal; aim with target.
  Quaternions are [x,y,z,w] (Blender: w,x,y,z). Key times are seconds; frames are 0-based
  (Blender frame f -> t=(f-1)/fps). Interpolation defaults to LINEAR; Blender-like ease is
  CUBICSPLINE with [t,v] keys.
  std/openpbr: glass tint is transmission_color and SSS colour is subsurface_color (base_color tints
  neither); specular_weight 1 = Blender Specular IOR Level 0.5; thin_film_thickness is micrometres.
  Path glass, including straight-through thin panes, sees the environment; opacity
  passthrough keeps the camera background. For Blender sky/sun see vork help lights.
  view_normal is camera space (Blender Normal: world space); motion
  is current-minus-previous with +Y down (Blender Vector: previous-minus-current).

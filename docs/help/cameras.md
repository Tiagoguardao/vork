CAMERAS
Omit camera.material for std/camera. To set optics, bind a material with
program:"std/camera" and set material.inputs (a material without program selects
std/openpbr). Use numeric material tracks. Custom rays: vork help light-paths.
std/camera follows Blender optics: lens/sensor in mm, clip/focus/ortho_scale in
metres, aperture_rotation in radians. focus_object supplies sampled camera-axis depth.
Exact fields: vork help field std/camera.lens, etc.
For DOF use perspective, use_dof:true and positive aperture_fstop. Both focus_distance
and sampled focus_object depth must exceed lens/1000 metres when used. A 50 mm lens can
focus at 2 m at f/2.8. For pinhole use use_dof:false.
In orthographic projection, Horizontal fit (horizontal) sets full width to
ortho_scale; Vertical (vertical) sets full height. The other axis follows image
aspect. Auto (auto) uses Horizontal at width/height > 1, otherwise Vertical.
Fit uses image aspect, independent of scene bounds.
<!-- all -->
Physical apertures can select path transport with the existing sample budget.
Optional camera_calibration declares projective acceleration; omission declares none.
See vork help light-paths --all for CameraRay clipping and
vork help camera-json for sidecar fields.

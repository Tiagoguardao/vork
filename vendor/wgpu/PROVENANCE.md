# wgpu 29.0.4

Source: the unmodified crates.io `wgpu` 29.0.4 package, except for the local
change below and this file. Registry metadata `.cargo-ok` and
`.cargo_vcs_info.json` are omitted. Upstream: https://github.com/gfx-rs/wgpu.
Licensed MIT OR Apache-2.0; both license files are retained.

Local patch: `src/backend/wgpu_core.rs`, `Device::poll` returns
`PollError::DeviceLost` (vendor/wgpu-types) when waiting finds the device lost.
Upstream treats every non-timeout poll error as fatal and panics ("Error in
Device::poll: Parent device is lost"), so a GPU timeout (NVIDIA Xid 109, a Metal
watchdog abort) or fault crashed the renderer instead of returning an error that
names the frame. Remove when upstream reports device loss from `poll`.

Local scheduling extension: `CommandEncoder::submission_boundary` records a typed
native recording boundary through the patched wgpu-core. This is internal to the
renderer; it closes a native buffer without splitting Vulkan queue submissions.

Local scheduling extension: `CommandEncoder::bound_passes` hands the preceding
recording to an owner callback before a pass that would exceed the bound; the owner
swaps in an empty encoder and keeps queue order. wgpu-core makes native command
buffers per pass when an encoder finishes and wgpu-hal Metal admits a bounded number
before submission, so the renderer keeps per-recording pass counts bounded.

Local GPU placement extension: native TLAS descriptor buffers enter the existing tracked
full-build path with an explicit retained BLAS reference set. Core validates range,
alignment, capacity, usage and device ownership, records initialization and storage/AS
transitions, and retains dependencies through command replay and submission. Trace conversion retains the native buffer and reference set, but recording native
builds in portable traces is refused because native references cannot survive resource
reallocation. Query submissions retain their exact BLAS dependency snapshot even
when later TLAS builds replace the reference set. Metal residency declarations
use that same submission tracker rather than re-reading mutable TLAS dependencies. Native templates and transform/mask
layout facts come from the Metal/Vulkan backend; shaders never synthesize addresses.
No raw-HAL renderer bypass, indirect build or update/refit mode is introduced. Existing
Metal AS fence ownership is preserved. Remove when upstream supplies an equivalent
tracked descriptor-buffer build and backend layout/template contract. Regressions:
core native descriptor range tests, renderer GPU packing/error tests and the real-device
build/query/rebuild lifetime contract in `scripts/check-metal-contracts`.

Local native geometry extension: axis-aligned bounding-box BLAS descriptions and
build inputs use the existing Vulkan/Metal HAL AABB implementation. Core checks
geometry counts, compatible flags, stride/alignment, buffer range, usage and device
ownership, and tracks initialization and build-input transitions.

Local optional-allocation extension: `src/optional_allocations.rs` and the native
buffer/texture/queue API wrappers reserve disposable render capacity before
backend allocation. Resource clones share a lease; upload staging retains its
lease through queue completion. Native allocation and headroom refresh are
serialized, and an epoch never reuses unconfirmed frees. A refused reservation
unwinds only the enclosing disposable scope without invoking the panic hook or
creating an invalid GPU handle. The caller owns recoverable jobs outside that
scope and discards its mutable render state; ordinary panics still propagate.
Abort-on-panic builds decline optional scopes. No ordinary allocation is refused
outside an optional scope. The renderer uses this for bounded 2D frame workers.

Native devices expose the binding size of a texture descriptor and of an allocated texture. Native texture handles share the first size query across their views and clones. Command encoders can attach a callback that receives their submission index when the queue accepts their command buffer. Capture admission uses that index to retire texture reads.

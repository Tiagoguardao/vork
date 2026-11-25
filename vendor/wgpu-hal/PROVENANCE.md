# wgpu-hal 29.0.4

Source: the unmodified crates.io `wgpu-hal` 29.0.4 package, except for the
local changes below and this file. Registry metadata `.cargo-ok` and
`.cargo_vcs_info.json` are omitted. Upstream: https://github.com/gfx-rs/wgpu.
Licensed MIT OR Apache-2.0; both license files are retained.

Local patch: Vulkan `add_raw_buffer` accounts for imported managed allocations,
matching `destroy_buffer`. OIDN shared allocations use `Buffer::from_raw_managed`;
without registration their destruction subtracts memory that was never counted.

Local patch: `src/metal/device.rs`, `ShaderInput::Msl` compilation, enables
`MTLCompileOptions::setPreserveInvariance(true)` with the same OS availability
check as the existing Naga compilation path. Slang vertex output positions are
marked invariant in `crates/vork-render/src/native.rs`; both halves are needed for
depth-equal prepass/forward agreement on Metal.

Remove this patch when upgrading to an upstream version that preserves
invariance for passthrough MSL.

Local patch: `src/metal/{completion,mod,device}.rs` checks every submitted
native buffer, including internal seed, empty signal and present buffers.
Completion handlers latch the first native error. Fence polls also inspect
retained statuses before retiring a contiguous successful queue prefix, so
neither delayed callbacks nor a later successful buffer can hide an abort.
The original command label and NSError reach `LostWithReason`; submit, poll
and wait check loss before publishing success, including already-reached fences.
Remove when upstream checks every buffer with equivalent error delivery.
Regression: `metal::completion::tests` (earlier abort/later success, out-of-order
completion, abort beyond an unfinished prefix). These tests need no GPU fault.

Local patch: `src/metal/conv.rs` sets
`allowDuplicateIntersectionFunctionInvocation` on triangle and bounding-box
descriptors from `NO_DUPLICATE_ANY_HIT_INVOCATION`. Upstream called the getter
and left Metal's default enabled. Remove when upstream applies the setter.

Local patch: Metal storage-buffer binding arrays use Tier-2 argument buffers of
device addresses (including binding offsets), with explicit read/write resource
usage declarations. Advertise `BUFFER_BINDING_ARRAY` alongside Tier-2 texture
arrays. Native Slang shaders use a ParameterBlock of StructuredBuffers for this
ABI. Remove when upstream supports Metal buffer binding arrays.

Local patch: `src/metal/command.rs` orders the next encoder-boundary timestamp
samples after a preceding counter resolve. The existing pre-resolve event only
orders writes before reads; counter reuse also needs reads before later writes.

Upstream examples (`halmark`, `raw-gles`, `ray-traced-triangle`), their target
declarations and example-only dev dependencies are omitted. The renderer uses
the library only.

Local scheduling change: Vulkan submits all native command buffers in one
`vkQueueSubmit` per HAL submit, including empty submissions. Recording boundaries
no longer partition queue submission or emit per-part relay operations. Core's
resource barriers remain in the ordered buffers; acquire waits, present/external
signals and the lifetime fence cover the whole submission. Upstream's single relay
between HAL submit calls remains to satisfy the HAL queue ordering contract.
A TOP_OF_PIPE semaphore wait orders subsequent execution; it is not a no-op.
A standalone ash repro (no vork or wgpu) reproduced Xid 31 command-fetch MMU
faults and Xid 109 timeouts on NVIDIA L4, driver 610.57.04 under a gVisor container sandbox:
a deep semaphore-carrying submit backlog plus same-device pipeline creation
requiring new code triggered device loss. Precreation, cache hits, another device,
and merged submissions avoided the tested failures. Serializing host API calls
did not. The original vork base, 536e57cb3, also completed with only submission
merging applied and late pipeline creation retained; frames were byte-identical.
The standalone usage passed core and synchronization validation. Binding reports
placed the fault address outside application allocations, and command-buffer
lifetime checks found no pending reset/free. This is a fault below vork, not an
established application lifetime or synchronization violation. NVIDIA driver versus
gVisor GPU proxy ownership, and the exact internal fault mechanism, remain
unresolved. These controls qualify the workaround on the tested configuration;
they do not establish universal safety of background compilation.

Vulkan makes one native submit call. No backend implements the former
accepted-prefix recovery hook; that hook and its unreachable core quarantine
branch are removed. Independent device-loss reporting, failed queue-drain
quarantine and submitted-resource lifetime tracking remain.
Metal pass-bounded recording segments are independent and unchanged.
Regression: renderer command_finish ordering/empty-submit tests and repeated L4
bottle, perfume, rainy-night and security-wall clips with synchronization validation.

Local patch: Metal binding-array argument buffers allocate the layout's declared
capacity and repeat the first supplied handle into the unused tail. This covers
textures, samplers, buffers and acceleration structures without changing core
partial-binding validation or tracking resources beyond the supplied prefix.
Remove when upstream allocates and initializes the full declared array.
Regression: `scripts/check-metal-contracts` executes native capacity, prefix and
padded-tail reads plus buffer offsets on a real Metal GPU, with API validation
off and on; missing hardware is a failure.

Local AS ordering patch: Metal AS builds and AS-binding compute/render encoders
wait on and update one queue-owned fence, including across command buffers.
AS barriers end build encoders. Remove only when upstream realizes equivalent
build/query/rebuild dependencies. Regression: `scripts/check-metal-contracts`.

Local shadow error-reporting extension: `DeviceError::LostWithReason` retains
Metal's native failed-command-buffer error and label through core's
loss callback. Core conversion and queue teardown recognize this loss variant.

Why each local patch exists and when to remove it: `../PATCHES.md`.

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

Local patch: `src/vulkan/adapter.rs` requires and enables
`runtimeDescriptorArray` with bindless descriptor indexing. Native Slang SPIR-V
uses unsized resource arrays; fixed Naga arrays alone did not require this bit.
Adapter feature reporting and the requested device feature chain agree.

Local Vulkan read-only depth/stencil fix: use STORE_OP_NONE for read-only layouts,
so render-pass end matches core's read-only access tracking. STORE and DONT_CARE
both introduce writes that per-part relay semaphores could hide. Admit read-only
attachments only with Vulkan 1.3 or KHR/EXT_load_store_op_none (enabled on older APIs);
otherwise core already tracks the attachment as writable. Metal is unchanged.

Local patch: Vulkan `Buffer::memory_properties` exposes the allocator-selected
memory properties for owned buffers (imported memory returns unknown). This lets
optional simulation retention identify a unique backing heap without mixing
host and device budgets. Remove when upstream exposes equivalent allocation
metadata. Heap ambiguity is covered by renderer `buffer_heap_tests`.

Local no-op release-preparation support: `src/noop/mod.rs` supplies a finite
64-byte mock TLAS instance layout/template and permissive texture format
capabilities for normal resource construction (including integer MSAA targets).
The no-op backend executes no shader or acceleration structure operation; the
mock reference is never dereferenced. Real Metal/Vulkan descriptors are unchanged.
Regression: GPU-free `scripts/prepare-shaders`; remove when upstream's no-op
backend supplies equivalent native descriptor construction.

Local patch: Metal `PipelineCache` holds `MTLBinaryArchive` segments (`src/metal/{mod,device,adapter}.rs`).
`PIPELINE_CACHE` is advertised where binary archives are supported. Loaded segments
are staged in temporary files and remain read-only; one fresh writable archive
receives this process's pipelines. Adding a render pipeline that shares a vertex
function with a different stored fragment partner can make a file-loaded archive's
serialization fail and empty it, losing earlier pipelines on the next save.
Pipeline creation first asks all loaded segments and the writable archive with
`FailOnBinaryArchiveMiss`; a miss compiles as without an archive and then records
the pipeline in the writable archive. Mesh render pipelines are not archived.
Cache data is an envelope: a little-endian u32 segment count, that many
little-endian u64 byte lengths, then each segment's bytes in order. Saves preserve
the loaded segments byte for byte and append the writable archive once it has
gained pipelines. There is no merge, compaction or segment cap. The validation key
hashes `wgpu-metal-binary-archives-v2`, the GPU name and the OS build that supplies
the Metal compiler, invalidating the former single-archive format.
An unchanged save reuses the writable archive's last serialized file rather than
asking Metal to serialize it again; repeated unchanged serialization can move its
backing file away. A serialization failure latches for the cache's lifetime:
subsequent data requests return `None`, so a later partial archive cannot replace
the persisted cache. Malformed envelopes and data Metal cannot open are
`PipelineCacheError::Rejected` (`src/lib.rs`),
a validation error, never device loss. Apple's shared shader cache evicts large
pipelines, so it is not a dependable warm path. Regression:
`metal::device::binary_archive_tests`. Remove when upstream persists Metal
pipelines through `PipelineCache`. `metal::release_archive_files_for_exit` removes
live caches' backing files for a process exit that skips destructors (vork's
one-shot `_exit`) and stops new ones; regression `exit_release_removes_backing_files_and_stops_new_ones`
and vork-cli `tests/metal_archive.rs`. `metal::Device::archive_additions` counts the archive misses a device
recorded, so the renderer serializes its archive only when it gained a pipeline. Creating a
pipeline never waits for a serialization, which holds the archive's lock for 0.02 to 0.1 s per
megabyte: an addition that meets one is deferred to whoever next holds the lock; regression
`additions_after_a_serialization_are_in_the_next_serialization`.

Local patch: `src/metal/compile_admission.rs` bounds the Metal compiler work of the
whole process. Metal compiles in one out-of-process `MTLCompilerService` per client
process; concurrent compilations share its memory, and when its allocator runs out
the service dies and every compilation in flight fails with
`XPC_ERROR_CONNECTION_INTERRUPTED`. Every library compilation (`ShaderInput::Msl`,
Naga MSL, flexible passthrough tables) and every pipeline-state compilation (render,
mesh and compute, including a binary archive miss together with its recording in the
archive, which is a compiler request too) is admitted by estimated service
memory: source bytes times the measured service bytes per source byte, against the
smaller of the measured service ceiling and half of physical memory. Archive hits do
not compile and are not admitted. A compilation starts when its estimate fits beside
the ones in flight, passing larger ones that wait, and one larger than the limit runs
alone. Passing is bounded: once later arrivals worth one full limit have passed a
waiting compilation, later arrivals wait behind it, so every compilation starts.
A compilation that fails with the interruption is retried once with nothing else
compiling; other failures are returned unchanged. The interruption is recognised
by the `NSError`'s own description beginning with the driver's report; Metal uses
several domains and codes for it, and a compiler diagnostic begins with its source
location, so a source that quotes the report is not retried. `PassthroughShader` and
`CompiledShader` carry their source size for this. Regression:
`metal::compile_admission::tests` (no GPU needed). Remove when Metal's compiler
service bounds its own memory across a client's concurrent compilations.

Local no-op test oracle: `src/noop/command.rs` retains the native acceleration
structure Build/Update modes in its command stream and exposes per-queue submitted Build/Update counters. Nonzero mock scratch sizes keep core from skipping descriptor recording; no native acceleration structure executes. The renderer's `rt_refit_tests.rs` observes actual
wgpu-core selection, including force_rebuild, without GPU hardware. Remove when
upstream provides equivalent no-op command inspection.

Local patch: `TlasInstanceLayout` describes the backend's triangle option word and
cull-disable/non-opaque masks. The renderer installs material sidedness there while
retaining backend-owned native references. Vulkan uses packed instance flags; Metal
uses `MTLAccelerationStructureInstanceOptions`; the no-op ABI mirrors Vulkan.
Regression: renderer `opaque_material_sides_use_hardware_traversal` exercises an
installed TLAS, both material sides and zero opaque candidate callbacks. Remove
when upstream exposes equivalent native-descriptor instance traversal options.

Local patch: Vulkan external-encoder signals can be reserved for the calling
thread's next submission. Concurrent canvas submissions cannot consume the NVENC
copy's completion signal. The ordinary queue-wide signal API is unchanged.
Regression: `vulkan::semaphore_list::thread_signal_tests` checks concurrent
producers without a GPU. Remove when upstream supports producer-scoped external
completion signals.

Local patch (diagnostic, opt-in): `src/vulkan/fault.rs`, with hooks in
`src/vulkan/adapter.rs` (extension list, `open_with_callback`, `device_from_raw`),
`src/vulkan/command.rs` (a checkpoint before each compute dispatch),
`src/vulkan/device.rs` (object names, buffer sizes), `src/vulkan/instance.rs`
(`VK_EXT_debug_utils` for address-range reports) and `src/vulkan/mod.rs`
(`get_lost_err`). With `VORK_DEVICE_FAULT=1`, the device is created with whichever of
`VK_EXT_device_fault` (and `deviceFaultVendorBinary` where offered),
`VK_EXT_device_address_binding_report`, `VK_NV_device_diagnostic_checkpoints` and
`VK_NV_device_diagnostics_config` the driver has. The first `VK_ERROR_DEVICE_LOST`
prints, with the prefix `vork gpu fault:`, the last compute dispatch each pipeline
stage reached (pass label, pipeline label, dispatch arguments), the driver's fault
description, address records (each mapped to the labelled buffer or image whose
address range holds it) and vendor codes, and writes any vendor binary to
`VORK_DEVICE_FAULT_DUMP` (default `vork-device-fault.bin`) and the full report beside
it as `.txt`. `VORK_DEVICE_FAULT=2` also asks the NVIDIA driver to keep shader debug
information. Without the variable nothing changes. Remove when upstream reports device
faults and checkpoint data.

Vulkan, Metal and D3D12 expose native texture allocation requirements and backing sizes for capture scheduling. Vulkan releases temporary pipeline shader modules after a failed driver creation and propagates explicit device loss.

Local patch: command encoders expose whether buffer or texture transitions can
record commands. Metal opts out alongside its two empty transition methods.
Other backends default to explicit transitions. Used by wgpu-core's lazy
pre-pass recording. Remove with the corresponding upstream pre-pass change.

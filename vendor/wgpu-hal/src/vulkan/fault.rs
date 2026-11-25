//! What the driver knows about a lost device, printed when the device is lost.
//!
//! Off unless `VORK_DEVICE_FAULT` is set (`1`, or `2` to also ask the NVIDIA
//! driver to keep shader debug information, which changes its pipeline cache).
//! When on, device creation enables whichever of these the driver has:
//!
//! - `VK_EXT_device_fault`: after `VK_ERROR_DEVICE_LOST` the driver describes
//!   the fault, the addresses involved and a vendor crash dump.
//! - `VK_EXT_device_address_binding_report`: the driver reports the GPU
//!   virtual address range of every buffer and image, so a faulting address
//!   can be named.
//! - `VK_NV_device_diagnostic_checkpoints`: a marker is recorded before every
//!   compute dispatch; after a loss the driver returns the last marker each
//!   pipeline stage reached, which names the dispatch that did not finish.
//! - `VK_NV_device_diagnostics_config`: resource tracking, automatic
//!   checkpoints and shader error reporting for the vendor crash dump.
//!
//! The report goes to stderr with the prefix `vork gpu fault:`. The vendor
//! dump (NVIDIA: an Aftermath-format crash dump, where the driver offers one)
//! goes to `VORK_DEVICE_FAULT_DUMP` (default `vork-device-fault.bin` in the
//! working directory), and the full report (every buffer and address range)
//! to the same path with the extension `txt`.

use alloc::{format, string::String, vec, vec::Vec};
use core::ffi::{c_void, CStr};
use core::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use ash::{ext, nv, vk};

fn level() -> u32 {
    static LEVEL: OnceLock<u32> = OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("VORK_DEVICE_FAULT") {
        Ok(value) if value == "2" => 2,
        Ok(value) if !value.is_empty() && value != "0" => 1,
        _ => 0,
    })
}

/// Whether fault reporting was asked for.
pub fn requested() -> bool {
    level() > 0
}

/// Device extensions to enable when the driver has them.
pub fn device_extensions() -> [&'static CStr; 4] {
    [
        ext::device_fault::NAME,
        ext::device_address_binding_report::NAME,
        nv::device_diagnostic_checkpoints::NAME,
        nv::device_diagnostics_config::NAME,
    ]
}

/// The structures fault reporting adds to `VkDeviceCreateInfo`.
#[derive(Default)]
pub struct CreateChain {
    fault: Option<vk::PhysicalDeviceFaultFeaturesEXT<'static>>,
    binding: Option<vk::PhysicalDeviceAddressBindingReportFeaturesEXT<'static>>,
    diagnostics: Option<(
        vk::PhysicalDeviceDiagnosticsConfigFeaturesNV<'static>,
        vk::DeviceDiagnosticsConfigCreateInfoNV<'static>,
    )>,
}

impl CreateChain {
    pub unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        enabled: &[&'static CStr],
    ) -> Self {
        let mut chain = Self::default();
        if !requested() {
            return chain;
        }
        if enabled.contains(&ext::device_fault::NAME) {
            let mut supported = vk::PhysicalDeviceFaultFeaturesEXT::default();
            let mut query = vk::PhysicalDeviceFeatures2::default().push_next(&mut supported);
            unsafe { instance.get_physical_device_features2(physical, &mut query) };
            VENDOR_BINARY.store(supported.device_fault_vendor_binary != 0, Ordering::SeqCst);
            chain.fault = Some(
                vk::PhysicalDeviceFaultFeaturesEXT::default()
                    .device_fault(supported.device_fault != 0)
                    .device_fault_vendor_binary(supported.device_fault_vendor_binary != 0),
            );
        }
        if enabled.contains(&ext::device_address_binding_report::NAME) {
            chain.binding = Some(
                vk::PhysicalDeviceAddressBindingReportFeaturesEXT::default()
                    .report_address_binding(true),
            );
        }
        if enabled.contains(&nv::device_diagnostics_config::NAME) {
            let mut flags = vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_RESOURCE_TRACKING
                | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_AUTOMATIC_CHECKPOINTS
                | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_SHADER_ERROR_REPORTING;
            if level() >= 2 {
                flags |= vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_SHADER_DEBUG_INFO;
            }
            chain.diagnostics = Some((
                vk::PhysicalDeviceDiagnosticsConfigFeaturesNV::default().diagnostics_config(true),
                vk::DeviceDiagnosticsConfigCreateInfoNV::default().flags(flags),
            ));
        }
        chain
    }

    pub fn add_to_device_create<'a>(
        &'a mut self,
        mut info: vk::DeviceCreateInfo<'a>,
    ) -> vk::DeviceCreateInfo<'a> {
        if let Some(fault) = self.fault.as_mut() {
            info = info.push_next(fault);
        }
        if let Some(binding) = self.binding.as_mut() {
            info = info.push_next(binding);
        }
        if let Some((features, config)) = self.diagnostics.as_mut() {
            info = info.push_next(features).push_next(config);
        }
        info
    }
}

struct State {
    device: ash::Device,
    queue: vk::Queue,
    fault: Option<ext::device_fault::Device>,
    checkpoints: Option<nv::device_diagnostic_checkpoints::Device>,
    messenger: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    enabled: Vec<&'static CStr>,
}

struct Marker {
    pass: String,
    pipeline: u64,
    what: String,
}

struct Range {
    bound: bool,
    internal: bool,
    base: u64,
    size: u64,
    object_type: vk::ObjectType,
    object: u64,
}

struct BufferRecord {
    handle: u64,
    size: u64,
    usage: vk::BufferUsageFlags,
}

#[derive(Default)]
struct Log {
    markers: Vec<Marker>,
    names: Vec<(u64, String)>,
    buffers: Vec<BufferRecord>,
    ranges: Vec<Range>,
}

impl Log {
    fn name(&self, handle: u64) -> &str {
        self.names
            .iter()
            .rev()
            .find(|(named, _)| *named == handle)
            .map_or("(unnamed)", |(_, name)| name.as_str())
    }
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static LOG: Mutex<Option<Log>> = Mutex::new(None);
static REPORTED: AtomicBool = AtomicBool::new(false);
/// Whether the driver will hand over a vendor crash dump after a loss.
static VENDOR_BINARY: AtomicBool = AtomicBool::new(false);

fn with_log<R>(run: impl FnOnce(&mut Log) -> R) -> R {
    let mut guard = LOG.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    run(guard.get_or_insert_with(Log::default))
}

unsafe extern "system" fn address_binding(
    _severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    if !kind.contains(vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING) || data.is_null() {
        return vk::FALSE;
    }
    let data = unsafe { &*data };
    let (object_type, object) = if data.object_count > 0 && !data.p_objects.is_null() {
        let object = unsafe { &*data.p_objects };
        (object.object_type, object.object_handle)
    } else {
        (vk::ObjectType::UNKNOWN, 0)
    };
    let mut next = data.p_next.cast::<vk::BaseInStructure<'_>>();
    while !next.is_null() {
        let header = unsafe { &*next };
        if header.s_type == vk::StructureType::DEVICE_ADDRESS_BINDING_CALLBACK_DATA_EXT {
            let binding =
                unsafe { &*next.cast::<vk::DeviceAddressBindingCallbackDataEXT<'_>>() };
            with_log(|log| {
                log.ranges.push(Range {
                    bound: binding.binding_type == vk::DeviceAddressBindingTypeEXT::BIND,
                    internal: binding
                        .flags
                        .contains(vk::DeviceAddressBindingFlagsEXT::INTERNAL_OBJECT),
                    base: binding.base_address,
                    size: binding.size,
                    object_type,
                    object,
                })
            });
        }
        next = header.p_next;
    }
    vk::FALSE
}

/// Start reporting for a newly created device.
pub unsafe fn install(
    entry: &ash::Entry,
    instance: &ash::Instance,
    instance_extensions: &[&'static CStr],
    device: &ash::Device,
    queue: vk::Queue,
    enabled: &[&'static CStr],
) {
    if !requested() {
        return;
    }
    // Extension functions and markers belong to one device. Report the first live one.
    if STATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).is_some() {
        std::eprintln!("vork gpu fault: another device is already reported; this one is not");
        return;
    }
    let fault = enabled
        .contains(&ext::device_fault::NAME)
        .then(|| ext::device_fault::Device::new(instance, device));
    let checkpoints = enabled
        .contains(&nv::device_diagnostic_checkpoints::NAME)
        .then(|| nv::device_diagnostic_checkpoints::Device::new(instance, device));
    let messenger = (enabled.contains(&ext::device_address_binding_report::NAME)
        && instance_extensions.contains(&ext::debug_utils::NAME))
    .then(|| {
        let functions = ext::debug_utils::Instance::new(entry, instance);
        let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING)
            .pfn_user_callback(Some(address_binding));
        unsafe { functions.create_debug_utils_messenger(&info, None) }
            .ok()
            .map(|messenger| (functions, messenger))
    })
    .flatten();
    let listed: Vec<_> = device_extensions()
        .into_iter()
        .map(|name| {
            format!(
                "{} {}",
                name.to_string_lossy(),
                if enabled.contains(&name) { "on" } else { "absent" }
            )
        })
        .collect();
    std::eprintln!(
        "vork gpu fault: reporting armed (level {}): {}; vendor crash dump {}; address ranges {}",
        level(),
        listed.join(", "),
        if VENDOR_BINARY.load(Ordering::SeqCst) { "offered" } else { "not offered by this driver" },
        if messenger.is_some() { "recorded" } else { "not recorded" }
    );
    *STATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(State {
        device: device.clone(),
        queue,
        fault,
        checkpoints,
        messenger,
        enabled: enabled.to_vec(),
    });
}

/// Stop reporting before `device` is destroyed.
pub unsafe fn uninstall(device: vk::Device) {
    if !requested() {
        return;
    }
    let mut guard = STATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if guard.as_ref().is_some_and(|state| state.device.handle() == device) {
        if let Some((functions, messenger)) = guard.take().and_then(|state| state.messenger) {
            unsafe { functions.destroy_debug_utils_messenger(messenger, None) };
        }
    }
}

/// Remember the label of a Vulkan object.
pub fn name(object_type: vk::ObjectType, handle: u64, name: &str) {
    if requested() {
        with_log(|log| log.names.push((handle, format!("{object_type:?} \"{name}\""))));
    }
}

/// Remember a buffer's size and usage.
pub fn buffer(handle: u64, size: u64, usage: vk::BufferUsageFlags) {
    if requested() {
        with_log(|log| log.buffers.push(BufferRecord { handle, size, usage }));
    }
}

/// Record a checkpoint in `cmd` describing the dispatch about to be recorded.
pub unsafe fn checkpoint(device: vk::Device, cmd: vk::CommandBuffer, pass: &str, pipeline: vk::Pipeline, what: String) {
    if !requested() {
        return;
    }
    let guard = STATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(functions) = guard.as_ref().filter(|state| state.device.handle() == device)
        .and_then(|state| state.checkpoints.as_ref()) else {
        return;
    };
    let id = with_log(|log| {
        log.markers.push(Marker {
            pass: pass.into(),
            pipeline: vk::Handle::as_raw(pipeline),
            what,
        });
        log.markers.len()
    });
    // The marker is the 1-based index of the record, carried as a pointer value.
    unsafe { functions.cmd_set_checkpoint(cmd, id as *const c_void) };
}

fn text(raw: &[core::ffi::c_char]) -> String {
    let bytes: Vec<u8> = raw.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Print everything the driver says about the device loss. Runs once.
pub fn report() {
    if !requested() || REPORTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let guard = STATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(state) = guard.as_ref() else {
        std::eprintln!("vork gpu fault: device lost, but no device had fault reporting installed");
        return;
    };
    let dump = std::env::var_os("VORK_DEVICE_FAULT_DUMP")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("vork-device-fault.bin"));
    let mut lines: Vec<String> = Vec::new();
    let mut detail: Vec<String> = Vec::new();
    lines.push(format!(
        "VK_ERROR_DEVICE_LOST; extensions enabled: {:?}",
        state.enabled.iter().filter(|name| device_extensions().contains(name)).collect::<Vec<_>>()
    ));

    // Checkpoints first: they do not depend on the fault query succeeding.
    let mut reached: Vec<(vk::PipelineStageFlags, usize)> = Vec::new();
    match state.checkpoints.as_ref() {
        Some(functions) => {
            let count = unsafe { functions.get_queue_checkpoint_data_len(state.queue) };
            let mut data = vec![vk::CheckpointDataNV::default(); count];
            if count > 0 {
                let mut written = count as u32;
                unsafe {
                    (functions.fp().get_queue_checkpoint_data_nv)(
                        state.queue,
                        &mut written,
                        data.as_mut_ptr(),
                    )
                };
                data.truncate(written as usize);
            }
            lines.push(format!("checkpoints: driver returned {}", data.len()));
            for entry in &data {
                reached.push((entry.stage, entry.p_checkpoint_marker as usize));
            }
        }
        None => lines.push("checkpoints: VK_NV_device_diagnostic_checkpoints not enabled".into()),
    }

    let mut addresses: Vec<vk::DeviceFaultAddressInfoEXT> = Vec::new();
    match state.fault.as_ref() {
        Some(functions) => {
            let query = functions.fp().get_device_fault_info_ext;
            let mut counts = vk::DeviceFaultCountsEXT::default();
            let first = unsafe { query(state.device.handle(), &mut counts, ptr::null_mut()) };
            lines.push(format!(
                "vkGetDeviceFaultInfoEXT counts: {first:?}; address infos {}, vendor infos {}, vendor binary {} bytes",
                counts.address_info_count, counts.vendor_info_count, counts.vendor_binary_size
            ));
            addresses = vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
            let mut vendors =
                vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
            let mut binary = vec![0u8; counts.vendor_binary_size as usize];
            let mut info = vk::DeviceFaultInfoEXT::default();
            if !addresses.is_empty() {
                info.p_address_infos = addresses.as_mut_ptr();
            }
            if !vendors.is_empty() {
                info.p_vendor_infos = vendors.as_mut_ptr();
            }
            if !binary.is_empty() {
                info.p_vendor_binary_data = binary.as_mut_ptr().cast();
            }
            let second = unsafe { query(state.device.handle(), &mut counts, &mut info) };
            addresses.truncate(counts.address_info_count as usize);
            vendors.truncate(counts.vendor_info_count as usize);
            binary.truncate(counts.vendor_binary_size as usize);
            lines.push(format!(
                "vkGetDeviceFaultInfoEXT: {second:?}; description \"{}\"",
                text(&info.description)
            ));
            for (index, address) in addresses.iter().enumerate() {
                let precision = address.address_precision.max(1);
                let low = address.reported_address & !(precision - 1);
                lines.push(format!(
                    "address info {index}: type {:?}, reported address {:#x}, precision {:#x} (range {:#x}..={:#x})",
                    address.address_type,
                    address.reported_address,
                    address.address_precision,
                    low,
                    low.saturating_add(precision - 1)
                ));
            }
            for (index, vendor) in vendors.iter().enumerate() {
                lines.push(format!(
                    "vendor info {index}: \"{}\", code {:#x}, data {:#x}",
                    text(&vendor.description),
                    vendor.vendor_fault_code,
                    vendor.vendor_fault_data
                ));
            }
            if !binary.is_empty() {
                let path = &dump;
                match std::fs::write(path, &binary) {
                    Ok(()) => lines.push(format!(
                        "vendor binary: {} bytes written to {}",
                        binary.len(),
                        path.display()
                    )),
                    Err(error) => lines.push(format!("vendor binary: not written: {error}")),
                }
            }
        }
        None => lines.push("fault info: VK_EXT_device_fault not enabled".into()),
    }
    drop(guard);

    with_log(|log| {
        for (stage, id) in &reached {
            let described = match id.checked_sub(1).and_then(|index| log.markers.get(index)) {
                Some(marker) => format!(
                    "marker {id} of {}: pass \"{}\", pipeline {:#x} {}, {}",
                    log.markers.len(),
                    marker.pass,
                    marker.pipeline,
                    log.name(marker.pipeline),
                    marker.what
                ),
                None => format!("marker {id:#x} (not one of the {} vork recorded)", log.markers.len()),
            };
            lines.push(format!("checkpoint: last reached by stage {stage:?}: {described}"));
        }
        // The address ranges still bound at the loss, newest record per range start.
        let mut live: Vec<&Range> = Vec::new();
        for range in &log.ranges {
            if range.bound {
                live.push(range);
            } else if let Some(at) = live
                .iter()
                .rposition(|bound| bound.base == range.base && bound.object == range.object)
            {
                live.remove(at);
            }
        }
        let describe = |range: &Range| {
            let size = log
                .buffers
                .iter()
                .rev()
                .find(|buffer| buffer.handle == range.object)
                .map_or(String::new(), |buffer| {
                    format!(", buffer size {} usage {:?}", buffer.size, buffer.usage)
                });
            format!(
                "{:#x}..{:#x} ({} bytes) {:?} {:#x} {}{}{}",
                range.base,
                range.base.saturating_add(range.size),
                range.size,
                range.object_type,
                range.object,
                log.name(range.object),
                size,
                if range.internal { " [driver internal]" } else { "" }
            )
        };
        let is_instruction = |address: &vk::DeviceFaultAddressInfoEXT| {
            [
                vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_UNKNOWN,
                vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_INVALID,
                vk::DeviceFaultAddressTypeEXT::INSTRUCTION_POINTER_FAULT,
            ]
            .contains(&address.address_type)
        };
        let instructions = addresses.iter().filter(|address| is_instruction(address)).count();
        if !addresses.is_empty() {
            lines.push(format!(
                "{} of {} address infos are instruction pointers (where shader code was executing), {} are memory addresses{}",
                instructions,
                addresses.len(),
                addresses.len() - instructions,
                if instructions == addresses.len() {
                    ": the driver reports no invalid memory access"
                } else {
                    ""
                }
            ));
        }
        for (index, address) in addresses.iter().enumerate() {
            if is_instruction(address) {
                continue;
            }
            let precision = address.address_precision.max(1);
            let low = address.reported_address & !(precision - 1);
            let high = low.saturating_add(precision - 1);
            let overlaps =
                |range: &&Range| range.base <= high && range.base.saturating_add(range.size) > low;
            let mut found = false;
            for range in live.iter().copied().filter(overlaps) {
                found = true;
                lines.push(format!("address info {index} lies in bound range {}", describe(range)));
            }
            for range in log.ranges.iter().filter(|range| !range.bound).filter(overlaps) {
                found = true;
                lines.push(format!(
                    "address info {index} lies in a range unbound before the loss: {}",
                    describe(range)
                ));
            }
            if !found {
                let below = live.iter().filter(|range| range.base <= low).max_by_key(|range| range.base);
                let above = live.iter().filter(|range| range.base > high).min_by_key(|range| range.base);
                lines.push(format!(
                    "address info {index} lies in no recorded range ({} recorded, {} bound)",
                    log.ranges.len(),
                    live.len()
                ));
                if let Some(range) = below {
                    lines.push(format!("  nearest bound range below: {}", describe(range)));
                }
                if let Some(range) = above {
                    lines.push(format!("  nearest bound range above: {}", describe(range)));
                }
            }
        }
        detail.push(format!("markers recorded: {}", log.markers.len()));
        detail.push(format!("buffers created: {}", log.buffers.len()));
        for buffer in &log.buffers {
            detail.push(format!(
                "buffer {:#x} size {} usage {:?} {}",
                buffer.handle,
                buffer.size,
                buffer.usage,
                log.name(buffer.handle)
            ));
        }
        detail.push(format!("address ranges bound at the loss: {}", live.len()));
        let mut sorted = live.clone();
        sorted.sort_by_key(|range| range.base);
        for range in sorted {
            detail.push(format!("range {}", describe(range)));
        }
    });

    for line in &lines {
        std::eprintln!("vork gpu fault: {line}");
    }
    let path = dump.with_extension("txt");
    let mut body = lines.join("\n");
    body.push('\n');
    body.push_str(&detail.join("\n"));
    body.push('\n');
    match std::fs::write(&path, body) {
        Ok(()) => std::eprintln!("vork gpu fault: full report written to {}", path.display()),
        Err(error) => std::eprintln!("vork gpu fault: full report not written: {error}"),
    }
}

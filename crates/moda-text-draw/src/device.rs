//! device.rs — excerpt of moda-engine's `GpuContext`: the device and queue the text pipelines draw with (see
//! PROVENANCE.md).

#[derive(Clone)]
pub struct GpuContext<'a> {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Called before allocating or uploading the packed resources of a capture page.
    pub admission: PageAdmission<'a>,
}

/// Native storage and written range, reported by the resource builder after packing.
#[derive(Clone, Copy, Default)]
pub struct PageAdmission<'a>(pub Option<&'a dyn Fn(u64, u64)>);

impl PageAdmission<'_> {
    pub fn charge(self, storage: u64, upload: u64) {
        if let Some(admit) = self.0 { admit(storage, upload); }
    }
    pub fn texture(self, device: &wgpu::Device, desc: &wgpu::TextureDescriptor) -> wgpu::Texture {
        if let Some(admit) = self.0 {
            let bytes = device.texture_binding_size(desc).unwrap_or_else(|| estimated_texture_bytes(
                desc.size, desc.dimension, desc.format, desc.mip_level_count, desc.sample_count));
            admit(bytes, 0);
        }
        device.create_texture(desc)
    }
}

/// Storage estimate for backends without native allocation sizes, including block rounding and mips.
pub fn estimated_texture_bytes(size: wgpu::Extent3d, dimension: wgpu::TextureDimension,
    format: wgpu::TextureFormat, levels: u32, samples: u32) -> u64 {
    (0..levels).map(|level| format.theoretical_memory_footprint(size.mip_level_size(level, dimension)))
        .sum::<u64>() * u64::from(samples)
}

/// How the host creates driver pipelines on a device. vork's renderer installs its own creation here, so Moda's
/// pipelines go through the persistent pipeline cache (a Metal binary archive on macOS) and are timed and counted
/// with every other pipeline; without a host, creation is the device's own.
pub trait PipelineHost: Send + Sync {
    /// Admit one optional family on a worker. The callback builds directly through
    /// the supplied cache so required allocation recovery never consumes its errors.
    #[cfg(not(target_arch = "wasm32"))]
    fn optional(&self, family: &'static str, build: OptionalBuild<'_>) -> anyhow::Result<()>;
    /// Hints from successful canvas builds select the descriptors to prepare on a later device.
    fn has_capture_hint(&self, _hint: &str) -> bool { false }
    fn remember_capture_hint(&self, _hint: &str) {}
    /// `descriptor.cache` is the host's to choose.
    fn render_pipeline(&self, descriptor: &wgpu::RenderPipelineDescriptor<'_>) -> wgpu::RenderPipeline;
    /// `descriptor.cache` is the host's to choose.
    fn compute_pipeline(&self, descriptor: &wgpu::ComputePipelineDescriptor<'_>) -> wgpu::ComputePipeline;
}

type Host = std::sync::Arc<dyn PipelineHost>;

struct DeviceHost {
    device: wgpu::Device,
    host: Option<Host>,
    #[cfg(not(target_arch = "wasm32"))]
    captures: std::sync::Mutex<std::collections::HashMap<std::any::TypeId, std::sync::Arc<dyn std::any::Any + Send + Sync>>>,
    #[cfg(not(target_arch = "wasm32"))]
    jobs: OptionalJobs,
}

static PIPELINE_HOSTS: std::sync::Mutex<Vec<std::sync::Arc<DeviceHost>>> = std::sync::Mutex::new(Vec::new());

/// Owns optional compilation for one open device. Jobs hold the device state, never
/// this registration, so dropping the registration can always join the worker.
pub struct PipelineDevice(std::sync::Arc<DeviceHost>);

impl PipelineDevice {
    pub fn new(device: &wgpu::Device, host: Option<Host>) -> Self {
        let owner = std::sync::Arc::new(DeviceHost {
            device: device.clone(), host,
            #[cfg(not(target_arch = "wasm32"))]
            captures: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            jobs: Default::default(),
        });
        let mut hosts = PIPELINE_HOSTS.lock().unwrap();
        hosts.push(owner.clone());
        Self(owner)
    }
}

impl Drop for PipelineDevice {
    fn drop(&mut self) {
        // Bootstrap can look up this device again to enqueue child pipelines. Keep
        // its closed queue visible until the worker has returned from driver code.
        #[cfg(not(target_arch = "wasm32"))]
        self.0.jobs.close();
        PIPELINE_HOSTS.lock().unwrap().retain(|owner| !std::sync::Arc::ptr_eq(owner, &self.0));
    }
}

fn device_host(device: &wgpu::Device) -> Option<std::sync::Arc<DeviceHost>> {
    PIPELINE_HOSTS.lock().unwrap().iter().rev().find(|owner| owner.device == *device).cloned()
}

fn pipeline_host(device: &wgpu::Device) -> Option<Host> {
    device_host(device).and_then(|owner| owner.host.clone())
}

/// The engine supplies the concrete type so the text crate need not depend on the engine.
#[cfg(not(target_arch = "wasm32"))]
pub fn capture_pipelines<T: Default + Send + Sync + 'static>(device: &wgpu::Device) -> std::sync::Arc<T> {
    device_host(device).filter(|owner| owner.host.is_some()).map_or_else(Default::default, |owner| {
        owner.captures.lock().unwrap().entry(std::any::TypeId::of::<T>())
            .or_insert_with(|| std::sync::Arc::new(T::default())).clone().downcast::<T>().expect("matched TypeId")
    })
}

#[cfg(not(target_arch = "wasm32"))]
type Job = Box<dyn FnOnce() + Send>;

/// One optional driver creation at a time per host. Required callers never use this queue.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct OptionalJobs(std::sync::Arc<std::sync::Mutex<Worker>>);

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
enum Worker {
    #[default]
    Idle,
    Running(std::sync::mpsc::Sender<Job>, std::thread::JoinHandle<()>),
    Closed,
}

#[cfg(not(target_arch = "wasm32"))]
impl OptionalJobs {
    fn enqueue(&self, job: Job) -> anyhow::Result<()> {
        let mut worker = self.0.lock().unwrap();
        if matches!(*worker, Worker::Idle) {
            let (send, recv) = std::sync::mpsc::channel::<Job>();
            let queue = std::sync::Arc::downgrade(&self.0);
            let thread = std::thread::Builder::new().name("canvas pipelines".into()).spawn(move || {
                for job in recv {
                    // Claim one job under the same lock as close, then compile outside it.
                    if !queue.upgrade().is_some_and(|q| matches!(*q.lock().unwrap(), Worker::Running(..))) {
                        break;
                    }
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                }
            })?;
            *worker = Worker::Running(send, thread);
        }
        match &*worker {
            Worker::Running(send, _) => send.send(job).map_err(|_| anyhow::anyhow!("canvas pipeline worker stopped")),
            _ => anyhow::bail!("canvas pipeline worker stopped"),
        }
    }

    fn close(&self) {
        let worker = std::mem::replace(&mut *self.0.lock().unwrap(), Worker::Closed);
        if let Worker::Running(sender, thread) = worker {
            drop(sender);
            if thread.thread().id() != std::thread::current().id() {
                let _ = thread.join();
            }
        }
    }
}

/// Every Moda render pipeline is created here.
pub fn create_render_pipeline(device: &wgpu::Device, descriptor: &wgpu::RenderPipelineDescriptor<'_>) -> wgpu::RenderPipeline {
    match pipeline_host(device) {
        Some(host) => host.render_pipeline(descriptor),
        None => device.create_render_pipeline(descriptor),
    }
}

/// Every Moda compute pipeline is created here.
pub fn create_compute_pipeline(device: &wgpu::Device, descriptor: &wgpu::ComputePipelineDescriptor<'_>) -> wgpu::ComputePipeline {
    match pipeline_host(device) {
        Some(host) => host.compute_pipeline(descriptor),
        None => device.create_compute_pipeline(descriptor),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub type OptionalBuild<'a> = Box<dyn FnOnce(Option<&wgpu::PipelineCache>) -> anyhow::Result<()> + 'a>;

/// Scopes are thread-local in native wgpu. Pop every scope before handing the
/// result back to the publisher. Device loss has its separate device callback.
#[cfg(not(target_arch = "wasm32"))]
pub fn scoped_creation<T>(device: &wgpu::Device, build: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
    let allocation = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = build();
    let errors = [validation, internal, allocation].map(|scope| pollster::block_on(scope.pop()));
    anyhow::ensure!(!device.is_lost(), "device lost during pipeline creation");
    if let Some(error) = errors.into_iter().flatten().next() { return Err(error.into()); }
    result
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct OptionalCompiler {
    device: wgpu::Device,
    owner: Option<std::sync::Arc<DeviceHost>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl OptionalCompiler {
    /// Keep the submitting device's host alive until this worker publishes.
    pub fn new(device: &wgpu::Device) -> Self { Self { device: device.clone(), owner: device_host(device) } }

    fn host(&self) -> Option<&Host> { self.owner.as_ref()?.host.as_ref() }

    pub fn has_capture_hint(&self, hint: &str) -> bool {
        self.host().is_some_and(|h| h.has_capture_hint(hint))
    }

    pub fn remember_capture_hint(&self, hint: &str) {
        if let Some(h) = self.host() { h.remember_capture_hint(hint); }
    }

    pub fn enqueue(&self, job: impl FnOnce() + Send + 'static) -> anyhow::Result<()> {
        let owner = self.owner.as_ref().ok_or_else(|| anyhow::anyhow!("canvas pipeline device is closed"))?;
        owner.jobs.enqueue(Box::new(job))
    }

    pub fn compile<T>(&self, family: &'static str, build: impl FnOnce(&wgpu::Device, Option<&wgpu::PipelineCache>) -> anyhow::Result<T>) -> anyhow::Result<T> {
        if let Some(host) = self.host() {
            let mut value = None;
            host.optional(family, Box::new(|cache| { value = Some(build(&self.device, cache)?); Ok(()) }))?;
            value.ok_or_else(|| anyhow::anyhow!("optional pipeline host omitted creation"))
        } else {
            scoped_creation(&self.device, || build(&self.device, None))
        }
    }
}

/// A shared descriptor builder can serve an ordinary draw or an admitted optional job.
#[derive(Clone, Copy)]
pub enum PipelineCreation<'a> {
    Required,
    Optional(Option<&'a wgpu::PipelineCache>),
}

impl PipelineCreation<'_> {
    pub fn render(self, device: &wgpu::Device, descriptor: &wgpu::RenderPipelineDescriptor<'_>) -> wgpu::RenderPipeline {
        match self {
            Self::Required => create_render_pipeline(device, descriptor),
            Self::Optional(cache) => device.create_render_pipeline(&wgpu::RenderPipelineDescriptor { cache, ..descriptor.clone() }),
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn close_discards_queued_jobs_and_joins_the_active_job() -> anyhow::Result<()> {
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        let timeout = Duration::from_secs(5);
        let jobs = Arc::new(OptionalJobs::default());
        let active = jobs.clone();
        let (started, compiling) = mpsc::channel();
        let (finished, joined) = mpsc::channel();
        jobs.enqueue(Box::new(move || {
            started.send(()).unwrap();
            let deadline = Instant::now() + timeout;
            while !matches!(*active.0.lock().unwrap(), Worker::Closed) {
                assert!(Instant::now() < deadline, "queue never closed");
                std::thread::yield_now();
            }
            finished.send(()).unwrap();
        }))?;
        compiling.recv_timeout(timeout)?;
        let (second, ran) = mpsc::channel();
        jobs.enqueue(Box::new(move || second.send(()).unwrap()))?;
        let (closed, done) = mpsc::channel();
        std::thread::spawn(move || {
            jobs.close();
            let _ = closed.send(());
        });
        done.recv_timeout(timeout)?;
        joined.try_recv()?;
        assert_eq!(ran.try_recv(), Err(mpsc::TryRecvError::Disconnected));
        Ok(())
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn closing_during_compilation_joins_the_job_and_rejects_its_child() -> anyhow::Result<()> {
        let adapter = pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))?;
        let (device, _queue) = pollster::block_on(adapter.request_device(&Default::default()))?;
        let registration = PipelineDevice::new(&device, None);
        let compiler = OptionalCompiler::new(&device);
        let worker = compiler.clone();
        let (started, compiling) = mpsc::sync_channel(0);
        let (finished, joined) = mpsc::channel();

        compiler.enqueue(move || {
            let result = worker.compile("teardown test", |device, _| {
                started.send(())?;
                let owner = worker.owner.as_ref().unwrap();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while !matches!(*owner.jobs.0.lock().unwrap(), Worker::Closed) {
                    anyhow::ensure!(std::time::Instant::now() < deadline, "device never closed its queue");
                    std::thread::yield_now();
                }
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
                });
                let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None, layout: None, module: &module, entry_point: Some("main"),
                    compilation_options: Default::default(), cache: None,
                });
                // The bootstrap looks up the device afresh when it requests the next family.
                anyhow::ensure!(OptionalCompiler::new(device).enqueue(|| {}).is_err(), "child escaped closing queue");
                Ok(())
            });
            finished.send(result).unwrap();
        })?;

        compiling.recv()?;
        drop(registration);
        joined.try_recv()??;
        assert!(compiler.enqueue(|| {}).is_err());
        assert!(OptionalCompiler::new(&device).enqueue(|| {}).is_err());
        Ok(())
    }
}

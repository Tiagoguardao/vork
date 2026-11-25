use std::{
    process::{Child, ChildStdout, Command},
    sync::{Arc, Mutex, Weak},
};

/// External decoders must be stopped with their read ends still open, including
/// when the CLI exits without dropping the GPU installation that owns the decoder.
pub struct DecoderProcess(Arc<Mutex<Child>>);
static DECODERS: Mutex<Vec<Weak<Mutex<Child>>>> = Mutex::new(Vec::new());

impl DecoderProcess {
    pub fn start(mut command: Command) -> std::io::Result<Self> {
        let mut live = DECODERS.lock().unwrap();
        let child = Arc::new(Mutex::new(command.spawn()?));
        live.retain(|child| child.strong_count() != 0);
        live.push(Arc::downgrade(&child));
        Ok(Self(child))
    }

    pub fn stdout(&self) -> ChildStdout {
        self.0
            .lock()
            .unwrap()
            .stdout
            .take()
            .expect("piped decoder output")
    }

    pub fn stderr(&self) -> Option<std::process::ChildStderr> {
        self.0.lock().unwrap().stderr.take()
    }

    pub fn wait(&self) -> std::io::Result<std::process::ExitStatus> {
        self.0.lock().unwrap().wait()
    }

    fn stop(child: &Mutex<Child>) {
        let mut child = child.lock().unwrap();
        let _ = child.kill();
        let _ = child.wait();
    }
}

impl Drop for DecoderProcess {
    fn drop(&mut self) {
        Self::stop(&self.0);
    }
}

pub fn stop_decoders_for_exit() {
    for child in DECODERS.lock().unwrap().iter().filter_map(Weak::upgrade) {
        DecoderProcess::stop(&child);
    }
}

#[path = "pixel_pipes.rs"]
mod pixel_pipes;
pub use pixel_pipes::PixelPipes;

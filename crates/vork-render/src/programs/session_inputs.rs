//! Compiler workers inherit the session candidate's input observations. Native
//! pipeline workers carry that candidate explicitly in the shader scene.
use std::{cell::RefCell, sync::{Arc, Mutex}};
use vork_slang::InstalledInputs;

pub(crate) type Capture = Arc<Mutex<InstalledInputs>>;
thread_local! {
    static CURRENT: RefCell<Option<Capture>> = const { RefCell::new(None) };
}

pub(crate) struct Scope {
    previous: Option<Capture>,
    _host: Option<vork_inputs::Scope>,
}
impl Drop for Scope {
    fn drop(&mut self) { CURRENT.with(|c| c.replace(self.previous.take())); }
}
pub(crate) fn enter(inputs: Option<Capture>) -> Scope {
    let host = inputs.as_ref().and_then(|capture| capture.lock().unwrap().host.enter_worker());
    Scope { previous: CURRENT.with(|c| c.replace(inputs)), _host: host }
}
pub(crate) fn current() -> Option<Capture> { CURRENT.with(|c| c.borrow().clone()) }
pub(crate) fn record(identity: &vork_slang::InputIdentity) -> anyhow::Result<()> {
    if let Some(inputs) = current() { inputs.lock().unwrap().record(identity)?; }
    Ok(())
}

//! A kept-alive worker keeps its device and installation across whole-document
//! reopens that only change values, and its pictures are byte-identical to
//! one-shot renders of the same documents.
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

fn binary() -> std::ffi::OsString {
    std::env::var_os("VORK_BIN").unwrap_or_else(|| env!("CARGO_BIN_EXE_vork").into())
}

fn scene(energy: f32, ball: f32) -> serde_json::Value {
    serde_json::json!({"version":2,"render":{"resolution":[64,48],"duration":1},"nodes":[
        {"type":"mesh","name":"ball","translation":[0,0.5,0],"geometry":{"std":"sphere","radius":ball,"segments":24,"rings":12},
            "materials":[{"inputs":{"base_color":[0.8,0.3,0.2],"specular_roughness":0.4}}]},
        {"type":"mesh","name":"floor","geometry":{"std":"grid","size":[20,20]},"materials":[{"inputs":{"base_color":[0.5,0.5,0.5],"specular_roughness":0.7}}]},
        {"type":"light","name":"key","shape":"point","translation":[1.5,3,2],"energy":energy,"radius":0,"color":[1,0.9,0.8]},
        {"type":"camera","name":"cam","translation":[0,1.2,3.5]}]})
}
/// The scene with two unnamed compositor entries whose order changes the picture.
fn composited(order: [&str; 2]) -> serde_json::Value {
    let mut document = scene(140., 0.5);
    document["programs"] = serde_json::json!({
        "offset": {"source": "struct Inputs { float amount=0.1; }; float4 composite(CompositeContext ctx, Inputs inputs) { float4 c=compositor_sample(ctx.uv); return float4(c.rgb+inputs.amount,c.a); }"},
        "gain": {"source": "struct Inputs { float gain=0.5; }; float4 composite(CompositeContext ctx, Inputs inputs) { float4 c=compositor_sample(ctx.uv); return float4(c.rgb*inputs.gain,c.a); }"}});
    document["materials"] = serde_json::json!({"offset": {"program": "offset"}, "gain": {"program": "gain"}});
    document["composition"]["space"] = "3d".into(); document["composition"]["compositor"] = serde_json::json!(order);
    document
}
const RENDER_ARGS: [&str; 6] = ["--profile", "std/draft", "--set", "samples=1", "--set", "denoise=false"];

fn cold_with(root: &Path, document: &serde_json::Value, name: &str, extra: &[&str]) -> Vec<u8> {
    let scene_path = root.join(format!("{name}.json"));
    std::fs::write(&scene_path, serde_json::to_vec(document).unwrap()).unwrap();
    let out = root.join(format!("{name}.png"));
    let output = Command::new(binary()).arg("render").arg(&scene_path).args(RENDER_ARGS).args(extra)
        .arg("--out").arg(&out).stdout(Stdio::piped()).stderr(Stdio::piped()).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    std::fs::read(out).unwrap()
}
fn cold(root: &Path, document: &serde_json::Value, name: &str) -> Vec<u8> { cold_with(root, document, name, &[]) }
/// The one-shot CLI's error line for a scene path, without the `Error: ` prefix.
fn cold_error(scene_path: &Path) -> String {
    let output = Command::new(binary()).arg("render").arg(scene_path).args(RENDER_ARGS)
        .arg("--out").arg(scene_path.with_extension("png")).stdout(Stdio::piped()).stderr(Stdio::piped()).output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    stderr.lines().find_map(|line| line.strip_prefix("Error: ")).unwrap_or_else(|| panic!("no error line in {stderr}")).to_owned()
}

struct Session { child: std::process::Child, replies: BufReader<std::process::ChildStdout> }
impl Session {
    fn start() -> Self {
        Self::command(Command::new(binary()))
    }
    fn profiled(log: &Path) -> Self {
        let mut command = Command::new(binary());
        command.env("MODA_RENDER_PROFILE", "1").stderr(std::fs::File::create(log).unwrap());
        Self::command(command)
    }
    fn command(mut command: Command) -> Self {
        let mut child = command.arg("session")
            .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let replies = BufReader::new(child.stdout.take().unwrap());
        Self { child, replies }
    }
    fn send(&mut self, request: serde_json::Value) -> serde_json::Value {
        let stdin = self.child.stdin.as_mut().unwrap();
        writeln!(stdin, "{request}").unwrap();
        stdin.flush().unwrap();
        let mut line = String::new();
        assert!(self.replies.read_line(&mut line).unwrap() > 0, "worker closed stdout after {request}");
        let reply: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(reply["id"], request["id"], "{reply}");
        reply
    }
    /// Open the whole document in memory; `base` is where its relative files live.
    fn open(&mut self, id: u32, document: &serde_json::Value, base: &Path) -> serde_json::Value {
        self.send(serde_json::json!({"id": id, "op": "open", "document": document, "base": base}))
    }
    fn render(&mut self, id: u32, out: &Path) -> serde_json::Value {
        let reply = self.send(serde_json::json!({"id": id, "op": "render", "out": out, "profile": "std/draft",
            "set": {"samples": 1, "denoise": false}}));
        assert_eq!(reply["ok"], true, "{reply}");
        assert_eq!(reply["result"]["frames"], 1, "{reply}");
        assert_eq!(reply["provisional"], false, "{reply}");
        reply
    }
}

#[test]
fn reopened_documents_reuse_what_they_can_and_match_one_shot_pictures() {
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("SKIP GPU session test: set VORK_GPU_TESTS=1 on a supported GPU");
        return;
    }
    let root = std::env::temp_dir().join(format!("vork-session-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let (dim, bright, bigger) = (scene(60., 0.5), scene(140., 0.5), scene(140., 0.7));
    let (first, second) = (composited(["offset", "gain"]), composited(["gain", "offset"]));
    let cold_dim = cold(&root, &dim, "cold-dim");
    let cold_bright = cold(&root, &bright, "cold-bright");
    let cold_bigger = cold(&root, &bigger, "cold-bigger");
    let cold_first = cold(&root, &first, "cold-first");
    let cold_second = cold(&root, &second, "cold-second");
    assert!(cold_dim != cold_bright && cold_bright != cold_bigger && cold_first != cold_second, "the edits must change the picture");
    let scene_path = root.join("cold-dim.json");

    let mut session = Session::start();
    // The first open has nothing to compare with.
    let reply = session.send(serde_json::json!({"id": 1, "op": "open", "scene": scene_path}));
    assert_eq!(reply, serde_json::json!({"id": 1, "ok": true, "revision": 1, "reuse": "reinstalled", "changed": []}));
    let rendered = session.render(2, &root.join("s0.png"));
    assert_eq!(rendered["reuse"], "reinstalled");
    assert_eq!(rendered["installs"], 1);
    // The first open prepared its candidate on the device under a default still; this
    // render's options differ, so the shot installs its own and the device was open.
    assert_eq!(rendered["device_reused"], true);
    assert_eq!(rendered["install_reused"], false, "{rendered}");
    assert_eq!(std::fs::read(root.join("s0.png")).unwrap(), cold_dim, "first session render differs from the one-shot render");

    // The identical document again: nothing dropped, nothing written.
    let reply = session.send(serde_json::json!({"id": 3, "op": "open", "scene": scene_path}));
    assert_eq!(reply, serde_json::json!({"id": 3, "ok": true, "revision": 2, "reuse": "reused", "changed": []}));
    let again = session.render(4, &root.join("s0b.png"));
    assert_eq!(again["reuse"], "reused", "{again}");
    assert_eq!(again["installs"], 1);
    assert_eq!(again["device_reused"], true);
    assert_eq!(again["install_reused"], true);
    assert_eq!(std::fs::read(root.join("s0b.png")).unwrap(), cold_dim);

    // A light energy change is written into the installed records in place.
    let reply = session.open(5, &bright, &root);
    assert_eq!(reply, serde_json::json!({"id": 5, "ok": true, "revision": 3, "reuse": "updated", "changed": [["nodes", "key", "energy"]]}));
    let updated = session.render(6, &root.join("s1.png"));
    assert_eq!(updated["reuse"], "updated", "{updated}");
    assert_eq!(updated["installs"], 1);
    assert_eq!(updated["install_reused"], true);
    assert_eq!(std::fs::read(root.join("s1.png")).unwrap(), cold_bright, "value-updated session render differs from the one-shot render");

    // Geometry is an installed fact: the document is reinstalled on the kept device.
    let reply = session.open(7, &bigger, &root);
    assert_eq!(reply, serde_json::json!({"id": 7, "ok": true, "revision": 4, "reuse": "reinstalled", "changed": [["nodes", "ball", "geometry", "radius"]]}));
    // The open prepared the candidate under the last render's options and kept
    // it, so this render draws from it without installing again.
    let reinstalled = session.render(8, &root.join("s2.png"));
    assert_eq!(reinstalled["reuse"], "reinstalled", "{reinstalled}");
    assert_eq!(reinstalled["installs"], 2);
    assert_eq!(reinstalled["device_reused"], true);
    assert_eq!(reinstalled["install_reused"], true, "{reinstalled}");
    assert_eq!(std::fs::read(root.join("s2.png")).unwrap(), cold_bigger, "reinstalled session render differs from the one-shot render");

    // A document the loader refuses fails at its authored path; the previous revision still renders.
    let mut broken = bigger.clone();
    broken["nodes"][2]["energy"] = "bright".into();
    let reply = session.open(9, &broken, &root);
    assert_eq!(reply["ok"], false);
    assert_eq!(reply["kind"], "scene", "{reply}");
    assert!(reply["error"].as_str().unwrap().contains("energy"), "{reply}");
    let kept = session.render(10, &root.join("s2b.png"));
    assert_eq!(kept["revision"], 4);
    assert_eq!(kept["reuse"], "reused", "{kept}");
    assert_eq!(kept["installs"], 2);
    assert_eq!(std::fs::read(root.join("s2b.png")).unwrap(), cold_bigger);

    // What only an install can refuse is refused by open too, with the one-shot's
    // error: a program that does not compile, a missing external buffer, a missing
    // scene file. The previous revision keeps rendering after each.
    let mut invalid = bigger.clone();
    invalid["nodes"][0]["materials"] = serde_json::json!([{"program": {"source": "THIS IS NOT A VALID SLANG PROGRAM"}}]);
    std::fs::write(root.join("invalid.json"), serde_json::to_vec(&invalid).unwrap()).unwrap();
    let invalid_error = cold_error(&root.join("invalid.json")).replace("invalid.json", "(document)");
    assert!(invalid_error.contains("materials[0].program"), "{invalid_error}");
    // The compiler's diagnostic runs over several lines; the one-shot's first line is the contract.
    let first_line = |reply: &serde_json::Value| reply["error"].as_str().unwrap().lines().next().unwrap().to_owned();
    let reply = session.open(30, &invalid, &root);
    assert_eq!(reply["ok"], false, "{reply}");
    assert_eq!(first_line(&reply), invalid_error, "{reply}");
    // The same refused document again is refused again, not remembered as current.
    let reply = session.open(36, &invalid, &root);
    assert_eq!(reply["ok"], false, "{reply}");
    assert_eq!(first_line(&reply), invalid_error, "{reply}");
    let mut unreadable = bigger.clone();
    unreadable["buffers"] = serde_json::json!({"positions": {"type": "vec3", "file": "missing.bin", "count": 3}});
    unreadable["nodes"][0]["geometry"] = serde_json::json!({"attributes": {"POSITION": "positions"}});
    std::fs::write(root.join("unreadable.json"), serde_json::to_vec(&unreadable).unwrap()).unwrap();
    let reply = session.open(31, &unreadable, &root);
    assert_eq!(reply["ok"], false, "{reply}");
    assert_eq!(reply["error"], cold_error(&root.join("unreadable.json")).replace("unreadable.json", "(document)"), "{reply}");
    let missing = root.join("nope.json");
    let reply = session.send(serde_json::json!({"id": 32, "op": "open", "scene": missing}));
    assert_eq!(reply["ok"], false, "{reply}");
    assert_eq!(reply["kind"], "io", "{reply}");
    assert_eq!(reply["error"], cold_error(&missing), "{reply}");
    let kept = session.render(33, &root.join("s2c.png"));
    assert_eq!(kept["revision"], 4);
    assert_eq!(kept["reuse"], "reused", "{kept}");
    assert_eq!(kept["installs"], 2);
    assert_eq!(std::fs::read(root.join("s2c.png")).unwrap(), cold_bigger);

    // set values spelled as the CLI spells them are read as the CLI reads them.
    let cold_small = cold_with(&root, &bigger, "cold-small", &["--set", "resolution=32x24"]);
    let reply = session.send(serde_json::json!({"id": 34, "op": "render", "out": root.join("small.png"), "profile": "std/draft",
        "set": {"samples": "1", "denoise": false, "resolution": "32x24"}}));
    assert_eq!(reply["ok"], true, "{reply}");
    assert_eq!(std::fs::read(root.join("small.png")).unwrap(), cold_small);
    let reply = session.send(serde_json::json!({"id": 35, "op": "render", "out": root.join("small.png"), "set": {"resolution": "wide"}}));
    assert_eq!(reply["kind"], "usage", "{reply}");

    // The first picture available: one sample, no denoise, labelled in the reply and
    // the sidecar; the final render after it is the ordinary one.
    let early = session.send(serde_json::json!({"id": 50, "op": "render", "out": root.join("early.png"), "profile": "std/draft",
        "set": {"samples": 4, "denoise": false}, "provisional": true}));
    assert_eq!(early["ok"], true, "{early}");
    assert_eq!(early["provisional"], true, "{early}");
    assert_eq!(early["report"]["render_settings"]["samples"], 1, "{early}");
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("early.png.meta.json")).unwrap()).unwrap();
    assert_eq!(meta["provisional"], true, "{meta}");
    let cold_four = cold_with(&root, &bigger, "cold-four", &["--set", "samples=4"]);
    let final_render = session.send(serde_json::json!({"id": 51, "op": "render", "out": root.join("final.png"), "profile": "std/draft",
        "set": {"samples": 4, "denoise": false}}));
    assert_eq!(final_render["ok"], true, "{final_render}");
    assert_eq!(final_render["provisional"], false, "{final_render}");
    assert_eq!(std::fs::read(root.join("final.png")).unwrap(), cold_four, "final after a provisional render differs from the one-shot render");
    assert_ne!(std::fs::read(root.join("early.png")).unwrap(), cold_four, "the provisional picture must differ from the final");
    let final_meta: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("final.png.meta.json")).unwrap()).unwrap();
    assert!(final_meta.get("provisional").is_none(), "{final_meta}");

    // A render that cannot read an external file fails before touching the device,
    // and nothing stale survives: restore the file and the installation is current again.
    let mut external = bigger.clone();
    external["buffers"] = serde_json::json!({"positions": {"type": "vec3", "file": "positions.bin", "count": 3}});
    external["nodes"][0]["geometry"] = serde_json::json!({"attributes": {"POSITION": "positions"}});
    let positions: Vec<u8> = [[-1f32, -1., 0.], [1., -1., 0.], [0., 1., 0.]].iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(root.join("positions.bin"), &positions).unwrap();
    let cold_triangle = cold(&root, &external, "cold-triangle");
    let reply = session.open(40, &external, &root);
    assert_eq!(reply["ok"], true, "{reply}");
    let before = session.render(41, &root.join("tri0.png"));
    assert_eq!(std::fs::read(root.join("tri0.png")).unwrap(), cold_triangle);
    std::fs::remove_file(root.join("positions.bin")).unwrap();
    let reply = session.send(serde_json::json!({"id": 42, "op": "render", "out": root.join("tri1.png"), "profile": "std/draft", "set": {"samples": 1, "denoise": false}}));
    assert_eq!(reply["ok"], false, "{reply}");
    assert_eq!(reply["kind"], "io", "{reply}");
    assert!(reply["error"].as_str().unwrap().contains("positions.bin"), "{reply}");
    std::fs::write(root.join("positions.bin"), &positions).unwrap();
    let after = session.render(43, &root.join("tri2.png"));
    assert_eq!(after["reuse"], "reused", "{after}");
    assert_eq!(after["installs"], before["installs"], "{after}");
    assert_eq!(after["device_reused"], true, "{after}");
    assert_eq!(std::fs::read(root.join("tri2.png")).unwrap(), cold_triangle, "recovered render differs from the one-shot render");

    // Unnamed compositor entries pair by position: a reorder reloads that part and
    // the picture equals a cold render of the reordered document.
    let reply = session.open(11, &first, &root);
    assert_eq!(reply["ok"], true, "{reply}");
    assert_eq!(reply["reuse"], "recompiled", "{reply}");
    // Render uses the authored installation already prepared by open.
    let composed = session.render(12, &root.join("s3.png"));
    assert_eq!(composed["reuse"], "recompiled", "{composed}");
    assert_eq!(composed["install_reused"], true, "{composed}");
    assert_eq!(std::fs::read(root.join("s3.png")).unwrap(), cold_first, "composited session render differs from the one-shot render");
    let reply = session.open(13, &second, &root);
    assert_eq!(reply, serde_json::json!({"id": 13, "ok": true, "revision": 7, "reuse": "reinstalled",
        "changed": [["composition", "compositor"]]}));
    let reordered = session.render(14, &root.join("s4.png"));
    assert_eq!(reordered["reuse"], "reinstalled", "{reordered}");
    assert_eq!(std::fs::read(root.join("s4.png")).unwrap(), cold_second, "reordered session render differs from the one-shot render");
    // An unchanged authored document keeps the already composited installation.
    let reply = session.open(19, &second, &root);
    assert_eq!(reply, serde_json::json!({"id": 19, "ok": true, "revision": 8, "reuse": "reused", "changed": []}));
    let again = session.render(20, &root.join("s4b.png"));
    assert_eq!(again["reuse"], "reused", "{again}");
    assert_eq!(std::fs::read(root.join("s4b.png")).unwrap(), cold_second);

    // An external program file is read again on every identical reopen: a broken file
    // is refused (twice), and the restored file is accepted and renders like cold.
    let gain = "import session_gain; struct Inputs { float gain=0.5; }; float4 composite(CompositeContext ctx, Inputs inputs) { float4 c=compositor_sample(ctx.uv); return float4(c.rgb*session_scale(inputs.gain),c.a); }";
    let imported = root.join("session_gain.slang");
    let original = "public float session_scale(float x) { return x; }";
    std::fs::write(&imported, original).unwrap();
    std::fs::write(root.join("gain.slang"), gain).unwrap();
    let mut filed = scene(140., 0.5);
    filed["programs"] = serde_json::json!({"gain": {"file": "gain.slang"}});
    filed["materials"] = serde_json::json!({"gain": {"program": "gain"}});
    filed["composition"]["space"] = "3d".into(); filed["composition"]["compositor"] = serde_json::json!(["gain"]);
    let cold_filed = cold(&root, &filed, "cold-filed");
    // The first successful open publishes inputs even before any delivery
    // settings or picture exist. A failed reopen must preserve those bytes.
    {
        let mut initial = Session::start();
        assert_eq!(initial.open(1, &filed, &root)["ok"], true);
        std::fs::write(&imported, "INVALID MODULE").unwrap();
        assert_eq!(initial.open(2, &filed, &root)["ok"], false);
        let reply = initial.render(3, &root.join("initial-rollback.png"));
        assert_eq!(reply["ok"], true, "{reply}");
        assert_eq!(reply["revision"], 1);
        assert_eq!(std::fs::read(root.join("initial-rollback.png")).unwrap(), cold_filed);
        std::fs::write(&imported, original).unwrap();
    }
    let reply = session.open(21, &filed, &root);
    assert_eq!(reply["ok"], true, "{reply}");
    session.render(22, &root.join("f0.png"));
    assert_eq!(std::fs::read(root.join("f0.png")).unwrap(), cold_filed);
    // The render reads the file again, not only the open: an edit between the two
    // invalidates the retained installation and renders like a one-shot of the edit.
    let reply = session.open(60, &filed, &root);
    assert_eq!(reply["ok"], true, "{reply}");
    std::fs::write(&imported, "public float session_scale(float x) { return 0.5*x; }").unwrap();
    let cold_dimmer = cold(&root, &filed, "cold-dimmer");
    assert_ne!(cold_dimmer, cold_filed);
    let cold_small_dimmer = cold_with(&root, &filed, "cold-dimmer-small", &["--set", "resolution=32x24"]);
    let edited = session.render(61, &root.join("f0b.png"));
    assert_eq!(edited["ok"], true, "{edited}");
    assert_eq!(edited["install_reused"], false, "{edited}");
    assert_eq!(std::fs::read(root.join("f0b.png")).unwrap(), cold_dimmer, "render after an external program edit differs from the one-shot of the edited state");
    std::fs::write(root.join("gain.slang"), "THIS IS NOT A VALID SLANG PROGRAM").unwrap();
    for id in [23, 24] {
        let reply = session.open(id, &filed, &root);
        assert_eq!(reply["ok"], false, "{reply}");
        assert!(reply["error"].as_str().unwrap().contains("gain.slang"), "{reply}");
    }
    // A rejected open keeps the last successfully rendered revision reproducible
    // even while the candidate's external source is still invalid.
    let reply = session.render(25, &root.join("f1.png"));
    assert_eq!(reply["ok"], true, "{reply}");
    assert_eq!(reply["revision"], edited["revision"], "{reply}");
    assert_eq!(std::fs::read(root.join("f1.png")).unwrap(), cold_dimmer);
    let smaller = session.send(serde_json::json!({"id": 250, "op": "render", "out": root.join("rollback-small.png"),
        "profile": "std/draft", "set": {"samples": 1, "denoise": false, "resolution": "32x24"}}));
    assert_eq!(smaller["ok"], true, "{smaller}");
    assert_eq!(std::fs::read(root.join("rollback-small.png")).unwrap(), cold_small_dimmer);
    std::fs::write(&imported, original).unwrap();
    std::fs::write(root.join("gain.slang"), gain).unwrap();
    let reply = session.open(26, &filed, &root);
    assert_eq!(reply["ok"], true, "{reply}");
    assert_eq!(reply["reuse"], "reinstalled", "{reply}");
    let restored = session.render(27, &root.join("f2.png"));
    assert_eq!(restored["reuse"], "reinstalled", "{restored}");
    assert_eq!(std::fs::read(root.join("f2.png")).unwrap(), cold_filed, "restored-program render differs from the one-shot render");

    // Usage errors are replies, not worker deaths; the removed patch request too.
    let reply = session.send(serde_json::json!({"id": 15, "op": "open", "scene": scene_path, "document": {}}));
    assert_eq!(reply["kind"], "usage", "{reply}");
    assert!(reply["error"].as_str().unwrap().contains("not both"), "{reply}");
    let reply = session.send(serde_json::json!({"id": 16, "op": "render", "out": root.join("x.jpg")}));
    assert_eq!(reply["kind"], "usage", "{reply}");
    let reply = session.send(serde_json::json!({"id": 17, "op": "patch", "ops": []}));
    assert_eq!(reply["kind"], "usage", "{reply}");
    // A render whose options are refused leaves nothing behind: the next open of a
    // valid document succeeds and its render matches the one-shot.
    let reply = session.send(serde_json::json!({"id": 62, "op": "render", "out": root.join("bad.png"), "profile": "std/draft", "set": {"samples": 0, "denoise": false}}));
    assert_eq!(reply["ok"], false, "{reply}");
    assert_eq!(reply["kind"], "usage", "{reply}");
    let reply = session.open(63, &bigger, &root);
    assert_eq!(reply["ok"], true, "{reply}");
    let recovered = session.render(64, &root.join("s2b.png"));
    assert_eq!(recovered["ok"], true, "{recovered}");
    assert_eq!(std::fs::read(root.join("s2b.png")).unwrap(), cold_bigger, "render after a refused render differs from the one-shot render");

    let reply = session.send(serde_json::json!({"id": 18, "op": "close"}));
    assert_eq!(reply, serde_json::json!({"id": 18, "ok": true}));
    let status = session.child.wait().unwrap();
    assert!(status.success(), "{status}");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn end_of_input_closes_the_worker_without_a_device() {
    let mut child = Command::new(binary()).arg("session")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "not json").unwrap();
        writeln!(stdin, r#"{{"id": 7, "op": "render", "out": "x.png"}}"#).unwrap();
        writeln!(stdin, r#"{{"id": 8, "op": "open", "scene": "/nonexistent/scene.json"}}"#).unwrap();
        writeln!(stdin, r#"{{"id": 9, "op": "open"}}"#).unwrap();
    }
    drop(child.stdin.take());
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{output:?}");
    let replies: Vec<serde_json::Value> = String::from_utf8_lossy(&output.stdout).lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(replies.len(), 4, "{replies:?}");
    assert_eq!(replies[0]["kind"], "usage");
    assert_eq!(replies[1]["id"], 7);
    assert!(replies[1]["error"].as_str().unwrap().contains("no scene is open"), "{replies:?}");
    assert_eq!(replies[2]["ok"], false);
    assert_eq!(replies[3]["kind"], "usage", "{replies:?}");
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"), "{output:?}");
}

/// Expansion consumes sources before file admission; compare its current output
/// with the actual installation, including after a render-time source edit.
#[test]
fn expanded_sources_refresh_without_reopening_and_keep_the_installed_snapshot() {
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("SKIP GPU session test: set VORK_GPU_TESTS=1 on a supported GPU");
        return;
    }
    let root = std::env::temp_dir().join(format!("vork-session-expansion-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let font = root.join("font.ttf");
    let regular = include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf");
    let bold = include_bytes!("../../vork-std/tests/fonts/LiberationSans-Bold.ttf");
    let mut session = Session::start();
    for placement in ["inline", "named", "nested"] {
        let geometry = serde_json::json!({"std":"text","body":"WWii","font":"font.ttf",
            "size":1,"align_x":"center","align_y":"center"});
        let mut doc = serde_json::json!({"version":2,"render":{"resolution":[128,64]},"nodes":[
            {"type":"mesh","geometry":geometry,"materials":[{"inputs":{"base_color":[0.05,0.05,0.05],"specular_weight":0}}]},
            {"type":"camera","translation":[0,0,4],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":1.5,"fit":"vertical"}}},
            {"type":"light","shape":"environment","color":[1,1,1]}]});
        if placement == "named" {
            doc["geometry"] = serde_json::json!({"letters":geometry});
            doc["nodes"][0]["geometry"] = "letters".into();
        } else if placement == "nested" {
            doc["nodes"][0] = serde_json::json!({"children":[doc["nodes"][0]]});
        }
        std::fs::write(&font, regular).unwrap();
        let before = cold(&root, &doc, &format!("{placement}-regular"));
        assert_eq!(session.open(1, &doc, &root)["ok"], true);
        let out = root.join("session.png");
        session.render(2, &out);
        assert_eq!(std::fs::read(&out).unwrap(), before);
        assert_eq!(session.render(3, &out)["install_reused"], true);
        // Reviewer's witness: replace the font at the same path without open.
        std::fs::write(&font, bold).unwrap();
        let after = cold(&root, &doc, &format!("{placement}-bold"));
        assert_ne!(before, after, "the replacement must change visible glyphs");
        let changed = session.render(4, &out);
        assert_eq!(changed["install_reused"], false, "{changed}");
        assert_eq!(changed["device_reused"], true, "{changed}");
        assert_eq!(std::fs::read(&out).unwrap(), after);
        assert_eq!(session.render(5, &out)["install_reused"], true);
        // Last-open was A; actual install is B. Reopening A must not keep B.
        std::fs::write(&font, regular).unwrap();
        let restored = session.open(6, &doc, &root);
        assert_eq!(restored["reuse"], "reinstalled", "{restored}");
        session.render(7, &out);
        assert_eq!(std::fs::read(&out).unwrap(), before);
        // A value edit after rendering B must compare with B's snapshot.
        std::fs::write(&font, bold).unwrap();
        session.render(8, &out);
        doc["nodes"][1]["translation"] = serde_json::json!([0.1,0,4]);
        let shifted = cold(&root, &doc, &format!("{placement}-shifted"));
        let updated = session.open(9, &doc, &root);
        assert_eq!(updated["reuse"], "updated", "{updated}");
        session.render(10, &out);
        assert_eq!(std::fs::read(&out).unwrap(), shifted);
        for corrupt in [false, true] {
            if corrupt { std::fs::write(&font, b"not a font").unwrap(); }
            else { std::fs::remove_file(&font).unwrap(); }
            let rejected = session.send(serde_json::json!({"id":11,"op":"render","out":out,"profile":"std/draft"}));
            assert_eq!(rejected["ok"], false, "{rejected}");
            assert!(rejected["error"].as_str().unwrap().contains("font"), "{rejected}");
            assert_eq!(session.open(14, &doc, &root)["ok"], false);
            session.render(15, &out);
            assert_eq!(std::fs::read(&out).unwrap(), shifted);
            std::fs::write(&font, bold).unwrap();
            let repaired = session.render(12, &out);
            assert_eq!(repaired["device_reused"], true, "{repaired}");
            assert_eq!(repaired["install_reused"], true, "{repaired}");
            assert_eq!(std::fs::read(&out).unwrap(), shifted);
        }
    }
    assert_eq!(session.send(serde_json::json!({"id":13,"op":"close"}))["ok"], true);
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn footage_still_reaps_the_compositor_decoder() {
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let root = std::env::temp_dir().join(format!("vork-footage-still-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let clip = Command::new("ffmpeg").current_dir(&root)
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=128x64:r=4:d=4",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "clip.mp4"])
        .output().expect("install ffmpeg for footage tests");
    assert!(clip.status.success(), "{}", String::from_utf8_lossy(&clip.stderr));

    let document = serde_json::json!({
        "version":2, "composition":{"space":"2d","color_space":"Linear Rec.709","compositor":["copy"]},
        "render":{"resolution":[128,64],"background":"transparent"},
        "nodes":[], "images":{"plate":{"file":"clip.mp4"}},
        "materials":{"copy":{"inputs":{"plate":{"image":"plate"}},"program":{"source":
            "import vork; struct Inputs { Image plate; }; float4 composite(CompositeContext c, Inputs i) { return compositor_sample(i.plate,c.uv); }"}}}
    });
    let scene = root.join("scene.json");
    std::fs::write(&scene, serde_json::to_vec(&document).unwrap()).unwrap();
    let render = || Command::new(binary()).arg("render").arg(&scene)
        .arg("--out").arg(root.join("still.png")).output().unwrap();
    let output = render();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    for message in ["Broken pipe", "Error muxing", "Error closing file", "Error writing trailer"] {
        assert!(!stderr.contains(message), "{stderr}");
    }
    let png = std::fs::read(root.join("still.png")).unwrap();
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));

    // A pinned descriptor has no filename extension for FFprobe to guess from.
    // An actual MP4 header still identifies the demuxer and its missing moov.
    let path = root.join("clip.mp4");
    let original = std::fs::read(&path).unwrap();
    assert_eq!(&original[4..8], b"ftyp");
    let header = u32::from_be_bytes(original[..4].try_into().unwrap()) as usize;
    for (input, cause) in [(&original[..header], "moov atom not found"),
                           (b"not a playable movie".as_slice(), "probe video")] {
        std::fs::write(&path, input).unwrap();
        let output = render();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        for expected in ["clip.mp4", "restore a playable movie", cause] {
            assert!(stderr.contains(expected), "{stderr}");
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

/// Admission and retention must agree for every sampled source, including changes
/// between requests. These fixtures use only builtins so authored programs cannot
/// accidentally disable the dependency path under test.
#[test]
fn sampled_sources_match_ordinary_renders_and_never_cache_missing_images() {
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("SKIP GPU session test: set VORK_GPU_TESTS=1 on a supported GPU");
        return;
    }
    let root = std::env::temp_dir().join(format!("vork-session-sources-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    const GREEN: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 16, 0, 0, 0, 16, 8, 2, 0, 0, 0, 144, 145, 104, 54, 0, 0, 0, 21, 73, 68, 65, 84, 120, 156, 99, 96, 248, 207, 64, 26, 26, 213, 48, 170, 97, 248, 106, 0, 0, 145, 234, 255, 1, 36, 164, 229, 140, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130];
    const MAGENTA: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 16, 0, 0, 0, 16, 8, 2, 0, 0, 0, 144, 145, 104, 54, 0, 0, 0, 22, 73, 68, 65, 84, 120, 156, 99, 248, 207, 240, 159, 36, 196, 48, 170, 97, 84, 195, 240, 213, 0, 0, 32, 211, 254, 16, 23, 153, 202, 184, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130];
    for (name, data) in [("plate.png", GREEN), ("0001.png", GREEN), ("0002.png", MAGENTA), ("magenta.png", MAGENTA)] {
        std::fs::write(root.join(name), data).unwrap();
    }
    let document = |file: &str| serde_json::json!({"version":2,
        "render":{"resolution":[64,48],"duration":1},
        "images":{"plate":{"file":file}},
        "materials":{"world":{"program":"std/environment_image","inputs":{"image":{"image":"plate"}}}},
        "nodes":[{"type":"light","shape":"environment","material":"world","energy":1},{"type":"camera"}]});
    let mut session = Session::start();
    // A real file retains; media follows ordinary enumeration/decoding and installs
    // again. Both opening a path and opening a document exercise the public worker.
    let movie = Command::new("ffmpeg").args(["-v", "error", "-y", "-loop", "1", "-i"])
        .arg(root.join("plate.png")).args(["-t", "0.1", "-c:v", "ffv1"])
        .arg(root.join("clip.mkv")).output().expect("install ffmpeg for movie tests");
    assert!(movie.status.success(), "{}", String::from_utf8_lossy(&movie.stderr));
    for (index, (file, retain)) in [("plate.png", true), ("%04d.png", false), ("clip.mkv", false)].iter().enumerate() {
        let doc = document(file);
        let name = format!("source-{index}");
        let expected = cold(&root, &doc, &name);
        let opened = session.send(serde_json::json!({"id":100,"op":"open","scene":root.join(format!("{name}.json"))}));
        assert_eq!(opened["ok"], true, "{opened}");
        for attempt in 0..2 {
            let path = root.join(format!("source-{index}-{attempt}.png"));
            let rendered = session.render(101, &path);
            if attempt == 1 { assert_eq!(rendered["install_reused"], *retain, "{rendered}"); }
            assert_eq!(std::fs::read(path).unwrap(), expected, "{file}");
        }
        assert_eq!(session.open(102, &doc, &root)["ok"], true);
        let reopened = session.render(103, &root.join("reopened.png"));
        assert_eq!(reopened["install_reused"], *retain, "{reopened}");
        assert_eq!(std::fs::read(root.join("reopened.png")).unwrap(), expected);
    }
    // A source can return to the last-open bytes after a render installed other
    // bytes. Opening must compare with the actual installation, not last-open.
    let mut file = document("plate.png");
    // Exercise disk replacement on a retained world with an authored consumer.
    file["programs"] = serde_json::json!({"pass": {"source": "struct Inputs {}; float4 composite(CompositeContext ctx, Inputs inputs) { return compositor_sample(ctx.uv); }"}});
    file["materials"]["pass"] = serde_json::json!({"program": "pass"});
    file["composition"] = serde_json::json!({"space":"3d", "compositor":["pass"]});
    assert_eq!(session.open(114, &file, &root)["ok"], true);
    session.render(115, &root.join("file-before.png"));
    std::fs::write(root.join("plate.png"), MAGENTA).unwrap();
    let edited = cold(&root, &file, "file-edited");
    let reply = session.render(116, &root.join("file-after.png"));
    assert_ne!(edited, std::fs::read(root.join("file-before.png")).unwrap());
    assert_eq!(reply["install_reused"], false, "{reply}");
    assert_eq!(std::fs::read(root.join("file-after.png")).unwrap(), edited);
    std::fs::write(root.join("plate.png"), GREEN).unwrap();
    let restored = session.open(117, &file, &root);
    assert_eq!(restored["reuse"], "reinstalled", "{restored}");
    session.render(118, &root.join("file-restored.png"));
    assert_eq!(std::fs::read(root.join("file-restored.png")).unwrap(), std::fs::read(root.join("file-before.png")).unwrap());
    let sequence = document("%04d.png");
    assert_eq!(session.open(104, &sequence, &root)["ok"], true);
    let later = cold_with(&root, &sequence, "later", &["--at", "0.1"]);
    let reply = session.send(serde_json::json!({"id":105,"op":"render","out":root.join("later-session.png"),
        "at":0.1,"profile":"std/draft","set":{"samples":1,"denoise":false}}));
    assert_eq!(reply["ok"], true, "{reply}");
    assert_eq!(std::fs::read(root.join("later-session.png")).unwrap(), later);
    assert_ne!(later, cold(&root, &sequence, "first-frame"));

    let missing = document("missing.png");
    let magenta = cold(&root, &document("magenta.png"), "expected-magenta");
    let missing_path = root.join("missing.json");
    std::fs::write(&missing_path, serde_json::to_vec(&missing).unwrap()).unwrap();
    let output = Command::new(binary()).arg("render").arg(&missing_path).args(RENDER_ARGS)
        .arg("--out").arg(root.join("cold-missing.png")).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(String::from_utf8_lossy(&output.stderr).contains("magenta"));
    assert_eq!(std::fs::read(root.join("cold-missing.png")).unwrap(), magenta);
    assert_eq!(session.open(106, &missing, &root)["ok"], true);
    for attempt in 0..2 {
        let path = root.join(format!("missing-{attempt}.png"));
        let reply = session.render(107, &path);
        assert_eq!(reply["install_reused"], false, "{reply}");
        assert!(reply["result"]["warnings"].to_string().contains("missing_image"), "{reply}");
        assert_eq!(std::fs::read(path).unwrap(), magenta);
    }
    // Repair, edit and remove the file without reopening the document.
    for (data, name) in [(GREEN, "repaired"), (MAGENTA, "edited")] {
        std::fs::write(root.join("missing.png"), data).unwrap();
        let expected = cold(&root, &missing, name);
        let path = root.join(format!("session-{name}.png"));
        let reply = session.render(108, &path);
        assert_eq!(reply["install_reused"], false, "{reply}");
        assert_eq!(std::fs::read(path).unwrap(), expected);
    }
    std::fs::remove_file(root.join("missing.png")).unwrap();
    let reply = session.render(109, &root.join("removed.png"));
    assert_eq!(reply["install_reused"], false, "{reply}");
    assert!(reply["result"]["warnings"].to_string().contains("missing_image"), "{reply}");
    assert_eq!(std::fs::read(root.join("removed.png")).unwrap(), magenta);
    assert_eq!(session.open(110, &missing, &root)["ok"], true);
    let mut unused = document("plate.png");
    unused["images"]["unused"] = serde_json::json!({"file":"unused-missing.png"});
    let expected = cold(&root, &unused, "unused");
    assert_eq!(session.open(112, &unused, &root)["ok"], true);
    let reply = session.render(113, &root.join("unused-session.png"));
    assert_eq!(reply["install_reused"], true, "{reply}");
    assert_eq!(std::fs::read(root.join("unused-session.png")).unwrap(), expected);
    assert_eq!(session.send(serde_json::json!({"id":111,"op":"close"}))["ok"], true);
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn randomized_canvas_edits_patch_one_install_and_match_fresh_frames() {
    use serde_json::json;
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let directory = std::env::temp_dir().join(format!("vork-canvas-edits-{}", std::process::id()));
    let root = directory.as_path();
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(root.join("font.ttf"), include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf")).unwrap();
    for (file, input) in [("a.mp4", "testsrc2=s=128x80:r=8:d=1"), ("b.mp4", "smptebars=s=128x80:r=8:d=1")] {
        let output = Command::new("ffmpeg").current_dir(root)
            .args(["-v", "error", "-f", "lavfi", "-i", input, "-c:v", "libx264", "-threads", "1", file]).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    }
    let mut doc = json!({"version":2, "composition":{"space":"2d"},
        "render":{"resolution":[128,80], "fps":8, "duration":1},
        "images":{"plate":{"file":"a.mp4"}}, "nodes":[
            {"type":"path","name":"plate","d":"M0 0H128V80H0Z","fill":{"image":"plate","fit":"fill"}},
            {"type":"path","name":"box","d":"M0 0H38V18H0Z","fill":"white"},
            {"type":"text","name":"title","content":"A","font":"font.ttf","font-size":18,"fill":"white","translation":[12,20,0]}]});
    let mut session = Session::start();
    let mut seed = 2907u32;
    let mut random = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); seed >> 8 };
    let mut edits = [0, 1, 2, 3];
    for step in 0..17 {
        if step > 0 {
            let slot = (step - 1) % edits.len();
            if slot == 0 {
                for i in (1..edits.len()).rev() { edits.swap(i, random() as usize % (i + 1)); }
            }
            match edits[slot] {
                0 => doc["nodes"][2]["content"] = format!("Edit {step}").into(),
                1 => doc["nodes"][2]["translation"] = json!([step + 2, 22 + random() % 18, 0]),
                2 => doc["nodes"][1]["fill"] = format!("#{:06x}", random()).into(),
                _ => doc["images"]["plate"]["file"] = if doc["images"]["plate"]["file"] == "a.mp4" { "b.mp4" } else { "a.mp4" }.into(),
            }
        }
        let opened = session.open(1, &doc, root);
        assert_eq!(opened["reuse"], if step == 0 { "reinstalled" } else { "updated" }, "step {step}: {opened}");
        let at = format!("{}/8", random() % 8);
        let out = root.join("look.png");
        let rendered = session.send(json!({"id":2,"op":"render","out":out,"at":at,"profile":"std/draft","set":{"samples":1,"denoise":false}}));
        assert_eq!(rendered["installs"], 1, "step {step}: {rendered}");
        assert_eq!(rendered["install_reused"], step > 0, "step {step}: {rendered}");
        let expected = cold_with(root, &doc, "fresh", &["--at", &at]);
        assert!(std::fs::read(&out).unwrap() == expected, "session differs from fresh at edit {step}, time {at}");
    }
    // Replace the bytes under the currently painted name. A refused candidate must still
    // render the previously published movie, including a seek that needs its pinned file.
    let current = doc["images"]["plate"]["file"].as_str().unwrap();
    let replacement = if current == "a.mp4" { "b.mp4" } else { "a.mp4" };
    let published = cold_with(root, &doc, "published", &["--at", "3/8"]);
    std::fs::copy(root.join(replacement), root.join("replacement.mp4")).unwrap();
    std::fs::rename(root.join("replacement.mp4"), root.join(current)).unwrap();
    let mut broken = doc.clone();
    broken["nodes"][2]["translation"] = json!([4, 8, 9]);
    assert_eq!(session.open(4, &broken, root)["ok"], false);
    let out = root.join("look.png");
    let look = json!({"id":5,"op":"render","out":out,"at":"3/8","profile":"std/draft","set":{"samples":1,"denoise":false}});
    assert_eq!(session.send(look.clone())["ok"], true);
    assert!(std::fs::read(&out).unwrap() == published, "refusal lost the pinned movie");
    let opened = session.open(6, &doc, root);
    assert_eq!(opened["reuse"], "updated", "{opened}");
    assert_eq!(session.send(look.clone())["installs"], 1);
    let replaced = cold_with(root, &doc, "replaced", &["--at", "3/8"]);
    assert!(replaced != published);
    assert!(std::fs::read(&out).unwrap() == replaced, "replacement kept the old decoder");

    // The current frame is 3/8. A title edit and another look there need no movie samples.
    doc["nodes"][2]["content"] = "New title".into();
    let opened = session.open(7, &doc, root);
    assert_eq!(opened["reuse"], "updated", "{opened}");
    let mut profiled = look;
    profiled["diagnostics"] = true.into();
    let rendered = session.send(profiled);
    assert_eq!(rendered["install_reused"], true, "{rendered}");
    let timings: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("look.png.timings.json")).unwrap()).unwrap();
    assert_eq!(timings["breakdown"]["input_decode_ms"].as_f64(), Some(0.), "title edit decoded the held movie");

    session.send(json!({"id":3,"op":"close"}));
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

/// A direct matte needs its image publication when the edited document is planned at open.
#[test]
fn replacing_a_canvas_image_publishes_it_before_mask_planning() {
    use serde_json::json;
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let root = std::env::temp_dir().join(format!("vork-session-matte-image-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let output = root.join("session.png");
    let mut doc = json!({"version":2, "composition":{"space":"2d"}, "render":{"resolution":[128,64]},
        "images":{"plate":{"file":"plate.mp4"}}, "nodes":[
            {"type":"path","name":"cut","d":"M0 0H64V64H0Z","fill":{"image":"plate"},"mask-type":"luminance"},
            {"type":"path","name":"body","d":"M0 0H64V64H0Z","fill":"red","mask":"cut"},
            {"type":"path","name":"swatch","d":"M64 0H128V64H64Z","fill":{"image":"plate"}}
        ]});
    let mut session = Session::start();
    let mut previous = None;
    for (file, source, reuse) in [
        ("plate.mp4", Some("color=black:s=64x64"), Some("reinstalled")),
        ("plate.mp4", Some("color=white:s=64x64"), Some("updated")),
        ("wide.mp4", Some("color=gray:s=128x64"), Some("updated")),
        ("still.png", Some("color=blue:s=64x64"), None),
        ("still.png", Some("color=white:s=128x64"), None),
        ("still.png", None, Some("reused")),
    ] {
        if let Some(source) = source {
            let encoded = Command::new("ffmpeg").args(["-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                source, "-frames:v", "1", "-threads", "1"]).arg(root.join(file)).output().unwrap();
            assert!(encoded.status.success(), "{}", String::from_utf8_lossy(&encoded.stderr));
        }
        doc["images"]["plate"]["file"] = file.into();
        let expected = cold(&root, &doc, "fresh");
        let opened = session.open(1, &doc, &root);
        assert_eq!(opened["ok"], true, "{file} {source:?}: {opened}");
        if let Some(reuse) = reuse { assert_eq!(opened["reuse"], reuse, "{opened}"); }
        session.render(2, &output);
        assert_eq!(std::fs::read(&output).unwrap(), expected, "{file} {source:?}");
        if let Some(before) = previous.as_ref() {
            assert_eq!(before == &expected, source.is_none(), "{file} {source:?}");
        }
        previous = Some(expected);
    }
    session.send(json!({"id":3,"op":"close"}));
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

// Each completed render prints the cumulative count, including work in the preceding open.
fn input_decode_calls(log: &Path) -> u64 {
    let contents = std::fs::read_to_string(log).unwrap();
    let totals = contents.lines().rev().find_map(|line| line.strip_prefix("canvas host profile (inclusive process totals): ")).unwrap();
    serde_json::from_str::<serde_json::Value>(totals).unwrap()["canvas.input_decode"]["calls"].as_u64().unwrap()
}

#[test]
fn ordinary_clip_swap_decodes_only_the_requested_frame() {
    use serde_json::json;
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let root = std::env::temp_dir().join(format!("vork-session-clip-swap-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for (file, pattern) in [("a.mp4", "testsrc2"), ("b.mp4", "smptebars")] {
        let output = Command::new("ffmpeg").args(["-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
            &format!("{pattern}=s=64x64:r=4:d=6"), "-c:v", "libx264", "-threads", "1"]).arg(root.join(file)).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    }
    let mut doc = json!({"version":2,"composition":{"space":"2d"},"render":{"resolution":[64,64],"fps":4,"duration":6},
        "images":{"clip":{"file":"a.mp4"}},"nodes":[{"name":"shot","type":"path","d":"M0 0H64V64H0Z","fill":{"image":"clip"}},
        {"name":"badge","type":"path","d":"M0 0H8V8H0Z","fill":"red"}]});
    let log = root.join("profile.log");
    let out = root.join("look.png");
    let mut session = Session::profiled(&log);
    let look = json!({"op":"render","at":5,"out":out,"profile":"std/draft","set":{"samples":1,"denoise":false}});
    assert_eq!(session.open(1, &doc, &root)["reuse"], "reinstalled");
    assert_eq!(session.send(look.clone())["ok"], true);
    let initial = input_decode_calls(&log);
    // Measure the installation checks shared by an edit and look with the movie held.
    doc["nodes"][1]["fill"] = "blue".into();
    assert_eq!(session.open(2, &doc, &root)["reuse"], "updated");
    assert_eq!(session.send(look.clone())["ok"], true);
    let mut last = input_decode_calls(&log);
    let edit_checks = last - initial;
    let mut previous = std::fs::read(&out).unwrap();
    for same_path in [false, true] {
        if same_path {
            std::fs::rename(root.join("a.mp4"), root.join("b.mp4")).unwrap();
        } else {
            doc["images"]["clip"]["file"] = "b.mp4".into();
        }
        let opened = session.open(3, &doc, &root);
        assert_eq!(opened["reuse"], "updated", "{opened}");
        assert_eq!(session.send(look.clone())["ok"], true);
        let count = input_decode_calls(&log);
        assert_eq!(count - last, edit_checks + 1, "expected only the requested movie frame, same path = {same_path}");
        last = count;
        let pixels = std::fs::read(&out).unwrap();
        assert!(pixels != previous && pixels == cold_with(&root, &doc, "fresh", &["--at","5"]));
        previous = pixels;
    }
    session.send(json!({"op":"close"}));
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn collapsed_matte_tracks_a_shared_movie_publication_across_seeks() {
    use serde_json::json;
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let root = std::env::temp_dir().join(format!("vork-shared-matte-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let movie = Command::new("ffmpeg").args(["-nostdin","-v","error","-y","-f","lavfi","-i",
        "color=white:s=32x32:r=1:d=3","-c:v","libx264","-threads","1"]).arg(root.join("film.mp4")).status().unwrap();
    assert!(movie.success());
    let doc = json!({"version":2,"composition":{"space":"2d"},"render":{"resolution":[64,32],"fps":1,"duration":3},
        "images":{"film":{"file":"film.mp4"}},"nodes":[
            {"type":"path","name":"reader","d":"M0 0H32V32H0Z","fill":{"image":"film"}},
            {"scale":[0,1,1],"children":[{"type":"path","name":"matte","d":"M32 0H64V32H32Z",
                "fill":{"image":"film"},"mask-type":"alpha"}]},
            {"type":"path","d":"M32 0H64V32H32Z","fill":"red","mask":"matte"}],
        "animation":[{"tracks":[{"path":["nodes","reader","visible"],"interpolation":"STEP",
            "keys":[[0,false],[1,true],[2,false]]}]}]});
    let mut session = Session::start();
    let opened = session.open(1, &doc, &root);
    assert_eq!(opened["ok"], true, "{opened}");
    let path = root.join("look.png");
    for at in [0, 1, 2, 1, 0] {
        let reply = session.send(json!({"op":"render","at":at,"out":path}));
        assert_eq!(reply["ok"], true, "{reply}");
        let pixels = Command::new("ffmpeg").args(["-v","error","-i"]).arg(&path)
            .args(["-pix_fmt","rgba","-f","rawvideo","-"]).output().unwrap();
        assert!(pixels.status.success());
        assert_eq!(pixels.stdout.len(), 64 * 32 * 4);
        for x in [16, 48] {
            assert_eq!(pixels.stdout[(16 * 64 + x) * 4 + 3], if at == 1 { 255 } else { 0 }, "x={x} at {at}");
        }
    }
    session.send(json!({"op":"close"}));
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn canvas_edit_restores_a_movie_matte_hidden_at_the_last_look() {
    use serde_json::json;
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let root = std::env::temp_dir().join(format!("vork-session-matte-seek-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let movie = Command::new("ffmpeg").args(["-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
        "testsrc2=s=64x64:r=4:d=6", "-c:v", "libx264", "-threads", "1"]).arg(root.join("clip.mp4")).output().unwrap();
    assert!(movie.status.success(), "{}", String::from_utf8_lossy(&movie.stderr));

    std::fs::copy(root.join("clip.mp4"), root.join("late.mp4")).unwrap();
    // Switching image paints hides one movie without changing the matte's retained row form.
    let mut doc = json!({"version":2,"composition":{"space":"2d"},"render":{"resolution":[64,64],"fps":4,"duration":6},
        "images":{"matte":{"file":"clip.mp4"},"late":{"file":"late.mp4"}},
        "nodes":[{"type":"path","name":"cut","d":"M0 0H64V64H0Z","fill":{"image":"matte","fit":"fill"},"mask-type":"luminance"},
            {"type":"path","name":"body","d":"M0 0H64V64H0Z","fill":"red","stroke":"white","stroke-width":1,"mask":"cut"}],
        "animation":[{"tracks":[{"path":["nodes","cut","fill"],"interpolation":"STEP",
            "keys":[[0,{"image":"matte","fit":"fill"}],[1,{"image":"late","fit":"fill"}]]}]}]});
    let out = root.join("look.png");
    let look = |session: &mut Session, at: u32| {
        let reply = session.send(json!({"op":"render","at":at,"out":out,"profile":"std/draft","diagnostics":true,
            "set":{"samples":1,"denoise":false}}));
        assert_eq!(reply["ok"], true, "{reply}");
        std::fs::read(&out).unwrap()
    };
    let log = root.join("profile.log");
    let mut session = Session::profiled(&log);
    assert_eq!(session.open(1, &doc, &root)["reuse"], "reinstalled");
    assert!(look(&mut session, 5) == cold_with(&root, &doc, "before", &["--at","5"]));
    doc["nodes"][1]["fill"] = "blue".into();
    let opened = session.open(2, &doc, &root);
    assert_eq!(opened["reuse"], "updated", "{opened}");
    let late = cold_with(&root, &doc, "late", &["--at","5"]);
    let zero = cold_with(&root, &doc, "zero", &["--at","0"]);
    assert!(late != zero);
    assert!(look(&mut session, 5) == late);
    assert!(look(&mut session, 0) == zero);

    // Refuse the same stroke once with its zero-time matte held, then with only the
    // late matte held. Looking again at the held instant needs no movie sample.
    doc["nodes"][1]["stroke-dasharray"] = json!([0.00001,0.00001]);
    let mut refusal_calls = Vec::new();
    for (at, held) in [(0, &zero), (5, &late)] {
        assert!(look(&mut session, at) == *held);
        let before = input_decode_calls(&log);
        let refused = session.open(3, &doc, &root);
        assert!(refused["error"].as_str().is_some_and(|s| s.starts_with("update scene") && s.contains("stroke-dasharray expansion")), "{refused}");
        assert!(look(&mut session, at) == *held, "refusal lost the held frame");
        let timings: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("look.png.timings.json")).unwrap()).unwrap();
        assert_eq!(timings["breakdown"]["input_decode_ms"].as_f64(), Some(0.), "rollback look decoded a movie");
        refusal_calls.push(input_decode_calls(&log) - before);
    }
    assert_eq!(refusal_calls[1], refusal_calls[0] + 1, "missing zero-time matte must be published before the stroke refusal");
    assert!(look(&mut session, 0) == zero, "refusal left the matte bound to the failed edit");
    session.send(json!({"op":"close"}));
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

/// Reopens retain compatible canvas rows and refresh changed external image bytes.
#[test]
fn canvas_sessions_reuse_the_document_and_match_one_shot_pictures() {
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("SKIP GPU session test: set VORK_GPU_TESTS=1 on a supported GPU");
        return;
    }
    let root = std::env::temp_dir().join(format!("vork-session-canvas-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    const GREEN: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 16, 0, 0, 0, 16, 8, 2, 0, 0, 0, 144, 145, 104, 54, 0, 0, 0, 21, 73, 68, 65, 84, 120, 156, 99, 96, 248, 207, 64, 26, 26, 213, 48, 170, 97, 248, 106, 0, 0, 145, 234, 255, 1, 36, 164, 229, 140, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130];
    const MAGENTA: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 16, 0, 0, 0, 16, 8, 2, 0, 0, 0, 144, 145, 104, 54, 0, 0, 0, 22, 73, 68, 65, 84, 120, 156, 99, 248, 207, 240, 159, 36, 196, 48, 170, 97, 84, 195, 240, 213, 0, 0, 32, 211, 254, 16, 23, 153, 202, 184, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130];
    std::fs::write(root.join("plate.png"), GREEN).unwrap();
    let document = |fill: &str| serde_json::json!({"version": 2, "composition": {"space": "2d"},
        "render": {"resolution": [96, 64]},
        "images": {"plate": {"file": "plate.png"}},
        "nodes": [
            {"name": "picture", "type": "path", "d": "M8 8H56V56H8Z", "fill": {"image": "plate", "fit": "fill"}},
            {"name": "box", "type": "path", "d": "M48 16H88V48H48Z", "fill": fill, "stroke": "white", "stroke-width": 2}]});
    let mut session = Session::start();
    let red = document("red");
    let expected = cold(&root, &red, "canvas-red");
    assert_eq!(session.open(1, &red, &root)["ok"], true);
    // The first render builds the document; the second draws from the kept one.
    let first = session.render(2, &root.join("red-1.png"));
    assert_eq!(first["install_reused"], false, "{first}");
    assert_eq!(std::fs::read(root.join("red-1.png")).unwrap(), expected);
    let second = session.render(3, &root.join("red-2.png"));
    assert_eq!((&second["device_reused"], &second["install_reused"], &second["reuse"]),
        (&serde_json::json!(true), &serde_json::json!(true), &serde_json::json!("reused")), "{second}");
    assert_eq!(std::fs::read(root.join("red-2.png")).unwrap(), expected);
    // The identical document again is kept as it is.
    let reopened = session.open(4, &red, &root);
    assert_eq!(reopened["reuse"], "reused", "{reopened}");
    // A paint edit patches the installed rows during open.
    let blue = document("blue");
    let expected = cold(&root, &blue, "canvas-blue");
    let opened = session.open(5, &blue, &root);
    assert_eq!((&opened["ok"], &opened["reuse"]), (&serde_json::json!(true), &serde_json::json!("updated")), "{opened}");
    let reply = session.render(6, &root.join("blue.png"));
    assert_eq!(reply["device_reused"], true, "{reply}");
    assert_eq!(std::fs::read(root.join("blue.png")).unwrap(), expected);
    // A changed image file is read again, not served from the process's image store.
    std::fs::write(root.join("plate.png"), MAGENTA).unwrap();
    let expected = cold(&root, &blue, "canvas-magenta");
    let reply = session.render(7, &root.join("magenta.png"));
    assert_eq!(reply["install_reused"], false, "{reply}");
    assert_eq!(std::fs::read(root.join("magenta.png")).unwrap(), expected);
    let reply = session.render(8, &root.join("magenta-2.png"));
    assert_eq!(reply["install_reused"], true, "{reply}");
    assert_eq!(std::fs::read(root.join("magenta-2.png")).unwrap(), expected);
    // A candidate that fails to open leaves the revision and the installation as they were.
    let revision = reply["revision"].clone();
    let mut broken = blue.clone();
    broken["nodes"].as_array_mut().unwrap().push(serde_json::json!(
        {"name": "label", "type": "text", "content": "x", "font": "missing.ttf", "font-size": 12}));
    let failed = session.open(9, &broken, &root);
    assert_eq!(failed["ok"], false, "{failed}");
    let reply = session.render(10, &root.join("after-failed-open.png"));
    assert_eq!((&reply["revision"], &reply["install_reused"], &reply["reuse"]),
        (&revision, &serde_json::json!(true), &serde_json::json!("reused")), "{reply}");
    assert_eq!(std::fs::read(root.join("after-failed-open.png")).unwrap(), expected);
    // Images no paint names are not read: a missing file or a movie among them does not stop reuse.
    let mut unused = blue.clone();
    unused["images"]["absent"] = serde_json::json!({"file": "absent.png"});
    unused["images"]["movie"] = serde_json::json!({"file": "unused.mp4"});
    assert_eq!(session.open(11, &unused, &root)["ok"], true);
    let reply = session.render(12, &root.join("unused-1.png"));
    assert_eq!(std::fs::read(root.join("unused-1.png")).unwrap(), expected);
    let installs = reply["installs"].clone();
    let reply = session.render(13, &root.join("unused-2.png"));
    assert_eq!((&reply["install_reused"], &reply["reuse"], &reply["installs"]),
        (&serde_json::json!(true), &serde_json::json!("reused"), &installs), "{reply}");
    assert_eq!(std::fs::read(root.join("unused-2.png")).unwrap(), expected);
    // A 3D scene showing a canvas picture of the same file, after the file changed: the picture reads the file
    // again, not the pixels the canvas before it left in the process's image store.
    std::fs::write(root.join("plate.png"), GREEN).unwrap();
    let world = serde_json::json!({"version": 2, "render": {"resolution": [128, 64], "motion_blur": {"shutter": 0}},
        "buffers": {
            "p": {"type": "vec3", "data": [[-2,-1,0],[2,-1,0],[2,1,0],[-2,1,0]]},
            "n": {"type": "vec3", "data": [[0,0,1],[0,0,1],[0,0,1],[0,0,1]]},
            "uv": {"type": "vec2", "data": [[0,1],[1,1],[1,0],[0,0]]},
            "idx": {"type": "u32", "data": [0,1,2,0,2,3]}},
        "geometry": {"screen": {"attributes": {"POSITION": "p", "NORMAL": "n", "TEXCOORD_0": "uv"}, "indices": "idx"}},
        "images": {"plate": {"file": "plate.png"}, "picture": {"render": {"root": "ui", "resolution": [128, 64]}}},
        "materials": {"screen": {"inputs": {"base_color": [0,0,0], "emission_color": {"image": "picture"}, "emission_luminance": 1}}},
        "nodes": [
            {"name": "camera", "type": "camera", "translation": [0,0,5], "material": {"program": "std/camera", "inputs": {"projection": "ortho", "ortho_scale": 4}}},
            {"name": "screen", "type": "mesh", "geometry": "screen", "materials": ["screen"]},
            {"name": "ui", "composition": {"space": "2d", "viewBox": [0,0,128,64]}, "children": [
                {"name": "bar", "type": "path", "d": "M10 10H118V54H10Z", "fill": {"image": "plate", "fit": "fill"}}]}]});
    let expected = cold(&root, &world, "world-green");
    assert_eq!(session.open(20, &world, &root)["ok"], true);
    session.render(21, &root.join("world-green-session.png"));
    assert_eq!(std::fs::read(root.join("world-green-session.png")).unwrap(), expected);
    std::fs::write(root.join("plate.png"), MAGENTA).unwrap();
    let expected = cold(&root, &world, "world-magenta");
    session.render(22, &root.join("world-magenta-session.png"));
    assert_eq!(std::fs::read(root.join("world-magenta-session.png")).unwrap(), expected);
    assert_eq!(session.send(serde_json::json!({"id": 14, "op": "close"}))["ok"], true);
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

/// Every look uses the same delivery options in a new process and the live worker.
fn compare_look(session: &mut Session, root: &Path, doc: &serde_json::Value,
    profile: &str, settings: serde_json::Value, at: f32) -> (Vec<u8>, serde_json::Value) {
    let path = root.join("fresh.json");
    let fresh = root.join("fresh.png");
    let kept = root.join("kept.png");
    std::fs::write(&path, serde_json::to_vec(doc).unwrap()).unwrap();
    let mut command = Command::new(binary());
    command.arg("render").arg(path).args(["--profile", profile, "--at", &at.to_string()]);
    for (key, value) in settings.as_object().unwrap() { command.args(["--set", &format!("{key}={value}")]); }
    let output = command.arg("--out").arg(&fresh).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let result = session.send(serde_json::json!({"id":1,"op":"render","out":kept,"at":at,"profile":profile,"set":settings}));
    assert_eq!(result["ok"], true, "{result}");
    let picture = std::fs::read(fresh).unwrap();
    assert_eq!(std::fs::read(kept).unwrap(), picture);
    (picture, result)
}

#[test]
fn moving_inside_a_dielectric_reinstalls_camera_containment() {
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let root = std::env::temp_dir().join(format!("vork-session-facts-{}",std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut session = Session::start();
    let mut doc = scene(140., 1.);
    doc["nodes"][0]["materials"][0]["inputs"] = serde_json::json!({"transmission_weight":1,"specular_roughness":0,
        "transmission_color":[0.1,0.7,0.3],"transmission_depth":0.3});
    let settings = serde_json::json!({"samples":2,"denoise":false});
    assert_eq!(session.open(1,&doc,&root)["ok"],true);
    let (outside, _) = compare_look(&mut session,&root,&doc,"std/path",settings.clone(),0.);
    doc["nodes"][3]["translation"] = serde_json::json!([0,0.5,0]);
    let opened = session.open(2,&doc,&root);
    let (inside, _) = compare_look(&mut session,&root,&doc,"std/path",settings.clone(),0.);
    assert_ne!(inside,outside);
    assert_eq!(opened["reuse"],"reinstalled","{opened}");

    session.send(serde_json::json!({"id":3,"op":"close"}));
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn scalar_edit_preserves_folded_material_tint() {
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let root = std::env::temp_dir().join(format!("vork-session-folded-{}",std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut session = Session::start();
    let settings = serde_json::json!({"samples":1,"denoise":false});
    let mut doc = scene(140.,0.8);
    doc["programs"] = serde_json::json!({"paint":{"source":"import vork; struct Inputs {float3 tint; float gain;}; Closure surface(ShadingContext c, Inputs i){Closure result={}; result.base_color=i.tint*i.gain; result.geometry_opacity=1; return result;}"}});
    doc["materials"] = serde_json::json!({"paint":{"program":"paint","inputs":{"tint":[1,0,0],"gain":1}}});
    doc["nodes"][0]["materials"] = serde_json::json!(["paint"]);
    doc["animation"] = serde_json::json!([{"tracks":[{"path":["materials","paint","inputs","tint"],"keys":[[0,[0,1,0]]]}]}]);
    assert_eq!(session.open(3,&doc,&root)["ok"],true);
    let (before,_) = compare_look(&mut session,&root,&doc,"std/previs",settings.clone(),0.);
    doc["materials"]["paint"]["inputs"]["gain"] = 0.3.into();
    assert_eq!(session.open(4,&doc,&root)["ok"],true);
    let (after,_) = compare_look(&mut session,&root,&doc,"std/previs",settings,0.);
    assert_ne!(before,after);
    session.send(serde_json::json!({"id":5,"op":"close"}));
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn changing_denoise_intent_reinstalls_promoted_previs() {
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let root = std::env::temp_dir().join(format!("vork-session-delivery-{}",std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut session = Session::start();
    let mut doc = scene(1.,0.6);
    doc["nodes"][0]["materials"][0]["inputs"]["emission_luminance"] = 20.into();
    doc["nodes"][0]["materials"][0]["inputs"]["emission_color"] = serde_json::json!([1,0.3,0.1]);
    assert_eq!(session.open(1,&doc,&root)["ok"],true);
    compare_look(&mut session,&root,&doc,"std/previs",serde_json::json!({}),0.);
    let (_, result) = compare_look(&mut session,&root,&doc,"std/previs",serde_json::json!({"denoise":false}),0.);
    assert_eq!(result["install_reused"],false,"{result}");
    // The emissive approximation warns and can force a fresh install. Hair
    // promotes to path without that warning and exercises the retained case.
    let mut hair = scene(140.,0.6);
    hair["nodes"].as_array_mut().unwrap().push(serde_json::json!({"type":"mesh","name":"strands",
        "geometry":{"topology":"curves","attributes":{"POSITION":{"type":"vec3","data":[[0,0,0],[0,1,0],[0.05,1.5,0],[0.1,2,0]]}},
            "widths":{"type":"f32","data":[0.15,0.15,0.1,0.05]},"curve_vertex_counts":{"type":"u32","data":[4]}},
        "materials":[{"program":"std/hair"}]}));
    assert_eq!(session.open(2,&hair,&root)["ok"],true);
    let (denoised,_) = compare_look(&mut session,&root,&hair,"std/previs",serde_json::json!({"samples":2}),0.);
    let (raw,_) = compare_look(&mut session,&root,&hair,"std/previs",serde_json::json!({"samples":2,"denoise":false}),0.);
    assert_ne!(denoised,raw);
    session.send(serde_json::json!({"id":3,"op":"close"}));
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn seeking_across_a_camera_cut_reinstalls_probe_placement() {
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let root = std::env::temp_dir().join(format!("vork-session-cut-{}",std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let mut session = Session::start();
    let mut doc = scene(140.,0.6);
    let mut other = doc["nodes"][3].clone();
    other["name"] = "second".into();
    other["translation"] = serde_json::json!([2,1,4]);
    doc["nodes"].as_array_mut().unwrap().push(other);
    doc["animation"] = serde_json::json!([{"tracks":[{"path":["render","camera"],"keys":[[0,"cam"],[0.5,"second"]]}]}]);
    assert_eq!(session.open(2,&doc,&root)["ok"],true);
    let settings = serde_json::json!({"denoise":false,"samples":1});
    let (first,_) = compare_look(&mut session,&root,&doc,"std/draft",settings.clone(),0.);
    let (second,result) = compare_look(&mut session,&root,&doc,"std/draft",settings,0.75);
    assert_ne!(first,second);
    assert_eq!(result["install_reused"],false,"{result}");
    session.send(serde_json::json!({"id":3,"op":"close"}));
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejected_open_replays_keyed_fonts() {
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let root = std::env::temp_dir().join(format!("vork-session-keyed-font-{}",std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let regular = include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf");
    let bold = include_bytes!("../../vork-std/tests/fonts/LiberationSans-Bold.ttf");
    std::fs::write(root.join("regular.ttf"),regular).unwrap();
    for rich_content in [false,true] {
        std::fs::write(root.join("keyed.ttf"),bold).unwrap();
        let mut doc = serde_json::json!({"version":2,"composition":{"space":"2d","viewBox":[0,0,128,64]},
            "render":{"resolution":[128,64]},"nodes":[{"name":"inputs","type":"text","font":"regular.ttf",
                "content":"WWii","font-size":32,"fill":"#ffffff","translation":[4,8,0],
                "animation":[{"tracks":[{"path":["font"],"keys":[[0,"keyed.ttf"]]}]}]}]});
        if rich_content {
            doc["nodes"][0].as_object_mut().unwrap().remove("animation");
            doc["animation"] = serde_json::json!([{"tracks":[{"path":["nodes","inputs","content"],"interpolation":"STEP",
                "keys":[[0,[{"content":"WWii","font":["keyed.ttf"]}]]]}]}]);
        }
        let small = cold_with(&root,&doc,"published-small",&["--set","resolution=64x32"]);
        let mut session = Session::start();
        let opened = session.open(1,&doc,&root);
        assert_eq!(opened["ok"],true,"{opened}");
        session.render(2,&root.join("large.png"));
        std::fs::write(root.join("keyed.ttf"),regular).unwrap();
        assert_ne!(small,cold_with(&root,&doc,"replacement-small",&["--set","resolution=64x32"]));
        assert_eq!(session.open(3,&serde_json::json!({"version":0}),&root)["ok"],false);
        let result = session.send(serde_json::json!({"id":4,"op":"render","out":root.join("small.png"),
            "profile":"std/draft","set":{"samples":1,"denoise":false,"resolution":"64x32"}}));
        assert_eq!(result["ok"],true,"{result}");
        assert_eq!(small,std::fs::read(root.join("small.png")).unwrap());
        session.send(serde_json::json!({"id":5,"op":"close"}));
        assert!(session.child.wait().unwrap().success());
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn eof_releases_assets_published_before_the_first_render() {
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let root = std::env::temp_dir().join(format!("vork-session-eof-files-{}",std::process::id()));
    let private = root.join("private");
    std::fs::create_dir_all(&private).unwrap();
    std::fs::write(root.join("font.ttf"),include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf")).unwrap();
    let doc = serde_json::json!({"version":2,"composition":{"space":"2d"},"render":{"resolution":[32,32]},
        "nodes":[{"type":"text","content":"A","font":"font.ttf","font-size":16}]});
    let mut command = Command::new(binary());
    command.env("TMPDIR",&private);
    let mut session = Session::command(command);
    let opened = session.open(1,&doc,&root);
    assert_eq!(opened["ok"],true,"{opened}");
    let copies = || std::fs::read_dir(&private).unwrap().flatten()
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("vork-revision-")).count();
    assert!(copies()>0);
    drop(session.child.stdin.take());
    assert!(session.child.wait().unwrap().success());
    assert_eq!(copies(),0);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejected_open_replays_executable_relative_filmic_lut() {
    if std::env::var("VORK_GPU_TESTS").as_deref() != Ok("1") { return; }
    let root = std::env::temp_dir().join(format!("vork-session-lut-{}",std::process::id()));
    let lut = Path::new("libraries/blender/filmic.exr");
    std::fs::create_dir_all(root.join(lut).parent().unwrap()).unwrap();
    let original = std::fs::canonicalize(binary()).unwrap();
    let package = original.parent().unwrap();
    // Give the child its own executable-relative data. Other jobs share the installed package.
    for name in ["vork","vork-slang-filesystem"] {
        let dest = root.join(name);
        if std::fs::hard_link(package.join(name), &dest).is_err() {std::fs::copy(package.join(name),dest).unwrap();}
    }
    std::fs::copy(package.join(lut),root.join(lut)).unwrap();
    let mut session = Session::command(Command::new(root.join("vork")));
    let mut doc = scene(140.,0.5);
    doc["composition"] = serde_json::json!({"space":"3d","view":{"program":"blender/filmic"}});
    let small = cold_with(&root,&doc,"published-small",&["--set","resolution=32x24"]);
    let opened = session.open(1,&doc,&root);
    assert_eq!(opened["ok"],true,"{opened}");
    session.render(2,&root.join("large.png"));
    std::fs::remove_file(root.join(lut)).unwrap();
    assert_eq!(session.open(3,&doc,&root)["ok"],false);
    let result = session.send(serde_json::json!({"id":4,"op":"render","out":root.join("small.png"),
        "profile":"std/draft","set":{"samples":1,"denoise":false,"resolution":"32x24"}}));
    assert_eq!(result["ok"],true,"{result}");
    assert_eq!(small,std::fs::read(root.join("small.png")).unwrap());
    session.send(serde_json::json!({"id":5,"op":"close"}));
    assert!(session.child.wait().unwrap().success());
    std::fs::remove_dir_all(root).unwrap();
}

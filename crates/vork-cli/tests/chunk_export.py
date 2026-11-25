#!/usr/bin/env python3
"""End-to-end chunk export regression. Run each case separately on a GPU, then run check."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import wave
import zlib

CASES = ("first", "second", "scene", "asset", "override", "build", "mix", "other-mix")
FRAME_COUNT = 4
RATE = "30000/1001"


def image(path, rgb):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes((0, *rgb)), level=0))
        + chunk(b"IEND", b"")
    )


def render(case, binary, out, other_build):
    base = out / case
    base.mkdir(parents=True, exist_ok=True)
    scene = {
        "version": 2,
        "composition": {"space": "2d", "viewBox": [0, 0, 128, 64]},
        "render": {"resolution": [128, 64], "fps": RATE, "duration": "1001/7500",
                   "background": "#17243a"},
        "images": {"tile": {"file": "tile.png"}},
        "nodes": [
            {"type": "path", "d": "M8 8H120V56H8Z", "fill": {"image": "tile", "fit": "cover"}},
            {"type": "path", "d": "M30 20H96V44H30Z", "fill": "#153348"},
            {"audio": {"file": "tone.wav"}},
        ],
    }
    if case == "scene":
        scene["nodes"][1]["fill"] = "#f5ab35"
    if case == "other-mix":
        scene["nodes"][2]["audio"]["gain"] = 0.5
    (base / "scene.json").write_text(json.dumps(scene))
    tile = base / "tile.png"
    image(tile, (55, 192, 142))
    original = tile.stat()
    if case == "asset":
        image(tile, (152, 56, 203))
    os.utime(tile, ns=(original.st_atime_ns, 1000000000000000000))

    with wave.open(str(base / "tone.wav"), "wb") as tone:
        tone.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        tone.writeframes(struct.pack("<96h", *([4000] * 48 + [-4000] * 48)) * 70)

    args = ["--profile", "std/draft"]
    if case == "override":
        args += ["--set", "seed=1"]
    if case.endswith("mix"):
        args += ["--out", base / "mix.wav", "--export-identity"]
    else:
        start = 0 if case == "first" else 2
        args += ["--out", base / "clip.mp4", "--no-audio", "--frame-start", str(start),
                 "--frame-end", str(start + 1)]
    if case == "build":
        binary = other_build
    subprocess.run([binary, "render", base / "scene.json", *args], check=True)


def packets(path):
    info = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_data_hash", "sha256", "-of", "json", path,
    ])
    return [packet["data_hash"] for packet in json.loads(info)["packets"]]


def check(out, stitch):
    first = out / "first" / "clip.mp4"
    second = out / "second" / "clip.mp4"
    joined = out / "joined.mp4"
    wav = out / "mix" / "mix.wav"

    def join(part, audio=None, *, expected=None):
        cmd = [sys.executable, stitch, "--frames", str(FRAME_COUNT), "--out", joined, first, part]
        if audio:
            cmd += ["--audio", audio]
        result = subprocess.run(cmd, text=True, capture_output=True)
        if expected is None:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode == 1 and expected in result.stderr, result.stderr

    join(second)
    join(second, wav)
    assert packets(joined) == packets(first) + packets(second)
    good_output = joined.read_bytes()
    for case in ("scene", "asset", "override", "build"):
        join(out / case / "clip.mp4", expected="export identity differs")
    join(second, out / "other-mix" / "mix.wav", expected="export identity differs")
    for correct, other in [(second, out / "scene" / "clip.mp4"), (wav, out / "other-mix" / "mix.wav")]:
        wrong = out / ("mispaired" + correct.suffix)
        shutil.copyfile(other, wrong)
        shutil.copyfile(str(correct) + ".meta.json", str(wrong) + ".meta.json")
        if correct == wav:
            join(second, wrong, expected="payload digest differs")
        else:
            join(wrong, expected="payload digest differs")
    legacy = out / "old.mp4"
    metadata = json.loads(second.with_name("clip.mp4.meta.json").read_text())
    del metadata["export_identity"]
    legacy.with_name("old.mp4.meta.json").write_text(json.dumps(metadata))
    join(legacy, expected="no supported export identity")
    assert joined.read_bytes() == good_output
    print("Matching chunks and WAV joined with unchanged H.264 packets. "
          "Scene, asset, override, build, WAV and legacy sidecar cases refused.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=shutil.which("vork"))
    parser.add_argument("--other-build", type=Path, help="Executable with a different embedded build version for the build case")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stitch", type=Path)
    parser.add_argument("case", choices=(*CASES, "check"))
    args = parser.parse_args()
    if args.case == "check":
        stitch = args.stitch or (Path(__file__).resolve().parents[3]
                                 / "skills" / "vork" / "scripts" / "chunk-export.py")
        check(args.out.resolve(), stitch.resolve())
    else:
        if args.case == "build" and args.other_build is None:
            parser.error("the build case needs --other-build")
        render(args.case, args.binary.resolve(), args.out.resolve(),
               args.other_build.resolve() if args.other_build else None)


if __name__ == "__main__":
    main()

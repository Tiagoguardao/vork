"""Render tiny tagged plates through the public still command and compare FFmpeg pixels."""
import argparse
import json
from fractions import Fraction
from pathlib import Path
import struct
import subprocess
import tempfile


def ffmpeg(*args):
    return subprocess.check_output(["ffmpeg", "-v", "error", "-nostdin", "-filter_threads", "1", *args])


def plate(case, folder, height):
    source = "color=c=0x689abd:s=64x32:r=30"
    decoder = []
    reference = []
    at = "0"
    if case in ("pq", "hlg"):
        transfer = {"pq": "smpte2084", "hlg": "arib-std-b67"}[case]
        # Two luminances, including an above-white patch. Encode the tagged samples losslessly.
        if height == 256:
            # Enough rows for two zscale slices. Coloured edges expose slice-dependent chroma.
            source = f"testsrc2=size=64x{height}:rate=30,format=yuv420p10le"
        else:
            source += ",drawbox=x=32:w=32:h=32:color=white:t=fill,format=yuv444p10le"
        path = folder / "plate.mkv"
        ffmpeg("-f", "lavfi", "-i", source, "-frames:v", "1", "-c:v", "ffv1",
               "-color_primaries", "bt2020", "-color_trc", transfer, "-colorspace", "bt2020nc", path)
        reference = ["zscale=transfer=linear:npl=100", "format=gbrpf32le", "zscale=primaries=709",
                     "tonemap=tonemap=clip", "zscale=transfer=iec61966-2-1:matrix=gbr:range=full"]
    elif case in ("hevc10", "prores422", "prores4444"):
        if case == "hevc10":
            path = folder / "plate.mp4"
            source = "testsrc2=size=64x32,format=yuv420p10le"
            codec = ["-c:v", "libx265", "-x265-params", "pools=1:frame-threads=1:log-level=error"]
        else:
            path = folder / "plate.mov"
            fmt = "yuv422p10le" if case == "prores422" else "yuva444p10le"
            source = f"testsrc2=size=64x32,format={fmt}"
            if case == "prores4444":
                source += ",geq=lum=lum(X\\,Y):cb=cb(X\\,Y):cr=cr(X\\,Y):a=1023*X/(W-1)"
            codec = ["-c:v", "prores_ks", "-profile:v", "2" if case == "prores422" else "4"]
        ffmpeg("-f", "lavfi", "-i", source, "-frames:v", "1", "-threads", "1", *codec, path)
    elif case in ("webm", "alpha-mkv"):
        path = folder / "plate.webm"
        source += ",format=yuva420p,geq=lum=lum(X\\,Y):cb=cb(X\\,Y):cr=cr(X\\,Y):a=255*X/(W-1)"
        ffmpeg("-f", "lavfi", "-i", source, "-frames:v", "1", "-c:v", "libvpx-vp9", "-lossless", "1", path)
        if case == "alpha-mkv":
            remux = path.with_suffix(".mkv")
            ffmpeg("-i", path, "-c", "copy", remux)
            path = remux
        decoder = ["-c:v", "libvpx-vp9"]
    else:
        at = "17/30"
        path = folder / ("cut.mkv" if case == "rounded-pts" else "cut.mp4")
        ffmpeg("-f", "lavfi", "-i", "color=red:s=64x32:r=30",
               "-vf", "drawbox=color=blue:t=fill:enable=gte(n\\,17)", "-frames:v", "19",
               "-c:v", "libx264", path)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "stream=time_base:frame=best_effort_timestamp", "-of", "json", path]))
        tick = Fraction(probe["streams"][0]["time_base"])
        pts = [int(f["best_effort_timestamp"]) for f in probe["frames"]]
        ordinal = max(i for i, t in enumerate(pts) if (t - pts[0]) * tick <= Fraction(at))
        assert ordinal == (16 if case == "rounded-pts" else 17)
        reference = [f"select=eq(n\\,{ordinal})"]
    return path, decoder, reference, at


def compare(binary, case):
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        height = 256 if case in ("pq", "hlg") else 32
        kind = "pq" if case.startswith("pq-") else case
        path, decoder, reference, at = plate(kind, folder, height)
        scene = folder / "scene.json"
        document = {
            "version": 2, "composition": {"space": "2d"},
            "render": {"resolution": [64, height], "fps": 30, "duration": 1, "background": "transparent"},
            "images": {"plate": {"file": path.name}},
            "nodes": [{"type": "path", "d": f"M0 0H64V{height}H0Z", "fill": {"image": "plate"}}]}
        if case == "pq-override":
            document["images"]["plate"]["color_space"] = "sRGB"
            reference = []
        elif case == "pq-graph":
            document["composition"].update(color_space="Linear Rec.709", compositor=["copy"])
            document["nodes"] = []
            document["programs"] = {"copy": {"source":
                "import vork; public struct Inputs { Image plate; }; "
                "public float4 composite(CompositeContext c, Inputs i) { return compositor_sample(i.plate,c.uv); }"}}
            document["materials"] = {"copy": {"program": "copy", "inputs": {"plate": {"image": "plate"}}}}
        scene.write_text(json.dumps(document))
        output = folder / "still.png"
        subprocess.run([binary, "render", scene, "--at", at, "--out", output], check=True)
        actual = ffmpeg("-i", output, "-pix_fmt", "rgba", "-f", "rawvideo", "-")
        filters = ["-vf", ",".join(reference)] if reference else []
        deep = ffmpeg(*decoder, "-i", path, *filters, "-frames:v", "1", "-pix_fmt", "rgba64le", "-f", "rawvideo", "-")
        expected = bytes((v + 128) // 257 for v, in struct.iter_unpack("<H", deep))
        assert len(actual) == len(expected) == 64 * height * 4
        if case in ("webm", "alpha-mkv", "prores4444"):
            assert len(set(expected[3::4])) > 32
            assert actual[3::4] == expected[3::4]
            # PNG is straight alpha; the sRGB paint stores rounded associated 8-bit colours.
            def associated(data, i):
                return (data[i] * data[i | 3] + 127) // 255
            assert max(abs(associated(actual, i) - associated(expected, i))
                       for i in range(len(actual)) if i % 4 != 3) <= 1
        else:
            assert actual == expected, (case, actual[:4], expected[:4],
                                        max(abs(a - b) for a, b in zip(actual, expected)))


if __name__ == "__main__":
    cases = ("pq", "pq-override", "pq-graph", "hlg", "cut", "rounded-pts", "webm", "alpha-mkv",
             "hevc10", "prores422", "prores4444")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("--case", choices=cases)
    args = parser.parse_args()
    for case in [args.case] if args.case else cases:
        compare(args.binary, case)

"""Generate small deterministic movies for the displayed-frame/seek oracle."""
import json
import random
import struct
import subprocess
import sys
from pathlib import Path


def encode(path, *options, source="testsrc2=s=64x48:r=30:d=3"):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi",
                    "-i", source, "-c:v", "libx264", "-threads", "1",
                    "-bf", "0", "-g", "12", *options, "-y", path], check=True)


def boxes(data):
    pos = 0
    while pos < len(data):
        size, kind = struct.unpack_from(">I4s", data, pos)
        assert 8 <= size <= len(data) - pos
        yield kind, data[pos + 8:pos + size]
        pos += size


def box(kind, payload):
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def edit(data, change):
    result = bytearray()
    for kind, payload in boxes(data):
        if kind in (b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"):
            payload = edit(payload, change)
        kind, payload = change(kind, payload)
        result.extend(box(kind, payload))
    return bytes(result)


def co64(kind, value):
    if kind == b"stco":
        offsets = struct.iter_unpack(">I", value[8:])
        return b"co64", value[:8] + b"".join(struct.pack(">Q", x) for (x,) in offsets)
    return kind, value

def descriptions(kind, value):
    if kind == b"stsd":
        value = value[:4] + struct.pack(">I", 2) + value[8:] * 2
    return kind, value


def conflicting_offsets(path, sample, offset, size="64x48"):
    # POC type 2 reference pictures in coded order. Shift one composition
    # timestamp across its neighbour, keeping all 90 distinct pictures.
    encode(path, "-profile:v", "high", "-video_track_timescale", "300", "-use_editlist", "0",
           source=f"nullsrc=s={size}:r=30:d=3,geq=lum='16+N':cb=128:cr=128")
    def conflict(kind, value):
        if kind == b"stbl":
            value += box(b"ctts", struct.pack(">8I", 0, 3, sample, 0, 1, offset, 89-sample, 0))
        return kind, value
    path.write_bytes(edit(path.read_bytes(), conflict))


def repeated_raw(path, conflict):
    # Raw PTS 130 identifies two different key pictures. The uninterrupted
    # repaired clock gives them distinct displayed times 130 and 140.
    def repeat_raw(kind, value):
        if kind == b"ctts":
            value = struct.pack(">12I", 0, 5, 5, 0, 1, 15, 6, 0, 1, 10, 77, 0)
        return kind, value
    path.write_bytes(edit(conflict.read_bytes(), repeat_raw))


def native_held(directory):
    conflict = directory / "conflict.mp4"
    conflicting_offsets(conflict, 5, 15, size="256x144")
    repeated_raw(directory / "repeated-raw.mp4", conflict)
    encode(directory / "control.mp4", source="testsrc2=s=256x144:r=30:d=3")


def native_damage(directory):
    clean = directory / "clean.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=s=320x180:r=30", "-c:v", "libvpx-vp9", "-threads", "1",
                    "-frames:v", "27", "-g", "11", "-lag-in-frames", "0",
                    "-auto-alt-ref", "0", "-cpu-used", "6", "-video_track_timescale", "30",
                    "-y", clean], check=True)
    packets = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pos,size,flags", "-of", "json", clean]))["packets"]
    assert "K" in packets[11]["flags"]
    pos, size = (int(packets[10][field]) for field in ("pos", "size"))
    assert size > 64
    # Retain the header, sample length and timestamps. The next key picture
    # makes the post-damage pixel reference independent of error concealment.
    data = bytearray(clean.read_bytes())
    data[pos + 64:pos + size] = bytes([255]) * (size - 64)
    (directory / "damaged.mp4").write_bytes(data)


def fixtures(directory):
    plain = directory / "control.mp4"
    encode(plain, "-video_track_timescale", "30", "-output_ts_offset", "1")
    data = plain.read_bytes()  # The moov follows mdat, so offsets survive resizing.
    assert [k for k, _ in boxes(data)].index(b"moov") > [k for k, _ in boxes(data)].index(b"mdat")

    # Preserve the NAL length, type and first slice byte, but make the
    # eleventh sample undecodable. Its table row and later sync samples survive.
    positions = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pos", "-of", "csv=p=0", plain]).splitlines()
    at = int(positions[10]) + 6
    damaged = bytearray(data)
    damaged[at:at + 4] = bytes(4)
    (directory / "damaged.mp4").write_bytes(damaged)

    def fractional(kind, value):
        if kind == b"elst":
            assert struct.unpack_from(">IIi", value, 4) == (2, 1000, -1)
            value = value[:20] + struct.pack(">I", 2967) + value[24:]
        if kind in (b"tkhd", b"mvhd"):
            at = 20 if kind == b"tkhd" else 16
            value = value[:at] + struct.pack(">I", 3967) + value[at + 4:]
        return kind, value

    def sparse_sync(kind, value):
        if kind == b"stss":
            entries = [n for (n,) in struct.iter_unpack(">I", value[8:]) if n != 13]
            assert len(entries) + 1 == struct.unpack_from(">I", value, 4)[0]
            value = value[:4] + struct.pack(">I", len(entries)) + b"".join(struct.pack(">I", n) for n in entries)
        return kind, value

    conflict = directory / "conflict.mp4"
    conflicting_offsets(conflict, 5, 15)
    repeated_raw(directory / "repeated-raw.mp4", conflict)

    def duplicates(kind, value):
        if kind == b"stbl":
            # Two samples have media PTS 13. Preserve tick zero and the edit.
            value += box(b"ctts", struct.pack(">IIIIIIII", 0, 3, 12, 0, 1, 1, 77, 0))
        return kind, value

    for name, change in [("fractional", fractional), ("co64", co64), ("sparse-sync", sparse_sync),
                         ("descriptions", descriptions), ("duplicates", duplicates)]:
        (directory / (name + ".mov")).write_bytes(edit(data, change))
    encode(directory / "refresh.mp4", "-x264-params", "intra-refresh=1:scenecut=0:keyint=12",
           source="testsrc2=s=128x96:r=30:d=3")
    encode(directory / "field.mov", "-flags", "+ildct+ilme", "-x264-params", "tff=1")
    encode(directory / "signed.mov", "-bf", "3", "-movflags", "+negative_cts_offsets")
    encode(directory / "vfr.mp4", "-vf", "select='mod(n,3)+eq(mod(n,7),0)'", "-fps_mode", "vfr")
    # Legal in-band configuration: the decoder obtains SPS/PPS from each IDR.
    # Keep the avc1 spelling to exercise the parameter-array admission gate.
    inband = directory / "inband.mp4"
    encode(inband, "-x264-params", "repeat-headers=1")
    data = bytearray(inband.read_bytes())
    config = data.index(b"avcC") + 4
    size = struct.unpack_from(">I", data, config - 8)[0] - 8
    data[config + 5:config + size] = b"\xe0\x00" + bytes(size - 7)
    inband.write_bytes(data)
    fast = directory / "short.mp4"
    encode(fast, "-movflags", "+faststart")
    data = fast.read_bytes()
    pos = data.index(b"mdat") - 4
    size = struct.unpack_from(">I", data, pos)[0] // 2
    fast.write_bytes(data[:pos] + struct.pack(">I", size) + data[pos + 4:pos + size])


def pages(directory):
    source = "nullsrc=s=16x16:r=25,geq=lum='16+mod(N,215)':cb=128:cr=128"
    encode(directory / "paged-table.mp4", "-frames:v", "8220", "-g", "32",
           "-x264-params", "scenecut=0", source=source)
    # A duplicate crosses each disk-page boundary, including a key picture.
    encode(directory / "paged-prefix.mkv", "-frames:v", "8220", "-c:v", "ffv1", "-g", "32",
           "-vf", r"setpts=N-gt(N\,0)*eq(mod(N\,4096)\,0)", "-fps_mode", "passthrough", source=source)


def variants(directory):
    rng = random.Random(2609)
    for i in range(64):
        b = rng.randrange(5)
        refresh = rng.randrange(4) == 0
        gop = rng.choice([6, 9, 12, 24])
        rate = rng.choice(["24", "30000/1001", "30", "60"])
        scale = rng.choice([120, 1000, 90000, 120000])
        options = ["-bf", str(b), "-x264-params",
                   f"keyint={gop}:min-keyint={gop}:scenecut=0:intra-refresh={int(refresh)}",
                   "-video_track_timescale", str(scale),
                   "-movie_timescale", str(rng.choice([60, 600, 1000, 1001])),
                   "-output_ts_offset", rng.choice(["0", "0.033", "1", "5.125"]),
                   "-use_editlist", str(rng.randrange(2))]
        if rng.randrange(2):
            options += ["-movflags", "+negative_cts_offsets"]
        if rng.randrange(2):
            options += ["-vf", f"select='mod(n,{rng.randrange(2,8)})'", "-fps_mode", "vfr"]
        path = directory / f"case-{i:02}.{rng.choice(['mov', 'mp4'])}"
        if i % 8 == 0:
            options += ["-movflags", "+faststart"]
        encode(path, *options, source=f"testsrc2=s=64x48:r={rate}:d=2")
        data = path.read_bytes()
        if i % 8 == 0:
            # The front moov and complete first samples survive loss of the tail.
            pos = data.index(b"mdat") - 4
            size = int(struct.unpack_from(">I", data, pos)[0] * rng.uniform(.55, .95))
            data = data[:pos] + struct.pack(">I", size) + data[pos+4:pos+size]
        else:
            if i % 3 == 0:
                data = edit(data, co64)
            if i % 5 == 0:
                data = edit(data, descriptions)
            if i % 7 == 0:
                def wide_edit(kind, value):
                    if kind == b"elst":
                        assert value[0] == 0
                        entries = struct.iter_unpack(">IiI", value[8:])
                        value = bytes([1]) + value[1:8] + b"".join(
                            struct.pack(">QqI", duration, start, rate) for duration, start, rate in entries)
                    return kind, value
                data = edit(data, wide_edit)
        path.write_bytes(data)

    for i in range(8):
        conflicting_offsets(directory / f"conflict-{i}.mp4", rng.randrange(2, 75), rng.randrange(11, 20))


if __name__ == "__main__":
    {"native-damage": native_damage, "native-held": native_held, "variants": variants, "pages": pages, "regressions": fixtures}[sys.argv[2]](Path(sys.argv[1]))

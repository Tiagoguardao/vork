#!/usr/bin/env python3
"""Stream-copy a full timeline from vork render ranges and an optional full WAV."""

import argparse
from fractions import Fraction
import json
import hashlib
import re
from math import lcm
from pathlib import Path
import subprocess
import tempfile


def sidecar(path):
    return json.loads(path.with_name(path.name + ".meta.json").read_text())


def export_identity(path, metadata):
    identity = metadata.get("export_identity")
    if not isinstance(identity, str) or not re.fullmatch(r"v2:[0-9a-f]{64}", identity):
        raise ValueError(f"{path}: no supported export identity; render with current vork "
                         "in a one-shot call with identifiable scene inputs. Pass --export-identity for a full WAV or MP4.")
    return identity


def probe(path, *options):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", *options, "-of", "json", str(path),
    ]))


def streams(path):
    return probe(path, "-show_streams", "-show_data_hash", "sha256")["streams"]


def check_video(path, stream, fps, count, starts):
    if stream["codec_name"] != "h264" or stream.get("has_b_frames") != 0:
        raise ValueError(f"{path}: expected H.264 without B-frames")
    step = 1 / (fps * Fraction(stream["time_base"]))
    packets = probe(path, "-select_streams", "v:0",
                    "-show_entries", "packet=pts,dts,duration,flags")["packets"]
    if step.denominator != 1 or len(packets) != count or stream["duration_ts"] != count * step:
        raise ValueError(f"{path}: expected {count} frames at {fps} fps")
    for index, packet in enumerate(packets):
        if (packet["pts"] != index * step or packet["dts"] != index * step
                or packet["duration"] != step):
            raise ValueError(f"{path}: incorrect timestamp or duration at frame {index}")
        if index in starts and "K" not in packet["flags"]:
            raise ValueError(f"{path}: frame {index} must be a keyframe")
    return tuple(stream.get(field) for field in (
        "extradata_hash", "width", "height", "sample_aspect_ratio", "pix_fmt",
        "color_range", "color_space", "color_transfer", "color_primaries",
    ))


def stitch(parts, output, frames, audio):
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must be an MP4")
    if output in parts or output == audio:
        raise ValueError("output must differ from each input")
    manifests = [(part, sidecar(part)) for part in parts]
    first, first_metadata = manifests[0]
    identity = export_identity(first, first_metadata)

    def require_match(path, metadata):
        if export_identity(path, metadata) != identity:
            raise ValueError(f"{path}: export identity differs from {first}; render all parts "
                             "from the same scene, assets, render settings and vork build")

    for path, metadata in manifests[1:]:
        require_match(path, metadata)
    if audio:
        audio_metadata = sidecar(audio)
        require_match(audio, audio_metadata)
    with tempfile.TemporaryDirectory(prefix=".chunk-export-", dir=output.parent) as work:
        work = Path(work)
        staged = [(stage_verified(part, work / f"{i}.mp4", metadata), metadata)
                  for i, (part, metadata) in enumerate(manifests)]
        sound = stage_verified(audio, work / "mix.wav", audio_metadata) if audio else None
        stitch_verified(staged, sound, audio_metadata if audio else None, output, frames, work)


def stage_verified(path, staged, metadata):
    expected = metadata.get("payload_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{path}: missing payload digest; render this part again")
    digest = hashlib.sha256()
    with path.open("rb") as source, staged.open("xb") as destination:
        while block := source.read(1024 * 1024):
            digest.update(block)
            destination.write(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"{path}: payload digest differs from its sidecar")
    return staged


def stitch_verified(manifests, audio, audio_metadata, output, frames, work):
    first_metadata = manifests[0][1]
    fps = Fraction(str(first_metadata["fps"]))
    if fps <= 0 or frames < 1:
        raise ValueError("frame count and fps must be positive")

    starts = set()
    offset = 0
    configuration = None
    for part, metadata in manifests:
        indices = [frame["index"] for frame in metadata["frames"]]
        if (not indices or indices != list(range(offset, offset + len(indices)))
                or Fraction(str(metadata["fps"])) != fps):
            raise ValueError(f"{part}: expected consecutive source frames starting at {offset}, at {fps} fps")
        tracks = streams(part)
        if len(tracks) != 1:
            raise ValueError(f"{part}: render the range with --no-audio")
        video = check_video(part, tracks[0], fps, len(indices), {0})
        settings = (metadata.get("output_display"),
                    metadata.get("settings"), metadata.get("denoiser"), video)
        if configuration is not None and settings != configuration:
            raise ValueError(f"{part}: render settings or video configuration differs from the first chunk")
        configuration = settings
        starts.add(offset)
        offset += len(indices)
    if offset != frames:
        raise ValueError(f"chunks cover {offset} frames, expected {frames}")

    inputs = ["-map", "0:v:0"]
    timescale = fps.numerator
    if audio:
        metadata = audio_metadata
        if (metadata.get("kind") != "audio" or metadata["frame_start"] != 0
                or metadata["frame_end"] != frames - 1 or Fraction(str(metadata["fps"])) != fps):
            raise ValueError(f"{audio}: render one WAV over the full timeline at {fps} fps")
        inputs = ["-i", str(audio), *inputs, "-map", "1:a:0", "-c:a", "aac", "-ar", "48000", "-ac", "2"]
        timescale = lcm(timescale, 48000)
        if timescale > 2**31 - 1:
            timescale = fps.numerator

    names = [f"file {part.name}\n" for part, _ in manifests]
    manifest = work / "parts.ffconcat"
    manifest.write_text("ffconcat version 1.0\n" + "".join(names), encoding="utf-8")
    movie = work / "joined.mp4"
    # Concat offsets are expressed in microseconds. Assign the exact rational
    # clock from packet order after verifying one input packet per frame.
    clock = f"setts=time_base={1/fps}:pts=N:dts=N:duration=1"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-xerror", "-f", "concat",
        "-auto_convert", "0", "-i", str(manifest), *inputs, "-c:v", "copy",
        "-bsf:v", clock, "-video_track_timescale", str(fps.numerator),
        "-movie_timescale", str(timescale), str(movie),
    ], check=True)
    tracks = streams(movie)
    if check_video(movie, tracks[0], fps, frames, starts) != configuration[-1]:
        raise ValueError("joined video configuration differs from the chunks")
    if audio:
        sound = tracks[1]
        if (sound["codec_name"] != "aac" or sound["start_pts"] != 0
                or sound["duration_ts"] * Fraction(sound["time_base"]) * 48000
                != metadata["audio"]["samples"]):
            raise ValueError("joined audio does not match the full-timeline WAV duration")
    movie.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="+", type=Path, metavar="CHUNK.mp4",
                        help="video-only chunks in timeline order, with their .meta.json sidecars")
    parser.add_argument("--frames", required=True, type=int, help="total frame count")
    parser.add_argument("--audio", type=Path, metavar="MIX.wav",
                        help="full-timeline vork WAV, with its .meta.json sidecar")
    parser.add_argument("--out", required=True, type=Path, metavar="MOVIE.mp4")
    args = parser.parse_args()
    try:
        stitch([p.resolve() for p in args.parts], args.out.resolve(), args.frames,
               args.audio.resolve() if args.audio else None)
    except (ValueError, KeyError, IndexError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()

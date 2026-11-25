#!/usr/bin/env python3
"""Compare two Linux source builds against shared scene-linear path references."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import profile_conformance_energy as energy

ROOT = energy.ROOT
CUTS = (
    ('Monster', 'Monster-Under-the-Bed-cut/shot.json', [1280, 720], 'std/draft'),
    ('Classroom', 'classroom/scene.json', [1280, 720], 'std/draft'),
    ('Bitterli', 'Bitterli-living-room/shot.json', [512, 288], 'std/previs'),
    ('Cozy-Kitchen', 'Cozy-Kitchen-cut/shot.json', [1280, 720], 'std/final'),
)
REFERENCE_SEEDS = (11, 29, 47, 71)
SEEDS = (101, 131, 151, 181)
SAMPLES = 4096
BOUNCES = 64
MODES = {'raw': False, 'denoised': True}
NO_EXCESS = 'candidate no further from the reference than base, within seed noise'


@cache
def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def cases(cuts):
    if cuts is None:
        return [dict(case, width=128, profiles=['std/final']) for case in energy.cases()]
    return [dict(name=name, path=str(cuts/scene), width=resolution[0], resolution=resolution, profiles=[profile])
            for name, scene, resolution, profile in CUTS]


def prepare(case, anchor):
    scene, regions = energy.prepare_case(case, case['width'], resolution=case.get('resolution'))
    inputs = {}

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, str) and value.startswith('/') and '\n' not in value:
            path = Path(value)
            if path.is_relative_to(anchor) and path.is_file():
                rel = path.relative_to(anchor).as_posix()
                inputs[rel] = file_hash(path)
                return {'asset': rel}
        return value

    at = 0.0 if 'path' in case else 0.5
    internal = case.get('internal_media', False)
    portable_scene = portable(scene)
    # A cut can have glTF sidecars or authored includes reached via its assets.
    if case['name'] in {cut[0] for cut in CUTS}:
        parent = Path(case['path']).parent
        inputs.update({p.relative_to(anchor).as_posix(): file_hash(p)
                       for p in parent.rglob('*') if p.is_file()})
    identity = hashlib.sha256(json.dumps(
        [portable_scene, inputs, at, internal or case.get('reference_internal_media', False)],
        sort_keys=True).encode()).hexdigest()
    return scene, regions, at, identity


def build_identity(build, harness):
    identity = harness.provenance(build/'vork')
    commit = build/'commit'
    identity.update(adapter='L4', binary_sha256=file_hash(build/'vork'),
                    worker_sha256=file_hash(build/'profile_conformance'),
                    source_revision=commit.read_text().strip() if commit.is_file() else None)
    return identity


def render_job():
    h = energy.harness()
    os.environ['WGPU_BACKEND'] = 'vulkan'
    config = json.loads(Path('job.json').read_text())
    os.environ['VORK_SOURCE_REVISION'] = config['revision']
    builds = {'candidate': Path(shutil.which('vork')).resolve().parent}
    if not config['collect']:
        builds['base'] = Path('base').resolve()
    identities = {side: build_identity(build, h) for side, build in builds.items()}
    save_json(Path('hardware.json'), dict(host=os.environ.get('GPU_TEST_HOST'),
              gpu=subprocess.check_output(
                  ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                  text=True).strip(), builds=identities))
    for case in config['cases']:
        if config['anchor'] == 'cuts':
            case['path'] = str(Path(case['path']).resolve())
        name = case['name']
        print(f'START {name}', flush=True)
        out = Path('results', name).resolve()
        out.mkdir(parents=True)
        try:
            scene, _, at, asset_identity = prepare(case, Path(config['anchor']).resolve())
            if asset_identity != case['identity']:
                raise RuntimeError('scene or asset bytes differ from the submitting host')
            render_case(h, case, scene, at, out, builds, identities, config)
        except Exception as error:
            save_json(out/'error.json', {'error': str(error)})
            print(f'{name}: {error}', flush=True)
        else:
            print(f'FINISHED {name}', flush=True)


def render_case(h, case, scene, at, out, builds, identities, config):
    with ExitStack() as stack:
        runners = {}
        for side, build in builds.items():
            directory = out/side
            directory.mkdir()
            identity = dict(identities[side], scene_inputs_sha256=case['identity'])
            runner = h.Runner(build/'vork', directory, identity, config['render_timeout'],
                              build/'profile_conformance')
            stack.callback(runner.close)
            runner.at = str(at)
            runner.scene_path = out/'scene.json'
            runner.internal_media = case.get('internal_media', False)
            runners[side] = runner

        if config['collect']:
            runner = runners['candidate']
            runner.internal_media |= case.get('reference_internal_media', False)
            runner.reference_bounces = BOUNCES
            batches = {}
            for seed in REFERENCE_SEEDS:
                _, _, _, key = runner.render(
                    scene, 'std/path', h.path_reference_settings(SAMPLES, seed),
                    f'{case["name"]}/reference/{seed}', reuse=True, preview=False)
                batches[seed] = key
            save_json(out/'renders.json', batches)
            return

        frames = {}
        for profile in case['profiles']:
            frames[profile] = {}
            for mode, denoise in MODES.items():
                frames[profile][mode] = []
                for seed in config['seeds']:
                    keys = {}
                    for side in ('base', 'candidate'):
                        runner = runners[side]
                        _, _, _, key = runner.render(
                            scene, profile, {'seed': seed, 'denoise': denoise},
                            f'{case["name"]}/{profile}/{mode}/{seed}/{side}',
                            fresh=not runner.internal_media, preview=False)
                        keys[side] = key
                    frames[profile][mode].append(keys)
        save_json(out/'renders.json', frames)


def stage_build(source, dest):
    dest.mkdir()
    for name in ('vork', 'vork-slang-filesystem', 'examples/profile_conformance', 'commit'):
        path = source/name
        if name == 'commit' and not path.exists():
            continue
        # The worker resolves packaged library images beside its executable.
        target = dest/Path(name).name
        shutil.copy2(path, target)
    if (source/'libraries').is_dir():
        shutil.copytree(source/'libraries', dest/'libraries')


def stage_job(job, group, args, revision, candidate):
    repo = job/'repo'
    scripts = repo/'scripts'
    scripts.mkdir(parents=True)
    for name in ('refcheck.py', 'profile-conformance.py', 'profile_conformance_energy.py',
                 'profile_conformance_scenes.py', 'render-invariants.py', 'render_exr.py'):
        shutil.copy2(ROOT/'scripts'/name, scripts/name)
    for name in ('packages/ts/scene.schema.json',
                 'crates/vork-render/tests/fixtures/fog-height-carriers.json'):
        target = repo/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    if args.cuts:
        for case in group:
            relative = Path(case['path']).relative_to(args.cuts)
            shutil.copytree(Path(case['path']).parent, job/'cuts'/relative.parent)
            case['path'] = str(Path('cuts')/relative)
    else:
        shutil.copytree(ROOT/'scenes/bugbash', repo/'scenes/bugbash')
    if not args.collect_references:
        stage_build(args.base, job/'base')
    save_json(job/'job.json', dict(cases=group, collect=args.collect_references,
              anchor='cuts' if args.cuts else 'repo', seeds=args.seeds,
              render_timeout=args.render_timeout, revision=revision))
    with job.with_suffix('.log').open('w') as log:
        result = subprocess.run(
            [args.gpu_test, '--timeout', str(args.job_timeout), '--dir', str(job),
             str(candidate), '--', 'python3', '-u', 'repo/scripts/refcheck.py', '--gpu-job'],
            stdout=log, stderr=subprocess.STDOUT)
    return job, result.returncode


def frame(out, side, key):
    return out/side/'renders'/key/'frame.000000.exr'


def install_reference(store, case, out, batches, hardware, side):
    """Publish complete, immutable batches under the scene/input identity."""
    import gzip
    import tempfile

    target = store/case['identity']
    with tempfile.TemporaryDirectory(dir=store) as pending:
        directory = Path(pending)/'reference'
        directory.mkdir()
        receipts = []
        for seed in REFERENCE_SEEDS:
            key = batches[str(seed)]
            source = frame(out, side, key)
            metadata = json.loads(source.with_name('frame.exr.meta.json').read_text())
            timing = json.loads(source.with_name('frame.exr.timings.json').read_text())
            settings = metadata['settings']
            at = 0 if 'path' in case else 0.5
            internal = bool(case.get('internal_media') or case.get('reference_internal_media'))
            if (settings['samples'] != SAMPLES or settings['seed'] != seed
                    or settings.get('denoiser') is not None
                    or settings.get('integrator') != 'path' or metadata['profile']['name'] != 'std/path'
                    or timing.get('reference_bounces') != BOUNCES or timing.get('at') != at
                    or bool(timing.get('internal_media')) != internal
                    or timing.get('backend') != 'Vulkan' or 'L4' not in timing.get('device', '')):
                raise RuntimeError(f'{out}: unexpected reference controls for seed {seed}')
            filename = f'{seed}.exr.gz'
            with source.open('rb') as raw, gzip.open(directory/filename, 'wb', compresslevel=3) as compressed:
                shutil.copyfileobj(raw, compressed)
            receipts.append(dict(seed=seed, file=filename, sha256=file_hash(directory/filename),
                                 render_key=key, settings=settings, timing=timing))
        save_json(directory/'reference.json', dict(
            case=case['name'], identity=case['identity'], samples_per_batch=SAMPLES, bounces=BOUNCES,
            hardware=hardware, batches=receipts))
        try:
            directory.rename(target)
        except OSError:
            if not target.is_dir():
                raise
            load_reference(store, case)


def load_reference(store, case):
    directory = store/case['identity']
    manifest = json.loads((directory/'reference.json').read_text())
    if (manifest['identity'] != case['identity']
            or manifest['samples_per_batch'] != SAMPLES or manifest['bounces'] != BOUNCES
            or tuple(batch['seed'] for batch in manifest['batches']) != REFERENCE_SEEDS):
        raise RuntimeError(f'{directory}: incompatible reference manifest')
    paths = []
    for batch in manifest['batches']:
        path = directory/f'{batch["seed"]}.exr.gz'
        if file_hash(path) != batch['sha256']:
            raise RuntimeError(f'{path}: reference checksum mismatch')
        paths.append(path)
    return manifest, paths


def pixels(path, resolution, compressed=False):
    import gzip
    import tempfile

    import numpy as np

    from render_exr import read_exr

    if compressed:
        with gzip.open(path, 'rb') as source, tempfile.NamedTemporaryFile() as decoded:
            shutil.copyfileobj(source, decoded)
            decoded.flush()
            rgb, width, height = read_exr(decoded.name)
    else:
        rgb, width, height = read_exr(path)
    if [width, height] != resolution:
        raise RuntimeError(f'{path}: expected {resolution}, got {[width, height]}')
    rgb = np.asarray([rgb[c] for c in 'RGB'], dtype=np.float64).reshape(3, height, width)
    if not np.isfinite(rgb).all():
        raise RuntimeError(f'{path}: nonfinite RGB')
    return rgb


def region_box(rectangle, width, height):
    x0, y0, x1, y1 = rectangle
    return int(x0*width), int(y0*height), int(x1*width), int(y1*height)


def crop(image, box):
    x0, y0, x1, y1 = box
    return image[:, y0:y1, x0:x1]


def rmse(image):
    import numpy as np
    return float(np.sqrt(np.mean(np.square(image))))


def paired_error(base, candidate, reference):
    base_residual = base-reference
    candidate_residual = candidate-reference
    b, c = rmse(base_residual), rmse(candidate_residual)
    # Difference of squared distances, divided by their sum, preserves low bits
    # when comparing almost identical renders far from the reference.
    delta = float(((candidate-base)*(candidate_residual+base_residual)).mean())/(b+c) if b+c else 0.0
    return b, c, delta


def score(case, out, frames, reference, regions, args, hardware):
    import statistics

    import numpy as np

    h = energy.harness()
    manifest, paths = reference
    resolution = json.loads((out/'scene.json').read_text())['render']['resolution']
    batches = [pixels(path, resolution, compressed=True) for path in paths]
    first, second = (np.mean(half, axis=0) for half in (batches[:2], batches[2:]))
    truth = (first+second)*0.5
    boxes = {region: region_box(rect, *resolution) for region, rect in regions.items()}
    boxes = {region: box for region, box in boxes.items() if box[0] < box[2] and box[1] < box[3]}
    reference_noise = {region: rmse(crop((first-second)*0.5, box)) for region, box in boxes.items()}
    tests = len(case['profiles'])*len(MODES)*len(boxes)
    rows = []
    for profile in case['profiles']:
        for mode in MODES:
            paths = [{side: frame(out, side, key) for side, key in pair.items()}
                     for pair in frames[profile][mode]]
            repeats = {region: [] for region in boxes}
            resolved = []
            for seed, pair in zip(args.seeds, paths, strict=True):
                base, candidate = (pixels(pair[side], resolution) for side in ('base', 'candidate'))
                for region, box in boxes.items():
                    repeats[region].append(paired_error(*(crop(rgb, box) for rgb in (base, candidate, truth))))
                resolved.append(dict(seed=seed, builds={
                    side: frame_settings(path, seed, mode, profile) for side, path in pair.items()}))
            summaries = {}
            for region, values in repeats.items():
                b, c, delta = (statistics.mean(column) for column in zip(*values))
                noise = energy.interval([pair[2] for pair in values], tests)
                summaries[region] = dict(base=b, candidate=c, delta=delta, seed_noise=noise,
                                         lower=delta-noise, upper=delta+noise, repeats=values,
                                         reference_half_difference=reference_noise[region])
            worst = max((region for region in summaries if region != 'whole'),
                        key=lambda region: (summaries[region]['lower'], summaries[region]['delta'],
                                            summaries[region]['candidate']))
            excess = max(summaries, key=lambda region: summaries[region]['lower'])
            result = summaries[excess]
            statement = (f'measured excess {result["delta"]:+.6g} ± {result["seed_noise"]:.3g} RGB RMSE ({excess})'
                         if result['lower'] > 0 else NO_EXCESS)
            # Show the paired seed with the largest observed excess in this crop.
            index = max(range(len(paths)), key=lambda i: repeats[worst][i][2])
            tiles = [crop(pixels(paths[index][side], resolution), boxes[worst])
                     for side in ('base', 'candidate')]
            tiles.append(crop(truth, boxes[worst]))
            panel = np.concatenate(tiles, axis=2)
            image = out/f'{profile.split("/")[1]}-{mode}-crop.png'
            h.preview_png(image, panel.reshape(3, -1), panel.shape[2], panel.shape[1])
            rows.append(dict(case=case['name'], profile=profile, mode=mode, resolution=resolution,
                             statement=statement, regions=summaries, worst=worst, crop_box=boxes[worst],
                             crop_seed=args.seeds[index], crop=str(image.relative_to(args.out)),
                             internal_media=case.get('internal_media', False),
                             reference=manifest, settings=resolved, hardware=hardware))
    return rows


def frame_settings(path, seed, mode, profile):
    metadata = json.loads(path.with_name('frame.exr.meta.json').read_text())
    settings = metadata['settings']
    if (metadata['profile']['name'] != profile or settings['seed'] != seed
            or (settings.get('denoiser') is None) != (mode == 'raw')):
        raise RuntimeError(f'{path}: unexpected profile, seed or denoiser for {profile} {mode} seed {seed}')
    return settings


def report(out, rows, errors, wall):
    save_json(out/'report.json', dict(rows=rows, errors=errors, wall_seconds=wall))
    lines = [
        '# Refcheck', '',
        'Errors are mean per-seed scene-linear RGB RMSE against the shared mean of four unclamped, '
        'undenoised 4096-sample path batches at depth 64. Positive Δ is candidate worsening. '
        'Paired Student-t bounds are simultaneous at 95% within each scene across the listed modes, '
        'profiles and regions. Reference half-differences in the JSON estimate residual sampling noise '
        'in the reference, separately from the render-seed bounds.', '',
        '| Scene / profile / mode | Base | Candidate | Δ ± seed noise | Reference noise | Worst region and crop | Result |',
        '|---|---:|---:|---:|---:|---|---|',
    ]
    for row in rows:
        whole = row['regions']['whole']
        worst = row['regions'][row['worst']]
        label = row['case'] + (' (internal media)' if row['internal_media'] else '')
        lines.append(f'| {label} / {row["profile"]} / {row["mode"]} | {whole["base"]:.8g} '
                     f'| {whole["candidate"]:.8g} | {whole["delta"]:+.6g} ± {whole["seed_noise"]:.3g} '
                     f'| {whole["reference_half_difference"]:.3g} | [{row["worst"]}]({row["crop"]}) {worst["delta"]:+.6g} ± {worst["seed_noise"]:.3g} '
                     f'| {row["statement"]} |')
    lines.extend(['', 'Worst region ranks by the lower bound of the excess. Crops show the seed with the highest region '
                  'excess, with base, candidate and shared reference from left to right in Reinhard + sRGB.',
                  f'Wall time {wall:.1f} s. Measured {len(rows)} scene/profile/mode combinations.'])
    if errors:
        lines.extend(['', '## Incomplete measurements', '', *[f'- {error}' for error in errors]])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(rows)} comparisons in {wall:.1f}s. {out/"report.md"}')
    excesses = [row for row in rows if row['statement'] != NO_EXCESS]
    for row in excesses:
        print(f'{row["case"]} / {row["profile"]} / {row["mode"]}: {row["statement"]}')
    if rows and not errors and not excesses:
        print(NO_EXCESS)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('base', type=Path, help='Linux build directory containing vork and its conformance worker')
    ap.add_argument('candidate', type=Path, help='Linux build directory with the same layout')
    ap.add_argument('--out', type=Path, required=True, help='new output directory, outside the repository')
    ap.add_argument('--cuts', type=Path, help='directory containing the four Landstep1b scene directories')
    ap.add_argument('--only', help='comma-separated names from the focused energy roster or the four cuts')
    ap.add_argument('--profiles', help='comma-separated profiles to override the case defaults')
    ap.add_argument('--seeds', default=SEEDS, type=lambda s: tuple(map(int, s.split(','))))
    ap.add_argument('--gpu-test', default=os.environ.get('GPU_TEST', 'gpu-test'))
    ap.add_argument('--jobs', type=int, default=4, help='maximum concurrent gpu-test reservations')
    ap.add_argument('--job-timeout', type=int, default=18000)
    ap.add_argument('--render-timeout', type=int, default=3600)
    ap.add_argument('--reference-cache', type=Path,
                    default=Path(os.environ.get('VORK_CACHE_DIR', Path.home()/'.cache/vork'))/'refcheck-v1')
    ap.add_argument('--collect-references', action='store_true',
                    help='fill missing fixed-reference batches with the candidate worker and exit')
    args = ap.parse_args()
    if not args.collect_references:
        try:
            import numpy
            import scipy.stats
        except ImportError as error:
            ap.error(f'install NumPy and SciPy on the submitting host ({error})')
    if args.jobs < 1 or args.job_timeout < 1 or args.render_timeout < 1:
        ap.error('job count and timeouts must be positive')
    if (len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 3
            or set(args.seeds).intersection(REFERENCE_SEEDS) or any(s < 0 or s > 2**32-1 for s in args.seeds)):
        ap.error('use at least three distinct unsigned render seeds disjoint from the reference seeds')
    args.base, args.candidate, args.out, args.reference_cache = (
        p.expanduser().resolve() for p in (args.base, args.candidate, args.out, args.reference_cache))
    if args.cuts:
        args.cuts = args.cuts.expanduser().resolve()
    selected = cases(args.cuts)
    if args.only:
        names = set(args.only.split(','))
        if names - {case['name'] for case in selected}:
            ap.error('unknown case: '+', '.join(sorted(names - {case['name'] for case in selected})))
        selected = [case for case in selected if case['name'] in names]
    if args.profiles:
        profiles = list(dict.fromkeys(args.profiles.split(',')))
        if set(profiles) - {'std/previs', 'std/draft', 'std/final', 'std/path'}:
            ap.error('expected comma-separated std/previs, std/draft, std/final or std/path')
        for case in selected:
            case['profiles'] = profiles
    refs, regions = {}, {}
    for case in selected:
        name = case['name']
        try:
            _, regions[name], _, case['identity'] = prepare(case, args.cuts or ROOT)
            refs[name] = load_reference(args.reference_cache, case)
        except FileNotFoundError as error:
            if 'identity' not in case:
                ap.error(str(error))
            if not args.collect_references:
                ap.error(f'{name}: missing reference {case["identity"]} in {args.reference_cache}')
    if args.collect_references:
        selected = [case for case in selected if case['name'] not in refs]
        if not selected:
            print('All selected references are already cached.')
            return 0
        args.reference_cache.mkdir(parents=True, exist_ok=True)
        # On/off fixtures can share exact inputs; render those references once.
        selected = list({case['identity']: case for case in selected}.values())
    started = time.monotonic()
    args.out.mkdir(parents=True)
    candidate = args.out/'candidate'
    stage_build(args.candidate, candidate)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    workers = min(args.jobs, len(selected))
    groups = [selected[i::workers] for i in range(workers)]
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(stage_job, args.out/f'job-{i}', group, args, revision, candidate): group
                for i, group in enumerate(groups)}
        for completed in as_completed(jobs):
            group = jobs[completed]
            job, status = completed.result()
            if status:
                errors.append(f'{job.with_suffix(".log")}: gpu-test exited {status}')
            for case in group:
                name = case['name']
                out = job/'results'/name
                try:
                    if (out/'error.json').exists():
                        raise RuntimeError(json.loads((out/'error.json').read_text())['error'])
                    frames = json.loads((out/'renders.json').read_text())
                    hardware = json.loads((job/'hardware.json').read_text())
                    if args.collect_references:
                        install_reference(args.reference_cache, case, out, frames, hardware, 'candidate')
                    else:
                        rows.extend(score(case, out, frames, refs[name], regions[name], args, hardware))
                except Exception as error:
                    errors.append(f'{name}: {error}')
    wall = time.monotonic()-started
    if args.collect_references:
        print(f'Collected {len(selected)} shared references in {wall:.1f}s.'
              if not errors else f'Reference collection incomplete after {wall:.1f}s.')
    else:
        order = {case['name']: i for i, case in enumerate(selected)}
        rows.sort(key=lambda row: order[row['case']])
        report(args.out, rows, errors, wall)
    for error in errors:
        print(error, file=sys.stderr)
    return int(bool(errors))


if __name__ == '__main__':
    sys.exit(render_job() if sys.argv[1:] == ['--gpu-job'] else main())

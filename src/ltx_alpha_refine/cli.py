# SPDX-License-Identifier: Apache-2.0
"""Standalone repair of matching RGB/Alpha frame sequences; no LTX inference here."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def frame_pairs(source, alpha, allow_alpha_tail=False):
    source, alpha = Path(source), Path(alpha)
    files = sorted(source.glob('*.png'))
    if not files or [p.name for p in files] != [f'{i:05d}.png' for i in range(len(files))]:
        raise ValueError('Source PNGs must be contiguous 00000.png ... with no missing frame.')
    matte = sorted(alpha.glob('*.png'))
    expected = [p.name for p in files]
    if [p.name for p in matte[:len(files)]] != expected or (not allow_alpha_tail and len(matte) != len(files)):
        raise ValueError('Alpha sequence must match source frame names/count; tail padding needs --allow-alpha-tail.')
    if allow_alpha_tail and [p.name for p in matte] != [f'{i:05d}.png' for i in range(len(matte))]:
        raise ValueError('Padded Alpha sequence must also be contiguous.')
    return [(p, alpha / p.name) for p in files]


def read_pair(source, alpha, raw_rgb_alpha=False):
    with Image.open(source) as im:
        rgb = np.array(im.convert('RGB'))
    with Image.open(alpha) as im:
        if raw_rgb_alpha:
            if im.mode != 'RGB':
                raise ValueError('Raw LTX decoder output must be RGB uint8.')
            matte = np.rint(np.array(im).astype(np.float32).mean(2)).astype(np.uint8)
        else:
            if im.mode != 'L':
                raise ValueError('Alpha must be 8-bit L PNG; use --raw-rgb-alpha for LTX RGB matte frames.')
            matte = np.array(im)
    if matte.shape != rgb.shape[:2]:
        raise ValueError('Source and Alpha dimensions differ; implicit resizing is not supported.')
    return rgb, matte


def validate_detections(rows, pairs):
    if [r.get('frame') for r in rows] != list(range(len(pairs))):
        raise ValueError('Detections must contain exactly one ordered row per source frame.')
    for row, (source, _) in zip(rows, pairs):
        with Image.open(source) as im:
            width, height = im.size
        for b in row['boxes']:
            box = b['box']
            if len(box) != 4 or any(type(v) is not int for v in box):
                raise ValueError('Boxes must have four integer coordinates.')
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError('Boxes must be nonempty and within the source image.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-frames', required=True)
    parser.add_argument('--alpha-frames', required=True)
    parser.add_argument('--vitmatte-dir', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--detections', help='JSON with frames [{frame, boxes: [{box, confidence}]}].')
    group.add_argument('--detector-dir', help='Local hand_detect_v1.0_s directory.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--raw-rgb-alpha', action='store_true')
    parser.add_argument('--allow-alpha-tail', action='store_true')
    parser.add_argument('--workers', type=int, choices=[1, 2, 4], default=4)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Output already exists; choose a fresh output directory.')
    start = time.perf_counter()
    pairs = frame_pairs(args.source_frames, args.alpha_frames, args.allow_alpha_tail)
    # Fail before inference or output writes if any input is malformed.
    for source, alpha in pairs:
        read_pair(source, alpha, args.raw_rgb_alpha)
    if args.detections:
        detections = json.loads(Path(args.detections).read_text())
    else:
        from .detector import Detector
        from .detection import filter_candidates
        detector = Detector(args.detector_dir)
        high = detector.threshold
        detector.threshold = .2
        candidates = []
        for i, (source, _) in enumerate(pairs):
            with Image.open(source) as im:
                boxes = detector(np.array(im.convert('RGB')))
            candidates.append({'frame': i, 'boxes': boxes})
        frames, recovered = filter_candidates(candidates, high)
        detections = {'frames': frames, 'recovered': recovered, 'candidates': candidates}
    validate_detections(detections['frames'], pairs)
    import torch
    from .repair import LearnedRepair
    from .parallel import FrameWorkers
    from .png import save
    if not torch.cuda.is_available():
        raise RuntimeError('This validated implementation requires CUDA/BF16; CPU repair is not validated.')
    model = LearnedRepair(args.vitmatte_dir)
    torch.set_num_threads(4)
    workers = FrameWorkers(model)
    output.mkdir(parents=True)
    (output / 'alpha').mkdir()
    (output / 'rgba-corrected').mkdir()
    (output / 'detections.json').write_text(json.dumps(detections, indent=2))
    report = {'state': 'running', 'frames': len(pairs), 'timing_scope': 'standalone refinement only, including validation, load and PNG I/O; excludes LTX', 'workers': args.workers}
    (output / 'run.json').write_text(json.dumps(report, indent=2))

    def work(i):
        rgb, matte = read_pair(*pairs[i], args.raw_rgb_alpha)
        aa, ff, changed = workers.repair(rgb, matte, detections['frames'][i]['boxes'])
        save(aa, output / 'alpha' / f'{i:05d}.png')
        save(np.dstack([ff, aa]), output / 'rgba-corrected' / f'{i:05d}.png')
        return changed

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            report['changed_alpha_pixels'] = sum(pool.map(work, range(len(pairs))))
        report.update(state='complete', seconds=time.perf_counter() - start, **workers.totals())
    except BaseException as exc:
        report.update(state='failed', error=str(exc), seconds=time.perf_counter() - start)
        raise
    finally:
        (output / 'run.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

# SPDX-License-Identifier: Apache-2.0
"""Run the optional CPU hand detector in an independent Python environment."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .detector import Detector
from .detection import filter_candidates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-frames', required=True, type=Path)
    parser.add_argument('--detector-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Choose a new detections output path.')
    files = sorted(args.source_frames.glob('*.png'))
    if not files or [p.name for p in files] != [f'{i:05d}.png' for i in range(len(files))]:
        raise ValueError('Source frames must be contiguous 00000.png ... .')
    model = Detector(args.detector_dir)
    high = model.threshold
    model.threshold = .2
    rows = []
    for i, path in enumerate(files):
        with Image.open(path) as im:
            boxes = model(np.array(im.convert('RGB')))
        rows.append({'frame': i, 'boxes': boxes})
    frames, recovered = filter_candidates(rows, high)
    result = {'frames': frames, 'recovered': recovered, 'candidates': rows}
    # Exclusive creation avoids replacing another process's output.
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()

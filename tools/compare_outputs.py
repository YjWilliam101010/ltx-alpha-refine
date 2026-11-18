# SPDX-License-Identifier: Apache-2.0
"""Exact decoded-pixel comparison, including sequence length and dimensions."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def compare(reference, candidate):
    result = {}
    for kind in ['alpha', 'rgba-corrected']:
        before = sorted((Path(reference)/kind).glob('*.png'))
        after = sorted((Path(candidate)/kind).glob('*.png'))
        if not before or [p.name for p in before] != [p.name for p in after]:
            raise ValueError(f'{kind}: missing or extra frames')
        changed = maximum = 0
        for a, b in zip(before, after):
            with Image.open(a) as im: x = np.array(im)
            with Image.open(b) as im: y = np.array(im)
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError(f'{kind}/{a.name}: shape or dtype mismatch')
            delta = np.abs(x.astype(np.int16)-y.astype(np.int16))
            changed += int(np.count_nonzero(delta))
            maximum = max(maximum, int(delta.max()))
        result[kind] = {'frames': len(before), 'changed_values': changed, 'maximum_absolute_error': maximum}
    result['exact'] = all(row['changed_values'] == 0 for row in result.values())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('candidate')
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['exact'] else 1)

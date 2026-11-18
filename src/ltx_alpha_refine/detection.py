# SPDX-License-Identifier: Apache-2.0
# Adapted from the tested reference integration; model weights are not included.
import numpy as np

def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = x * y
    return inter / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

def follows(a, b):
    a = np.array(a, float)
    b = np.array(b, float)
    sa = np.prod(a[2:] - a[:2])
    sb = np.prod(b[2:] - b[:2])
    return iou(a, b) > 0.1 and max(sa, sb) < 2.5 * max(1, min(sa, sb)) and (np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2) < 0.9 * max(np.linalg.norm(a[2:] - a[:2]), np.linalg.norm(b[2:] - b[:2])))

def filter_candidates(frames, high):

    def supported(i, b, direction):
        nodes = [b]
        for distance in range(1, 4):
            j = i + direction * distance
            if j < 0 or j >= len(frames):
                break
            nodes = [c for c in frames[j]['boxes'] if any((follows(x['box'], c['box']) for x in nodes))]
            if any((x['confidence'] >= high for x in nodes)):
                return True
            if not nodes:
                break
        return False
    out = []
    added = []
    for i, row in enumerate(frames):
        strong = [b for b in row['boxes'] if b['confidence'] >= high]
        keep = list(strong)
        for b in sorted(row['boxes'], key=lambda b: -b['confidence']):
            if b['confidence'] >= high:
                continue
            if any((iou(b['box'], v['box']) > 0.5 for v in keep)):
                continue
            valid = supported(i, b, -1) and supported(i, b, 1)
            if not valid and 0 < i < len(frames) - 1:
                for left in frames[i - 1]['boxes']:
                    if left['confidence'] < high:
                        continue
                    for right in frames[i + 1]['boxes']:
                        if right['confidence'] < high:
                            continue
                        aa = np.array(left['box'], float)
                        bb = np.array(right['box'], float)
                        mid = (aa + bb) / 2
                        span = max(np.linalg.norm(aa[2:] - aa[:2]), np.linalg.norm(bb[2:] - bb[:2]))
                        if np.linalg.norm((aa[:2] + aa[2:] - bb[:2] - bb[2:]) / 2) > 2.5 * span:
                            continue
                        if iou(mid, b['box']) > 0.35:
                            valid = True
            if valid:
                c = dict(b, recovered=True)
                keep.append(c)
                added.append({'frame': i, **c})
        out.append({'frame': row['frame'], 'boxes': keep})
    return (out, added)

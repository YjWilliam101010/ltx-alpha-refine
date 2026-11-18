# SPDX-License-Identifier: Apache-2.0
# Adapted from the tested reference integration; model weights are not included.
import cv2, numpy as np
cv2.setNumThreads(1)

def regions(rgb, alpha, boxes, pad=24):
    h, w = alpha.shape
    out = np.zeros((h, w), bool)
    for b in boxes:
        x0, y0, x1, y1 = b['box']
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)
        patch = rgb[y0:y1, x0:x1]
        aa = alpha[y0:y1, x0:x1]
        contrast = np.linalg.norm(255 - patch.astype(np.float32), axis=2)
        samples = contrast[(aa > 240) & (contrast > 12)]
        if len(samples) < 16 or np.median(samples) < 40:
            continue
        out[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)] = True
    return out

def islands(rgb, a, f, boxes):
    reg = regions(rgb, a, boxes, 64)
    a = a.copy()
    f = f.copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats((a > 26).astype(np.uint8), connectivity=8)
    white = cv2.inRange(rgb, (247, 247, 247), (255, 255, 255)) > 0
    removed = []
    for j in range(1, n):
        if stats[j, 4] > 512:
            continue
        core = lab == j
        if not np.all(white[core]) or not np.all(reg[core]):
            continue
        ext = cv2.dilate(core.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & white & reg & ((lab == j) | (a <= 26))
        a[ext] = 0
        f[ext] = rgb[ext]
        removed.append({'box': stats[j].tolist(), 'pixels': int(ext.sum())})
    return (a, f, removed)

def project(rgb, a, f, boxes, mode='palette'):
    h, w = a.shape
    out = a.astype(np.float32) / 255
    ff = f.copy()
    reg = regions(rgb, a, boxes, 24)
    details = []
    for b in boxes:
        bx0, by0, bx1, by1 = b['box']
        x0 = max(0, bx0 - 24)
        y0 = max(0, by0 - 24)
        x1 = min(w, bx1 + 24)
        y1 = min(h, by1 + 24)
        use = reg[y0:y1, x0:x1]
        I = rgb[y0:y1, x0:x1].astype(np.float32) / 255
        prior = out[y0:y1, x0:x1].copy()
        if not np.any(use):
            continue
        if mode == 'bound':
            goal = np.maximum(prior, 1 - I.min(2) - 8 / 255)
            valid = use & (I.min(2) < 240 / 255) & (goal > prior)
        else:
            anchor = cv2.erode((prior > 0.98).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & (I.min(2) < 235 / 255)
            if anchor.sum() < 16:
                continue
            dist, lab = cv2.distanceTransformWithLabels((~anchor).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            palette = np.vstack([np.zeros((1, 3), np.float32), I[anchor]])
            F = palette[lab]
            u = 1 - I
            v = 1 - F
            vv = (v * v).sum(2)
            uu = (u * u).sum(2)
            dot = (u * v).sum(2)
            estimate = np.clip(dot / np.maximum(vv, 1e-06), 0, 1)
            cos = dot / np.sqrt(np.maximum(vv * uu, 1e-09))
            confidence = np.clip(1 - dist / 32, 0, 1) * np.clip((cos - 0.98) / 0.02, 0, 1)
            goal = prior + confidence * np.maximum(0, estimate - prior)
            valid = use & (prior < 0.98) & (I.min(2) < 242 / 255) & (vv > 0.025) & (goal > prior)
        out[y0:y1, x0:x1][valid] = goal[valid]
        rgbnew = np.clip((I - (1 - goal[:, :, None])) / np.maximum(goal[:, :, None], 0.08), 0, 1)
        ff[y0:y1, x0:x1][valid] = np.uint8(rgbnew[valid] * 255 + 0.5)
        details.append({'box': b['box'], 'changed': int(valid.sum())})
    return (np.uint8(np.clip(out * 255 + 0.5, 0, 255)), ff, details)

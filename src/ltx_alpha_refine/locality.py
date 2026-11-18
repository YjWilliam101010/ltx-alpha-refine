# SPDX-License-Identifier: Apache-2.0
# Adapted from the tested reference integration; model weights are not included.
import numpy as np, cv2

def restrict_update(rgb, alpha, new_alpha, new_rgb, boxes):
    if not boxes:
        return (alpha.copy(), rgb.copy(), 0)
    h, w = alpha.shape
    blend = np.zeros((h, w), np.float32)
    for b in boxes:
        mask = np.zeros((h, w), np.uint8)
        x0, y0, x1, y1 = b['box']
        mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 1
        dist = cv2.distanceTransform(1 - mask, cv2.DIST_C, 3)
        local = np.clip(1 - dist / 8, 0, 1) if b.get('legacy') else np.clip((32 - dist) / 8, 0, 1)
        blend = np.maximum(blend, local)
    active = blend > 0
    aa = alpha.astype(np.float32) / 255
    bb = new_alpha.astype(np.float32) / 255
    out_a = aa * (1 - blend) + bb * blend
    numerator = rgb.astype(np.float32) * (aa * (1 - blend))[:, :, None] + new_rgb.astype(np.float32) * (bb * blend)[:, :, None]
    ff = rgb.copy()
    ff[active] = np.uint8(np.clip(numerator[active] / np.maximum(out_a[active, None], 1e-06) + 0.5, 0, 255))
    out = np.uint8(np.clip(out_a * 255 + 0.5, 0, 255))
    return (out, ff, int(np.count_nonzero(out != alpha)))

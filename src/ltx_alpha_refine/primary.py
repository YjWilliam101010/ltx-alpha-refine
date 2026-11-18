# SPDX-License-Identifier: Apache-2.0
# Adapted from the tested reference integration; model weights are not included.
import torch, cv2, numpy as np, time
from pathlib import Path
from transformers import VitMatteForImageMatting
cv2.setNumThreads(1)
torch.set_num_threads(2)

class LearnedRepair:

    def __init__(self, model_path):
        t = time.perf_counter()
        self.model = VitMatteForImageMatting.from_pretrained(str(Path(model_path)), local_files_only=True).eval().to(device='cuda', dtype=torch.bfloat16)
        self.load_seconds = time.perf_counter() - t
        self.params = sum((p.numel() for p in self.model.parameters()))

    @torch.inference_mode()
    def repair(self, rgb, alpha, boxes, method='contrast_guarded'):
        if not boxes:
            return (alpha.copy(), rgb.copy(), 0)
        h, w = alpha.shape
        a = alpha.astype(np.float32) / 255
        f = rgb.astype(np.float32) / 255
        fg = (alpha > 127).astype(np.uint8)
        inner = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
        outer = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5)

        def background(th):
            _, lab = cv2.connectedComponents(cv2.inRange(rgb, (th, th, th), (255, 255, 255)), connectivity=8)
            flags = np.zeros(int(lab.max()) + 1, bool)
            flags[np.unique(lab[alpha < 26])] = True
            flags[0] = False
            return flags[lab]
        strict = background(243)
        si = cv2.distanceTransform((~strict).astype(np.uint8), cv2.DIST_L2, 5)
        so = cv2.distanceTransform(strict.astype(np.uint8), cv2.DIST_L2, 5)
        base = np.full(alpha.shape, 0.5, np.float32)
        base[inner > 8] = 1
        base[outer > 8] = 0
        base[(si > 2) & (alpha > 127) & (cv2.inRange(rgb, (0, 0, 0), (179, 179, 179)) > 0)] = 1
        base[(so > 2) & (alpha < 220)] = 0
        hand = np.zeros(alpha.shape, bool)
        for b in boxes:
            x0, y0, x1, y1 = b['box']
            hand[max(0, y0 - 24):min(h, y1 + 24), max(0, x0 - 24):min(w, x1 + 24)] = True
        reachable = background(239)
        bg = reachable & (cv2.inRange(rgb, (243, 243, 243), (255, 255, 255)) > 0)
        bi = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 5)
        bo = cv2.distanceTransform(bg.astype(np.uint8), cv2.DIST_L2, 5)
        tri = base.copy()
        tri[(bi <= 4) & hand] = 0.5
        tri[(bi > 2) & hand & (outer > 8)] = 0.5
        tri[(bo > 2) & hand] = 0
        self.diagnostics = {'promoted_strict_background': int((bg & ~strict & hand).sum()), 'boxes': []}
        for b in boxes:
            bx0, by0, bx1, by1 = b['box']
            bx0 = max(0, bx0)
            bx1 = min(w, bx1)
            by0 = max(0, by0)
            by1 = min(h, by1)
            contrast = np.linalg.norm(255 - rgb[by0:by1, bx0:bx1].astype(np.float32), axis=2)
            reliable = (inner[by0:by1, bx0:bx1] > 8) & (contrast > 12)
            values = contrast[reliable]
            median = float(np.median(values)) if values.size else 0.0
            fallback = method == 'contrast_guarded' and (values.size < 16 or median < 40)
            selected = base if fallback or method == 'conservative' else tri
            self.diagnostics['boxes'].append({'median_background_contrast': median, 'anchor_pixels': int(values.size), 'fallback_to_conservative': bool(fallback)})
            x0 = max(0, bx0 - 32)
            x1 = min(w, bx1 + 32)
            y0 = max(0, by0 - 32)
            y1 = min(h, by1 + 32)
            I = rgb[y0:y1, x0:x1].astype(np.float32) / 255
            T = selected[y0:y1, x0:x1]
            if not np.any(T == 0.5):
                continue
            inp = np.concatenate([I * 2 - 1, T[:, :, None]], axis=2).transpose(2, 0, 1).copy()
            p = torch.from_numpy(inp)[None].to(device='cuda', dtype=torch.bfloat16)
            hh, ww = I.shape[:2]
            p = torch.nn.functional.pad(p, (0, -ww % 32, 0, -hh % 32))
            pred = self.model(pixel_values=p).alphas[0, 0, :hh, :ww].float().cpu().numpy()
            pred[T == 1] = 1
            pred[T == 0] = 0
            yy, xx = np.indices((hh, ww))
            edge = np.minimum.reduce([xx + 1, ww - xx, yy + 1, hh - yy])
            blend = np.clip(edge / 8, 0, 1)
            aa = a[y0:y1, x0:x1] * (1 - blend) + pred * blend
            ff = np.clip((I - (1 - aa[:, :, None])) / np.maximum(aa[:, :, None], 0.08), 0, 1)
            use = (blend > 0.999) & (aa < 0.999)
            f[y0:y1, x0:x1][use] = ff[use]
            a[y0:y1, x0:x1] = aa
        au = np.uint8(np.clip(a * 255 + 0.5, 0, 255))
        fu = np.uint8(np.clip(f * 255 + 0.5, 0, 255))
        from .locality import restrict_update
        scopes = [dict(b, legacy=d['fallback_to_conservative']) for b, d in zip(boxes, self.diagnostics['boxes'])]
        return restrict_update(rgb, alpha, au, fu, scopes)

# SPDX-License-Identifier: Apache-2.0
# Adapted from the tested reference integration; model weights are not included.
from .constrained import LearnedRepair as Base
import torch, cv2, numpy as np, time

class LearnedRepair(Base):

    def __init__(self, model_path):
        super().__init__(model_path)
        self.secondary_forwards = 0
        self.secondary_seconds = 0.0
        self.secondary_commits = 0

    @torch.inference_mode()
    def repair(self, rgb, alpha, boxes, method=None):
        aa, ff, _ = super().repair(rgb, alpha, boxes)
        if not boxes:
            return (aa, ff, 0)
        h, w = aa.shape
        roi = np.zeros((h, w), np.uint8)
        for b in boxes:
            x0, y0, x1, y1 = b['box']
            roi[max(0, y0 - 16):min(h, y1 + 16), max(0, x0 - 16):min(w, x1 + 16)] = 1
        white = cv2.inRange(rgb, (243, 243, 243), (255, 255, 255)) > 0
        candidates = (white & (aa < 26) & (roi > 0)).astype(np.uint8)
        n, labels, stats, centers = cv2.connectedComponentsWithStats(candidates, 8)
        rim = (roi > 0) & (cv2.erode(roi, np.ones((3, 3), np.uint8)) == 0)
        rim[0] = True
        rim[-1] = True
        rim[:, 0] = True
        rim[:, -1] = True
        excluded = set(np.unique(labels[rim]).tolist())
        excluded.add(0)
        accepted = []
        primary = aa.copy()
        for label in range(1, n):
            if label in excluded:
                continue
            x, y, cw, ch, area = map(int, stats[label])
            if area > 24000 or max(cw, ch) > 288:
                continue
            core = labels[y:y + ch, x:x + cw] == label
            values = rgb[y:y + ch, x:x + cw][core]
            if np.median(np.minimum(np.minimum(values[:, 0], values[:, 1]), values[:, 2])) >= 252:
                continue
            bx0 = max(0, x - 3)
            by0 = max(0, y - 3)
            bx1 = min(w, x + cw + 3)
            by1 = min(h, y + ch + 3)
            local = (labels[by0:by1, bx0:bx1] == label).astype(np.uint8)
            ring = (cv2.dilate(local, np.ones((5, 5), np.uint8)) > 0) & (local == 0)
            strong = primary[by0:by1, bx0:bx1][ring] > 230
            if strong.sum() < 2 or strong.mean() < 0.2:
                continue
            support = rgb[by0:by1, bx0:bx1][ring][strong]
            if np.mean(np.minimum(np.minimum(support[:, 0], support[:, 1]), support[:, 2]) >= 230) < 0.75:
                continue
            pw = max(96, cw + 64)
            ph = max(96, ch + 64)
            px0 = max(0, min(w - pw, x + (cw - pw) // 2))
            py0 = max(0, min(h - ph, y + (ch - ph) // 2))
            px1 = min(w, px0 + pw)
            py1 = min(h, py0 + ph)
            I = rgb[py0:py1, px0:px1].astype(np.float32) / 255
            pA = primary[py0:py1, px0:px1]
            core = labels[py0:py1, px0:px1] == label
            mask = (cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & white[py0:py1, px0:px1] & (roi[py0:py1, px0:px1] > 0)
            T = np.full(pA.shape, 0.5, np.float32)
            T[pA >= 240] = 1
            T[pA <= 15] = 0
            T[mask] = 0.5
            inp = np.concatenate([I * 2 - 1, T[:, :, None]], 2).transpose(2, 0, 1).copy()
            p = torch.from_numpy(inp)[None].to('cuda', dtype=torch.bfloat16)
            hh, ww = I.shape[:2]
            p = torch.nn.functional.pad(p, (0, -ww % 32, 0, -hh % 32))
            tick = time.perf_counter()
            pred = self.model(pixel_values=p).alphas[0, 0, :hh, :ww].float().cpu().numpy()
            self.secondary_seconds += time.perf_counter() - tick
            self.secondary_forwards += 1
            vals = pred[core]
            if np.median(vals) < 0.95 or np.quantile(vals, 0.1) < 0.9:
                continue
            prior = aa[py0:py1, px0:px1]
            proposal = np.uint8(np.clip(pred * 255 + 0.5, 0, 255))
            use = mask & (proposal > prior)
            aa[py0:py1, px0:px1][use] = proposal[use]
            a = aa[py0:py1, px0:px1].astype(np.float32) / 255
            color = np.uint8(np.clip((I - (1 - a[:, :, None])) / np.maximum(a[:, :, None], 0.08), 0, 1) * 255 + 0.5)
            ff[py0:py1, px0:px1][use] = color[use]
            self.secondary_commits += 1
            accepted.append({'box': [x, y, cw, ch], 'pixels': int(use.sum()), 'median_alpha': float(np.median(vals))})
        self.proposal_diagnostics = accepted
        return (aa, ff, int(np.count_nonzero(aa != alpha)))

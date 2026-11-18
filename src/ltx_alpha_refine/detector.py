# SPDX-License-Identifier: Apache-2.0
# Adapted from the tested reference integration; model weights are not included.
from pathlib import Path
import time, json
import cv2, numpy as np, onnxruntime as ort
from PIL import Image
cv2.setNumThreads(1)

class Detector:

    def __init__(self, model_directory, size=512):
        level = 's'
        t = time.perf_counter()
        self.level = level
        self.size = size
        path = Path(model_directory)
        self.threshold = float(json.loads((path / 'threshold.json').read_text())['threshold'])
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 2
        opt.inter_op_num_threads = 1
        opt.log_severity_level = 3
        self.session = ort.InferenceSession(str(path / 'model.onnx'), sess_options=opt, providers=['CPUExecutionProvider'])
        self.load_seconds = time.perf_counter() - t

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        nw = self.size
        nh = int(np.ceil(h / w * nw / 32) * 32)
        im = np.asarray(Image.fromarray(rgb).resize((nw, nh)), np.float32) / 255
        raw = self.session.run(None, {'images': np.ascontiguousarray(im.transpose(2, 0, 1)[None])})[0][0]
        if raw.shape[0] == 5:
            raw = raw.T
        if raw.shape[1] == 5:
            good = raw[raw[:, 4] > self.threshold]
            scores = good[:, 4]
            boxes = good[:, :4].copy()
            if not len(boxes):
                return []
            boxes[:, :2] -= boxes[:, 2:] / 2
            pick = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.threshold, 0.7)
            result = []
            for idx in np.asarray(pick).reshape(-1):
                x, y, bw, bh = boxes[idx]
                bb = [max(0, int(round(x / nw * w))), max(0, int(round(y / nh * h))), min(w, int(round((x + bw) / nw * w))), min(h, int(round((y + bh) / nh * h)))]
                result.append({'box': bb, 'confidence': float(scores[idx])})
            return result
        raise RuntimeError(('unexpected output', raw.shape))

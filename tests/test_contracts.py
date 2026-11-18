# SPDX-License-Identifier: Apache-2.0
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ltx_alpha_refine.cli import frame_pairs, read_pair, validate_detections
from ltx_alpha_refine.detection import filter_candidates
from ltx_alpha_refine.locality import restrict_update
from ltx_alpha_refine.png import encode
from ltx_alpha_refine.projection import project


class Contracts(unittest.TestCase):
    def test_weak_detection_uses_current_geometry(self):
        boxes = [[10, 10, 30, 30], [13, 10, 33, 30], [16, 10, 36, 30]]
        rows = [{'frame': i, 'boxes': [{'box': b, 'confidence': .25 if i == 1 else .8}]} for i, b in enumerate(boxes)]
        final, recovered = filter_candidates(rows, .395)
        self.assertEqual(final[1]['boxes'][0]['box'], boxes[1])
        self.assertEqual(len(recovered), 1)
        self.assertEqual(rows[1]['boxes'][0], {'box': boxes[1], 'confidence': .25})

    def test_no_interpolated_detection_is_created(self):
        rows = [{'frame': i, 'boxes': [] if i == 1 else [{'box': [10, 10, 30, 30], 'confidence': .8}]} for i in range(3)]
        final, recovered = filter_candidates(rows, .395)
        self.assertEqual(final[1]['boxes'], [])
        self.assertEqual(recovered, [])

    def test_isolated_weak_detection_is_rejected(self):
        final, _ = filter_candidates([{'frame': 0, 'boxes': [{'box': [10, 10, 30, 30], 'confidence': .25}]}], .395)
        self.assertEqual(final[0]['boxes'], [])

    def test_locality_preserves_every_pixel_outside_support(self):
        rng = np.random.default_rng(4)
        rgb = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
        a = rng.integers(0, 256, (160, 160), dtype=np.uint8)
        aa, ff, _ = restrict_update(rgb, a, 255-a, 255-rgb, [{'box': [64, 64, 88, 88]}])
        outside = np.ones(a.shape, bool)
        outside[32:120, 32:120] = False
        self.assertTrue(np.array_equal(aa[outside], a[outside]))
        self.assertTrue(np.array_equal(ff[outside], rgb[outside]))

    def test_no_hand_is_an_identity(self):
        rgb = np.full((16, 16, 3), 123, np.uint8)
        a = np.arange(256, dtype=np.uint8).reshape(16, 16)
        aa, ff, count = restrict_update(rgb, a, 255-a, 255-rgb, [])
        self.assertTrue(np.array_equal(a, aa))
        self.assertTrue(np.array_equal(rgb, ff))
        self.assertEqual(count, 0)

    def test_white_composite_bound_does_not_reduce_alpha(self):
        rgb = np.full((64, 64, 3), 90, np.uint8)
        a = np.full((64, 64), 255, np.uint8)
        a[28:34, 28:34] = 20
        out, _, _ = project(rgb, a, rgb, [{'box': [16, 16, 48, 48]}], 'bound')
        self.assertTrue(np.all(out >= a))
        self.assertGreater(int(out[30, 30]), int(a[30, 30]))

    def test_png_is_lossless_for_all_supported_channels(self):
        rng = np.random.default_rng(3)
        for shape in [(17, 21), (17, 21, 3), (17, 21, 4)]:
            values = rng.integers(0, 256, shape, dtype=np.uint8)
            self.assertTrue(np.array_equal(values, np.array(Image.open(io.BytesIO(encode(values))))))

    def test_frame_alignment_and_explicit_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            s, a = Path(temp)/'source', Path(temp)/'alpha'
            s.mkdir(); a.mkdir()
            for i in range(2):
                Image.new('RGB', (16, 16)).save(s/f'{i:05d}.png')
                Image.new('L', (16, 16)).save(a/f'{i:05d}.png')
            self.assertEqual(len(frame_pairs(s, a)), 2)
            Image.new('L', (16, 16)).save(a/'00002.png')
            with self.assertRaises(ValueError): frame_pairs(s, a)
            self.assertEqual(len(frame_pairs(s, a, True)), 2)
            (s/'00000.png').rename(s/'00003.png')
            with self.assertRaises(ValueError): frame_pairs(s, a, True)

    def test_raw_rgb_alpha_conversion_is_not_pillow_luma(self):
        with tempfile.TemporaryDirectory() as temp:
            s, a = Path(temp)/'s.png', Path(temp)/'a.png'
            Image.new('RGB', (4, 4), (255, 0, 0)).save(s)
            Image.new('RGB', (4, 4), (255, 0, 0)).save(a)
            with self.assertRaises(ValueError): read_pair(s, a)
            _, matte = read_pair(s, a, True)
            self.assertTrue(np.all(matte == 85))

    def test_detection_misalignment_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'00000.png'
            Image.new('RGB', (16, 16)).save(p)
            with self.assertRaises(ValueError):
                validate_detections([{'frame': 1, 'boxes': []}], [(p, p)])
            with self.assertRaises(ValueError):
                validate_detections([{'frame': 0, 'boxes': [{'box': [0, 0, 17, 17]}]}], [(p, p)])


if __name__ == '__main__':
    unittest.main()

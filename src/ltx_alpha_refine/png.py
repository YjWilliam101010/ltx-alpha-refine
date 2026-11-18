# SPDX-License-Identifier: Apache-2.0
# Adapted from the tested reference integration; model weights are not included.
from pathlib import Path
import struct, zlib, time, io, json, os, numpy as np
from PIL import Image

def chunk(kind, payload):
    return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload) & 4294967295)

def encode(array):
    a = np.ascontiguousarray(array, dtype=np.uint8)
    h, w = a.shape[:2]
    ch = 1 if a.ndim == 2 else a.shape[2]
    typ = {1: 0, 3: 2, 4: 6}[ch]
    rows = np.empty((h, w * ch + 1), np.uint8)
    rows[:, 0] = 0
    rows[:, 1:] = a.reshape(h, -1)
    data = zlib.compress(rows.tobytes(), 1)
    return bytes([137, 80, 78, 71, 13, 10, 26, 10]) + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, typ, 0, 0, 0)) + chunk(b'IDAT', data) + chunk(b'IEND', b'')

def save(array, path):
    Path(path).write_bytes(encode(array))

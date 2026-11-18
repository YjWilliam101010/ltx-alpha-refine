# SPDX-License-Identifier: Apache-2.0
"""Verify independently downloaded refinement weights against the tested bytes."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--vitmatte-dir', required=True, type=Path)
parser.add_argument('--detector-dir', type=Path)
args = parser.parse_args()
manifest = json.loads((Path(__file__).resolve().parents[1]/'model-assets.json').read_text())
for row in manifest:
    if row['repo'].startswith('hustvl/'):
        path = args.vitmatte_dir/row['file']
    elif args.detector_dir:
        path = args.detector_dir/Path(row['file']).name
    else:
        continue
    if path.stat().st_size != row['bytes']:
        raise ValueError(f'Size mismatch: {path.name}')
    with path.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    if sha != row['sha256']:
        raise ValueError(f'SHA256 mismatch: {path.name}')
    print('verified', path.name)

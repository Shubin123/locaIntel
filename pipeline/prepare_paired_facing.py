"""Prepare labeled two-view CFV pairs, without inventing temporal timestamps.

Requires numpy/Pillow. The two photographs share a CFV identity and nearby
labeled angles; they are NOT verified same-camera traffic sequences.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

REVISION = 'e17aaeb4860b33757c582a57e5a05e70ec40b10e'
SHA256 = {'train.zip': '3831521f520895e8ce016286f5dca2d0a84a5c5c42bc28f3a21ba3f601383812',
          'test.zip': 'c5b26548dbcf9417c8d852b50813d50bedc7c03350daaf66972f1d80ed3fbf62'}
VALID_IDS = [2, 11, 12, 29, 33, 34, 36, 45, 54, 55]


def circular(a, b):
    return abs((a - b + 180) % 360 - 180)


def build(data, output):
    output.mkdir(parents=True, exist_ok=True)
    crops, pairs, identities = [], [], {}
    with (output / 'crops.u8').open('wb') as packed:
        for name in ('train.zip', 'test.zip'):
            archive_path = data / name
            with archive_path.open('rb') as source:
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
            if digest != SHA256[name]:
                raise ValueError('Pinned dataset checksum mismatch: ' + name)
            with zipfile.ZipFile(archive_path) as archive:
                groups = {}
                for line in archive.read('metadata.jsonl').splitlines():
                    row = json.loads(line)
                    if row['x1'] < 0 or row['y1'] < 0 or row['x2'] <= row['x1'] or row['y2'] <= row['y1']:
                        continue
                    groups.setdefault(row['identity'], []).append(row)
                identities[name] = sorted(groups)
                cached = {}
                def crop(row, resolution):
                    key = (row['file_name'], resolution)
                    if key not in cached:
                        image = Image.open(io.BytesIO(archive.read(row['file_name']))).convert('RGB')
                        image = image.crop(tuple(row[k] for k in ('x1', 'y1', 'x2', 'y2')))
                        if resolution == '40px':
                            image = image.resize((40, 40), Image.Resampling.BILINEAR)
                        image = image.resize((224, 224), Image.Resampling.BILINEAR)
                        index = len(crops)
                        packed.write(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy().tobytes())
                        cached[key] = index
                        crops.append({'index': index, 'identity': row['identity'], 'angle': row['angle'],
                                      'label': int(math.ceil((row['angle'] - 22.5) / 45)) % 8,
                                      'file_name': row['file_name'], 'resolution': resolution})
                    return cached[key]
                for identity, rows in sorted(groups.items()):
                    split = 'test' if name == 'test.zip' else 'validation' if identity in VALID_IDS else 'train'
                    seen = set()
                    for center in range(0, 360, 45):
                        for offset in (-10, 10):
                            current = min(rows, key=lambda r: circular(r['angle'], (center + offset) % 360))
                            earlier = [r for r in rows if r['file_name'] != current['file_name'] and 3 <= circular(r['angle'], current['angle']) <= 8]
                            if not earlier:
                                continue
                            previous = min(earlier, key=lambda r: circular(r['angle'], (current['angle'] - 5) % 360))
                            pair_key = (previous['file_name'], current['file_name'])
                            if pair_key in seen:
                                continue
                            seen.add(pair_key)
                            for resolution in ('clean', '40px'):
                                pairs.append({'split': split, 'identity': identity, 'resolution': resolution,
                                              'previous': crop(previous, resolution), 'current': crop(current, resolution)})
            print(name, 'prepared', len(crops), 'crops;', len(pairs), 'pairs', flush=True)
    assert not set(identities['train.zip']) & set(identities['test.zip'])
    manifest = {'dataset': 'fort-cyber/CFV-Dataset', 'revision': REVISION, 'archive_sha256': SHA256,
                'pair_provenance': 'Distinct CFV stills, same identity, labeled angular separation 3–8 degrees. No verified camera ID, chronology or timestamp; not real temporal training data.',
                'validation_ids': VALID_IDS, 'identities': identities, 'input': 'uint8 RGB CHW 224x224; clean or 40px bilinear degradation',
                'crops': crops, 'pairs': pairs}
    (output / 'pairs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'crops': len(crops), 'pairs': len(pairs), 'by_split': {s: sum(p['split'] == s for p in pairs) for s in ('train', 'validation', 'test')}}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('.cache/cfv-data'))
    p.add_argument('--output', type=Path, default=Path('.cache/paired-facing'))
    args = p.parse_args()
    build(args.data, args.output)

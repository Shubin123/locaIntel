"""Prepare a reproducible local-only batch of distinct native labeled CFV JPEGs.

These are still photographs, not verified camera sequences. This script never
downloads data or writes into the published website. Production input must match
the pinned official-test archive; an explicit checksum supports small test fixtures.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import zipfile

from PIL import Image

REVISION = 'e17aaeb4860b33757c582a57e5a05e70ec40b10e'
ARCHIVE_SHA256 = 'c5b26548dbcf9417c8d852b50813d50bedc7c03350daaf66972f1d80ed3fbf62'
TEST_IDENTITIES = frozenset(range(14, 28))
ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def label(angle):
    return int(math.ceil((angle - 22.5) / 45)) % 8


def candidate(source, row, seen):
    content = source.read(row['file_name'])
    digest = hashlib.sha256(content).hexdigest()
    if digest in seen:
        return None
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != 'JPEG':
                return None
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            width, height = image.size
        box = [row.get(k) for k in ('x1', 'y1', 'x2', 'y2')]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            return None
        x, y, xx, yy = box
        if not (0 <= x < xx <= width and 0 <= y < yy <= height and xx-x >= 20 and yy-y >= 16):
            return None
    except (OSError, ValueError, SyntaxError):
        return None
    return dict(hash=digest, identity=row['identity'], angle=row['angle'], label=label(row['angle']),
                width=width, height=height, box=[x/width, y/height, xx/width, yy/height],
                bytes=len(content), original_file=row['file_name']), content


def build(archive=Path('.cache/cfv-data/test.zip'), output=Path('.cache/facing-verification'), count=1000, *, expected_sha=ARCHIVE_SHA256):
    archive, output = Path(archive), Path(output).absolute()
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Verification count must be a positive integer')
    if expected_sha is None or digest_file(archive) != expected_sha:
        raise ValueError('Official CFV test archive checksum differs from the pinned input')
    public = (ROOT / 'web').resolve()
    if output.is_symlink() or output.resolve() == public or public in output.resolve().parents or output.resolve() in ROOT.resolve().parents or output.resolve() == ROOT.resolve() or output.resolve() in archive.resolve().parents:
        raise ValueError('Verification photos must remain in a separate local directory, outside web')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.facing-verification-', dir=output.parent))
    photos, seen = [], set()
    try:
        (temporary / 'images').mkdir()
        with zipfile.ZipFile(archive) as source:
            buckets = defaultdict(list)
            for line in source.read('metadata.jsonl').splitlines():
                row = json.loads(line)
                if row.get('identity') not in TEST_IDENTITIES:
                    raise ValueError('Verification archive contains a non-test vehicle identity')
                angle = row.get('angle')
                if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle) or not 0 <= angle <= 360:
                    raise ValueError('Invalid labeled orientation angle')
                name = row.get('file_name')
                if not isinstance(name, str):
                    raise ValueError('Invalid original photo filename')
                buckets[(row['identity'], int((angle % 360) // 5))].append(row)
            # Cycle identities and, independently, each identity's angle strata.
            # Invalid crops must not underrepresent an otherwise available car.
            # Filename hashing varies angles within each five-degree interval.
            for key, rows in buckets.items():
                buckets[key] = deque(sorted(rows, key=lambda r: (hashlib.sha256(('locaintel-1000-v1:' + r['file_name']).encode()).hexdigest(), r['file_name'])))
            identities = sorted({identity for identity, _ in buckets})
            strata = {identity: deque((identity, angle_bin) for angle_bin in range(72) if (identity, angle_bin) in buckets) for identity in identities}
            while len(photos) < count and any(strata.values()):
                for identity in identities:
                    if len(photos) == count:
                        break
                    order, selected = strata[identity], None
                    while order and selected is None:
                        rows = buckets[order[0]]
                        while rows and selected is None:
                            selected = candidate(source, rows.popleft(), seen)
                        if rows:
                            order.rotate(-1)
                        else:
                            order.popleft()
                    if selected:
                        photo, content = selected
                        index, filename = len(photos), 'images/%06d.jpg' % len(photos)
                        (temporary / filename).write_bytes(content)
                        photos.append(dict(index=index, file=filename, **photo))
                        seen.add(photo['hash'])
        if len(photos) != count:
            raise ValueError('Expected exactly %d distinct valid native JPEG photos; found %d' % (count, len(photos)))
        manifest = dict(version=1, dataset='fort-cyber/CFV-Dataset', revision=REVISION,
                        archive_sha256=expected_sha, count=len(photos), photos_bytes=sum(p['bytes'] for p in photos),
                        sampling='Deterministic round robin over official-test identity × five-degree orientation bins; filename-hash ordering within bins.',
                        provenance='Native labeled CFV still photographs. No verified camera IDs, capture timestamps or chronological sequences. Not a measured traffic-camera accuracy set.',
                        photos=photos)
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        if output.exists():
            if not output.is_dir():
                raise ValueError('Verification output must be a directory')
            shutil.rmtree(output)
        temporary.replace(output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('.cache/cfv-data/test.zip'))
    parser.add_argument('--output', type=Path, default=Path('.cache/facing-verification'))
    parser.add_argument('--count', type=int, default=1000)
    args = parser.parse_args()
    result = build(args.archive, args.output, args.count)
    print(json.dumps({key: result[key] for key in ('count', 'photos_bytes', 'archive_sha256')}))

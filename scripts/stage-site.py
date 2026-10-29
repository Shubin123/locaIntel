"""Stage public application files and a metadata-only camera catalog for Pages.

This intentionally does not recursively copy web/. Camera snapshots, browser
profiles, traces, caches and arbitrary new files are never publication inputs.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit

APP_FILES = (
    'index.html', 'style.css', 'app.js', 'sources.js', 'motion.js', 'detector.js',
    'monitor.js', 'smooth.js', 'temporal.js', 'orientation.js', 'road.js',
    'heading.js', 'paired-facing.js', 'capture-pair.js', 'facing-cache.js', 'sw.js',
    'routes.js', 'routes-ui.js', 'matcher.js', 'gps.js', 'map.js', 'track-ui.js',
    'orientation-worker.js', 'detect-worker.js', 'road-worker.js', 'inference-runtime.js',
)
MODEL_FILES = (
    'README.md', 'CFV-LICENSE', 'TIMM-LICENSE', 'SegFormer-LICENSE',
    'paired-facing-v1.json', 'paired-facing-training-v1.json',
    'vehicle-facing-v3.json', 'facing-training-v3.json', 'drivebc-facing-v3.onnx',
    'webgl-v1.json', 'road-v1.json', 'yolo11n.onnx', 'yolo11n-webgl.onnx',
    'segformer-b0-road-256.onnx', 'segformer-b0-road-256-webgl.onnx',
)
VENDOR_FILES = (
    'leaflet.js', 'leaflet.css', 'leaflet-LICENSE', 'leaflet-heat.js', 'leaflet-heat-LICENSE',
    'ort/LICENSE', 'ort/ort.webgpu.min.js', 'ort/ort.webgl.min.js', 'ort/ort.wasm.min.js',
    'ort/ort-wasm-simd-threaded.wasm', 'ort/ort-wasm-simd-threaded.mjs',
    'ort/ort-wasm-simd-threaded.jsep.wasm', 'ort/ort-wasm-simd-threaded.jsep.mjs',
)
VENDOR_IMAGES = ('images/layers.png', 'images/layers-2x.png', 'images/marker-icon.png',
                 'images/marker-icon-2x.png', 'images/marker-shadow.png')
CAMERA_FIELDS = frozenset(('id', 'name', 'region', 'highway', 'caption', 'image', 'source', 'lat', 'lon', 'orientation'))
LATEST_FIELDS = frozenset(('storage_mode', 'generated', 'cameras', 'catalog_size', 'errors', 'model', 'schema_version'))


def regular_file(source, relative, required=True):
    """Check every component before opening: directory symlinks are unsafe too."""
    path = source
    for component in Path(relative).parts:
        path = path / component
        if path.is_symlink():
            raise ValueError('Publication input is a symlink: ' + relative)
    if not path.exists() and not required:
        return None
    if not path.is_file():
        raise ValueError('Missing regular publication file: ' + relative)
    return path


def reject_inline(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {'hash', 'frame', 'frames', 'detections', 'blob', 'photos', 'photo'}:
                raise ValueError('Camera evidence cannot be published: ' + str(key))
            reject_inline(item)
    elif isinstance(value, list):
        for item in value:
            reject_inline(item)
    elif isinstance(value, str) and any(token in value.lower() for token in ('data:', 'blob:', ';base64,')):
        raise ValueError('Inline images or browser blobs cannot be published')


def read_json(path):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid_constant)


def camera_catalog(value):
    if not isinstance(value, list) or not value or len(value) > 10000:
        raise ValueError('Expected a nonempty camera metadata catalog')
    result, ids = [], set()
    for camera in value:
        if isinstance(camera, dict):
            # Look direction (N, SE, …) lets browsers turn facing into a compass heading.
            camera = {**camera, 'orientation': camera.get('orientation') or ''}
        if not isinstance(camera, dict) or set(camera) != CAMERA_FIELDS:
            raise ValueError('Only camera catalog metadata fields may be published')
        for field in CAMERA_FIELDS - {'lat', 'lon'}:
            if not isinstance(camera[field], str) or len(camera[field]) > 10000:
                raise ValueError('Invalid camera catalog text: ' + field)
        if not camera['id'] or camera['id'] in ids:
            raise ValueError('Camera catalog IDs must be distinct')
        ids.add(camera['id'])
        for field, maximum in [('lat', 90), ('lon', 180)]:
            number = camera[field]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or abs(number) > maximum:
                raise ValueError('Invalid camera coordinate: ' + field)
        for field in ('image', 'source'):
            url = urlsplit(camera[field])
            if url.scheme != 'https' or not url.hostname or not (url.hostname == 'drivebc.ca' or url.hostname.endswith('.drivebc.ca')) or url.username or url.password or url.port not in (None, 443):
                raise ValueError('Only HTTPS DriveBC catalog URLs may be published')
        result.append({field: camera[field] for field in sorted(CAMERA_FIELDS)})
    return result


def sanitized_data(source):
    latest = read_json(regular_file(source, 'data/latest.json'))
    cameras = read_json(regular_file(source, 'data/cameras.json'))
    patterns = read_json(regular_file(source, 'data/patterns.json'))
    for value in (latest, cameras, patterns):
        reject_inline(value)
    if not isinstance(latest, dict) or set(latest) != LATEST_FIELDS or latest.get('storage_mode') != 'browser-only':
        raise ValueError('latest.json must contain only the browser-only camera catalog')
    if latest['errors'] != [] or latest['schema_version'] != 2 or not isinstance(latest['generated'], str) or not isinstance(latest['model'], str):
        raise ValueError('Invalid public catalog metadata')
    cleaned = camera_catalog(latest['cameras'])
    if camera_catalog(cameras) != cleaned or isinstance(latest['catalog_size'], bool) or latest['catalog_size'] != len(cleaned):
        raise ValueError('Published camera catalogs disagree')
    if patterns != {'cameras': {}, 'hourly': []}:
        raise ValueError('Remote camera observation patterns cannot be published')
    return {'latest.json': {**latest, 'cameras': cleaned}, 'cameras.json': cleaned, 'patterns.json': patterns}


def image_content(path):
    with path.open('rb') as source:
        header = source.read(256)
    return (header.startswith((b'\x89PNG\r\n\x1a\n', b'\xff\xd8\xff', b'GIF87a', b'GIF89a', b'BM', b'II*\0', b'MM\0*'))
            or header[:4] == b'RIFF' and header[8:12] == b'WEBP'
            or b'<svg' in header.lower()
            or header[4:8] == b'ftyp' and header[8:12] in (b'avif', b'avis', b'heic', b'heix'))


def stage(source=Path('web'), output=Path('.cache/site')):
    source, output = Path(source).absolute(), Path(output).absolute()
    if any(path.is_symlink() for path in (source, *source.parents)) or not source.is_dir():
        raise ValueError('Publication source must be a real directory')
    if any(path.is_symlink() for path in (output, *output.parents)) or output.resolve() == source.resolve() or source.resolve() in output.resolve().parents or output.resolve() in source.resolve().parents:
        raise ValueError('Publication output must be separate from its source')
    payloads = sanitized_data(source)
    files = []
    for relative in (*APP_FILES, *('models/' + name for name in MODEL_FILES), *('vendor/' + name for name in VENDOR_FILES)):
        path = regular_file(source, relative)
        if image_content(path):
            raise ValueError('Image bytes found outside curated vendor icons: ' + relative)
        files.append((relative, path))
    for relative in ('vendor/' + name for name in VENDOR_IMAGES):
        path = regular_file(source, relative, required=False)
        if path:
            if not path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('Curated Leaflet icons must be PNG files')
            files.append((relative, path))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.site-stage-', dir=output.parent))
    try:
        for relative, path in files:
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        (temporary / 'data').mkdir()
        for name, value in payloads.items():
            (temporary / 'data' / name).write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
        if output.exists():
            if not output.is_dir():
                raise ValueError('Publication output must be a directory')
            shutil.rmtree(output)
        temporary.replace(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'files': len(files) + len(payloads), 'cameras': len(payloads['cameras.json']), 'storage_mode': 'browser-only'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('web'))
    parser.add_argument('--output', type=Path, default=Path('.cache/site'))
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.output)))

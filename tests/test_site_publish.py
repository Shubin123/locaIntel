"""Verify the Pages artifact boundary excludes local camera evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/stage-site.py'
SPEC = importlib.util.spec_from_file_location('stage_site', SCRIPT)
publish = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish)


class SitePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.output = self.root / 'web', self.root / 'site'
        for relative in (*publish.APP_FILES, *('models/' + name for name in publish.MODEL_FILES), *('vendor/' + name for name in publish.VENDOR_FILES)):
            file = self.source / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('public application fixture\n')
        self.camera = dict(id='238', name='Camera', region='North', highway='1', caption='Road',
                           image='https://www.drivebc.ca/images/238.jpg',
                           source='https://images.drivebc.ca/bchighwaycam/pub/html/123.html', lat=49, lon=-123, orientation='N')
        self.latest = dict(storage_mode='browser-only', generated='2026-10-20T12:00:00Z',
                           cameras=[self.camera], catalog_size=1, errors=[], model='On-device inference', schema_version=2)
        self.write_data()

    def write_data(self):
        for name, value in [('latest', self.latest), ('cameras', self.latest['cameras']), ('patterns', {'cameras': {}, 'hourly': []})]:
            file = self.source / 'data' / (name + '.json')
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(value))

    def test_allowlist_excludes_every_photo_cache_trace_and_unlisted_asset(self):
        private = ('frames/238.jpg', 'photos/secret.png', '.cache/state.json', 'data/history.json',
                   'trace.zip', 'screenshots/camera.png', 'models/private-training.jpg', 'private.js')
        for name in private:
            file = self.source / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'PRIVATE CAMERA EVIDENCE')
        result = publish.stage(self.source, self.output)
        self.assertEqual(result['storage_mode'], 'browser-only')
        self.assertEqual(result['cameras'], 1)
        for name in private:
            self.assertFalse((self.output / name).exists(), name)
        actual = {p.relative_to(self.output).as_posix() for p in self.output.rglob('*') if p.is_file()}
        expected = set(publish.APP_FILES) | {'models/' + n for n in publish.MODEL_FILES} | {'vendor/' + n for n in publish.VENDOR_FILES} | {'data/latest.json', 'data/cameras.json', 'data/patterns.json'}
        self.assertEqual(actual, expected)
        self.assertEqual(json.loads((self.output / 'data/latest.json').read_text()), self.latest)

    def test_existing_output_is_rebuilt_so_stale_camera_photos_cannot_survive(self):
        publish.stage(self.source, self.output)
        leaked = self.output / 'frames' / 'old.jpg'
        leaked.parent.mkdir()
        leaked.write_bytes(b'old private photo')
        publish.stage(self.source, self.output)
        self.assertFalse(leaked.exists())

    def test_camera_hash_frame_detections_and_other_analysis_fields_are_rejected(self):
        for field, value in [('hash', 'private-hash'), ('frame', 'frames/238.jpg'), ('detections', []),
                             ('analysis', {'motion': {}}), ('counts', {'car': 2}), ('facing', {}),
                             ('photo', 'private'), ('HASH', 'private')]:
            with self.subTest(field=field):
                self.camera[field] = value
                self.write_data()
                with self.assertRaises(ValueError):
                    publish.stage(self.source, self.output)
                del self.camera[field]
        self.assertFalse(self.output.exists())

    def test_inline_image_or_browser_blob_is_rejected_in_any_catalog_text(self):
        for value in ('data:image/png;base64,SECRET', 'DATA:image/svg+xml,<svg/>',
                      '<img src="data:image/jpeg;base64,PRIVATE">', 'blob:https://example.test/private'):
            with self.subTest(value=value):
                self.camera['caption'] = value
                self.write_data()
                with self.assertRaises(ValueError):
                    publish.stage(self.source, self.output)

    def test_non_browser_catalog_or_remote_observation_patterns_are_rejected(self):
        for mode in (None, 'server', 'browser-only '):
            self.latest['storage_mode'] = mode
            self.write_data()
            with self.assertRaises(ValueError):
                publish.stage(self.source, self.output)
        self.latest['storage_mode'] = 'browser-only'
        self.write_data()
        (self.source / 'data/patterns.json').write_text(json.dumps({'cameras': {'238': {'samples': 1}}, 'hourly': []}))
        with self.assertRaisesRegex(ValueError, 'observation patterns'):
            publish.stage(self.source, self.output)

    def test_catalogs_must_agree_and_remote_urls_are_restricted_to_drivebc(self):
        (self.source / 'data/cameras.json').write_text(json.dumps([{**self.camera, 'id': 'other'}]))
        with self.assertRaisesRegex(ValueError, 'disagree'):
            publish.stage(self.source, self.output)
        for url in ('https://drivebc.ca.evil.test/238.jpg', 'http://www.drivebc.ca/238.jpg',
                    'https://user:password@www.drivebc.ca/238.jpg', 'https://www.drivebc.ca:444/238.jpg'):
            self.camera['image'] = url
            self.write_data()
            with self.assertRaises(ValueError):
                publish.stage(self.source, self.output)

    def test_symlinks_never_enter_artifact_including_parent_directories(self):
        private = self.root / 'private'
        private.write_bytes(b'private image')
        allowed = self.source / 'app.js'
        allowed.unlink()
        allowed.symlink_to(private)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            publish.stage(self.source, self.output)

        allowed.unlink()
        allowed.write_text('public')
        vendor = self.source / 'vendor'
        vendor.rename(self.root / 'real-vendor')
        vendor.symlink_to(self.root / 'real-vendor', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            publish.stage(self.source, self.output)

    def test_source_parent_symlinks_cannot_bypass_component_checks(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'real directory'):
            publish.stage(alias / 'web', self.output)

    def test_renaming_a_camera_image_as_an_allowed_model_does_not_publish_it(self):
        (self.source / 'models/yolo11n.onnx').write_bytes(b'\x89PNG\r\n\x1a\nPRIVATE IMAGE')
        with self.assertRaisesRegex(ValueError, 'Image bytes'):
            publish.stage(self.source, self.output)

    def test_only_curated_leaflet_png_icons_may_be_copied(self):
        icon = self.source / 'vendor/images/marker-icon.png'
        icon.parent.mkdir()
        icon.write_bytes(b'\x89PNG\r\n\x1a\npublic icon')
        (icon.parent / 'private.jpg').write_bytes(b'private image')
        publish.stage(self.source, self.output)
        self.assertEqual((self.output / 'vendor/images/marker-icon.png').read_bytes(), icon.read_bytes())
        self.assertFalse((self.output / 'vendor/images/private.jpg').exists())
        icon.write_bytes(b'not a PNG')
        with self.assertRaisesRegex(ValueError, 'PNG'):
            publish.stage(self.source, self.output)

    def test_invalid_input_does_not_replace_a_previously_safe_artifact(self):
        publish.stage(self.source, self.output)
        old = (self.output / 'data/latest.json').read_bytes()
        self.latest['cameras'][0]['frame'] = 'private.jpg'
        self.write_data()
        with self.assertRaises(ValueError):
            publish.stage(self.source, self.output)
        self.assertEqual((self.output / 'data/latest.json').read_bytes(), old)

    def test_source_and_output_must_not_overlap_or_follow_output_symlinks(self):
        for output in (self.source, self.source / 'site', self.root):
            with self.assertRaisesRegex(ValueError, 'separate'):
                publish.stage(self.source, output)
        self.output.symlink_to(self.root / 'another', target_is_directory=True)
        with self.assertRaises(ValueError):
            publish.stage(self.source, self.output)

    def test_missing_app_assets_and_nonfinite_metadata_fail_before_publication(self):
        (self.source / 'app.js').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing regular'):
            publish.stage(self.source, self.output)
        (self.source / 'app.js').write_text('public')
        self.camera['lat'] = float('nan')
        self.write_data()
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            publish.stage(self.source, self.output)


if __name__ == '__main__':
    unittest.main()

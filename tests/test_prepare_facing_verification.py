import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
import zipfile

from PIL import Image
from pipeline.prepare_facing_verification import ARCHIVE_SHA256, ROOT, build, label


def jpeg(seed):
    image = Image.frombytes('RGB', (64, 48), random.Random(seed).randbytes(64*48*3))
    target = io.BytesIO()
    image.save(target, format='JPEG', quality=85)
    return target.getvalue()


class FacingVerificationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive, self.output = self.root / 'test.zip', self.root / 'verification'

    def archive_rows(self, rows, images=None):
        images = images or {}
        with zipfile.ZipFile(self.archive, 'w') as target:
            target.writestr('metadata.jsonl', '\n'.join(json.dumps(r) for r in rows))
            for i, row in enumerate(rows):
                target.writestr(row['file_name'], images.get(row['file_name'], jpeg(i)))
        return hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def rows(self, identities=(14, 15), angles=(0, 45, 90, 135, 180, 225, 270, 315)):
        return [dict(file_name=f'test/{identity}/{angle}.jpg', identity=identity, angle=angle,
                     x1=5, y1=6, x2=55, y2=40) for identity in identities for angle in angles]

    def test_exactly_1000_distinct_native_jpegs_cover_identities_and_all_angle_bins(self):
        rows = self.rows(range(14, 28), range(0, 360, 5))
        sha = self.archive_rows(rows)
        result = build(self.archive, self.output, 1000, expected_sha=sha)
        self.assertEqual(result['count'], 1000)
        self.assertEqual(len(result['photos']), 1000)
        self.assertEqual(len({p['hash'] for p in result['photos']}), 1000)
        self.assertEqual({p['identity'] for p in result['photos']}, set(range(14, 28)))
        self.assertEqual({int(p['angle']//5) for p in result['photos']}, set(range(72)))
        per_id = [sum(p['identity'] == identity for p in result['photos']) for identity in range(14, 28)]
        self.assertLessEqual(max(per_id)-min(per_id), 1)
        self.assertEqual(result['photos_bytes'], sum(p['bytes'] for p in result['photos']))
        with zipfile.ZipFile(self.archive) as source:
            for p in result['photos']:
                content = (self.output / p['file']).read_bytes()
                self.assertEqual(content, source.read(p['original_file']))
                self.assertEqual(hashlib.sha256(content).hexdigest(), p['hash'])
                self.assertEqual((p['width'], p['height']), (64, 48))
                self.assertEqual(p['box'], [5/64, 6/48, 55/64, 40/48])
                self.assertNotIn('cameraId', p)
                self.assertNotIn('observedAt', p)

    def test_selection_and_manifest_are_deterministic_across_rebuilds(self):
        sha = self.archive_rows(self.rows())
        first = build(self.archive, self.output, 10, expected_sha=sha)
        encoded = (self.output / 'manifest.json').read_bytes()
        second = build(self.archive, self.output, 10, expected_sha=sha)
        self.assertEqual(first, second)
        self.assertEqual((self.output / 'manifest.json').read_bytes(), encoded)

    def test_invalid_angle_strata_do_not_unbalance_available_identities(self):
        rows = self.rows(angles=(0, 45, 90, 135))
        rows[0]['x2'] = 0
        sha = self.archive_rows(rows)
        photos = build(self.archive, self.output, 6, expected_sha=sha)['photos']
        self.assertEqual(sum(p['identity'] == 14 for p in photos), 3)
        self.assertEqual(sum(p['identity'] == 15 for p in photos), 3)

    def test_pinned_archive_checksum_is_mandatory_unless_fixture_checksum_is_explicit(self):
        sha = self.archive_rows(self.rows())
        self.assertNotEqual(sha, ARCHIVE_SHA256)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            build(self.archive, self.output, 2)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            build(self.archive, self.output, 2, expected_sha=None)
        self.assertFalse(self.output.exists())

    def test_duplicate_original_bytes_do_not_count_as_distinct_photos(self):
        rows = self.rows(angles=(0, 45, 90))
        images = {r['file_name']: jpeg(1) for r in rows}
        sha = self.archive_rows(rows, images)
        with self.assertRaisesRegex(ValueError, 'exactly 2.*found 1'):
            build(self.archive, self.output, 2, expected_sha=sha)
        self.assertFalse(self.output.exists())

    def test_invalid_boxes_corrupt_photos_and_non_jpeg_inputs_are_not_selected(self):
        rows = self.rows(angles=(0, 45, 90, 135, 180, 225, 270, 315))[:8]
        rows[0]['x1'] = -1
        rows[1]['x2'] = 1000
        rows[2]['x2'] = rows[2]['x1'] + 19
        rows[3]['y2'] = rows[3]['y1'] + 15
        rows[4]['x1'] = float('nan')
        png = io.BytesIO()
        Image.new('RGB', (64, 48)).save(png, format='PNG')
        images = {rows[5]['file_name']: b'invalid image', rows[6]['file_name']: png.getvalue()}
        sha = self.archive_rows(rows, images)
        result = build(self.archive, self.output, 1, expected_sha=sha)
        self.assertEqual(result['photos'][0]['original_file'], rows[7]['file_name'])

    def test_non_test_identities_and_invalid_angles_are_rejected(self):
        for field, value in [('identity', 1), ('angle', -1), ('angle', 361), ('angle', True), ('angle', float('inf'))]:
            with self.subTest(field=field, value=value):
                rows = self.rows()
                rows[0][field] = value
                sha = self.archive_rows(rows)
                with self.assertRaises(ValueError):
                    build(self.archive, self.output, 1, expected_sha=sha)

    def test_inclusive_cfv_class_boundaries_match_training_labels(self):
        self.assertEqual([label(a) for a in (0, 22.5, 22.5001, 67.5, 337.5, 359.9)], [0, 0, 1, 1, 7, 0])
        sha = self.archive_rows(self.rows(identities=(14,), angles=(360,)))
        result = build(self.archive, self.output, 1, expected_sha=sha)
        self.assertEqual(result['photos'][0]['angle'], 360)
        self.assertEqual(result['photos'][0]['label'], 0)

    def test_short_input_does_not_replace_a_previous_valid_verification_set(self):
        sha = self.archive_rows(self.rows())
        build(self.archive, self.output, 2, expected_sha=sha)
        saved = (self.output / 'manifest.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'exactly 1000'):
            build(self.archive, self.output, 1000, expected_sha=sha)
        self.assertEqual((self.output / 'manifest.json').read_bytes(), saved)

    def test_website_output_and_archive_ancestor_output_are_forbidden(self):
        sha = self.archive_rows(self.rows())
        for output in (ROOT / 'web/facing-verification', self.root):
            with self.assertRaisesRegex(ValueError, 'local directory'):
                build(self.archive, output, 1, expected_sha=sha)
        self.assertTrue(self.archive.exists())


if __name__ == '__main__':
    unittest.main()

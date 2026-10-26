import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError

from pipeline.gps_replay import IMAGE, LISTING, fetch_passes, select, stamp_time

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
JPEG = b'\xff\xd8\xff\xe0replay'


def passed(camera, start, end=None, closest=None):
    end = end or start
    return {'cameraId': camera, 'name': 'Cam ' + camera, 'start': start, 'end': end, 'expectedFacing': 'toward_camera',
            'closest': {'time': closest or end, 'distance': 22.2}}


def export(*passes):
    return {'format': 'locaintel-own-vehicle-v1', 'passes': list(passes)}


class GpsReplayTests(unittest.TestCase):
    def test_replay_stamps_are_utc_capture_times(self):
        self.assertEqual(stamp_time('20261026230014'), datetime(2026, 10, 26, 23, 0, 14, tzinfo=timezone.utc))

    def test_frames_during_the_pass_plus_nearest_each_side_within_the_window(self):
        start, end = datetime(2026, 10, 27, 9, 0, 0, tzinfo=timezone.utc), datetime(2026, 10, 27, 9, 0, 12, tzinfo=timezone.utc)
        stamps = ['20261027085800', '20261027085930', '20261027085945', '20261027090005', '20261027090100', '20261027090300', '../../etc', 7]
        self.assertEqual(select(stamps, start, end, timedelta(seconds=60)), ['20261027085945', '20261027090005', '20261027090100'])
        self.assertEqual(select(stamps, start, end, timedelta(seconds=10)), ['20261027090005'])

    def test_live_passes_are_downloaded_once_and_expired_passes_are_not_requested(self):
        calls = []

        def fetch(url):
            calls.append(url)
            if url == LISTING.format(id='238'):
                return json.dumps(['20261027085945', '20261027090005', '20261027091000']).encode()
            return JPEG
        data = export(passed('238', '2026-10-27T09:00:00.000Z', '2026-10-27T09:00:12.000Z', '2026-10-27T09:00:09.000Z'),
                      passed('238', '2026-10-27T09:09:50.000Z'), passed('864', '2026-10-26T11:00:00.000Z'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fetch_passes(data, root, fetch, now=NOW)
            self.assertEqual(calls, [LISTING.format(id='238'), IMAGE.format(id='238', stamp='20261027085945'),
                                     IMAGE.format(id='238', stamp='20261027090005'), IMAGE.format(id='238', stamp='20261027091000')])
            self.assertEqual([p['status'] for p in manifest], ['saved', 'saved', 'expired'])
            first = manifest[0]['frames']
            self.assertEqual([f['file'] for f in first], ['238/20261027085945.jpg', '238/20261027090005.jpg'])
            self.assertEqual([f['car_expected'] for f in first], [False, True])
            self.assertEqual(first[1]['seconds_from_closest'], -4.0)
            self.assertEqual((root / '238' / '20261027090005.jpg').read_bytes(), JPEG)
            written = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(written['source'], 'DriveBC Replay the Day')
            self.assertEqual(len(written['passes']), 3)
            calls.clear()
            fetch_passes(data, root, fetch, now=NOW)
            self.assertEqual(calls, [LISTING.format(id='238')], 'saved frames are not downloaded again')

    def test_failures_are_recorded_per_pass_and_never_written_as_photos(self):
        def fetch(url):
            if '864' in url:
                raise URLError('offline')
            return json.dumps(['20261027090005']).encode() if 'api' in url else b'<html>not found</html>'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fetch_passes(export(passed('864', '2026-10-27T09:00:00.000Z'), passed('238', '2026-10-27T09:00:00.000Z', '2026-10-27T09:00:10.000Z')), root, fetch, now=NOW)
            self.assertEqual(manifest[0]['status'], 'error: <urlopen error offline>')
            self.assertRegex(manifest[1]['status'], 'did not return a JPEG')
            self.assertEqual(sorted(p.name for p in root.rglob('*.jpg')), [])

    def test_unexpected_exports_and_camera_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'evaluation set'):
                fetch_passes({'passes': []}, Path(directory), lambda url: b'', now=NOW)
            with self.assertRaisesRegex(ValueError, 'camera id'):
                fetch_passes(export(passed('../238', '2026-10-27T09:00:00.000Z')), Path(directory), lambda url: b'', now=NOW)

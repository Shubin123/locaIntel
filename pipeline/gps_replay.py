"""Download DriveBC Replay the Day frames for the camera passes in a GPS evaluation export.

DriveBC keeps about 24 hours of images per camera. Browsers cannot read the
replay listing (it sends no CORS header), so this local helper does. For each
pass in the JSON exported by the web app's vehicle tracking panel, it saves the
replay frames captured during the pass, plus the nearest one on each side within
the window, to .cache/gps-replay/<camera>/<UTC stamp>.jpg with a manifest.json.
Frames flagged car_expected were captured while the vehicle was in view. Import
the output folder in the web app with "Import replay frames" to verify them there.
"""
import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

LISTING = 'https://www.drivebc.ca/api/webcams/{id}/replayTheDay/'
IMAGE = 'https://www.drivebc.ca/images/replaytheday/{id}/{stamp}.jpg'
RETENTION = timedelta(hours=24)
# v1: the original own-car evaluation set; v2: any vehicle profile. Both carry passes.
FORMATS = ('locaintel-own-vehicle-v1', 'locaintel-vehicle-track-v2')


def moment(value):
    """ISO time from the export; the Z suffix needs replacing before Python 3.11."""
    return datetime.fromisoformat(str(value).replace('Z', '+00:00'))


def stamp_time(stamp):
    """Replay stamps are UTC capture times, YYYYMMDDHHMMSS."""
    return datetime.strptime(stamp, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)


def iso(value):
    return value.isoformat().replace('+00:00', 'Z')


def select(stamps, start, end, window):
    """Stamps captured during the pass, plus the nearest one each side within window."""
    times = sorted((stamp_time(s), s) for s in stamps if isinstance(s, str) and re.fullmatch(r'\d{14}', s))
    before = [s for t, s in times if start - window <= t < start][-1:]
    during = [s for t, s in times if start <= t <= end]
    after = [s for t, s in times if end < t <= end + window][:1]
    return before + during + after


def default_fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'LocaIntel GPS replay'}), timeout=30) as response:
        return response.read()


def fetch_passes(export, output=Path('.cache/gps-replay'), fetch=default_fetch, now=None, window=timedelta(seconds=60)):
    if not isinstance(export, dict) or export.get('format') not in FORMATS or not isinstance(export.get('passes'), list):
        raise ValueError('Expected a tracking set exported by the vehicle tracking panel')
    now = now or datetime.now(timezone.utc)
    output = Path(output)
    listings, manifest = {}, []
    for item in export['passes']:
        camera = str(item.get('cameraId', ''))
        # The id becomes a URL and a directory name: digits only.
        if not re.fullmatch(r'\d{1,7}', camera):
            raise ValueError('Unexpected DriveBC camera id: ' + camera)
        start, end, closest = moment(item['start']), moment(item['end']), moment(item['closest']['time'])
        entry = {'camera_id': camera, 'name': item.get('name'), 'start': item['start'], 'end': item['end'],
                 'closest': item['closest'], 'expected_facing': item.get('expectedFacing'), 'frames': []}
        manifest.append(entry)
        if now - start > RETENTION:
            entry['status'] = 'expired'
            continue
        try:
            if camera not in listings:
                listings[camera] = json.loads(fetch(LISTING.format(id=camera)))
            for stamp in select(listings[camera], start, end, window):
                path = output / camera / (stamp + '.jpg')
                if not path.exists():
                    data = fetch(IMAGE.format(id=camera, stamp=stamp))
                    if not data.startswith(b'\xff\xd8'):
                        raise ValueError('DriveBC replay did not return a JPEG for ' + stamp)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                captured = stamp_time(stamp)
                entry['frames'].append({'file': path.relative_to(output).as_posix(), 'captured_at': iso(captured),
                                        'seconds_from_closest': round((captured - closest).total_seconds(), 1),
                                        'car_expected': start <= captured <= end})
            entry['status'] = 'saved' if entry['frames'] else 'no replay frame near the pass'
        except (OSError, URLError, ValueError) as error:
            entry['status'] = 'error: ' + str(error)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps({'source': 'DriveBC Replay the Day', 'generated': iso(now), 'passes': manifest}, indent=1) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('export', type=Path, help='tracking set JSON exported by the vehicle tracking panel')
    parser.add_argument('--output', type=Path, default=Path('.cache/gps-replay'))
    parser.add_argument('--window-seconds', type=float, default=60, help='also keep the nearest frame this far before/after each pass')
    args = parser.parse_args()
    passes = fetch_passes(json.loads(args.export.read_text(encoding='utf-8')), args.output, window=timedelta(seconds=args.window_seconds))
    print(json.dumps({'passes': len(passes), 'frames': sum(len(p['frames']) for p in passes),
                      'expired': sum(p['status'] == 'expired' for p in passes), 'output': str(args.output)}))

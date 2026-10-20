"""Publish camera metadata only. Camera photos are fetched by browsers, never CI."""
import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

CATALOG = 'https://catalogue.data.gov.bc.ca/dataset/6b39a910-6c77-476f-ac96-7b4f18849b1c/resource/a9d52d85-8402-4ce7-b2ac-a2779837c48a/download/webcams.csv'


def publish(output=Path('web/data'), fetch=None):
    if fetch is None:
        def fetch(url):
            with urlopen(Request(url, headers={'User-Agent': 'LocaIntel camera catalog'}), timeout=45) as response:
                return response.read().decode('utf-8-sig')
    cameras = []
    for row in csv.DictReader(io.StringIO(fetch(CATALOG))):
        cameras.append(dict(id=row['id'], name=row['camName'], region=row['region_name'],
                            highway=row['highway_number'], caption=row['caption'],
                            image=row['links_imageURL'], source=row['links_camDetailsPage'],
                            lat=float(row['latitude']), lon=float(row['longitude'])))
    if not cameras:
        raise ValueError('Camera catalog is empty')
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(storage_mode='browser-only', generated=datetime.now(timezone.utc).isoformat(),
                   cameras=cameras, catalog_size=len(cameras), errors=[], model='On-device YOLO + facing', schema_version=2)
    for name, value in [('latest', payload), ('cameras', cameras), ('patterns', {'cameras': {}, 'hourly': []})]:
        (output / (name+'.json')).write_text(json.dumps(value, separators=(',', ':'))+'\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('web/data'))
    print(json.dumps({'cameras': publish(parser.parse_args().output)['catalog_size'], 'storage': 'browser-only'}))

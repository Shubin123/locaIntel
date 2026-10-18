# LocaIntel

Public DriveBC camera observatory with cached snapshots and YOLO11 object detection.

## Run locally

Python 3.11+ recommended.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pipeline.collect --limit 72
uvicorn pipeline.server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. The single-worker service polls a rotating selection of 48 cameras every 120 seconds. Configure `CAMERA_LIMIT` and `POLL_SECONDS` via environment variables. GPU inference follows Ultralytics defaults; a CPU also works. We use pretrained YOLO11 nano, not a model trained specifically on BC roads.

- `/api/events`: SSE JSON updates and heartbeats.
- `/api/cameras/864/stream`: multipart JPEG stream of cached snapshots.
- `/data/latest.json`: latest analysis; normalized detection boxes, per-class counts, observed timestamp, upstream Last-Modified when supplied, hash, and fetch errors.
- `/data/cameras.json`: complete upstream camera catalog.

## GitHub Pages

The workflow ingests and analyzes 72 cameras per run, rotates through least recently checked cameras, persists cache with GitHub Actions Cache, and deploys `web/`. It runs on pushes, manually, and about every 20 minutes. GitHub can delay schedules; this is not a real-time service guarantee. The dashboard polls the published JSON every minute. GitHub Pages cannot run Python or host SSE; use the included server for continuous collection and streams.

Image bytes are SHA-256 deduplicated before inference. Downloads use six concurrent workers, a 25-second request timeout, a 120-second TTL, and a daily camera catalog cache. Only one latest frame per camera is retained. Cache and inference model persist between scheduled runs; cache eviction can cause a cold start. Fetch failures preserve prior good frames and expose errors. Displayed observation times are fetch/inference times, not guaranteed camera capture times. Unchanged images retain their original observation time.

Filters, search, overlays, activity sorting, stale labels, and paginated rendering work on real analyzed data. Counts represent objects in a still frame, not unique vehicles across frames, speed, traffic volume, road safety, or incident detection. Weather, nighttime, occlusion, and camera angle affect accuracy.

## Verification

```sh
python -m unittest discover -s tests
```

## Sources and licensing

Camera catalog: [DriveBC HighwayCams](https://catalogue.data.gov.bc.ca/dataset/bc-highway-cams). Images and data © Province of British Columbia, provided under the [Open Government Licence – BC](https://www2.gov.bc.ca/gov/content/data/open-data/open-government-licence-bc). This independent project is not affiliated with DriveBC.

Source code: AGPL-3.0; Ultralytics dependencies and model licensing also apply. See [Ultralytics licensing](https://www.ultralytics.com/license).

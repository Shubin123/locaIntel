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

## Motion, road membership, and traffic patterns

The dashboard now combines YOLO11 objects with NVIDIA SegFormer B0 (Cityscapes road segmentation), optical flow, and an explicit **rule-based meta-classifier**. This ensemble has not been trained or probability-calibrated on DriveBC data. It abstains on weak or conflicting evidence. SegFormer is run at 512×512 for CPU inference. A patch just below a vehicle's bounding box supplies road support; segmentation errors and perspective can affect this estimate. Road scores are model scores, not calibrated probabilities. `on_road: null` means unavailable/uncertain, while `false` means road support was low, not proof of illegal/off-road travel.

Individual motion is evaluated only when trusted frame timestamps are at most two seconds apart. Background features validate camera stability; forward/backward Lucas–Kanade optical flow, appearance similarity, detection class and predicted location must agree on an unambiguous match. Camera shifts, duplicate frames, long gaps and weak features yield **unknown**, not stationary. Thresholds use a maximum 640px-wide analysis image. Arrows indicate image-space direction; no metric speed or vehicle identity across sparse snapshots is inferred. A `track_id` is only the previous-frame detection index, not a persistent identity.

Scheduled DriveBC stills are generally too sparse for reliable individual motion. Their published map shows sampled **visible-vehicle density**, with optional road-supported and motion-supported layers. Sparse sampling cannot determine traffic throughput, congestion, turning movements, lane direction, or unique vehicle counts. Camera locations, fields of view and varying observation times bias geographic density; brightness is normalized to the displayed maximum. Gray camera markers remain visible when no vehicles qualify. Model load failures are disclosed and road membership remains unknown.

The per-camera heatmap accumulates vehicle bottom-center positions across up to 48 distinct frames from the past 24 hours and divides by sample count. The hourly chart averages counts per camera first, then weights each observed camera equally. It shows gaps, sample counts and changing camera coverage. History starts when this version first runs; there is no fabricated historical traffic. Each pane and visible camera card has a loaded/updated timestamp and briefly highlights when its relevant content changes, with an accessible announcement and recent update log. Refreshes which return unchanged data do not trigger update notifications.

### Video and Meta DINOv2

Meta's DINOv2-small can provide crop appearance embeddings for short-gap vehicle association. It is used by the video input mode, not falsely applied as a vehicle identity classifier to far-apart webcam stills. Optional histogram mode is lighter but less discriminative. The code was checked with real downloaded DINOv2 embeddings and synthetic short-gap motion tests; this is not field validation of vehicle tracking accuracy.

Run collection once to initialize the catalog, then supply an authorized video that actually corresponds to the selected camera:

```sh
python -m pipeline.video --source /path/to/camera-video.mp4 --camera-id 864 --frames 120
# Lighter association, without DINOv2:
python -m pipeline.video --source /path/to/camera-video.mp4 --camera-id 864 --appearance histogram
```

Recorded videos use the file's frame rate for motion and are labeled as recorded; displayed observation time is analysis time. Live sources use frame arrival time and abstain when processing gaps exceed two seconds. Recordings are for inspecting motion; do not mix archived recordings into a live traffic deployment. Run a single writer: the video CLI and snapshot collector should not write concurrently. To serve video updates instead of running the snapshot collector in the server:

```sh
VIDEO_SOURCE=/path/to/camera-video.mp4 VIDEO_CAMERA_ID=864 uvicorn pipeline.server:app --host 0.0.0.0 --port 8000
```

The server repeats bounded video sessions; recorded files restart between sessions. Models download on first use. GitHub Actions persists Hugging Face models under `.cache/huggingface`. `--road-model none` disables segmentation for a lightweight CLI run. DINOv2 is not downloaded by scheduled snapshot jobs.

Compass direction requires a surveyed per-camera road axis in `calibrations.json`; camera-facing direction alone is insufficient. Keys are camera IDs, for example:

```json
{"CAMERA_ID": {"road_axis": [[0.5, 0.9], [0.5, 0.1]], "away_bearing": 90}}
```

These illustrative coordinates must be replaced with measured geometry. `road_axis` runs from the near-road point to the far-road point in normalized image coordinates; `away_bearing` is that road direction's compass bearing in degrees clockwise from north. Opposing motion is assigned the reciprocal bearing. Curves, multiple roads, ramps, and cameras that change view need richer calibration; leave them uncalibrated. No production calibrations are supplied.

Model and map sources: [Meta DINOv2](https://github.com/facebookresearch/dinov2), [SegFormer model and license](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024), [OpenCV optical flow](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html), [Leaflet](https://leafletjs.com/), and [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). Vendored map library licenses are in `web/vendor/`. Model-specific license terms apply independently of this repository's source license.

### Reading motion arrows

The camera explorer starts with heat and road shading off to keep motion legible. Green outlined arrows end at the current vehicle position; their tails show the background-compensated displacement enlarged by the selected 1×/4×/8×/16× display scale. Long arrows are shortened to fit the image. Their length is not speed. The numbered vehicle list gives image direction, analysis-pixel displacement, and frame gap; selecting a row isolates that vehicle. Selection resets when a new frame arrives because detection numbers are not persistent vehicle identities. Dashed amber boxes mean unknown motion, blue boxes indicate stationary evidence, and sparse snapshots show an explicit motion-unavailable explanation. `node --test tests/motion.test.cjs` checks vector geometry and evidence gating.

### Refresh camera images from DriveBC

**Refresh from DriveBC** now makes direct, cache-bypassing browser requests to the upstream image URLs for every visible camera and the selected camera. It also checks the published analysis. Image requests are limited to six at a time, time out after 12 seconds, and are shared between the camera card and explorer. Newly displayed cameras load from the source when filtering or loading more feeds. The latest-source view refreshes every minute while open; the analyzed-frame view remains selected during automatic analysis checks.

Each camera shows its source-check time separately from its analyzed-frame age. A source refresh cannot make DriveBC capture a newer frame; an unchanged upstream image can legitimately remain unchanged. Failed requests retain the last successful source image, or fall back to the cached analyzed image, with a visible failure status. Refreshing source images still works when the published analysis request fails, provided the page already has its camera list.

Source image bytes are hashed with SHA-256. Published boxes, vectors, road shading and sampled heat are shown only on matching images. Fresh images without matching published analysis run through the browser detector described below, so new boxes and per-camera counts appear as local inference completes. Geographic maps, history and network totals still refer to published analysis. **View analyzed frame** switches back to the matching cached frame; **Motion-only view** also selects that frame. On GitHub Pages, fresh images do not trigger Python inference: YOLO/SegFormer analysis remains on its separate scheduled workflow. No GitHub token is exposed to visitors.

`node --test tests/*.test.cjs` checks direct source requests, cache bypass, bounded downloads, unchanged-image reuse, timeouts and last-good-image retention, as well as vector geometry.

### Bounding boxes on refreshed images

Fresh DriveBC images now run through YOLO11 nano **in the browser**, using a vendored ONNX Runtime Web worker. The first unmatched source image downloads the model (~10 MB) and WASM runtime (~11 MB); the UI shows loading and per-camera detection progress. The selected camera is prioritized, inference runs one image at a time off the UI thread, and results are cached by source-image SHA-256. Repeated images reuse the same boxes; a new image receives new detections. Refresh retries failed detection. Browser detections stay local and do not modify the published history or aggregate map.

Fresh-source boxes and per-camera counts appear as inference completes. Confidence is 0.30 and class-specific NMS IoU is 0.45. Normalized boxes undo square-input letterboxing so coordinates align with the original image. Single-frame browser detections have unknown movement and road membership; cached road masks, vectors and historical heat are not projected onto different images. If browser inference fails or the device lacks WebAssembly support, the page discloses the error and **View analyzed frame** restores the last server-analyzed image and boxes.

The checked-in model can be rebuilt with `python -m pipeline.export_browser_model` after installing the optional ONNX export dependency noted in that module. Model/runtime licensing and origin are documented in `web/models/README.md` and `web/vendor/ort/LICENSE`. Test coverage includes letterbox coordinate mapping, class/confidence filtering, nonmaximum suppression, serial inference, image-hash deduplication, and retry handling.

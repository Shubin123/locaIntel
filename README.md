# LocaIntel

DriveBC camera observatory with browser-local photos, YOLO11 detection and GPU facing inference.

## Run locally

Python 3.11+ recommended. Publish camera metadata, then serve the static app:

```sh
python -m pipeline.publish_catalog
python -m http.server 8000 --bind 127.0.0.1 --directory web
```

Open http://localhost:8000. This server delivers app/model assets and metadata;
your browser fetches photos directly from DriveBC and retains photos and results
locally. The pretrained models are not trained specifically on BC roads.

## GitHub Pages

The workflow publishes only allowlisted app/model assets and public camera
metadata. It runs unit/privacy checks and browser E2E before deployment, then
verifies the deployed site. It runs on pushes, pull requests, manually and daily;
releases are serialized. Camera photos and browser data are never uploaded or
cached by the workflow. Old remote photo caches/artifacts have been removed.

Visible source images load directly from DriveBC under the device budget. Browser
YOLO, road and facing models process them locally; matching results and photos
persist in IndexedDB. Observation times are not guaranteed capture times. Counts
describe visible objects, not unique vehicles, traffic volume, speed or safety.

## Optional local Python analysis

The older Python collector/video server is available for local research only:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pipeline.collect --limit 72
uvicorn pipeline.server:app --host 127.0.0.1 --port 8000
```

This optional utility writes photos to the local filesystem and polls cameras;
it is not the Pages pipeline and must not be run as cloud photo storage. Its
`/api/events`, `/api/cameras/864/stream` and detection-rich `/data/latest.json`
endpoints belong to that local server. Pages publishes metadata-only JSON.

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

The server repeats bounded video sessions; recorded files restart between sessions. Models download on first use. The local Python utility stores Hugging Face models under `.cache/huggingface`. `--road-model none` disables segmentation for a lightweight CLI run. DINOv2 is not downloaded by scheduled snapshot jobs.

Compass direction requires a surveyed per-camera road axis in `calibrations.json`; camera-facing direction alone is insufficient. Keys are camera IDs, for example:

```json
{"CAMERA_ID": {"road_axis": [[0.5, 0.9], [0.5, 0.1]], "away_bearing": 90}}
```

These illustrative coordinates must be replaced with measured geometry. `road_axis` runs from the near-road point to the far-road point in normalized image coordinates; `away_bearing` is that road direction's compass bearing in degrees clockwise from north. Opposing motion is assigned the reciprocal bearing. Curves, multiple roads, ramps, and cameras that change view need richer calibration; leave them uncalibrated. No production calibrations are supplied.

Model and map sources: [Meta DINOv2](https://github.com/facebookresearch/dinov2), [SegFormer model and license](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024), [OpenCV optical flow](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html), [Leaflet](https://leafletjs.com/), and [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). Vendored map library licenses are in `web/vendor/`. Model-specific license terms apply independently of this repository's source license.

### Reading motion arrows

The camera explorer starts with heat and road shading off to keep motion legible. Green outlined arrows end at the current vehicle position; their tails show the background-compensated displacement enlarged by the selected 1×/4×/8×/16× display scale. Long arrows are shortened to fit the image. Their length is not speed. The numbered vehicle list gives image direction, analysis-pixel displacement, and frame gap; selecting a row isolates that vehicle. Selection resets when a new frame arrives because detection numbers are not persistent vehicle identities. Dashed amber boxes mean unknown motion, blue boxes indicate stationary evidence, and sparse snapshots show an explicit motion-unavailable explanation. `node --test tests/motion.test.cjs` checks vector geometry and evidence gating.

### Refresh camera images from DriveBC

**Refresh from DriveBC** now makes direct, cache-bypassing browser requests to the upstream image URLs for every visible camera and the selected camera. It also checks the published analysis. Image requests use the selected device budget (one to six at a time), time out after 12 seconds, and are shared between the camera card and explorer. Newly displayed cameras load from the source when filtering or loading more feeds. Automatic source refresh follows the device power budget; the analyzed-frame view remains selected during automatic checks.

Each camera shows its source-check time separately from its analyzed-frame age. A source refresh cannot make DriveBC capture a newer frame; an unchanged upstream image can legitimately remain unchanged. Failed requests retain the last successful source image, or fall back to the cached analyzed image, with a visible failure status. Refreshing source images still works when the published analysis request fails, provided the page already has its camera list.

Source image bytes are hashed with SHA-256. Published boxes, vectors, road shading and sampled heat are shown only on matching images. Fresh images without matching published analysis run through the browser detector described below, so new boxes and per-camera counts appear as local inference completes. Maps and totals follow the latest fully analyzed frames in Live, or saved frames at the history cursor. **View analyzed frame** switches back to the matching cached frame; **Motion-only view** also selects that frame. On GitHub Pages, fresh images do not trigger Python inference: YOLO/SegFormer/facing inference runs locally in the browser. No GitHub token is exposed to visitors.

`node --test tests/*.test.cjs` checks direct source requests, cache bypass, bounded downloads, unchanged-image reuse, timeouts and last-good-image retention, as well as vector geometry.

### Bounding boxes on refreshed images

Fresh DriveBC images now run through YOLO11 nano **in the browser**, using a vendored ONNX Runtime Web worker. The first unmatched source image downloads the model (~10 MB) and WASM runtime (~11 MB); the UI shows loading and per-camera detection progress. The selected camera is prioritized, inference runs one image at a time off the UI thread, and results are cached by source-image SHA-256. Repeated images reuse the same boxes; a new image receives new detections. Refresh retries failed detection. Browser detections stay local and contribute to this browser’s monitoring map/history; they do not modify the published server dataset.

Fresh-source boxes and per-camera counts appear as inference completes. Confidence is 0.30 and class-specific NMS IoU is 0.45. Normalized boxes undo square-input letterboxing so coordinates align with the original image. Single-frame browser detections have unknown movement and road membership; cached road masks, vectors and historical heat are not projected onto different images. If browser inference fails or the device lacks WebAssembly support, the page discloses the error and **View analyzed frame** restores the last server-analyzed image and boxes.

The checked-in model can be rebuilt with `python -m pipeline.export_browser_model` after installing the optional ONNX export dependency noted in that module. Model/runtime licensing and origin are documented in `web/models/README.md` and `web/vendor/ort/LICENSE`. Test coverage includes letterbox coordinate mapping, class/confidence filtering, nonmaximum suppression, serial inference, image-hash deduplication, and retry handling.

## Unified browser monitoring

The website now includes one time cursor for the camera explorer, feed cards, detection totals, geographic map, and sampled-traffic chart. **Live** uses the latest available, fully analyzed frame per camera. While a new photo is being processed, the previous photo and its matching boxes remain visible. Keyed updates preserve camera image elements, selection, focus and map popups. New image/overlay pairs are decoded and swapped together, with reduced-motion preferences respected.

The timeline uses local IndexedDB history. Scrub, pick a date/time, or use previous/next and playback to select saved observations; playback steps through the selected camera's frames while every pane follows the same time. Historical views choose the last saved frame **at or before** the cursor, and their charts exclude future observations. A camera with no saved frame at that time does not borrow its current photo. History review pauses new live collection and queued inference; already-running requests can finish and be archived without moving the cursor. **Return to Live** and manual **Refresh** resume current source requests. The selected saved photo can be downloaded.

### Local photo retention and storage

The default is **500 MB of photo data, up to seven days**, as requested. Storage and retention selectors change these limits; oldest observations are removed when either limit is reached. There is no guarantee that 500 MB can hold seven complete days for every camera. Only viewed/selected feeds are collected while the app is open; this is not a background recording service when the browser is closed.

Images are deduplicated by SHA-256, stored once as IndexedDB blobs, and linked to camera observations and their exact-frame detections. Existing analyzed snapshots are also saved when available. Photo history, matching boxes and settings survive ordinary page reloads and site updates. Saved browser detections are reused when their model version still matches, reducing repeated work. A bounded set of decoded/live image URLs stays in memory; old photos are read from disk when reviewing them.

The storage pane reports exact saved photo bytes and the selected budget, photo/frame counts, and the browser's estimated origin usage/quota. The photo budget excludes metadata and HTTP-cached model/runtime files. Browser policy, private mode, user clearing and storage pressure can affect local data; errors are shown without stopping live monitoring. **Clear saved history** deletes this app's saved photos and metadata after confirmation, while leaving preferences and ongoing live monitoring intact. Photos and browser-generated detections are not uploaded to GitHub or another service.

Timeline observation times describe source fetches or known server observations, not guaranteed camera capture times. Repeated identical images update their last-seen metadata without inventing additional distinct photo samples.

### Device power slider

The slider controls browser work, not an operating-system CPU percentage, GPU setting, or power consumption in watts. Detection input size and model accuracy settings stay fixed. These are the current budgets:

| Level | Automatic refresh | Concurrent downloads | Idle gap between detections |
| --- | --- | --- | --- |
| Eco | 5 minutes | 1 | 1.5 seconds |
| Low | 3 minutes | 2 | 0.9 seconds |
| Balanced (default) | 90 seconds | 3 | 0.35 seconds |
| Fast | 60 seconds | 4 | 0.1 seconds |
| Maximum | 30 seconds | 6 | none |

The setting persists on this device. Hidden tabs and historical review pause new scheduled work; active operations finish safely. Manual refresh remains available, subject to the chosen download and inference budgets. The separate GitHub Actions/Python collection service is unaffected by a visitor's browser slider.

Run browser-side unit tests with `npm ci && npm test`; Python tests remain `python -m unittest discover -s tests`. The browser history and transaction tests use `fake-indexeddb`. Real Chrome QA also checks indexed storage across reloads, matching photo/box swaps, stable camera DOM elements, timeline pinning, power settings, and phone layout.

### Cached playback and temporal evidence

Hover a camera card (or focus one of its controls) to loop its last 24 locally saved photos at the shared timeline cursor. Touch users can tap the photo to start/stop. Playback reads IndexedDB, makes no source requests, shows observation timestamps and exact-frame boxes, and restores the current presentation on exit. Frame spacing is irregular; playback is not real-time video. Photos retain the configured 500 MB / seven-day default.

Browser temporal analysis compares three distinct saved frames using native-resolution crop appearance, unique class/size/position matching, and a background-change gate. Moving vectors require consistent displacement over two steps, both no longer than 15 seconds. Persistent position can be detected across gaps up to 30 minutes; this is evidence of occupied position, not proof of continuous identity or a parked vehicle. Byte-distinct photos with identical downsampled pixels do not establish stationary evidence. Camera changes, ambiguous matches and sparse moving tracks remain unresolved. This is a conservative heuristic ensemble, not a calibrated tracking model; no road direction or speed is inferred from image axes. Sparse DriveBC refreshes often cannot supply moving-track evidence.

Temporal evidence, frame hashes, observation windows, state counts and unmatched arrivals/departures are stored with each analyzed photo and pruned with it. The explorer's **Saved motion evidence** panel exports the selected camera's log as JSON; historical exports exclude observations after the shared cursor. Existing heat maps show sampled occupancy, while supported vectors show individual image-space movement.

### Vehicle facing on every car

The **Vehicle facing** widget gives every detected car, truck and bus at least 8×6 source pixels a facing estimate (toward the camera, away from it, or a side view with the nose direction on the image), with a **high / medium / low** confidence tier. On 1,298 sampled DriveBC frames (3,906 vehicles), the browser policy produces an estimate for **99.9%** of vehicles; the previous CFV cascade abstained on **82%** of them (crop-size gates, mirror disagreement and ambiguous scores).

The old model was trained only on street-level CFV car photos and got just **50–53%** of front/rear/side right on elevated DriveBC views. The new model (`models/drivebc-facing-v3.onnx`, MobileNetV3, ~16 MiB) is adapted to DriveBC:

1. 657 hand-labeled DriveBC crops from 342 cameras (`pipeline/data/drivebc-facing-labels.json`, visible side only; unclear crops were left out).
2. A frozen DINOv2-base + linear teacher (79% held-out) soft-labels 3,245 unlabeled DriveBC crops.
3. The student trains on CFV (8 views + angle), the hand labels and the pseudo-labels together.

Camera-grouped 5-fold cross-validation (no camera in both train and test; each fold's teacher sees only that fold's training labels):

| | Front/rear/side accuracy |
| --- | --- |
| Previous CFV model | 50% |
| DriveBC-adapted student (all vehicles) | **75%** |
| High tier (53% of vehicles) | 90% |
| Medium tier (31%) | 64% |
| Low tier (16%) | 44% |

Going from 474 to 657 labels kept overall accuracy about level (74% → 75%) but moved more vehicles into the 90%-accurate high tier (36% → 53%). Labeling more crops is the most direct way to improve further. DINOv2-large did not make a better teacher (78% vs 79–81%).

The tier thresholds are chosen on held-out predictions, so treat low-confidence results as guesses. The crop and its mirror run as one batch-2 WebGPU dispatch (~4 ms per vehicle on the test Mac; WASM fallback). Clipped crops and paired lamps that contradict the model lower the tier by one step. Low-tier facings never steer route links or spatial headings. Facing describes appearance, not measured movement.

Grow the dataset and re-measure:

```bash
python -m pipeline.facing_coverage collect --count 300   # adds DriveBC frames + YOLO boxes to .cache/drivebc-facing
python -m pipeline.facing_coverage coverage              # replays the browser policy on every vehicle
python -m pipeline.train_drivebc_facing --folds 5 --distill   # cross-validation (needs transformers, timm, CFV data)
python -m pipeline.train_drivebc_facing --final --distill && python -m pipeline.export_drivebc_facing
```

### Vehicle routes through time

The **Vehicle routes** widget turns each vehicle's facing plus the camera's look direction into a compass heading. The look direction comes from the published catalog `orientation`, or "looking north" in the caption. It shows two things:

- **Direction of travel through time.** A per-corridor chart of N/E/S/W-bound vehicles per time bin, plus flow arrows on the map. This uses facing alone (medium/high tiers), so it works on every sampled camera.
- **Cross-camera links.** A sighting is linked to one at one of the next two sampled cameras along its heading when: the colour profile and class agree; both ends have a heading; and the implied speed is 35–130 km/h. Speed uses upstream capture time from DriveBC's `Expires` header (capture + 1 s), not fetch time. The link must also be mutual-best with no near-equal rival. White, black and grey vehicles need a closer colour match and never rate "high".

Every refresh, the widget runs a **decoy test**: it repeats the linking with each camera's clock shifted by ±10–20 minutes, so true vehicles no longer line up. Routes are only presented as routes when real links clearly outnumber decoy links; otherwise they are listed as candidates "not distinguishable from chance".

Measured on Hwy 1 in the Lower Mainland (98 cameras, 40 rounds at 45 s, `python -m pipeline.route_survey` then `DECOYS=20 node scripts/route-survey.cjs .cache/route-survey/1.jsonl`): most cameras refresh only every ~10 minutes, and the same vehicle is rarely photographed at two cameras. Over 357 frames and 1,633 sightings (74% with a heading), the linker found **17** links, while 30 decoy runs averaged **13.6** (range 7–21): not distinguishable from chance. Reliable route tracking needs cameras that refresh every few seconds. On DriveBC stills, the direction-of-travel series is the dependable "through time" signal.

### GPU acceleration and road-aligned direction

Detection, road segmentation and vehicle facing select WebGPU automatically where supported, with WASM fallback and visible backend/timing status. Identical crop pixels across refreshed photos reuse existing facing results. Road-supported and selected cars are prioritized. GPU jobs use shorter idle gaps under the same power slider; hidden tabs and history review still pause new work.

A compact SegFormer model predicts road surface on fresh source images, so the road overlay and road-supported map metric work alongside live browser YOLO. General road direction combines road-mask centerlines, supported vehicle corridors and consistent crop-facing estimates. Cyan **Road est.** labels are approximate image directions; green motion vectors still require measured temporal evidence. Ambiguous intersections and conflicting axes abstain. Road results are cached with each photo and restored at the shared timeline cursor. Model provenance, numerical validation, measured GPU/CPU timings and the road model's research/evaluation terms are documented in [the model notes](web/models/README.md#gpu-execution-and-fast-road-context).

### WebGL fallback

When a browser has no working WebGPU, vehicle detection and road segmentation now run on WebGL before falling back to CPU (on the test Mac: ~1.6× and ~2.4× faster than WASM, with matching outputs). Status lines show **GPU / WebGL**. Car facing still uses the CPU in that case. Details, validation and rebuild steps: [WebGL fallback](web/models/README.md#webgl-fallback-browsers-without-webgpu).

## Multi-car GPU inference batches

> **Facing model v3:** the slider still sets how many cars share one facing dispatch. The DriveBC-adapted model takes a dynamic batch of crop/mirror pairs, so the static 2/4/8-car cascade graphs described below were removed.


The **GPU inference batching** slider under Device power budget selects **Off,
2, 4, or 8 cars** per inference call. Four cars is the default. This controls real
WebGPU batches in the vehicle-facing cascade: every car retains its original and
mirrored 224×224 views, and both the compact screening stage and full-resolution
checker process multiple cars together. YOLO and road-scene inference keep their
existing frame-at-a-time execution. The power slider remains independent.

Higher settings improve sustained throughput at the cost of more GPU memory.
Partial batches run immediately, with unused slots padded, using the selected
static shape so short queues do not trigger graph recompilation. A batch of
eight needs at least 9.2 MiB of GPU input buffers; intermediate activations,
weights and runtime buffers require substantially more. Changing the setting
releases the previous batch sessions. An unsupported batch or memory/device
failure returns to individual inference, with a reason in status; fixed batch
tensors are never passed to an incompatible CPU graph. CPU-only browsers use
individual inference. Selected crops keep priority and all results retain their
exact frame, crop fingerprint and detection index.

Static graphs use ORT graph optimizations, preallocated GPU input/output buffers,
and command capture/replay when supported. The compact individual path now also
uses capture. Batches share the existing checksummed weights rather than
redownloading a model for each setting: all nine new shape graphs total about
1.7 MiB, and one setting normally caches about 0.3 MiB for the selected precision
and compact model. Model-cache storage is separate from the photo-history limit.
The FP16 and compact graphs reference tensor byte ranges in the original ONNX
files; the FP32 graphs reuse the original external weight shards.

Local **GTX 1080 Ti**, real Chromium/Vulkan WebGPU, FP32 (no `shader-f16`), with
14 eligible, unlabeled DriveBC car crops:

| Mode | Median ms / car | Cars / second |
| --- | ---: | ---: |
| Before changes, individual | 32.2 | 31.1 |
| Optimized individual | 28.7 | 34.8 |
| 2-car GPU batch | 25.7 | 39.0 |
| 4-car GPU batch | 24.4 | 41.1 |
| 8-car GPU batch | 23.8 | 42.1 |

Eight-car throughput improved about **35%** over the previous individual path.
Each measurement uses three warmups and fifteen timed calls on the first N crops;
separate equivalence checks cover all fourteen crops, including partial batches.
Every batch retained identical view scores and angles on this GPU. All nine
exports also matched their original graphs on independent deterministic inputs
using native ORT CPU, including FP16 exports. These checks establish export
fidelity, not accuracy. Warm timings exclude downloads, model initialization
(about 2–3 seconds in this run), image decoding and crop preparation. Hardware,
drivers, queue occupancy and other workloads affect speed; these measurements
are not a universal guarantee of optimal kernels. Full results and crop
provenance: [benchmark report](docs/benchmarks/2026-10-20-gtx1080ti.json).

Reproduce browser timing (Node 20+ and a working hardware WebGPU adapter):

```sh
npm ci
npx playwright install chromium
npm run benchmark:gpu
# Use an installed Chrome/Chromium instead:
GPU_BENCH_CHROME=/path/to/chrome npm run benchmark:gpu
# Optionally supply N crops as packed float32 RGB CHW, each 3×224×224:
npm run benchmark:gpu -- --crops=/path/to/crops.f32
```

The default benchmark uses deterministic synthetic inputs and explicitly reports
this. It rejects CPU fallback and software GPU adapters, reports warm throughput
and cold latency, and checks every input against individual inference. To compare
an earlier worker, pass `--baseline-worker=/path/to/original-worker.js` alongside
`--crops`. On Linux the benchmark requests Vulkan WebGPU; the interactive app
uses the browser's normal adapter selection.

Rebuild/validate the batch graphs without downloading training checkpoints:

```sh
python -m pip install onnx onnxruntime numpy
python -m pipeline.export_gpu_batches
python scripts/validate-gpu-batches.py
npm test
```

## Complete facing coverage and spatial direction evidence

> **Superseded for facing by v3** (see *Vehicle facing on every car*). Only boxes under 8×6 pixels abstain now; clipped crops get a lower confidence tier instead of abstaining. The spatial-direction material below still applies.


The browser now runs the facing model on **every valid YOLO car box** on viewed
feeds. Tiny and frame-edge crops keep their numerical diagnostics but still
abstain from direction estimates, preserving the existing quality thresholds.
Unsupported vehicle classes remain unresolved. Status distinguishes model runs,
quality abstentions, queued work and errors; **Retry failed facing jobs** retries
execution errors. Existing cached pre-inference abstentions are refreshed under
the new facing version.

Temporal results now add motion only to matching detection fingerprints. They
cannot replace, remove or reorder the current YOLO boxes. Facing results are
saved by camera, photo hash and exact crop fingerprint, including browser YOLO
layouts that differ from a published layout. Late temporal writes are rejected
when their detector layout is stale.

The camera explorer has a **Show all vehicle direction vectors** overlay and
**Draw direction references** control. Drag from an arrow's tail to its head to
place several spatial references, or enter normalized coordinates under
**Reference arrows and coordinates**. Up to 64 references can be retained per
camera. Undo or clear annotations with the adjacent buttons.

A conservative **rule-based meta-classifier** combines mirror-verified facing,
road geometry, validated temporal motion and nearby user references. References
refine the local axis only when the trained facing evidence agrees. Crossings,
opposing arrows and road/facing conflicts abstain. References cannot supply a
missing vehicle identity or turn an uncertain crop into measured movement.
Annotations transfer between frames only after aspect-ratio and background
stability checks. Cyan vectors are appearance-based facing estimates; green
vectors require validated displacement. These are image-space vectors, not
compass bearings, metric speed or a statistically calibrated traffic model.

Photos, detector/facing diagnostics, road context, temporal evidence and fused
heading records are retained in **IndexedDB**, attached to the exact saved frame.
Camera reference arrows also persist there. Model weights use the separate,
checksummed Cache Storage cache. No inference, photo or annotation data is
uploaded to a storage server. The existing photo size/age limits still prune
photos and frame records; annotations are cleared separately. **Keep data on
this device** requests persistent storage from the browser; the browser may
refuse, and clearing site data still removes everything.

An installed service worker caches the static app shell and runtime files, so
saved photos, annotations and results can be reopened offline after the initial
visit. Camera photos are deliberately excluded from that shell cache so they
remain subject to the IndexedDB retention limits. If published JSON is
unavailable, the dashboard can restore a view from locally retained frames. A
static HTTPS host (or localhost) is needed for the first visit; no running
storage backend is needed for retention or later offline review. Python server
SSE is now opt-in with `?serverEvents=1`; ordinary static hosting makes no API
connection.

Additional local efficiency measurements on the GTX 1080 Ti:

| Work | Previous median | Optimized median |
| --- | ---: | ---: |
| Prepare 8 real car crops | 26.7 ms | 10.3 ms |
| YOLO GPU worker call | 19.8 ms | 17.2 ms |
| Road GPU worker call | 13.4 ms | 8.8 ms |

Crop preparation reuses a bounded photo-decode cache and canvases; the benchmark
needed one decode instead of 240 across 30 groups, with identical crop pixel
hashes. Batched inference reuses its packing buffer and writes mirrored pixels
directly. Batch mode no longer initializes an unused individual compact session.
YOLO and road models now capture and replay their static GPU command graphs.
Device-loss observers release old sessions when batch settings change, avoiding
retention of obsolete batch buffers. Facing weights, model scores, resolution
and confidence/mirror gates remain unchanged. Warm facing kernels are still
compute-bound; these changes improve surrounding work rather than promising a
faster ConvNeXt kernel. See the [efficiency report](docs/benchmarks/2026-10-20-spatial-efficiency.json).

Run the browser regression with real hardware WebGPU:

```sh
npm ci
npx playwright install chromium
npm run test:browser
# Or select an installed browser:
GPU_BENCH_CHROME=/path/to/chrome npm run test:browser
```

It checks full model coverage, low-quality abstentions, stale temporal layouts,
persistent annotations, a narrow viewport, and an offline reload **after stopping
the fixture's static host**. The fixture is synthetic; numerical model fidelity
is separately checked against the real crops above. Spatial fusion does not
have a labeled DriveBC accuracy benchmark.

## Facing across cached screenshots

> **Legacy:** two-screenshot confirmation required matching screenshots within 15 s, which left most live vehicles unresolved. It now applies only to archived results from the retired CFV cascade. v3 estimates show a confidence tier from a single photo instead.


Facing confirmation now requires **at least two distinct screenshots from the
same camera**, observed no more than 15 seconds apart. Every valid YOLO car crop
still runs through the GPU cascade, including tiny/clipped crops that retain
raw diagnostics but abstain. Crop-plus-mirror verification and two-screenshot
confirmation are separate checks. Up to five cached screenshots check agreement;
ambiguous associations, changed views, duplicate decoded pixels, missing evidence
and conflicting facing predictions leave direction unresolved. This is
appearance association, not a guaranteed vehicle identity or measured movement.

While viewing live source images with facing enabled, a bounded scheduler requests
a follow-up after five seconds and retries once if the upstream photo is unchanged
or unavailable. Sparse upstream images may never provide a valid pair. History
review and hidden tabs pause scheduled work. Observation timestamps do not establish
camera capture times. The browser retains appearance features, raw model evidence,
and separate paired results in IndexedDB. Pruning a supporting frame invalidates
its derived confirmation. Offline reopening reuses local evidence without a
storage server; retention limits and browser eviction still apply.

The paired residual head was actually trained on 2,050 labeled CFV pair conditions
(1,025 distinct pairs evaluated clean and at 40 pixels), using 4,080 distinct
crop-condition inputs through the production WebGPU cascade. Train/validation/test
identities are disjoint: 42/10/14. These are nearby labeled views of the same car,
**not verified same-camera temporal traffic sequences**. No browser predictions
are used as training labels. More unlabeled cache improves consistency checks;
it does not demonstrate continual supervised accuracy improvement.

Held-out eight-view results (224 examples per condition):

| Input | Latest frame | Mean of two | Learned raw head | Shipping confirmation |
| --- | ---: | ---: | ---: | --- |
| Clean | 97.77% | 98.66% | 98.66% | 215 accepted, 1 wrong (95.98% coverage) |
| 40px | 94.64% | 95.98% | 96.43% | 180 accepted, 0 wrong (80.36% coverage) |

The shipping gate preserves the latest verified class and matches the simpler
verified-two-frame baseline on validation/test. It establishes **no additional
production accuracy gain from the learned gate**. Class boundaries are
underrepresented; the two degradation conditions are correlated. The official
test identities were previously evaluated for the upstream cascade. DriveBC
sequence accuracy remains unmeasured. An independently supervising agent reproduced
the shipped JavaScript predictions and reviewed cache/association failure cases.
Full provenance, selection policy, coverage and limitations are in
[paired-facing-training-v1.json](web/models/paired-facing-training-v1.json).

Reproduce after obtaining the pinned CFV `train.zip` and `test.zip` archives in
`.cache/cfv-data` (revision and required checksums are enforced by preparation):

```sh
.venv/bin/python -m pipeline.prepare_paired_facing
GPU_BENCH_CHROME=/path/to/chrome node scripts/extract-paired-facing.cjs
.venv/bin/python -m pipeline.train_paired_facing
npm test
GPU_BENCH_CHROME=/path/to/chrome npm run test:browser
GPU_BENCH_CHROME=/path/to/chrome npm run benchmark:gpu
```

The latest local full-resolution WebGPU benchmark on the Pascal GPU measured
34.60 cars/s at batch 1 and 43.15 cars/s at batch 8 (warm synthetic crops,
15 samples, GPU command capture enabled). All eight views matched across sizes,
with zero score/angle differences. Cold initialization took about 1.9–2.4 seconds;
these are local measurements, not a performance guarantee for other devices.

## Browser-only publishing and automatic local retries

GitHub Pages publishes the app, model assets and the public camera metadata
catalog. The workflow does **not** run the photo collector, save camera-image
Actions caches, or publish camera frames. `pipeline.publish_catalog` downloads
only the catalog CSV; `scripts/stage-site.py` builds an exact allowlist and rejects
photo evidence, inline images, symlinks and unexpected assets. All camera photos
come directly from DriveBC into the end user's browser. Browser profiles, E2E
screenshots, videos and traces are never uploaded by CI.

The browser keeps checksummed facing weights in Cache Storage and photo bytes,
raw facing results and paired evidence in IndexedDB. A low-priority local queue
finds missing/current-version facing jobs throughout retained history, not just
visible feeds. Transient crop/model errors retry after 2 and 10 seconds, with
attempts retained across reloads; exhausted jobs remain visible and the retry
button resets them. Backfill reads only locally saved photo bytes and never
fetches a remote replacement photo. It pauses during history review, when facing
is disabled, and in hidden tabs. Unknown facing and quality abstentions are
completed results, not errors to loop on.

The **Core model updates** panel shows queued downloads, checksum verification,
local saves, cache hits and failures. Web Locks serialize these cache updates
across model workers and browser tabs; browsers without Web Locks report that
only per-worker serialization is available. This does not serialize GPU batches.
A cached file with the wrong checksum is ignored and replaced only after a
checksum-verified download. Downloads use the expected checksum in their URL to
avoid stale HTTP/CDN copies; a failed update leaves the previous cached bytes intact.

On pushes and pull requests, GitHub Actions runs the real facing-model browser
E2E using WASM on its CPU runner, plus unit and privacy tests. Deployment waits
for those checks. A second browser E2E checks the deployed GitHub Pages code,
catalog privacy contract, local facing persistence, two-screenshot confirmation,
offline reload, an injected transient failure on a nonvisible retained photo,
local automatic retry, and absence of uploads. Deterministic synthetic photo and
detection fixtures keep CI independent of live camera availability; the facing
model itself runs normally. Core releases/deployments are sequential. These
checks enforce the contract on tested releases; browser quota, eviction, private
mode and user-cleared storage can still remove local caches.

The legacy `pipeline.collect` command remains a **local-only** analysis utility;
it is not invoked or staged by the Pages publishing workflow. The optional Python analysis sections describe local research utilities.


## Local 1,000-image scale verification

The verification cohort contains **1,000 distinct original JPEGs**, SHA-256 checked
against the pinned official CFV test archive. All 14 test identities are balanced
at 71–72 photos each, covering all 72 five-degree angle bins. Preparation performs
no downloads and writes only to `.cache/facing-verification`, outside the site.

```sh
.venv/bin/python -m pipeline.prepare_facing_verification
GPU_BENCH_CHROME=/path/to/chrome npm run verify:1000
```

The command verifies every native photo in IndexedDB, runs the production facing
cascade through the local retry queue, compares all 1,000 outputs with individual
GPU inference, reopens offline, retries ten missed jobs (including three injected
transient failures), renders a 1,000-entry dashboard, and tests retention pruning.
Photos, prediction files and the report stay local. No verification photos,
browser profiles or traces are staged for Pages or uploaded to CI.

Measured locally on the Pascal WebGPU adapter:

| Check | Result |
| --- | --- |
| Native photos retained | 1,000; 85,308,516 photo bytes |
| Timed GPU batching | 20.065 s; 49.84 images/s |
| Timed individual GPU reference | 23.344 s; 42.84 images/s |
| Original/mirror probability or angle differences | Zero across all 1,000 |
| Eight-view / front-rear-side accuracy | 91.6% / 96.2% |
| Single-frame acceptance | 870/1,000; group accuracy 850/870 (97.70%) |
| Offline cache | 1,000 results; zero inference workers needed |
| Offline missed-job retry | 10/10 recovered, including 3 transient failures |
| Dashboard | 1,000 entries; 24 cards; initial render ~204 ms |
| Decoded bitmap peak | 2.81 MiB; transient 5th image before 4-frame cache eviction |
| Observed main renderer JS heap peak | ~39.4 MiB; excludes native/GPU allocations |
| Retention pruning | 497 valid photo/result pairs retained under half-size limit |

Timing excludes three full-path synthetic warmups per pass; their startup cost
was 3.23 s batched and 2.05 s individual. Timed runs include local reads, decode,
crop preparation, queue handling and result persistence in both passes.
The scalar evidence reset occurs outside the timed phase.
The 125 eight-car worker requests contain 125 compact-stage calls and 310 two-car
checker calls; they are **not** 125 total GPU executions. Small checker
microbatches avoid padding every difficult crop to eight cars. Cached backfill
loads at most eight local photos at once and completion wakes preempt polling
idle time. Retained result caches remain capped at 512 and exact-crop cache at 256.

CFV supplies the crop boxes, so this does not measure YOLO recall. These are
all-angle stills, not verified traffic-camera sequences; synthetic cache IDs and
timestamps test storage/UI scale only. The test identities were previously
evaluated upstream. No confidence, mirror or lamp thresholds were retuned.
The broader all-angle accuracy is not comparable to the earlier class-center
paired-view benchmark. One hardware run verifies this scale, not arbitrary data
sizes or all devices. The complete local report is
`.cache/facing-verification/report.json`; raw predictions remain alongside it.
When a browser has no working WebGPU, vehicle detection and road segmentation now run on WebGL before falling back to CPU (on the test Mac: ~1.6× and ~2.4× faster than WASM, with matching outputs). Status lines show **GPU / WebGL**. Vehicle facing uses the CPU in that case (~16 MiB model). Details, validation and rebuild steps: [WebGL fallback](web/models/README.md#webgl-fallback-browsers-without-webgpu).

const $ = id => document.getElementById(id);
const vehicleLabels = ['car', 'truck', 'bus', 'motorcycle', 'bicycle'];
const count = c => vehicleLabels.reduce((n, k) => n + (c.counts?.[k] || 0), 0);
const vehicles = c => (c.detections || []).filter(d => vehicleLabels.includes(d.label));
const motionState = d => d.motion?.state === 'moving' ? (validVector(d) ? 'moving' : 'unknown') : d.motion?.state || 'unknown';
const signature = value => JSON.stringify(value);
let data = null, patterns = null, limit = 24, selectedCamera = '', refreshing = false;
let map = null, heatLayer = null, markerLayer = null;
let selectedVehicle = null, selectedVehicleFrame = '';
const paneVersions = new Map(), updateLog = [];
let announcementTimer;
const cameraSources = new CameraSourceCache();
let sourceLoading = false, sourceJob = Promise.resolve([]);
function sourceEntry(c) { return $('source-view').checked ? cameraSources.get(c.id) : null; }
function unmatchedSource(c) { const source = sourceEntry(c); return !!source?.url && source.hash !== c.hash; }
function sourceLabel(c) {
  const source = sourceEntry(c);
  if (!$('source-view').checked) return 'Analyzed frame · ' + age(c);
  if (source?.url) return (source.error ? 'Source refresh failed · last good source image' : 'DriveBC checked ' + new Date(source.checked).toLocaleTimeString()) + (source.modified ? ' · upstream ' + new Date(source.modified).toLocaleString() : '') + (unmatchedSource(c) ? ' · no matching analysis' : ' · matching analysis');
  return source?.error ? 'Source unavailable · cached analyzed frame retained' : 'Cached analyzed frame · fetching DriveBC image…';
}
function visibleSourceCameras() {
  const rows = filteredCameras().sort($('sort').value === 'name' ? (a,b) => a.name.localeCompare(b.name) : $('sort').value === 'fresh' ? (a,b) => Date.parse(b.observed)-Date.parse(a.observed) : (a,b) => count(b)-count(a));
  const selected = data?.cameras.find(c => c.id === selectedCamera);
  return selected ? [...rows.slice(0,limit), selected] : rows.slice(0,limit);
}
async function loadVisibleSources(force = false) {
  if (!data || !$('source-view').checked) return [];
  if (sourceLoading) { await sourceJob; return force ? loadVisibleSources(true) : []; }
  sourceLoading = true;
  sourceJob = cameraSources.refresh(visibleSourceCameras(), force);
  try { return await sourceJob; }
  finally { sourceLoading = false; render(); }
}
function scheduleSources() {
  if (!refreshing && !sourceLoading && $('source-view').checked && visibleSourceCameras().some(c => !cameraSources.get(c.id))) {
    queueMicrotask(() => { if (!sourceLoading) loadVisibleSources(); });
  }
}
function node(tag, text, cls) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function link(text, url) {
  const a = node('a', text); a.href = url; a.target = '_blank'; a.rel = 'noreferrer'; return a;
}
function notifyPane(key, label, version, badge, pane, sourceTime) {
  const serialized = signature(version), prior = paneVersions.get(key);
  if (prior?.version === serialized) {
    if (badge) badge.textContent = prior.text;
    return;
  }
  const clock = new Date().toLocaleTimeString();
  const text = `${prior ? 'Updated' : 'Loaded'} ${clock}`;
  paneVersions.set(key, { version: serialized, text });
  if (badge) {
    badge.textContent = text; badge.title = sourceTime ? `Source published ${new Date(sourceTime).toLocaleString()}` : 'Display last changed at this time';
    badge.classList.add('fresh'); setTimeout(() => badge.classList.remove('fresh'), 3000);
  }
  if (pane && prior) {
    pane.classList.remove('pane-flash'); void pane.offsetWidth; pane.classList.add('pane-flash');
  }
  updateLog.unshift(`${clock} · ${label} ${prior ? 'updated' : 'loaded'}`);
  updateLog.splice(12);
  $('update-log').replaceChildren(...updateLog.map(t => node('li', t)));
  clearTimeout(announcementTimer);
  announcementTimer = setTimeout(() => { $('pane-announcer').textContent = 'Widget updates: ' + updateLog.slice(0, 6).join('. '); }, 250);
}
function filteredCameras() {
  if (!data) return [];
  const search = $('search').value.toLowerCase();
  return data.cameras.filter(c => (!$('region').value || c.region === $('region').value) &&
    `${c.name} ${c.highway} ${c.caption}`.toLowerCase().includes(search));
}
function age(c) {
  const minutes = Math.max(0, Math.floor((Date.now() - Date.parse(c.observed)) / 60000));
  return `${minutes < 60 ? minutes + 'm' : Math.floor(minutes / 60) + 'h'} ago${minutes > 60 ? ' · stale' : ''}`;
}
function motionCounts(rows) {
  const states = { moving: 0, stationary: 0, unknown: 0 };
  for (const c of rows) for (const d of vehicles(c)) states[motionState(d)]++;
  return states;
}
function drawPicture(c, detail = false) {
  const picture = node('div', undefined, 'picture'), image = node('img');
  image.loading = detail ? 'eager' : 'lazy'; image.alt = c.caption;
  const source = sourceEntry(c), mismatched = unmatchedSource(c);
  image.src = source?.url || `${c.frame}?v=${c.hash.slice(0, 12)}`;
  image.onerror = () => { image.alt = 'Cached image unavailable'; };
  image.onload = () => {
    if (image.naturalWidth && image.naturalHeight) {
      picture.style.aspectRatio = `${image.naturalWidth} / ${image.naturalHeight}`;
      if (!mismatched) renderVectorLayer(picture, c, detail, image.naturalWidth, image.naturalHeight, selectedVehicle);
    }
  };
  picture.append(image, node('span', `HWY ${c.highway}`, 'tag'), node('span', source?.url ? (mismatched ? 'SOURCE · NOT ANALYZED' : 'SOURCE · ANALYSIS MATCHED') : 'ANALYZED FRAME', 'frame-kind'));
  if (!mismatched && detail && ($('heat-overlay').checked || $('road-overlay').checked)) {
    const canvas = node('canvas'); canvas.width = 640; canvas.height = 400;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      if ($('road-overlay').checked && c.analysis?.road_mask) {
        const grid = c.analysis.road_mask;
        for (let y = 0; y < grid.length; y++) for (let x = 0; x < grid[y].length; x++) {
          if (grid[y][x] < .6) continue;
          ctx.fillStyle = `rgba(72,180,244,${grid[y][x] * .4})`;
          ctx.fillRect(x * 640 / grid[y].length, y * 400 / grid.length, 640 / grid[y].length + 1, 400 / grid.length + 1);
        }
      }
      if ($('heat-overlay').checked) {
        const grid = patterns?.cameras?.[c.id]?.occupancy_grid || c.analysis?.occupancy_grid;
        const max = grid ? Math.max(...grid.flat(), .001) : 1;
        if (grid) for (let y = 0; y < grid.length; y++) for (let x = 0; x < grid[y].length; x++) {
          if (!grid[y][x]) continue;
          const px = (x + .5) * 640 / grid[y].length, py = (y + .5) * 400 / grid.length;
          const gradient = ctx.createRadialGradient(px, py, 1, px, py, 28);
          gradient.addColorStop(0, `rgba(255,105,45,${Math.min(.75, grid[y][x] / max * .75)})`);
          gradient.addColorStop(.5, 'rgba(248,218,87,.25)'); gradient.addColorStop(1, 'rgba(248,218,87,0)');
          ctx.fillStyle = gradient; ctx.fillRect(px - 28, py - 28, 56, 56);
        }
      }
    }
    picture.append(canvas);
  }
  if (!mismatched && ($('boxes').checked || detail)) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 1000 1000'); svg.setAttribute('preserveAspectRatio', 'none');
    for (const [i, d] of (c.detections || []).entries()) {
      const [x, y, xx, yy] = d.box, state = motionState(d);
      const focused = !detail || selectedVehicle == null || selectedVehicle === i;
      const color = state === 'moving' ? '#b8ed86' : state === 'stationary' ? '#69b9ff' : '#efbd7b';
      const r = document.createElementNS(svg.namespaceURI, 'rect');
      for (const [k, v] of Object.entries({ x: x * 1000, y: y * 1000, width: (xx - x) * 1000, height: (yy - y) * 1000, fill: 'none', stroke: color, 'stroke-width': 2 })) r.setAttribute(k, v);
      r.setAttribute('vector-effect', 'non-scaling-stroke');
      r.setAttribute('stroke-width', detail ? 2 : 1.5);
      r.setAttribute('opacity', focused ? 1 : .18);
      if (state === 'unknown') r.setAttribute('stroke-dasharray', '4 4');
      const title = document.createElementNS(svg.namespaceURI, 'title');
      title.textContent = `#${i + 1} ${d.label} ${Math.round(d.confidence * 100)}% · ${state}`; r.append(title); svg.append(r);
      if (detail) {
        const label = document.createElementNS(svg.namespaceURI, 'text');
        label.setAttribute('x', x * 1000); label.setAttribute('y', Math.max(24, y * 1000 - 8));
        label.setAttribute('fill', color); label.setAttribute('font-size', '30');
        label.setAttribute('stroke', '#071014'); label.setAttribute('stroke-width', '6'); label.setAttribute('paint-order', 'stroke'); label.setAttribute('opacity', focused ? 1 : .18); label.textContent = `#${i + 1}`; svg.append(label);
      }

    }
    picture.append(svg);
  }
  if (!mismatched) renderVectorLayer(picture, c, detail, 640, 400, selectedVehicle);
  return picture;
}
function selectCamera(id) { if (id !== selectedCamera) selectedVehicle = null; selectedCamera = id; $('camera-select').value = id; renderDetail(); scheduleSources(); }
function renderDetail() {
  const c = data?.cameras.find(c => c.id === selectedCamera);
  if (!c) { $('detail-title').textContent = 'Select a camera'; $('detail-caption').textContent = 'No camera available for this filter.'; $('detail-picture').replaceChildren(); $('vehicle-list').replaceChildren(); $('motion-explanation').textContent = ''; $('show-all-vectors').hidden = true; notifyPane('detail', 'Camera explorer', null, $('detail-updated'), $('detail-pane')); return; }
  if (selectedVehicle != null && (selectedVehicleFrame !== c.hash || selectedVehicle >= c.detections.length)) selectedVehicle = null;
  $('detail-title').textContent = c.name;
  $('detail-picture').replaceWith(Object.assign(drawPicture(c, true), { id: 'detail-picture' }));
  const sample = patterns?.cameras?.[c.id], analysis = c.analysis;
  const measured = vehicles(c).filter(d => validVector(d)).length;
  $('motion-explanation').textContent = unmatchedSource(c) ? 'New DriveBC image loaded. No matching analysis has been published; older boxes, motion vectors and heat are hidden. Choose View analyzed frame to inspect the last matching results.' : motionExplanation(c, measured) + (selectedVehicle != null ? ` Only vehicle #${selectedVehicle + 1} is highlighted: ${motionDescription(c.detections[selectedVehicle])}.` : '');
  $('motion-explanation').classList.toggle('has-vectors', measured > 0 && !unmatchedSource(c));
  $('show-all-vectors').hidden = selectedVehicle == null;
  const gap = analysis?.motion?.gap_seconds;
  const cadence = gap == null ? 'No trustworthy short-gap frame pair.' : `Frame gap ${gap.toFixed(2)}s; motion status: ${analysis.motion.status.replaceAll('_', ' ')}.`;
  $('detail-caption').textContent = `${sourceLabel(c)}. Cached analysis: ${sample?.samples || 1} distinct samples · ${age(c)}. ${cadence} ${analysis?.road_model ? 'Blue overlay: predicted road surface.' : 'Road model unavailable; road membership unknown.'} Heat shows relative visible-vehicle locations, not speed.${c.media_note ? ' ' + c.media_note : ''}`;
  const list = vehicles(c).map(d => {
    const i = c.detections.indexOf(d), row = node('button', undefined, 'vehicle-row'), road = d.on_road === true ? 'road-supported' : d.on_road === false ? 'road not supported' : 'road unknown';
    row.type = 'button'; row.setAttribute('aria-pressed', String(selectedVehicle === i));
    row.onclick = () => { selectedVehicle = selectedVehicle === i ? null : i; selectedVehicleFrame = c.hash; renderDetail(); };
    row.append(node('span', `#${i + 1} ${d.label} · ${Math.round(d.confidence * 100)}%`), node('span', `${motionDescription(d)} · ${road}`));
    return row;
  });
  $('vehicle-list').replaceChildren(...(unmatchedSource(c) ? [node('p', 'Vehicle measurements belong to the previous analyzed frame. Select View analyzed frame to see them.', 'widget-note')] : list.length ? list : [node('p', 'No vehicles detected.', 'widget-note')]));
  notifyPane('detail', 'Camera explorer', [c.id, c.hash, c.analysis, c.detections, sample, selectedVehicle, $('vector-overlay').checked, $('vector-scale').value, $('heat-overlay').checked, $('road-overlay').checked, sourceEntry(c)?.hash, sourceEntry(c)?.error], $('detail-updated'), $('detail-pane'), data.generated);
}
function initMap() {
  if (!window.L || map) return;
  map = L.map('traffic-map', { scrollWheelZoom: false }).setView([53.1, -124.2], 5);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' }).addTo(map);
  markerLayer = L.layerGroup().addTo(map);
  if (L.heatLayer) heatLayer = L.heatLayer([], { radius: 28, blur: 22, maxZoom: 10, minOpacity: .2, gradient: { .2: '#467bd0', .4: '#49c7ad', .6: '#e7e775', .8: '#f19a4b', 1: '#ed5252' } }).addTo(map);
}
function renderMap(rows) {
  initMap();
  const fresh = $('fresh-only').checked, metric = $('map-metric').value, now = Date.now();
  const samples = rows.filter(c => !fresh || now - Date.parse(c.observed) < 3600000).map(c => {
    const value = metric === 'road' ? (c.analysis?.road_model ? c.analysis.on_road : null) : metric === 'history' ? patterns?.cameras?.[c.id]?.mean_vehicles : metric === 'moving' ? motionCounts([c]).moving : count(c);
    return { c, value };
  }).filter(s => s.value != null);
  if (map) {
    markerLayer.clearLayers(); const max = Math.max(1, ...samples.map(s => s.value));
    heatLayer?.setLatLngs(samples.filter(s => s.value > 0).map(({ c, value }) => [c.lat, c.lon, value / max]));
    for (const { c, value } of samples) {
      const content = node('div'); content.append(node('strong', c.name), node('p', `${value} ${metric === 'road' ? 'road-supported vehicles' : metric === 'moving' ? 'motion-supported' : metric === 'history' ? 'average visible vehicles' : 'visible vehicles'} · ${age(c)}`));
      const button = node('button', 'Inspect camera'); button.onclick = () => { selectCamera(c.id); $('detail-pane').scrollIntoView({ behavior: 'smooth', block: 'center' }); }; content.append(button);
      L.circleMarker([c.lat, c.lon], { radius: 4, color: value ? '#b8ed86' : '#92a2ac', weight: 1, fillOpacity: .7 }).bindPopup(content).addTo(markerLayer);
    }
  } else {
    $('traffic-map').hidden = true; $('map-fallback').hidden = false;
    $('map-fallback').replaceChildren(node('p', 'Map library unavailable. Camera locations:', 'widget-note'), ...samples.slice(0, 20).map(({ c, value }) => {
      const button = node('button', `${c.name} (${c.lat}, ${c.lon}) · ${value}`); button.onclick = () => selectCamera(c.id); return button;
    }));
  }
  const stale = samples.filter(({ c }) => now - Date.parse(c.observed) > 3600000).length;
  $('map-caption').textContent = `${samples.length} sampled cameras · ${stale} over 60 minutes old. ${metric === 'moving' ? 'Only motion-supported detections contribute; an empty heat layer means no qualifying measurements.' : 'Relative visible-vehicle density around camera locations. Brightness depends on camera coverage and field of view, not measured traffic flow.'}`;
  notifyPane('map', 'Traffic heatmap', [metric, fresh, samples.map(({ c, value }) => [c.id, c.hash, value])], $('map-updated'), $('map-pane'), data.generated);
}
function renderTrend(rows) {
  const hours = new Map((patterns?.hourly || []).map(h => [h.hour, h]));
  const bins = [], end = new Date(patterns?.generated || Date.now()); end.setUTCMinutes(0, 0, 0);
  for (let i = 23; i >= 0; i--) { const time = new Date(end.getTime() - i * 3600000), key = time.toISOString().replace('.000Z', 'Z'); bins.push({ time, sample: hours.get(key) }); }
  const max = Math.max(1, ...bins.map(b => b.sample?.mean_vehicles || 0));
  $('hourly-chart').replaceChildren(...bins.map(({ time, sample }) => {
    const bin = node('div', undefined, 'hour-bin'), bar = node('div', undefined, 'bar' + (sample ? '' : ' no-sample'));
    bar.style.height = sample ? `${Math.max(2, sample.mean_vehicles / max * 140)}px` : '2px';
    bin.title = `${time.toISOString().slice(11, 13)}:00 UTC · ${sample ? sample.mean_vehicles + ' vehicles/frame · ' + sample.cameras + ' cameras · ' + sample.samples + ' samples' : 'No samples'}`;
    bin.append(bar, node('span', time.toISOString().slice(11, 13))); return bin;
  }));
  const measured = bins.filter(b => b.sample).length;
  $('trend-caption').textContent = `${measured}/24 hours sampled across the whole network. Hourly mean visible vehicles per frame, equally weighted by camera. Hover bars for camera/sample counts; gaps mean no samples. The camera set can vary by hour. Search and region filters apply to the map and feeds; this chart covers the full network.`;
  notifyPane('trend', '24h traffic pattern', patterns?.hourly || [], $('trend-updated'), $('trend-pane'), patterns?.generated);
}
function renderFeeds(rows) {
  rows = [...rows].sort($('sort').value === 'name' ? (a, b) => a.name.localeCompare(b.name) : $('sort').value === 'fresh' ? (a, b) => Date.parse(b.observed) - Date.parse(a.observed) : (a, b) => count(b) - count(a));
  $('shown').textContent = `${rows.length} available`; $('feeds').replaceChildren();
  for (const c of rows.slice(0, limit)) {
    const card = node('article', undefined, 'card'), details = node('div', undefined, 'details'), h = node('h3'); h.append(link(c.name, c.source)); details.append(h);
    const meta = node('div', undefined, 'meta'); meta.append(node('span', c.region), node('span', `${count(c)} vehicles in cached analysis`)); details.append(meta);
    const counts = node('div', undefined, 'counts'); for (const [label, n] of Object.entries(c.counts || {})) counts.append(node('span', `${n} ${label}`)); if (!counts.childNodes.length) counts.append(node('span', 'No objects detected')); details.append(counts);
    const states = motionCounts([c]); details.append(node('div', `Cached motion: ${states.moving} moving · ${states.stationary} stationary · ${states.unknown} unknown`, 'motion-tag'));
    const sources = node('div', undefined, 'camera-sources'); for (const [text, url] of [['DriveBC camera', c.source], ['Original image', c.image], ['Cached frame', c.frame]]) sources.append(link(text + ' ↗', url)); details.append(sources);
    details.append(node('div', sourceLabel(c), 'source-status'));
    details.append(node('div', `${c.error ? 'Analysis fetch failed · ' : ''}Analyzed frame ${age(c)}`, 'age'));
    const inspect = node('button', 'Inspect motion & heat', 'inspect-button'); inspect.onclick = () => { selectCamera(c.id); $('detail-pane').scrollIntoView({ behavior: 'smooth', block: 'center' }); }; details.append(inspect);
    const badge = node('span', undefined, 'update-badge'); details.append(badge); card.append(drawPicture(c), details); $('feeds').append(card);
    notifyPane(`camera-${c.id}`, c.name, [c.hash, c.detections, c.error, sourceEntry(c)?.hash, sourceEntry(c)?.error], badge, card, c.observed);
  }
  $('more').hidden = rows.length <= limit;
  $('message').textContent = rows.length ? 'Refresh requests visible images from DriveBC. Counts, maps and history describe cached analysis; overlays are hidden on unmatched source images.' : 'No matching analyzed cameras.';
  notifyPane('feeds', 'Camera feeds', [rows.slice(0, limit).map(c => [c.id, c.hash, c.detections, c.error, sourceEntry(c)?.hash, sourceEntry(c)?.error]), $('boxes').checked, $('source-view').checked], $('feeds-updated'), $('feeds-pane'), data.generated);
}
function render() {
  if (!data) return;
  const rows = filteredCameras();
  const current = selectedCamera;
  $('camera-select').replaceChildren(...rows.map(c => new Option(c.name, c.id)));
  selectedCamera = rows.some(c => c.id === current) ? current : rows[0]?.id || '';
  if (selectedCamera !== current) selectedVehicle = null;
  $('camera-select').value = selectedCamera;
  renderFeeds(rows); renderMap(rows); renderTrend(rows); renderDetail();
  const states = motionCounts(rows);
  for (const state of ['moving', 'stationary', 'unknown']) $(state + '-count').textContent = states[state].toLocaleString();
  const roads = rows.filter(c => c.analysis?.road_model).length;
  const onRoadMoving = rows.reduce((n,c) => n + vehicles(c).filter(d => d.on_road === true && motionState(d) === 'moving').length, 0);
  $('model-status').replaceChildren(node('p', `YOLO11: vehicle detection · SegFormer: ${roads}/${rows.length} camera road masks`), node('p', `${onRoadMoving} moving vehicles have predicted-road support.`), node('p', 'Rule-based meta-classifier: detector + appearance matching + forward/backward optical flow + camera stability.'), node('p', 'Meta DINOv2: available for short-gap video inputs. Webcam snapshots do not support cross-frame vehicle identity.'));
  for (const error of data.model_errors || []) $('model-status').append(node('p', `${error.model} unavailable: ${error.error}`));
  notifyPane('motion', 'Motion evidence', [states, roads, onRoadMoving, data.model_errors], $('motion-updated'), $('motion-pane'), data.generated);
  notifyPane('controls', 'Network controls', [$('search').value, $('region').value, $('sort').value, $('boxes').checked, $('source-view').checked, [...$('region').options].map(o => o.value)], $('controls-updated'), $('controls-pane'));
  scheduleSources();
}
function apply(payload, history) {
  data = payload; if (history) patterns = history;
  const summary = { catalog: data.catalog_size, analyzed: data.cameras.length, vehicles: data.cameras.reduce((n, c) => n + count(c), 0) };
  for (const [key, value] of Object.entries(summary)) {
    $(key).textContent = value.toLocaleString(); notifyPane(key, { catalog: 'Camera catalog', analyzed: 'Analyzed feeds', vehicles: 'Vehicle count' }[key], value, $(key + '-updated'), $(key).parentElement, data.generated);
  }
  notifyPane('models', 'Model summary', [data.model, data.schema_version, data.model_errors], $('models-updated'), $('models-updated').parentElement, data.generated);
  $('status').textContent = `Cache published ${new Date(data.generated).toLocaleString()}${data.errors?.length ? ' · ' + data.errors.length + ' fetch errors' : ''}`;
  const selected = $('region').value; $('region').replaceChildren(new Option('All of British Columbia', ''));
  for (const r of [...new Set(data.cameras.map(c => c.region))].sort()) $('region').add(new Option(r, r));
  if ([...$('region').options].some(o => o.value === selected)) $('region').value = selected;
  render();
}
async function refresh(fromSource = false) {
  if (refreshing) return;
  refreshing = true; const button = $('refresh'); button.disabled = true; button.textContent = '↻ Refreshing…'; button.setAttribute('aria-busy', 'true');
  if (fromSource) $('source-view').checked = true;
  $('refresh-result').textContent = fromSource ? 'Requesting visible cameras from DriveBC and checking analysis…' : 'Checking published analysis…';
  let analysisError = false, historyError = false;
  try {
    const stamp = Date.now();
    const results = await Promise.allSettled(['latest', 'patterns'].map(async name => {
      const r = await fetch(`data/${name}.json?t=${stamp}`, { cache: 'no-store' }); if (!r.ok) throw Error('HTTP ' + r.status); return r.json();
    }));
    if (results[0].status === 'fulfilled') apply(results[0].value, results[1].status === 'fulfilled' ? results[1].value : null);
    else { analysisError = true; if (!data) throw results[0].reason; }
    historyError = results[1].status === 'rejected';
    const sources = fromSource ? await loadVisibleSources(true) : [];
    const failed = sources.filter(s => s.error).length;
    $('refresh-result').textContent = (fromSource ? (sources.length ? `${sources.length - failed}/${sources.length} visible camera images checked directly with DriveBC.${failed ? ' ' + failed + ' source requests failed; last good frames retained.' : ''}` : 'No visible cameras match the current filters.') : 'Published analysis checked.') + (analysisError ? ' Analysis refresh failed; previous results retained.' : ' Analysis published ' + new Date(data.generated).toLocaleString() + '.') + (historyError ? ' Pattern history unavailable; prior history retained.' : '');
  } catch (e) {
    $('refresh-result').textContent = 'Refresh failed. Try again shortly.' + (data ? ' Keeping the last loaded analysis.' : '');
    if (!data) { $('status').textContent = 'Snapshot cache unavailable'; $('message').textContent = 'The workflow has not published a camera catalog yet, or the network is unavailable.'; }
  } finally { refreshing = false; button.disabled = false; button.textContent = '↻ Refresh from DriveBC'; button.removeAttribute('aria-busy'); scheduleSources(); }
}

for (const id of ['search', 'region', 'sort', 'boxes', 'map-metric', 'fresh-only', 'source-view']) $(id).addEventListener('input', () => { limit = 24; render(); });
for (const id of ['heat-overlay', 'road-overlay', 'vector-overlay', 'vector-scale']) $(id).addEventListener('input', renderDetail);
$('show-analyzed').onclick = () => { $('source-view').checked = false; render(); };
$('motion-only').onclick = () => { $('source-view').checked = false; $('heat-overlay').checked = false; $('road-overlay').checked = false; $('vector-overlay').checked = true; render(); };
$('show-all-vectors').onclick = () => { selectedVehicle = null; renderDetail(); };
$('camera-select').addEventListener('change', () => selectCamera($('camera-select').value));
$('refresh').onclick = () => refresh(true); $('more').onclick = () => { limit += 24; render(); }; refresh(true); setInterval(() => refresh($('source-view').checked), 60000);
if (!location.hostname.endsWith('github.io')) {
  const stream = new EventSource('api/events'); stream.onmessage = async e => { const p = JSON.parse(e.data); if (p.cameras) { apply(p); refresh(); } }; stream.onerror = () => stream.close();
}

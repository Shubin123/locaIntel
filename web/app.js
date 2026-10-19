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
// One local history clock drives every data pane.
let timelineAt = null, historyRows = [], historyPatterns = null, playbackTimer = null;
let archiveRevision=0,patternMemo=null,patternMemoKey='',lastMapVersion='';
let powerProfile = Monitoring.profile((() => { try { return localStorage.getItem('locaintel.power'); } catch (_) { return 3; } })());
let autoRefreshTimer, archivePaintTimer, archiveReady = Promise.resolve(), archiveError = '', seekSerial = 0;
const archiveMetadata = new Map(), historyPhotoSources = new Map(), publishedPhotoSources = new Map(), livePresentations = new Map();
const historyPhotoLoads = new Map();
const imageSizes = new Map(), cardNodes = new Map(), cardVersions = new Map(), markerNodes = new Map();
const publishedImports = new Set(); let publishedImportRunning = false;
const photoArchive = new Monitoring.PhotoArchive({ onChange: archiveChanged });
function isLive() { return timelineAt == null; }
function viewCameras() { return isLive() ? (data?.cameras || []) : historyRows; }
function viewPatterns(){if(!isLive())return historyPatterns;const key=archiveRevision+':'+Math.floor(Date.now()/60000);if(patternMemoKey!==key){patternMemoKey=key;patternMemo=Monitoring.patternsAt([...archiveMetadata.values()],Date.now());}return patternMemo;}
function currentTime() { return isLive() ? Date.now() : timelineAt; }
function queueViewPaint() {
  if(detectionPaintTimer)return;
  detectionPaintTimer = setTimeout(() => {
    detectionPaintTimer=null;
    if (!data) return;
    const currentRows=filteredCameras(),snapshots=currentRows.map(displayedCamera);renderFeeds(currentRows);renderDetail();renderBrowserStatus();renderMap(snapshots);renderTrend(currentRows);renderSummary();renderMotionEvidence(snapshots);renderFacingStatus(snapshots);renderRoadStatus(snapshots);scheduleRoads();scheduleFacing();releaseOldPhotoURLs();
  }, powerProfile.paintMs);
}
function cacheFailure(error) { archiveError = error?.message || String(error); $('cache-status').textContent = 'Photo history could not be saved: ' + archiveError + '. Live monitoring continues.'; $('cache-status').classList.add('cache-error'); }
function archiveChanged(event = {}) {
  archiveRevision++;
  if (event.clear) archiveMetadata.clear();
  if (event.row) { archiveMetadata.set(event.row.key, event.row); scheduleTemporal(event.row);queueViewPaint(); }
  for (const key of event.removedKeys || []) archiveMetadata.delete(key);
  if (event.reload) { photoArchive.metadata().then(rows => { archiveMetadata.clear(); for (const row of rows) archiveMetadata.set(row.key,row); updateArchiveUI(); }).catch(cacheFailure); }
  clearTimeout(archivePaintTimer); archivePaintTimer = setTimeout(updateArchiveUI, 500);
}
async function recordSource(camera, source) {
  if (source.error || !source.blob) return;
  try {
    await archiveReady;
    const local=browserDetector.get(source.hash);
    await photoArchive.saveFrame(camera,source,source.hash===camera.hash?{observedAt:Date.parse(camera.observed),analysisMode:'published',detections:camera.detections,counts:camera.counts,analysis:camera.analysis}:local?.status==='ready'?{analysisMode:'browser',detectorVersion:local.version,detections:local.detections,counts:local.counts,analysis:{road_model:null,motion:{status:'insufficient_cadence',gap_seconds:null}}}:{});
  } catch (error) { cacheFailure(error); }
}
function recordDetection(source, result) {
  if (result.status === 'ready' && source.cameraId) archiveReady.then(() => photoArchive.saveAnalysis(source.cameraId,source.hash,result)).catch(cacheFailure);
}
async function updateArchiveUI() {
  try {
    const stats = await photoArchive.stats();
    $('storage-used').textContent = Monitoring.bytes(stats.usedBytes) + ' / ' + Monitoring.bytes(stats.budgetBytes);
    $('storage-progress').max = stats.budgetBytes; $('storage-progress').value = stats.usedBytes;
    $('storage-detail').textContent = `${stats.frames.toLocaleString()} saved camera frames · ${stats.photos.toLocaleString()} unique photos · up to ${Math.round(stats.retentionMs/3600000)} hours. Photo bytes exclude model files and metadata.`;
    $('cache-limit').value = String(Math.round(stats.budgetBytes/Monitoring.MB)); $('cache-retention').value = String(Math.round(stats.retentionMs/3600000));
    if (!archiveError) $('cache-status').textContent = 'Photos stay in this browser. Oldest photos are removed at the size or age limit; collection runs while this app is open.';
    if (navigator.storage?.estimate) {
      const estimate = await navigator.storage.estimate();
      $('storage-origin').textContent=`Estimated app-origin storage: ${Monitoring.bytes(estimate.usage||0)} · browser quota: ${Monitoring.bytes(estimate.quota||0)}.`;
      $('storage-progress').title = `Photo cache: ${Monitoring.bytes(stats.usedBytes)}. Estimated app-origin storage: ${Monitoring.bytes(estimate.usage || 0)}; browser quota: ${Monitoring.bytes(estimate.quota || 0)}.`;
    }
    renderTimeline();
  } catch (error) { cacheFailure(error); }
}
function localDateValue(time) { const date = new Date(time); return new Date(time-date.getTimezoneOffset()*60000).toISOString().slice(0,19); }
function renderTimeline() {
  const rows = [...archiveMetadata.values()], times = Monitoring.observationTimes(rows);
  const first = times[0], last = Math.max(Date.now(),times[times.length-1]||0), range = $('timeline-position');
  range.disabled = !times.length; range.min = isLive()?(first||0):Math.min(first||timelineAt,timelineAt); range.max = last; range.step = 1; range.value = isLive() ? last : timelineAt;
  $('timeline-date').disabled = !times.length; $('timeline-date').min = first ? localDateValue(first) : ''; $('timeline-date').max = localDateValue(last);
  if (document.activeElement !== $('timeline-date')) $('timeline-date').value = localDateValue(currentTime());
  $('timeline-start').textContent = first ? new Date(first).toLocaleString() : 'No saved photos yet';
  $('timeline-end').textContent = 'Now · ' + new Date(last).toLocaleTimeString();
  $('timeline-title').textContent = isLive() ? 'Live monitoring' : new Date(timelineAt).toLocaleString();
  $('go-live').classList.toggle('live-active',isLive()); $('go-live').textContent = isLive() ? '● Live' : 'Return to Live';
  $('timeline-pane').classList.toggle('reviewing',!isLive());
  const unknown = isLive() ? 0 : historyRows.filter(c => !c._analysisAvailable).length;
  $('timeline-caption').textContent = `${rows.length.toLocaleString()} saved frames across ${new Set(rows.map(r=>r.cameraId)).size} cameras. ${isLive() ? 'Live updates are active. Drag back to review saved photos across all panes.' : `Review paused live work. Each pane shows the latest saved frame at or before this time.${unknown ? ' '+unknown+' saved views have no detection results.' : ''}`} Timeline times are observation times, not guaranteed camera capture times.`;
  const bins = Array(80).fill(0); if (first != null) for (const row of rows) bins[Math.min(79,Math.floor((row.observedAt-first)/Math.max(1,last-first)*80))]++;
  const maximum = Math.max(1,...bins); const bars = bins.map((n,i)=>{const bar=node('span');bar.dataset.key=String(i);bar.style.height=Math.max(1,n/maximum*28)+'px';return bar;}); SmoothUI.list($('timeline-density'),bars);
  $('timeline-play').textContent = playbackTimer ? 'Pause history' : 'Play history';
  const cameraTimes=Monitoring.observationTimes(rows,selectedCamera),at=isLive()?Date.now()+1:timelineAt;
  $('timeline-play').disabled=cameraTimes.length<2;$('timeline-prev').disabled=!cameraTimes.some(t=>t<at);$('timeline-next').disabled=!cameraTimes.some(t=>t>at);
  const selected = viewCameras().find(c => c.id === selectedCamera), source = selected ? (selected._archive ? historyPhotoSources.get(selected.hash) : displayedCamera(selected)._renderSource) : null;
  $('download-photo').hidden = !source?.url;
  if (source?.url) { $('download-photo').href=source.url; $('download-photo').download=`drivebc-${selected.id}-${new Date(selected.observed).toISOString().replace(/[:.]/g,'-')}.jpg`; }
}
async function seekHistory(time, { playback=false } = {}) {
  if(!Number.isFinite(Number(time)))return;
  if (!playback) stopPlayback();
  const serial = ++seekSerial; timelineAt = Number(time); updateBackgroundBudget();
  const rows = Monitoring.latestAt([...archiveMetadata.values()],timelineAt);
  historyRows = rows.map(row => ({...row.camera,hash:row.hash,observed:new Date(row.observedAt).toISOString(),counts:row.counts||{},detections:row.detections||[],analysis:row.analysis||{},_archive:true,_analysisAvailable:Array.isArray(row.detections),browser_analysis:row.analysisMode==='browser',archiveKey:row.key,_analysisMode:row.analysisMode,_photoWidth:row.width,_photoHeight:row.height,source_modified:row.sourceModified,model:row.analysisMode==='browser'?'Browser YOLO11':'Published YOLO11'}));
  historyPatterns = Monitoring.patternsAt([...archiveMetadata.values()],timelineAt);
  if (serial !== seekSerial) return;
  render(); renderTimeline(); await loadHistoryPhotos();
}
async function loadHistoryPhotos() {
  if (isLive()) return;
  const wanted = [...filteredCameras().slice(0,limit)]; const selected=historyRows.find(c=>c.id===selectedCamera);if(selected)wanted.unshift(selected);
  await Promise.all(wanted.map(c=>{
    if(historyPhotoSources.has(c.hash))return;
    if(!historyPhotoLoads.has(c.hash))historyPhotoLoads.set(c.hash,(async()=>{try{const blob=await photoArchive.photo(c.hash);if(blob)historyPhotoSources.set(c.hash,{url:URL.createObjectURL(blob),blob,hash:c.hash,checked:c.observed,archive:true,width:c._photoWidth,height:c._photoHeight});}catch(error){cacheFailure(error);}finally{historyPhotoLoads.delete(c.hash);}})());
    return historyPhotoLoads.get(c.hash);
  }));
  if (!isLive()) { queueViewPaint(); renderTimeline(); }
  if(historyPhotoSources.size>96){const keep=new Set([...filteredCameras().slice(0,limit),...historyRows.filter(c=>c.id===selectedCamera)].map(c=>c.hash));for(const [hash,source]of historyPhotoSources)if(!keep.has(hash)){URL.revokeObjectURL(source.url);historyPhotoSources.delete(hash);if(historyPhotoSources.size<=96)break;}}
}
function stopPlayback() { if(playbackTimer)clearInterval(playbackTimer);playbackTimer=null; }
function stepHistory(direction, playback=false) {
  const times=Monitoring.observationTimes([...archiveMetadata.values()],selectedCamera);if(!times.length)return;
  const at=isLive()?Date.now()+1:timelineAt;
  const target=direction<0?[...times].reverse().find(t=>t<at):times.find(t=>t>at);
  if(target!=null)seekHistory(target,{playback});else if(playback){stopPlayback();renderTimeline();}
}
function startPlayback(){
  if(playbackTimer){stopPlayback();renderTimeline();return;}
  const times=Monitoring.observationTimes([...archiveMetadata.values()],selectedCamera);if(!times.length)return;
  if(isLive()||timelineAt>=times[times.length-1])seekHistory(times[0],{playback:true});
  playbackTimer=setInterval(()=>stepHistory(1,true),1000);renderTimeline();
}
function goLive({refreshSources=true}={}){stopPlayback();timelineAt=null;seekSerial++;historyRows=[];historyPatterns=null;updateBackgroundBudget();render();renderTimeline();scheduleSources();scheduleDetection();if(refreshSources&&!refreshing)refresh(true);}
function updateBackgroundBudget(){
  cameraSources.setConcurrency(powerProfile.downloads);browserDetector.setBudget({delayMs:powerProfile.inferenceDelay});facingAnalyzer.setBudget({delayMs:powerProfile.inferenceDelay});roadAnalyzer.setBudget({delayMs:powerProfile.inferenceDelay});
  $('background-status').textContent = !isLive() ? 'History review: live fetching and queued detection paused.' : document.hidden ? 'Hidden tab: background work paused.' : `Live · automatic refresh about every ${powerProfile.refreshMs/1000} seconds.`;
  clearTimeout(autoRefreshTimer);
  if(isLive()&&!document.hidden)autoRefreshTimer=setTimeout(async()=>{if(isLive()&&!document.hidden)await refresh($('source-view').checked);updateBackgroundBudget();},powerProfile.refreshMs);
  if(isLive()&&!document.hidden){browserDetector.wake();roadAnalyzer.wake();facingAnalyzer.wake();}
}
function setPower(value){
  powerProfile=Monitoring.profile(value);$('power-budget').value=powerProfile.level;$('power-label').value=powerProfile.name;$('power-budget').setAttribute('aria-valuetext',powerProfile.name);
  $('power-description').textContent = `${powerProfile.name}: ${powerProfile.level<3?'slower updates and more idle time to reduce battery and network use':powerProfile.level===3?'a balance between responsiveness and battery use':'faster updates with more background work'}. Detection quality stays the same.`;
  try{localStorage.setItem('locaintel.power',String(powerProfile.level));}catch(_){}
  updateBackgroundBudget();
}
function releaseOldPhotoURLs(){
  const active=new Set([...filteredCameras().slice(0,limit).map(c=>c.id),selectedCamera]);
  if(livePresentations.size>128)for(const id of livePresentations.keys())if(!active.has(id)){livePresentations.delete(id);if(livePresentations.size<=128)break;}
  const keep=new Set([...cameraSources.entries.values()].map(s=>s.url));
  for(const value of livePresentations.values())if(value._renderSource?.url)keep.add(value._renderSource.url);
  for(const image of document.querySelectorAll('.picture img'))if(image.src)keep.add(image.src);
  cameraSources.releaseUnused(keep);
  if(publishedPhotoSources.size>128)for(const [hash,source]of publishedPhotoSources)if(!keep.has(source.url)){URL.revokeObjectURL(source.url);publishedPhotoSources.delete(hash);if(publishedPhotoSources.size<=128)break;}
}
async function cachePublishedFrames(){
  if(publishedImportRunning||!isLive()||document.hidden)return;publishedImportRunning=true;
  try{
    const wanted=visibleSourceCameras();
    for(const c of wanted){
      const key=c.id+':'+c.hash;if(publishedImports.has(key))continue;if(!isLive()||document.hidden)break;publishedImports.add(key);
      await cameraSources.slot();
      try{
        const response=await fetch(`${c.frame}?v=${c.hash.slice(0,12)}`);if(!response.ok)continue;
        const blob=await response.blob();const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await blob.arrayBuffer())),b=>b.toString(16).padStart(2,'0')).join('');if(hash!==c.hash)continue;
        let dimensions=imageSizes.get(hash);if(!dimensions&&typeof createImageBitmap==='function'){const bitmap=await createImageBitmap(blob);dimensions=[bitmap.width,bitmap.height];bitmap.close();}
        const source={hash,url:URL.createObjectURL(blob),blob,cameraId:c.id,checked:c.observed,width:dimensions?.[0],height:dimensions?.[1]};publishedPhotoSources.set(hash,source);
        await archiveReady;await photoArchive.saveFrame(c,source,{observedAt:Date.parse(c.observed),analysisMode:'published',detections:c.detections,counts:c.counts,analysis:c.analysis});
      }catch(error){if(error.name==='QuotaExceededError')cacheFailure(error);}finally{cameraSources.release();}
    }
  }finally{publishedImportRunning=false;}
}

const cameraSources = new CameraSourceCache({concurrency:powerProfile.downloads,onUpdate:recordSource});
let detectionPaintTimer;
const browserDetector = new BrowserDetector({
  onChange: queueViewPaint,
  onResult: recordDetection,
  delayMs: powerProfile.inferenceDelay,
  isPaused: () => !isLive() || document.hidden,
  isWanted: hash => isLive() && $('source-view').checked && detectionTargets().some(c => cameraSources.get(c.id)?.hash === hash)

});
function localAnalysis(c){const source=sourceEntry(c);if(!source?.hash)return null;const current=browserDetector.get(source.hash);if(current)return current;const saved=archiveMetadata.get(c.id+':'+source.hash);const result=saved?.analysisMode==='browser'&&saved.detectorVersion===browserDetector.modelVersion?{version:saved.detectorVersion,detections:saved.detections,counts:saved.counts,observed:new Date(saved.observedAt).toISOString()}:saved?.browserResult;if(result?.version===browserDetector.modelVersion&&Array.isArray(result.detections))return{...result,status:'ready',hash:source.hash};return null;}
function withTemporal(c){
  const t=archiveMetadata.get(c.id+':'+c.hash)?.temporal;
  if(!t)return c;
  return {...c,detections:t.detections,analysis:{...c.analysis,states:t.log.states,motion:{status:t.log.status,temporal:t.log}}};
}
const publishedRoads=new Map(),roadWrites=new Set();
function persistRoad(source,index,result){const key=source.cameraId+':'+source.hash;if(roadWrites.has(key))return;roadWrites.add(key);archiveReady.then(()=>photoArchive.saveRoadScene(key,result)).catch(cacheFailure).finally(()=>roadWrites.delete(key));}
const roadAnalyzer=new VehicleFacing.CropOrientation({
  workerFactory:()=>new Worker('road-worker.js?v=gpu-road-1'),prepareCrop:RoadVision.prepare,delayMs:powerProfile.inferenceDelay,
  isPaused:()=>!isLive()||document.hidden||(browserDetector.running&&browserDetector.backend!=='webgpu'),
  isWanted:hash=>isLive()&&visibleSourceCameras().some(c=>baseDisplayedCamera(c).hash===hash),
  onChange:queueViewPaint,onResult:persistRoad
});
function roadScene(c){
  const saved=archiveMetadata.get(c.id+':'+c.hash)?.roadScene;if(saved?.version===RoadVision.VERSION)return saved;
  const current=!c._archive?roadAnalyzer.get({cameraId:String(c.id),hash:c.hash},RoadVision.scene):null;if(current?.state==='ready')return current;
  if(c.analysis?.road_mask&&c.analysis?.road_model){if(!publishedRoads.has(c.hash)){publishedRoads.set(c.hash,{version:'published',model:c.analysis.road_model,road_mask:c.analysis.road_mask,skeleton:RoadVision.skeleton(c.analysis.road_mask),backend:'published',width:imageSizes.get(c.hash)?.[0]||640,height:imageSizes.get(c.hash)?.[1]||400});if(publishedRoads.size>128)publishedRoads.delete(publishedRoads.keys().next().value);}return publishedRoads.get(c.hash);}return null;
}
function withRoad(c){const result=roadScene(c);return result?RoadVision.attach(c,result):c;}
function scheduleRoads(){if(!data||!isLive()||document.hidden)return;for(const original of visibleSourceCameras()){const c=baseDisplayedCamera(original),source=c._renderSource;if(!source?.blob||c._waitingForSource)continue;const scene=roadScene(c),identity={...source,cameraId:String(c.id)},row=archiveMetadata.get(c.id+':'+c.hash);if(!scene)roadAnalyzer.request(identity,RoadVision.scene,0,c.id===selectedCamera?100:0);else if(scene.version===RoadVision.VERSION&&row?.roadScene?.version!==RoadVision.VERSION&&row)persistRoad(identity,0,scene);}roadAnalyzer.wake();}
const facingWrites=new Set();
function persistFacing(source,index,result){
  const key=source.cameraId+':'+source.hash,task=key+':'+index;if(facingWrites.has(task))return;facingWrites.add(task);
  archiveReady.then(()=>photoArchive.saveOrientation(key,index,result)).catch(cacheFailure).finally(()=>facingWrites.delete(task));
}
const facingAnalyzer=new VehicleFacing.CropOrientation({
  delayMs:powerProfile.inferenceDelay,
  isPaused:()=>!isLive()||document.hidden||!$('facing-enabled').checked||(browserDetector.running&&browserDetector.backend!=='webgpu'&&facingAnalyzer.backend!=='webgpu')||(roadAnalyzer.running&&roadAnalyzer.backend!=='webgpu'&&facingAnalyzer.backend!=='webgpu'),
  isWanted:hash=>isLive()&&visibleSourceCameras().some(c=>displayedCamera(c).hash===hash),
  onChange:queueViewPaint,
  onResult:persistFacing
});
function withFacing(c){
  const saved=archiveMetadata.get(c.id+':'+c.hash)?.facing,source={cameraId:String(c.id),hash:c.hash};
  return {...c,detections:(c.detections||[]).map((d,i)=>{const persisted=saved?.version===VehicleFacing.VERSION&&saved.byDetection?.[i]?.version===VehicleFacing.VERSION&&saved.byDetection?.[i]?.fingerprint===VehicleFacing.fingerprint(d)?saved.byDetection[i]:null;const o=persisted||(!c._archive?facingAnalyzer.get(source,d):null)||null;return {...d,orientation:o?{...o,road_direction:RoadVision.direction(d,o)}:null};})};
}
function displayedCamera(c) { return withFacing(withRoad(withTemporal(baseDisplayedCamera(c)))); }
function scheduleFacing(){
  if(!data||!isLive()||document.hidden||!$('facing-enabled').checked)return;
  for(const original of visibleSourceCameras()){
    const c=displayedCamera(original),source=c._renderSource;
    if(!source?.blob||c._waitingForSource)continue;
    for(const [i,d]of (c.detections||[]).entries())if(vehicleLabels.includes(d.label)){
      const identity={...source,cameraId:String(c.id)},row=archiveMetadata.get(c.id+':'+c.hash);
      if(!d.orientation||d.orientation.state==='queued')facingAnalyzer.request(identity,d,i,c.id===selectedCamera&&i===(selectedVehicle??c.detections.findIndex(d=>d.label==='car'))?1000:(c.id===selectedCamera?100:0)+(d.on_road===true?30:d.on_road===false?0:10));
      else if(['estimated','unknown'].includes(d.orientation.state)&&row&&VehicleFacing.fingerprint(row.detections?.[i]||{})===d.orientation.fingerprint&&(row.facing?.byDetection?.[i]?.version!==VehicleFacing.VERSION||row.facing?.byDetection?.[i]?.fingerprint!==d.orientation.fingerprint))persistFacing(identity,i,d.orientation);
    }
  }
  facingAnalyzer.wake();
}
function baseDisplayedCamera(c) {
  if(c._archive)return c;
  if(!$('source-view').checked)return {...c,_renderSource:publishedPhotoSources.get(c.hash)||null};
  const local=localAnalysis(c),source=sourceEntry(c);
  if(source?.url&&(source.hash===c.hash||local?.status==='ready')){
    const states={moving:0,stationary:0,unknown:local?.detections?.filter(d=>vehicleLabels.includes(d.label)).length||0};
    const view=source.hash===c.hash?{...c,_renderSource:source}:{...c,hash:source.hash,detections:local.detections,counts:local.counts,observed:local.observed,browser_analysis:true,_renderSource:source,analysis:{states,road_model:null,motion:{status:'insufficient_cadence',gap_seconds:null}}};
    livePresentations.set(c.id,view);return view;
  }
  const previous=livePresentations.get(c.id)||{...c,_renderSource:publishedPhotoSources.get(c.hash)||null};
  return {...previous,_waitingForSource:!!source?.url&&source.hash!==previous.hash,_pendingSourceHash:source?.hash};
}

function detectionTargets() {
  if(!isLive())return [];
  const targets = visibleSourceCameras();
  const selected = data?.cameras.find(c => c.id === selectedCamera);
  return [...new Map((selected ? [selected, ...($('boxes').checked ? targets : [])] : targets).map(c => [c.id, c])).values()];
}
function scheduleDetection() {
  if (!data || !isLive() || document.hidden || !$('source-view').checked) return;
  for (const c of detectionTargets()) {
    const source = sourceEntry(c);
    if (source?.blob && source.hash !== c.hash && localAnalysis(c)?.status!=='ready') browserDetector.request(source, c.id === selectedCamera);
  }
}
function renderBrowserStatus() {
  if(!isLive()){$('browser-detector-status').textContent='History review: showing saved detections; live detection is paused.';return;}
  const targets = data ? detectionTargets() : [];
  const needing = targets.filter(c => sourceEntry(c)?.url && unmatchedSource(c));
  const completed = needing.filter(c => localAnalysis(c)?.status === 'ready').length;
  const failed = needing.filter(c => localAnalysis(c)?.status === 'error');
  $('browser-detector-status').textContent = failed.length ? `Browser YOLO: ${completed}/${needing.length} fresh frames detected; ${failed.length} failed. ${localAnalysis(failed[0]).error}. Refresh retries detection, or use View analyzed frame.` :
    needing.length ? `Browser YOLO: ${completed}/${needing.length} fresh frames detected${browserDetector.backend==='webgpu'?' · GPU / WebGPU':browserDetector.backend==='wasm'?' · CPU / WASM':''}${Number.isFinite(browserDetector.lastInferenceMs)?' · '+browserDetector.lastInferenceMs+' ms':''}${browserDetector.state === 'loading' ? ' · downloading model on first use' : completed < needing.length ? ' · processing images in the background' : ' · current-source boxes ready'}.` : 'YOLO boxes match the displayed analyzed frames. Fresh-image detection loads when needed.';
}
let sourceLoading = false, sourceJob = Promise.resolve([]);
function sourceEntry(c) {return c._archive?historyPhotoSources.get(c.hash):$('source-view').checked?cameraSources.get(c.id):null;}
function unmatchedSource(c) { const source = sourceEntry(c); return !!source?.url && source.hash !== c.hash; }
function sourceLabel(c) {
  if(c._archive)return 'Saved locally · observed '+new Date(c.observed).toLocaleString()+(c._analysisAvailable?' · matching saved boxes':' · no saved analysis');
  const view=displayedCamera(c);if(view._waitingForSource)return localAnalysis(c)?.status==='error'?'Detection failed for the new photo · previous photo and boxes retained; Refresh retries':'Updating source boxes · previous photo and detections retained';
  const source = sourceEntry(c);
  if (!$('source-view').checked) return 'Analyzed frame · ' + age(c);
  if (source?.url) return (source.error ? 'Source refresh failed · last good source image' : 'DriveBC checked ' + new Date(source.checked).toLocaleTimeString()) + (source.modified ? ' · upstream ' + new Date(source.modified).toLocaleString() : '') + (unmatchedSource(c) ? localAnalysis(c)?.status === 'ready' ? ' · browser YOLO boxes ready' : localAnalysis(c)?.status === 'error' ? ' · browser detection failed' : ' · detecting fresh boxes…' : ' · matching published analysis');
  return source?.error ? 'Source unavailable · cached analyzed frame retained' : 'Cached analyzed frame · fetching DriveBC image…';
}
function visibleSourceCameras() {
  if(!isLive())return [];
  const rows = filteredCameras().sort($('sort').value === 'name' ? (a,b) => a.name.localeCompare(b.name) : $('sort').value === 'fresh' ? (a,b) => Date.parse(b.observed)-Date.parse(a.observed) : (a,b) => count(b)-count(a));
  const selected = data?.cameras.find(c => c.id === selectedCamera);
  return selected ? [selected,...rows.slice(0,limit)] : rows.slice(0,limit);
}
async function loadVisibleSources(force = false) {
  if (!data || !isLive() || document.hidden || !$('source-view').checked) return [];
  if (sourceLoading) { await sourceJob; return force ? loadVisibleSources(true) : []; }
  sourceLoading = true;
  sourceJob=Promise.all([...new Map(visibleSourceCameras().map(c=>[c.id,c])).values()].map(c=>cameraSources.load(c,force||sourceIsStale(c))));
  try { return await sourceJob; }
  finally { sourceLoading = false; render(); scheduleDetection(); }
}
function sourceIsStale(c){const source=cameraSources.get(c.id);return !source||Date.now()-(Date.parse(source.attempted||source.checked)||0)>=powerProfile.refreshMs;}
function scheduleSources() {
  if (isLive() && !document.hidden && !refreshing && !sourceLoading && $('source-view').checked && visibleSourceCameras().some(sourceIsStale)) {
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
    pane.classList.add('pane-flash'); setTimeout(()=>pane.classList.remove('pane-flash'),1000);
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
  return viewCameras().filter(c => (!$('region').value || c.region === $('region').value) &&
    `${c.name} ${c.highway} ${c.caption}`.toLowerCase().includes(search));
}
function age(c) {
  const minutes = Math.max(0, Math.floor((currentTime() - Date.parse(c.observed)) / 60000));
  return `${minutes < 60 ? minutes + 'm' : Math.floor(minutes / 60) + 'h'} ${isLive()?'ago':'before cursor'}${minutes > 60 ? ' · stale' : ''}`;
}
function motionCounts(rows) {
  const states = { moving: 0, stationary: 0, unknown: 0 };
  for (const c of rows) for (const d of vehicles(c)) states[motionState(d)]++;
  return states;
}
function drawPicture(original, detail = false) {
  const c = displayedCamera(original);
  const picture = node('div', undefined, 'picture'), image = node('img');
  picture.dataset.key='picture';picture.dataset.frameKey=c.hash;image.dataset.key='photo';
  image.loading = detail ? 'eager' : 'lazy'; image.alt = c.caption;
  const source = c._archive?sourceEntry(c):c._renderSource;
  const missingPhoto=c._archive&&!source?.url;
  const mismatched=missingPhoto; 
  if(!missingPhoto)image.src=source?.url||`${c.frame}?v=${c.hash.slice(0,12)}`;
  image.onerror = () => { image.alt = 'Cached image unavailable'; };
  image.onload = function(){
    const target=this.closest('.picture');if(!target)return;
    if(this.naturalWidth&&this.naturalHeight){imageSizes.set(c.hash,[this.naturalWidth,this.naturalHeight]);target.style.aspectRatio=`${this.naturalWidth} / ${this.naturalHeight}`;if(!mismatched)renderVectorLayer(target,c,detail,this.naturalWidth,this.naturalHeight,selectedVehicle);}
  };
  const knownSize=imageSizes.get(c.hash)||(source?.width&&source?.height?[source.width,source.height]:null);if(knownSize)picture.style.aspectRatio=`${knownSize[0]} / ${knownSize[1]}`;
  picture.append(image,node('span',`HWY ${c.highway}`,'tag'),node('span',c._archive?'SAVED PHOTO':c._waitingForSource?'UPDATING · PREVIOUS FRAME':source?.url?(c.browser_analysis?'CURRENT SOURCE · LOCAL YOLO':'ANALYSIS MATCHED'):'ANALYZED FRAME','frame-kind'));
  if(missingPhoto){picture.dataset.loading='true';image.alt='Loading saved photo…';return picture;}

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
        const grid = viewPatterns()?.cameras?.[c.id]?.occupancy_grid || c.analysis?.occupancy_grid;
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
  if (!mismatched) {renderVectorLayer(picture,c,detail,...(imageSizes.get(c.hash)||[640,400]),selectedVehicle);renderFacingLayer(picture,c,detail,selectedVehicle);}
  return picture;
}
function selectCamera(id) { if (id !== selectedCamera) selectedVehicle = null; selectedCamera = id; $('camera-select').value = id; renderDetail(); scheduleSources(); scheduleDetection();scheduleRoads();scheduleFacing(); }
function renderDetail() {
  renderTemporalLog();
  const original = viewCameras().find(c => c.id === selectedCamera);
  const c = original ? displayedCamera(original) : null;
  if (!c) { $('facing-inspector').replaceChildren(); $('detail-title').textContent = data?.cameras.find(c=>c.id===selectedCamera)?.name||'Select a camera'; $('detail-caption').textContent = isLive()?'No camera available for this filter.':'No saved photo for this camera at or before the timeline cursor.'; $('detail-picture').replaceChildren(); $('vehicle-list').replaceChildren(); $('motion-explanation').textContent = ''; $('show-all-vectors').hidden = true; notifyPane('detail', 'Camera explorer', null, $('detail-updated'), $('detail-pane')); return; }
  if (selectedVehicle != null && (selectedVehicleFrame !== c.hash || selectedVehicle >= c.detections.length)) selectedVehicle = null;
  $('detail-title').textContent = c.name;
  SmoothUI.picture($('detail-picture'),Object.assign(drawPicture(original,true),{id:'detail-picture'}));
  const sample = c.browser_analysis ? null : viewPatterns()?.cameras?.[c.id], analysis = c.analysis;
  const measured = vehicles(c).filter(d => validVector(d)).length;
  $('motion-explanation').textContent = c._waitingForSource ? 'Updating this camera in the background. The previous photo and its matching boxes stay visible until the next frame is ready.' : c._archive&&!c._analysisAvailable?'This saved photo has no stored detections. No boxes or motion measurements are assumed.':c.analysis?.motion?.temporal ? `Local temporal log: ${c.analysis.motion.temporal.status.replaceAll('_',' ')} · ${c.analysis.motion.temporal.frameHashes.length} distinct frames. ${c.analysis.states.moving} moving · ${c.analysis.states.stationary} persistent-position stationary · ${c.analysis.states.unknown} unresolved. Moving tracks require consistent appearance and direction across three observations, each gap ≤15s. Stationary means persistent position, not proven identity. Sparse or ambiguous matches remain unresolved.` : c.browser_analysis ? 'YOLO boxes match this photo. Collecting three distinct cached observations for temporal evidence; repeated source photos do not count.' : unmatchedSource(original) ? (localAnalysis(original)?.status === 'error' ? 'Fresh-image detection failed. Choose View analyzed frame for the last available bounding boxes, or Refresh to retry.' : 'Computing YOLO bounding boxes for this fresh source image… The first run downloads the model; detection continues in the background.') : motionExplanation(c, measured) + (selectedVehicle != null ? ` Only vehicle #${selectedVehicle + 1} is highlighted: ${motionDescription(c.detections[selectedVehicle])}.` : '');
  $('motion-explanation').classList.toggle('has-vectors', measured > 0 && !unmatchedSource(c));
  $('show-all-vectors').hidden = selectedVehicle == null;
  const gap = analysis?.motion?.gap_seconds;
  const cadence = gap == null ? 'No trustworthy short-gap frame pair.' : `Frame gap ${gap.toFixed(2)}s; motion status: ${analysis.motion.status.replaceAll('_', ' ')}.`;
  $('detail-caption').textContent = c._archive?`${sourceLabel(c)}. Photo ${age(c)}; ${c._analysisAvailable?c._analysisMode+' detections saved with this frame':'no saved measurements'}. All panes use this same timeline cursor.`:c.browser_analysis ? `${sourceLabel(original)}. Browser YOLO detected ${vehicles(c).length} vehicles at ${new Date(c.observed).toLocaleTimeString()}. Facing is estimated from isolated crops; movement requires temporal evidence. ${c.analysis?.road_model?c.analysis.on_road+' road-supported vehicles; road axes follow this same image.':'Road surface prediction is queued for this exact frame.'} Published analysis: ${age(original)}.` : `${sourceLabel(original)}. Cached analysis: ${sample?.samples || 1} distinct samples · ${age(c)}. ${cadence} ${analysis?.road_model ? 'Blue overlay: predicted road surface.' : 'Road model unavailable; road membership unknown.'} Heat shows relative visible-vehicle locations, not speed.${c.media_note ? ' ' + c.media_note : ''}`;
  const list = vehicles(c).map(d => {
    const i = c.detections.indexOf(d), row = node('button', undefined, 'vehicle-row'), road = d.on_road === true ? 'road-supported' : d.on_road === false ? 'road not supported' : 'road unknown';
    row.dataset.key=c.hash+':'+i;row.type = 'button'; row.setAttribute('aria-pressed', String(selectedVehicle === i));
    row.onclick = () => { selectedVehicle = selectedVehicle === i ? null : i; selectedVehicleFrame = c.hash; renderDetail();scheduleFacing(); };
    row.append(node('span', `#${i + 1} ${d.label} · ${Math.round(d.confidence * 100)}%`), node('span', `${VehicleFacing.description(d.orientation)}${d.orientation?.road_direction?.state==='estimated'?' · road direction '+d.orientation.road_direction.direction:''} · ${motionDescription(d)} · ${road}`));
    return row;
  });
  renderFacingInspector(c);
  SmoothUI.list($('vehicle-list'),list.length?list:[Object.assign(node('p',c._archive&&!c._analysisAvailable?'No saved analysis for this photo.':'No vehicles detected.','widget-note'),{id:'no-detections'})]);
  renderTimeline();
  notifyPane('detail', 'Camera explorer', [c.id, c.hash, c.analysis, c.detections, sample, selectedVehicle, $('vector-overlay').checked, $('vector-scale').value, $('heat-overlay').checked, $('road-overlay').checked, sourceEntry(original)?.hash, sourceEntry(original)?.error, localAnalysis(original)?.status], $('detail-updated'), $('detail-pane'), data.generated);
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
  const fresh = $('fresh-only').checked, metric = $('map-metric').value, now = currentTime();
  const samples = rows.filter(c => !fresh || now - Date.parse(c.observed) < 3600000).map(c => {
    const facing=vehicles(c).filter(d=>d.orientation?.state==='estimated');
    const value = c._analysisAvailable===false?null:metric.startsWith('facing_')?(facing.length?facing.filter(d=>d.orientation.facing===(metric==='facing_toward'?'toward_camera':'away_from_camera')).length:null):metric === 'road' ? (c.analysis?.road_model ? c.analysis.on_road : null) : metric === 'history' ? viewPatterns()?.cameras?.[c.id]?.mean_vehicles : metric === 'moving' ? motionCounts([c]).moving : count(c);
    return { c, value };
  });
  const mapVersion=signature([metric,fresh,timelineAt,samples.map(({c,value})=>[c.id,c.hash,value,Math.floor((now-Date.parse(c.observed))/60000)])]);
  if(lastMapVersion===mapVersion)return;lastMapVersion=mapVersion;
  if (map) {
    const wantedMarkers=new Set(samples.map(s=>s.c.id));for(const [id,marker]of markerNodes)if(!wantedMarkers.has(id)){markerLayer.removeLayer(marker);markerNodes.delete(id);}
    const max = Math.max(1, ...samples.map(s => s.value));
    heatLayer?.setLatLngs(samples.filter(s => s.value > 0).map(({ c, value }) => [c.lat, c.lon, value / max]));
    for (const { c, value } of samples) {
      const content = node('div'); content.append(node('strong', c.name), node('p', `${value==null?'Analysis unavailable':value} ${metric.startsWith('facing_')?'appearance-based facing estimates':metric === 'road' ? 'road-supported vehicles' : metric === 'moving' ? 'motion-supported' : metric === 'history' ? 'average visible vehicles' : 'visible vehicles'} · ${age(c)}`));
      const button = node('button', 'Inspect camera'); button.onclick = () => { selectCamera(c.id); $('detail-pane').scrollIntoView({ behavior: 'smooth', block: 'center' }); }; content.append(button);
      let marker=markerNodes.get(c.id);if(!marker){marker=L.circleMarker([c.lat,c.lon],{radius:4,color:value?'#b8ed86':'#92a2ac',weight:1,fillOpacity:.7}).bindPopup(content).addTo(markerLayer);markerNodes.set(c.id,marker);}else{marker.setLatLng([c.lat,c.lon]);marker.setStyle({color:value?'#b8ed86':'#92a2ac'});const old=marker.getPopup().getContent();if(old?.nodeType)SmoothUI.patch(old,content);else marker.setPopupContent(content);}
    }
  } else {
    $('traffic-map').hidden = true; $('map-fallback').hidden = false;
    $('map-fallback').replaceChildren(node('p', 'Map library unavailable. Camera locations:', 'widget-note'), ...samples.slice(0, 20).map(({ c, value }) => {
      const button = node('button', `${c.name} (${c.lat}, ${c.lon}) · ${value}`); button.onclick = () => selectCamera(c.id); return button;
    }));
  }
  const stale = samples.filter(({ c }) => now - Date.parse(c.observed) > 3600000).length;
  $('map-caption').textContent = `${samples.length} sampled cameras · ${samples.filter(s=>s.value==null).length} without analysis · ${stale} over 60 minutes old. ${metric.startsWith('facing_')?'Appearance-based, camera-relative facing estimates at the cursor; no estimated speed or measured traffic flow. Unresolved cameras do not contribute.':metric === 'moving' ? 'Only motion-supported detections contribute; an empty heat layer means no qualifying measurements.' : 'Relative visible-vehicle density around camera locations. Brightness depends on camera coverage and field of view, not measured traffic flow.'}`;
  notifyPane('map', 'Traffic heatmap', [metric, fresh, samples.map(({ c, value }) => [c.id, c.hash, value])], $('map-updated'), $('map-pane'), data.generated);
}
function renderTrend(rows) {
  const activePatterns=viewPatterns();
  const hours = new Map((activePatterns?.hourly || []).map(h => [h.hour, h]));
  const bins = [], end = new Date(activePatterns?.generated || currentTime()); end.setUTCMinutes(0, 0, 0);
  for (let i = 23; i >= 0; i--) { const time = new Date(end.getTime() - i * 3600000), key = time.toISOString().replace('.000Z', 'Z'); bins.push({ time, sample: hours.get(key) }); }
  const max = Math.max(1, ...bins.map(b => b.sample?.mean_vehicles || 0));
  SmoothUI.list($('hourly-chart'),bins.map(({ time, sample }) => {
    const bin = node('div', undefined, 'hour-bin');bin.dataset.key=time.toISOString();const bar = node('div', undefined, 'bar' + (sample ? '' : ' no-sample'));
    bar.style.height = sample ? `${Math.max(2, sample.mean_vehicles / max * 140)}px` : '2px';
    bin.title = `${time.toISOString().slice(11, 13)}:00 UTC · ${sample ? sample.mean_vehicles + ' vehicles/frame · ' + sample.cameras + ' cameras · ' + sample.samples + ' samples' : 'No samples'}`;
    bin.append(bar, node('span', time.toISOString().slice(11, 13))); return bin;
  }));
  const measured = bins.filter(b => b.sample).length;
  $('trend-caption').textContent = `${measured}/24 hours sampled across saved cameras. Hourly mean visible vehicles per frame, equally weighted by camera. Hover bars for camera/sample counts; gaps mean no samples. The camera set can vary by hour. ${isLive()?'Saved local frames up to now; filters apply to map and feeds.':'Local saved samples up to the timeline cursor; future photos are excluded.'}`;
  notifyPane('trend', '24h traffic pattern', activePatterns?.hourly || [], $('trend-updated'), $('trend-pane'), activePatterns?.generated);
}
function renderFeeds(rows) {
  rows=[...rows].sort($('sort').value==='name'?(a,b)=>a.name.localeCompare(b.name):$('sort').value==='fresh'?(a,b)=>Date.parse(b.observed)-Date.parse(a.observed):(a,b)=>count(b)-count(a));
  $('shown').textContent=`${rows.length} available`;
  const nodes=[],wanted=new Set();
  for(const original of rows.slice(0,limit)){
    const c=displayedCamera(original),key=String(c.id);wanted.add(key);
    const version=signature([c.hash,c.detections,c.counts,c.error,c.name,c.region,c._waitingForSource,c._archive,c._analysisAvailable,sourceLabel(original),age(c),c._renderSource?.url,sourceEntry(original)?.url,$('boxes').checked,localAnalysis(original)?.status]);
    let card=cardNodes.get(key);
    if(card?.dataset.previewing==='true'){nodes.push(card);continue;}
    if(card&&cardVersions.get(key)===version){nodes.push(card);continue;}
    const next=node('article',undefined,'card');next.dataset.key=key;
    const details=node('div',undefined,'details');details.dataset.key='details';const h=node('h3');h.append(link(c.name,c.source));details.append(h);
    const meta=node('div',undefined,'meta');meta.append(node('span',c.region),node('span',c._archive?(c._analysisAvailable?`${count(c)} saved vehicle detections`:'Analysis unavailable'):c.browser_analysis?`${count(c)} vehicles detected in source`:`${count(c)} vehicles in analyzed frame`));details.append(meta);
    const counts=node('div',undefined,'counts');for(const [label,n]of Object.entries(c.counts||{}))counts.append(node('span',`${n} ${label}`));if(!counts.childNodes.length)counts.append(node('span',c._archive&&!c._analysisAvailable?'No stored analysis':'No objects detected'));details.append(counts);
    const facing=vehicles(c).filter(d=>d.orientation?.state==='estimated');details.append(node('div',`${facing.length} facing estimates · ${facing.filter(d=>d.orientation.facing==='toward_camera').length} toward / ${facing.filter(d=>d.orientation.facing==='away_from_camera').length} away from camera`,'facing-tag'));
    const states=motionCounts([c]);details.append(node('div',`${c._archive?'Saved':'Frame'} motion: ${states.moving} moving · ${states.stationary} stationary · ${states.unknown} unknown`,'motion-tag'));
    const sources=node('div',undefined,'camera-sources');for(const [text,url]of[['DriveBC camera',c.source],['Original image',c.image]])if(url)sources.append(link(text+' ↗',url));details.append(sources);
    details.append(node('div',sourceLabel(original),'source-status'),node('div',`${c._archive?'Saved observation':c.browser_analysis?'Local detection':'Analyzed frame'} ${age(c)}`,'age'));
    const inspect=node('button','Inspect motion & heat','inspect-button');inspect.onclick=()=>{selectCamera(c.id);$('detail-pane').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'center'});};details.append(inspect,node('div','Hover or focus this card to play saved frames · up to 24 photos','source-status'),node('span',undefined,'update-badge'));
    next.append(drawPicture(original),details);
    if(card)SmoothUI.patch(card,next);else card=next;
    installHoverPlayback(card,key);
    cardNodes.set(key,card);cardVersions.set(key,version);nodes.push(card);
    notifyPane(`camera-${key}`,c.name,[c.hash,c.detections,c.error,c._waitingForSource,sourceEntry(original)?.error],card.querySelector('.update-badge'),card,c.observed);
  }
  SmoothUI.list($('feeds'),nodes);
  for(const key of cardNodes.keys())if(!wanted.has(key)){cardNodes.delete(key);cardVersions.delete(key);}
  $('more').hidden=rows.length<=limit;
  $('message').textContent=isLive()?'Photos and matching boxes swap together. New frames are processed in the background; previous frames remain visible.':'Local photos and their matching saved measurements at the shared timeline cursor.';
  notifyPane('feeds','Camera feeds',[nodes.map(n=>n.dataset.key+':'+n.querySelector('.picture')?.dataset.frameKey),$('boxes').checked,timelineAt],$('feeds-updated'),$('feeds-pane'),isLive()?data.generated:new Date(timelineAt).toISOString());
}

function render() {
  if (!data) return;
  const rows = filteredCameras();
  const current = selectedCamera;
  const options=(data?.cameras||[]).filter(c=>(!$('region').value||c.region===$('region').value)&&`${c.name} ${c.highway} ${c.caption}`.toLowerCase().includes($('search').value.toLowerCase()));
  selectedCamera=options.some(c=>c.id===current)?current:options[0]?.id||'';
  SmoothUI.options($('camera-select'),options.map(c=>[c.name,c.id]),selectedCamera);
  if (selectedCamera !== current) selectedVehicle = null;
  $('camera-select').value = selectedCamera;
  const snapshots=rows.map(displayedCamera);
  renderFeeds(rows);renderMap(snapshots);renderTrend(rows);renderDetail();
  renderMotionEvidence(snapshots);
  notifyPane('controls', 'Network controls', [$('search').value, $('region').value, $('sort').value, $('boxes').checked, $('source-view').checked, [...$('region').options].map(o => o.value)], $('controls-updated'), $('controls-pane'));
  scheduleSources(); scheduleDetection();scheduleRoads();scheduleFacing();renderRoadStatus(snapshots);renderFacingStatus(snapshots);renderBrowserStatus();renderSummary();if(isLive())cachePublishedFrames();else loadHistoryPhotos();
}
function renderMotionEvidence(snapshots){
  const states=motionCounts(snapshots);
  for (const state of ['moving', 'stationary', 'unknown']) $(state + '-count').textContent = states[state].toLocaleString();
  const roads = snapshots.filter(c => c.analysis?.road_model).length;
  const onRoadMoving = snapshots.reduce((n,c) => n + vehicles(c).filter(d => d.on_road === true && motionState(d) === 'moving').length, 0);
  $('model-status').replaceChildren(node('p', `YOLO11: vehicle detection · SegFormer: ${roads}/${snapshots.length} camera road masks`), node('p', `${onRoadMoving} moving vehicles have predicted-road support.`), node('p', 'Rule-based meta-classifier: detector + appearance matching + forward/backward optical flow + camera stability.'), node('p', 'Meta DINOv2: available for short-gap video inputs. Webcam snapshots do not support cross-frame vehicle identity.'));
  for (const error of data.model_errors || []) $('model-status').append(node('p', `${error.model} unavailable: ${error.error}`));
  notifyPane('motion', 'Motion evidence', [states, roads, onRoadMoving, data.model_errors], $('motion-updated'), $('motion-pane'), data.generated);
}
function renderSummary(){if(!data)return;
  const summary = { catalog: data.catalog_size, analyzed: viewCameras().map(displayedCamera).filter(c=>c._analysisAvailable!==false).length, vehicles: viewCameras().map(displayedCamera).reduce((n, c) => n + count(c), 0) };
  for (const [key, value] of Object.entries(summary)) {
    $(key).textContent = value.toLocaleString(); notifyPane(key, { catalog: 'Camera catalog', analyzed: 'Analyzed feeds', vehicles: 'Vehicle count' }[key], value, $(key + '-updated'), $(key).parentElement, data.generated);
  }
  notifyPane('models', 'Model summary', [data.model, data.schema_version, data.model_errors], $('models-updated'), $('models-updated').parentElement, data.generated);
  $('status').textContent = `${isLive()?'Published analysis '+new Date(data.generated).toLocaleString():'Local history at '+new Date(timelineAt).toLocaleString()}${isLive()&&data.errors?.length ? ' · ' + data.errors.length + ' fetch errors' : ''}`;
}
function apply(payload, history) {
  data = payload; if (history) patterns = history;
  const selected=$('region').value,regions=[...new Set(data.cameras.map(c=>c.region))].sort();
  SmoothUI.options($('region'),[['All of British Columbia',''],...regions.map(r=>[r,r])],regions.includes(selected)?selected:'');
  render();
}
async function refresh(fromSource = false) {
  if (refreshing || (!fromSource && (!isLive() || document.hidden))) return;
  if(fromSource&&!isLive())goLive({refreshSources:false});
  refreshing = true; const button = $('refresh'); button.disabled = true; button.textContent = '↻ Refreshing…'; button.setAttribute('aria-busy', 'true');
  if (fromSource) { $('source-view').checked = true; browserDetector.retryFailed();facingAnalyzer.retryFailed();roadAnalyzer.retryFailed(); }
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
$('refresh').onclick=()=>refresh(true);$('more').onclick=()=>{limit+=24;render();};
if (!location.hostname.endsWith('github.io')) {
  const stream = new EventSource('api/events'); stream.onmessage = async e => { const p = JSON.parse(e.data); if(p.cameras){if(isLive())apply(p);else data=p;} }; stream.onerror = () => stream.close();
}

// Monitoring preferences and persistent local history.
$('power-budget').addEventListener('input',()=>setPower($('power-budget').value));
let seekTimer;
$('timeline-position').addEventListener('input',()=>{stopPlayback();const time=Number($('timeline-position').value);timelineAt=time;updateBackgroundBudget();$('timeline-title').textContent='Reviewing '+new Date(time).toLocaleString();clearTimeout(seekTimer);seekTimer=setTimeout(()=>seekHistory(time),120);});
$('timeline-position').addEventListener('change',()=>{clearTimeout(seekTimer);seekHistory(Number($('timeline-position').value));});
$('timeline-date').addEventListener('change',()=>{const time=new Date($('timeline-date').value).getTime();if(Number.isFinite(time))seekHistory(time);});
$('go-live').onclick=goLive;$('timeline-prev').onclick=()=>stepHistory(-1);$('timeline-next').onclick=()=>stepHistory(1);$('timeline-play').onclick=startPlayback;
async function configureStorage(){try{archiveError='';$('cache-status').classList.remove('cache-error');await archiveReady;await photoArchive.configure({budgetBytes:Number($('cache-limit').value)*Monitoring.MB,retentionMs:Number($('cache-retention').value)*3600000});await updateArchiveUI();}catch(error){cacheFailure(error);}}
$('cache-limit').addEventListener('change',configureStorage);$('cache-retention').addEventListener('change',configureStorage);
$('clear-history').onclick=async()=>{if(!confirm('Delete saved camera photos from this browser? This cannot be undone.'))return;try{await photoArchive.clear();for(const source of historyPhotoSources.values())URL.revokeObjectURL(source.url);historyPhotoSources.clear();goLive({refreshSources:false});await updateArchiveUI();}catch(error){cacheFailure(error);}};
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopPlayback();updateBackgroundBudget();if(!document.hidden&&isLive()){scheduleSources();scheduleDetection();}});
window.addEventListener('monitor-frame-commit',releaseOldPhotoURLs);
archiveReady=photoArchive.initialize().then(async()=>{const rows=await photoArchive.metadata();for(const row of rows){archiveMetadata.set(row.key,row);scheduleTemporal(row);}for(const row of rows.filter(r=>r.analysisMode==='browser'&&r.detectorVersion===browserDetector.modelVersion&&Array.isArray(r.detections)).sort((a,b)=>b.observedAt-a.observedAt).slice(0,128))browserDetector.entries.set(row.hash,{status:'ready',hash:row.hash,version:row.detectorVersion,detections:row.detections,counts:row.counts||{},observed:new Date(row.observedAt).toISOString()});archiveRevision++;await updateArchiveUI();}).catch(cacheFailure);
setPower(powerProfile.level);refresh(true);

// Temporal work runs serially with the same device budget as inference.
const temporalPending=new Set(),hoverCards=new WeakSet();let temporalRunning=false;
function scheduleTemporal(row){
  if(!row.detections||!isLive())return;
  temporalPending.add(row.cameraId);setTimeout(runTemporal,0);
}
async function runTemporal(){
  if(temporalRunning||!isLive()||document.hidden||!temporalPending.size)return;
  temporalRunning=true;const id=temporalPending.values().next().value;temporalPending.delete(id);
  try{
    const rows=[...archiveMetadata.values()].filter(r=>r.cameraId===id&&r.detections).sort((a,b)=>a.observedAt-b.observedAt).slice(-3),latest=rows.at(-1);
    if(latest&&signature(latest.temporal?.log?.frameHashes)!==signature(rows.map(r=>r.hash))){const frames=[];
      for(const row of rows){const blob=await photoArchive.photo(row.hash);if(!blob)continue;frames.push({...row,...await TemporalTraffic.features(blob,row.detections)});}
      if(frames.length){const result=TemporalTraffic.analyze(frames);await photoArchive.saveTemporal(latest.key,result);queueViewPaint();}
    }
  }catch(error){console.warn('Temporal analysis unavailable',error);}
  finally{temporalRunning=false;if(temporalPending.size)setTimeout(runTemporal,powerProfile.inferenceDelay+50);}
}
function installHoverPlayback(card,id){
  if(hoverCards.has(card))return;hoverCards.add(card);
  let token=0,timer=null,preview=null,urls=[];
  const stop=()=>{token++;clearTimeout(timer);preview?.remove();preview=null;urls.forEach(URL.revokeObjectURL);urls=[];delete card.dataset.previewing;cardVersions.delete(id);queueViewPaint();};
  const start=async()=>{
    if(card.dataset.previewing)return;const serial=++token,previewClock=timelineAt;
    const rows=[...archiveMetadata.values()].filter(r=>r.cameraId===id&&r.observedAt<=currentTime()).sort((a,b)=>a.observedAt-b.observedAt).slice(-24);
    if(rows.length<2)return;
    card.dataset.previewing='true';preview=node('div',undefined,'hover-playback');preview.setAttribute('aria-live','off');const img=node('img'),caption=node('span');preview.append(img,caption);card.querySelector('.picture').append(preview);
    let i=0;
    const show=async()=>{if(!card.isConnected||document.hidden||timelineAt!==previewClock){stop();return;}const row=rows[i++%rows.length],blob=await photoArchive.photo(row.hash);if(serial!==token)return;if(!blob){timer=setTimeout(show,700);return;}
      const url=URL.createObjectURL(blob);urls.push(url);img.onload=()=>{while(urls.length>2)URL.revokeObjectURL(urls.shift());};img.src=url;
      preview.querySelector('svg')?.remove();const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 1000 1000');svg.setAttribute('preserveAspectRatio','none');
      for(const d of row.temporal?.detections||row.detections||[]){const rect=document.createElementNS(svg.namespaceURI,'rect'),[x,y,xx,yy]=d.box;for(const [k,v]of Object.entries({x:x*1000,y:y*1000,width:(xx-x)*1000,height:(yy-y)*1000,fill:'none',stroke:d.motion?.state==='moving'?'#b8ed86':d.motion?.state==='stationary'?'#69b9ff':'#efbd7b','stroke-width':3}))rect.setAttribute(k,v);svg.append(rect);}preview.append(svg);renderVectorLayer(preview,{detections:row.temporal?.detections||row.detections||[]},false,row.width||640,row.height||400);renderFacingLayer(preview,withFacing({...row.camera,hash:row.hash,detections:row.temporal?.detections||row.detections||[],_archive:true}),false,null);
      caption.textContent=`Cached playback · ${((i-1)%rows.length)+1}/${rows.length} · ${new Date(row.observedAt).toLocaleString()}`;timer=setTimeout(show,matchMedia('(prefers-reduced-motion: reduce)').matches?1800:700);
    };await show();
  };
  card.addEventListener('pointerenter',start);card.addEventListener('pointerleave',stop);card.addEventListener('focusin',start);card.addEventListener('focusout',e=>{if(!card.contains(e.relatedTarget))stop();});
  // A tap on the photo also previews on touch devices; controls keep their actions.
  card.addEventListener('click',e=>{if(e.pointerType==='touch'&&e.target.closest('.picture'))card.dataset.previewing?stop():start();});
}

let motionDownloadURL=null;
function renderTemporalLog(){
  let pane=$('temporal-log');if(!pane){pane=node('details');pane.id='temporal-log';$('detail-pane').append(pane);}
  const rows=[...archiveMetadata.values()].filter(r=>r.cameraId===selectedCamera&&r.observedAt<=currentTime()&&r.temporal).sort((a,b)=>a.observedAt-b.observedAt);
  const version=signature([selectedCamera,rows.map(r=>[r.hash,r.temporal.log,r.facing,r.roadScene])]);if(pane.dataset.version===version)return;pane.dataset.version=version;
  const summary=node('summary',`Saved motion evidence · ${rows.length} observations`),explain=node('p','Three-frame appearance matching with a stable-background gate. Blue means persistent position across distinct images; it does not prove the same vehicle stayed continuously. Green vectors require two consistent steps ≤15 seconds apart. Longer gaps retain occupancy and turnover evidence; individual movement remains unresolved.');
  const list=node('ol');for(const row of rows.slice(-20).reverse()){const l=row.temporal.log;list.append(node('li',`${new Date(row.observedAt).toLocaleString()} · ${l.status.replaceAll('_',' ')} · ${l.states.moving} moving / ${l.states.stationary} persistent position / ${l.states.unknown} unresolved${l.turnover?` · ${l.turnover.arrivals} unmatched arrivals / ${l.turnover.departures} unmatched departures`:''}`));}
  if(motionDownloadURL)URL.revokeObjectURL(motionDownloadURL);motionDownloadURL=URL.createObjectURL(new Blob([JSON.stringify(rows.map(r=>({cameraId:r.cameraId,hash:r.hash,...r.temporal,facing:r.facing||null,road:r.roadScene||null})),null,2)],{type:'application/json'}));const download=node('a','Download camera motion log (JSON)');download.href=motionDownloadURL;download.download=`camera-${selectedCamera}-motion.json`;pane.replaceChildren(summary,explain,list,download);
}

try{$('facing-enabled').checked=localStorage.getItem('locaintel.facing')!=='off';}catch(_){}
$('facing-enabled').addEventListener('change',()=>{try{localStorage.setItem('locaintel.facing',$('facing-enabled').checked?'on':'off');}catch(_){}facingAnalyzer.setBudget({delayMs:powerProfile.inferenceDelay});roadAnalyzer.setBudget({delayMs:powerProfile.inferenceDelay});scheduleFacing();queueViewPaint();});
function renderFacingStatus(snapshots){
  const ds=snapshots.flatMap(vehicles),totals={toward_camera:0,away_from_camera:0,side:0,unresolved:0};let completed=0;
  for(const d of ds){if(d.orientation&&!['queued','processing'].includes(d.orientation.state))completed++;if(d.orientation?.state!=='estimated'){totals.unresolved++;continue;}if(d.orientation.facing==='toward_camera'||d.orientation.facing==='away_from_camera')totals[d.orientation.facing]++;else totals.side++;}
  SmoothUI.list($('facing-totals'),Object.entries(totals).map(([key,n])=>{const cell=node('div');cell.dataset.key=key;cell.append(node('strong',String(n)),node('span',{toward_camera:'FACING CAMERA',away_from_camera:'FACING AWAY',side:'SIDE VIEW',unresolved:'UNRESOLVED / WAITING'}[key]));return cell;}));
  $('facing-status').textContent=!isLive()?'Saved facing estimates at the timeline cursor; no new crop inference.':!$('facing-enabled').checked?'New facing inference paused; saved estimates remain visible.':facingAnalyzer.phase==='loading'?'Loading the cached vehicle-view model (GPU where available); YOLO boxes stay visible.':facingAnalyzer.phase==='error'?'Facing model unavailable. Refresh retries failed crops; vehicle boxes and cached history remain available.':`${ds.filter(d=>d.orientation?.road_direction?.state==='estimated').length} road-aligned directions · ${completed}/${ds.length} filtered vehicle crops have results · ${facingAnalyzer.queue.length} queued${facingAnalyzer.backend==='webgpu'?' · GPU / WebGPU':facingAnalyzer.backend==='wasm'?' · CPU / WASM':''}${Number.isFinite(facingAnalyzer.lastInferenceMs)?' · '+(facingAnalyzer.lastReused?'exact crop reused':facingAnalyzer.lastInferenceMs+' ms / crop'):''}${facingAnalyzer.running?' · analyzing one crop':''}. Analysis runs on viewed feeds. Tiny, clipped and unsupported crops abstain. Scores are not calibrated probabilities.`;
  notifyPane('facing','Vehicle facing',[totals,completed,facingAnalyzer.phase,isLive(),$('facing-enabled').checked],$('facing-updated'),$('facing-pane'));
}
function renderFacingLayer(picture,c,detail,focused){
  picture.querySelector('.facing-layer')?.remove();if(!detail&&!$('boxes').checked)return;
  const layer=document.createElementNS('http://www.w3.org/2000/svg','svg');layer.classList.add('facing-layer');layer.setAttribute('viewBox','0 0 1000 1000');layer.setAttribute('preserveAspectRatio','none');layer.setAttribute('aria-label','Estimated car facing, separate from measured motion');
  for(const [i,d]of (c.detections||[]).entries()){
    const o=d.orientation;if(detail&&focused!=null&&focused!==i)continue;
    if(detail&&$('road-overlay').checked&&d.road_axis){const axis=d.road_axis.vector,center=[(d.box[0]+d.box[2])/2,d.box[3]],line=document.createElementNS(layer.namespaceURI,'line');for(const [key,value]of Object.entries({x1:Math.max(0,Math.min(1,center[0]-axis[0]*.06))*1000,y1:Math.max(0,Math.min(1,center[1]-axis[1]*.06))*1000,x2:Math.max(0,Math.min(1,center[0]+axis[0]*.06))*1000,y2:Math.max(0,Math.min(1,center[1]+axis[1]*.06))*1000,stroke:'#88e2f2','stroke-width':2,'stroke-dasharray':'4 4','vector-effect':'non-scaling-stroke'}))line.setAttribute(key,value);const title=document.createElementNS(layer.namespaceURI,'title');title.textContent='Estimated local road axis; no travel direction is assumed';line.append(title);layer.append(line);}
    if(o?.state!=='estimated')continue;
    const text=document.createElementNS(layer.namespaceURI,'text');text.setAttribute('x',String(d.box[0]*1000));text.setAttribute('y',String(Math.max(18,d.box[1]*1000-5)));text.textContent=(o.facing_glyph?(o.road_direction?.state==='estimated'?'Road est. '+({right:'→','down-right':'↘',down:'↓','down-left':'↙',left:'←','up-left':'↖',up:'↑','up-right':'↗'}[o.road_direction.direction]||o.facing_glyph):'Est. '+o.facing_glyph)+' · ':'')+(o.facing==='toward_camera'?'front':o.facing==='away_from_camera'?'rear':'side');const title=document.createElementNS(layer.namespaceURI,'title');title.textContent=VehicleFacing.description(o)+(o.road_direction?.state==='estimated'?'; road axis supports '+o.road_direction.direction:'')+'; estimated appearance and geometry, not measured movement';text.append(title);layer.append(text);
    if(detail)for(const [x,y]of o.lights?.points||[]){const circle=document.createElementNS(layer.namespaceURI,'circle');for(const [key,value]of Object.entries({cx:x*1000,cy:y*1000,r:5,fill:'none',stroke:o.lights.type==='taillight_pair'?'#ff8b8b':'#fff3b0','stroke-width':2}))circle.setAttribute(key,value);layer.append(circle);}
  }
  picture.append(layer);
}
function renderFacingInspector(c){
  const pane=$('facing-inspector'),index=selectedVehicle!=null?selectedVehicle:c.detections.findIndex(d=>d.label==='car'),d=c.detections[index];
  if(!d){pane.replaceChildren(node('p','Select a detected car to inspect its isolated crop.','widget-note'));delete pane.dataset.cropKey;return;}
  const source=c._archive?sourceEntry(c):c._renderSource,key=c.hash+':'+index+':'+signature(d.orientation);
  if(pane.dataset.cropKey===key)return;pane.dataset.cropKey=key;
  const canvas=node('canvas');canvas.width=224;canvas.height=224;canvas.setAttribute('role','img');canvas.setAttribute('aria-label',`Isolated vehicle #${index+1}`);
  const details=node('div'),o=d.orientation;details.append(node('h3',`Isolated vehicle #${index+1} · ${d.label}`),node('p',VehicleFacing.description(o),'facing-caption'));
  const lightText={headlight_pair:'Paired bright white/yellow pixels: possible headlights.',taillight_pair:'Paired red pixels: possible taillights.',ambiguous:'Both lamp colors present; lamp evidence is ambiguous.',none:'No reliable paired-lamp cue.'}[o?.lights?.type]||'Lamp evidence pending.';
  details.append(node('p',d.road_axis?'Local road axis available · '+(o?.road_direction?.state==='estimated'?'road-aligned estimated direction: '+o.road_direction.direction:o?.road_direction?.reason==='facing_road_axis_disagree'?'axis and facing disagree; direction unresolved':'waiting for consistent facing to choose its direction'):d.on_road===false?'Road model does not support this car footprint.':'Road-axis evidence is unavailable or ambiguous here.','widget-note'),node('p',lightText,'widget-note'),node('p',o?.model?`${o.view?.replaceAll('_',' ')} · model group score ${o.score?.toFixed(2)} (uncalibrated)${Number.isFinite(o.mirror_error_degrees)?' · mirror error '+Math.round(o.mirror_error_degrees)+'°':''} · ${o.crop?.width||'?'}×${o.crop?.height||'?'} source pixels`:'Car crops are evaluated by the pretrained eight-view model when enough source pixels are available.','widget-note'),node('p','This is an appearance estimate. Movement and stationary evidence stay separate.','widget-note'));
  pane.replaceChildren(canvas,details);
  (async()=>{const blob=source?.blob||await photoArchive.photo(c.hash);if(!blob||pane.dataset.cropKey!==key)return;const image=await createImageBitmap(blob);try{if(pane.dataset.cropKey!==key)return;const [x,y,xx,yy]=d.box;canvas.height=Math.max(1,Math.round(224*(yy-y)*image.height/((xx-x)*image.width)));const ctx=canvas.getContext('2d');ctx.drawImage(image,x*image.width,y*image.height,(xx-x)*image.width,(yy-y)*image.height,0,0,224,canvas.height);for(const [px,py]of o?.lights?.points||[]){ctx.beginPath();ctx.arc((px-x)/(xx-x)*224,(py-y)/(yy-y)*canvas.height,5,0,Math.PI*2);ctx.strokeStyle=o.lights.type==='taillight_pair'?'#ff8b8b':'#fff3b0';ctx.lineWidth=2;ctx.stroke();}}finally{image.close();}})().catch(()=>{if(pane.dataset.cropKey===key)canvas.setAttribute('aria-label','Saved crop unavailable');});
}

function renderRoadStatus(snapshots){
  const ready=snapshots.filter(c=>c.analysis?.road_model),directions=snapshots.flatMap(vehicles).filter(d=>d.orientation?.road_direction?.state==='estimated'),axes=snapshots.flatMap(vehicles).filter(d=>d.road_axis);
  $('road-status').textContent=`${ready.length}/${snapshots.length} filtered frames have road predictions · ${axes.length} local road axes · ${directions.length} road-aligned facing estimates. ${Number.isFinite(roadAnalyzer.lastInferenceMs)?roadAnalyzer.lastInferenceMs+' ms / scene · ':''}${roadAnalyzer.running?'Predicting one scene… ':''}${roadAnalyzer.backend==='webgpu'?'GPU / WebGPU':roadAnalyzer.backend==='wasm'?'CPU / WASM':roadAnalyzer.phase==='loading'?'Loading compact road model…':''}`;
  notifyPane('road','Road surface and direction',[ready.map(c=>[c.id,c.hash,c.analysis.road_backend]),axes.length,directions.length,roadAnalyzer.phase],$('road-updated'),$('road-pane'));
}

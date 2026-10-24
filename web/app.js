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
const gpuBatchSizes=[1,2,4,8];
let gpuBatchSize=(()=>{try{const n=Number(localStorage.getItem('locaintel.gpuBatch')||4);return gpuBatchSizes.includes(n)?n:4;}catch(_){return 4;}})();
let autoRefreshTimer, archivePaintTimer, archiveReady = Promise.resolve(), archiveError = '', seekSerial = 0;
const archiveMetadata = new Map(), historyPhotoSources = new Map(), publishedPhotoSources = new Map(), livePresentations = new Map();
const historyPhotoLoads = new Map();
const imageSizes = new Map(), cardNodes = new Map(), cardVersions = new Map(), markerNodes = new Map();
const publishedImports = new Set(); let publishedImportRunning = false;
const directionReferences=new Map(),headingWrites=new Map();let drawReference=false,referenceDrag=null,localHistoryOnly=false;
let pairedModel=null,pairedModelError='';
const modelUpdates=new Map();
const pairedContexts=new Map(),pairedWrites=new Map(),pairedHistoryLoads=new Map();
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
function modelCacheUpdate(update){
  modelUpdates.delete(update.file);modelUpdates.set(update.file,update);while(modelUpdates.size>12)modelUpdates.delete(modelUpdates.keys().next().value);
  const names={queued:'Queued',checking_local_cache:'Checking local cache',repairing_cache:'Replacing invalid cache',downloading:'Downloading',verifying:'Verifying checksum',cached:'Verified local cache',saved_locally:'Verified and saved locally',cache_unavailable:'Loaded; browser cache unavailable',error:'Failed; local retry pending'};
  $('model-cache-status').textContent=([...modelUpdates.values()].some(v=>v.scope==='worker_only')?'This browser supports one model-cache update at a time per worker; cross-worker locking is unavailable. ':'Core model updates run one at a time across workers and tabs. ')+'Photos never enter this model cache.';
  $('model-cache-updates').replaceChildren(...[...modelUpdates.values()].reverse().map(v=>node('li',`${v.file.split('/').pop()}: ${names[v.state]||v.state}${v.error?' · '+v.error:''}${v.bytes?' · '+Monitoring.bytes(v.bytes):''}`)));
}
function cacheFailure(error) { archiveError = error?.message || String(error); $('cache-status').textContent = 'Photo history could not be saved: ' + archiveError + '. Live monitoring continues.'; $('cache-status').classList.add('cache-error'); }
function archiveChanged(event = {}) {
  archiveRevision++;pairedContexts.clear();localFacingQueue.wake();
  if (event.clear) archiveMetadata.clear();
  if (event.row) { archiveMetadata.set(event.row.key, event.row); scheduleTemporal(event.row);queueViewPaint(); }
  for (const key of event.removedKeys || []) archiveMetadata.delete(key);
  if(event.clear||event.removedKeys?.length)queueViewPaint();
  if (event.reload) { photoArchive.metadata().then(rows => { archiveMetadata.clear(); for (const row of rows) archiveMetadata.set(row.key,row); updateArchiveUI(); }).catch(cacheFailure); }
  clearTimeout(archivePaintTimer); archivePaintTimer = setTimeout(updateArchiveUI, 500);
}
async function recordSource(camera, source) {
  if (source.error || !source.blob) return;
  try {
    await archiveReady;
    const local=browserDetector.get(source.hash);
    await photoArchive.saveFrame(camera,source,source.hash===camera.hash?{observedAt:Date.parse(camera.observed),analysisMode:'published',detections:camera.detections,counts:camera.counts,analysis:camera.analysis}:local?.status==='ready'?{analysisMode:'browser',detectorVersion:local.version,detections:local.detections,counts:local.counts,analysis:{road_model:null,motion:{status:'insufficient_cadence',gap_seconds:null}}}:{});
    if(visibleSourceCameras().some(c=>String(c.id)===String(camera.id)))pairCapture.request(camera.id,source.hash,Date.parse(source.checked));
  } catch (error) { cacheFailure(error); }
}
window.addEventListener('online',()=>localFacingQueue.wake());
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
  if(isLive()&&!document.hidden){browserDetector.wake();roadAnalyzer.wake();facingAnalyzer.wake();pairCapture.wake();localFacingQueue.wake();}else{pairCapture.pause();localFacingQueue.pause();}
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
  if(publishedImportRunning||!isLive()||document.hidden||data?.storage_mode==='browser-only')return;publishedImportRunning=true;
  try{
    const wanted=visibleSourceCameras();
    for(const c of wanted){
      const key=c.id+':'+c.hash;if(publishedImports.has(key))continue;if(!isLive()||document.hidden)break;
      await cameraSources.slot();
      try{
        await archiveReady;let blob=await photoArchive.photo(c.hash);
        if(!blob){const response=await fetch(`${c.frame}?v=${c.hash.slice(0,12)}`);if(!response.ok)continue;blob=await response.blob();}const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await blob.arrayBuffer())),b=>b.toString(16).padStart(2,'0')).join('');if(hash!==c.hash)continue;
        let dimensions=imageSizes.get(hash);if(!dimensions&&typeof createImageBitmap==='function'){const bitmap=await createImageBitmap(blob);dimensions=[bitmap.width,bitmap.height];bitmap.close();}
        const source={hash,url:URL.createObjectURL(blob),blob,cameraId:c.id,checked:c.observed,width:dimensions?.[0],height:dimensions?.[1]};publishedPhotoSources.set(hash,source);
        await archiveReady;await photoArchive.saveFrame(c,source,{observedAt:Date.parse(c.observed),analysisMode:'published',detections:c.detections,counts:c.counts,analysis:c.analysis});publishedImports.add(key);
      }catch(error){if(error.name==='QuotaExceededError')cacheFailure(error);}finally{cameraSources.release();}
    }
  }finally{publishedImportRunning=false;}
}

const cameraSources = new CameraSourceCache({concurrency:powerProfile.downloads,onUpdate:recordSource});
let detectionPaintTimer;
const browserDetector = new BrowserDetector({
  onChange: queueViewPaint,onModelUpdate:modelCacheUpdate,
  onResult: recordDetection,
  delayMs: powerProfile.inferenceDelay,
  isPaused: () => !isLive() || document.hidden,
  isWanted: hash => isLive() && $('source-view').checked && detectionTargets().some(c => cameraSources.get(c.id)?.hash === hash)

});
function localAnalysis(c){const source=sourceEntry(c);if(!source?.hash)return null;const current=browserDetector.get(source.hash);if(current)return current;const saved=archiveMetadata.get(c.id+':'+source.hash);const result=saved?.analysisMode==='browser'&&saved.detectorVersion===browserDetector.modelVersion?{version:saved.detectorVersion,detections:saved.detections,counts:saved.counts,observed:new Date(saved.observedAt).toISOString()}:saved?.browserResult;if(result?.version===browserDetector.modelVersion&&Array.isArray(result.detections))return{...result,status:'ready',hash:source.hash};return null;}
function withTemporal(c){
  const t=archiveMetadata.get(c.id+':'+c.hash)?.temporal;if(!t)return c;
  const detections=SpatialHeading.mergeTemporal(c.detections||[],t),states={moving:0,stationary:0,unknown:0};
  for(const d of detections)if(vehicleLabels.includes(d.label))states[motionState(d)]++;
  const sameLayout=t.log?.detectionSignature===SpatialHeading.detectionSignature(c.detections||[]);
  return {...c,detections,analysis:{...c.analysis,states,...(sameLayout?{motion:{status:t.log.status,temporal:t.log}}:{})}};
}

const publishedRoads=new Map(),roadWrites=new Set();
function persistRoad(source,index,result){const key=source.cameraId+':'+source.hash;if(roadWrites.has(key))return;roadWrites.add(key);archiveReady.then(()=>photoArchive.saveRoadScene(key,result)).catch(cacheFailure).finally(()=>roadWrites.delete(key));}
// WebGL and WebGPU jobs don't compete with CPU work, so they need not serialize.
const onGPU=backend=>backend==='webgpu'||backend==='webgl';
const roadAnalyzer=new VehicleFacing.CropOrientation({
  onModelUpdate:modelCacheUpdate,workerFactory:()=>new Worker('road-worker.js?v=local-1'),prepareCrop:RoadVision.prepare,delayMs:powerProfile.inferenceDelay,
  isPaused:()=>!isLive()||document.hidden||(browserDetector.running&&!onGPU(browserDetector.backend)),
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
  onModelUpdate:modelCacheUpdate,batchSize:gpuBatchSize,delayMs:powerProfile.inferenceDelay,
  isPaused:()=>!isLive()||document.hidden||!$('facing-enabled').checked||(browserDetector.running&&!onGPU(browserDetector.backend)&&!onGPU(facingAnalyzer.backend))||(roadAnalyzer.running&&!onGPU(roadAnalyzer.backend)&&!onGPU(facingAnalyzer.backend)),
  isWanted:hash=>isLive()&&(localFacingQueue.wants(hash)||visibleSourceCameras().some(c=>{const view=baseDisplayedCamera(c),row=archiveMetadata.get(c.id+':'+view.hash);return view.hash===hash||row&&PairedFacing.rowsFor(row,[...archiveMetadata.values()]).some(r=>r.hash===hash);})),
  onChange:()=>{queueViewPaint();localFacingQueue.wake();},
  onResult:persistFacing
});
const localFacingQueue=new LocalFacingQueue({rows:()=>[...archiveMetadata.values()],photo:hash=>photoArchive.photo(hash),analyzer:facingAnalyzer,isPaused:()=>!isLive()||document.hidden||!$('facing-enabled').checked,onError:cacheFailure});
function pairContext(c){
  const key=c.id+':'+c.hash,row=archiveMetadata.get(key);
  if(!pairedContexts.has(key)){if(pairedContexts.size>=128)pairedContexts.delete(pairedContexts.keys().next().value);pairedContexts.set(key,PairedFacing.context(row,[...archiveMetadata.values()]));}
  return pairedContexts.get(key);
}
// v3 results are single-photo estimates with a confidence tier; the paired
// screenshot confirmer only applies to archived results from the retired cascade.
function savedFacing(row,d){const key=VehicleFacing.fingerprint(d),r=row?.facing?.byFingerprint?.[key]||Object.values(row?.facing?.byDetection||{}).find(r=>r?.fingerprint===key);return r?.version===VehicleFacing.VERSION?r:PairedFacing.baseFor(row,d);}
function rawFacing(c,d){const saved=savedFacing(archiveMetadata.get(c.id+':'+c.hash),d),local=!c._archive?facingAnalyzer.get({cameraId:String(c.id),hash:c.hash},d):null;return saved?.state==='error'?local||saved:saved||local||null;}
function withFacing(c){
  const context=pairContext(c);
  return {...c,detections:(c.detections||[]).map(d=>{const raw=rawFacing(c,d),o=raw?.version===PairedFacing.BASE_VERSION?PairedFacing.combine(raw,d,context,pairedModel):raw;return {...d,orientation:o?{...o,road_direction:RoadVision.direction(d,o)}:null};})};
}
const pairCapture=new PairCapture({
  isPaused:()=>!isLive()||document.hidden||!$('facing-enabled').checked||!$('source-view').checked||localHistoryOnly,
  capture:async id=>{const camera=visibleSourceCameras().find(c=>String(c.id)===String(id));if(!camera)throw Error('Camera no longer viewed');const source=await cameraSources.load(camera,true);scheduleDetection();scheduleFacing();queueViewPaint();return source;},
  onError:()=>{} // Normal source status reports errors; retries are bounded.
});
async function schedulePairedHistory(c){
  for(const candidate of pairContext(c).candidates){
    const row=candidate.row,needed=[...new Set(candidate.matches.values())].filter(i=>!savedFacing(row,row.features.detections[i]));
    if(!needed.length||pairedHistoryLoads.has(row.key))continue;
    pairedHistoryLoads.set(row.key,true);
    try{const blob=await photoArchive.photo(row.hash);if(!blob||!isLive()||document.hidden)continue;const source={hash:row.hash,cameraId:String(row.cameraId),blob,width:row.features.width,height:row.features.height};for(const i of needed)facingAnalyzer.request(source,row.features.detections[i],i,200);}
    finally{pairedHistoryLoads.delete(row.key);}
  }
}
function persistPaired(c){
  const key=c.id+':'+c.hash,row=archiveMetadata.get(key);if(!row)return;
  const byFingerprint=Object.fromEntries(c.detections.filter(d=>d.orientation?.paired).map(d=>[VehicleFacing.fingerprint(d),{...d.orientation,road_direction:undefined}]));
  if(!Object.keys(byFingerprint).length)return;
  const result={version:PairedFacing.VERSION,byFingerprint},value=signature(result);
  if(signature(row.pairedFacing)===value||pairedWrites.get(key)===value)return;pairedWrites.set(key,value);
  archiveReady.then(()=>photoArchive.savePairedFacing(key,result)).catch(cacheFailure).finally(()=>{if(pairedWrites.get(key)===value)pairedWrites.delete(key);});
}
function displayedCamera(c) { const view=withFacing(withRoad(withTemporal(baseDisplayedCamera(c)))),size=imageSizes.get(view.hash);if(size){view._photoWidth=size[0];view._photoHeight=size[1];}return SpatialHeading.attach(view,archiveMetadata.get(view.id+':'+view.hash),directionReferences.get(String(view.id))); }
function scheduleFacing(){
  if(!data||!isLive()||document.hidden||!$('facing-enabled').checked)return;
  for(const original of visibleSourceCameras()){
    const c=displayedCamera(original),source=c._renderSource;
    if(!source?.blob)continue;
    for(const [i,d]of (c.detections||[]).entries())if(vehicleLabels.includes(d.label)){
      const identity={...source,cameraId:String(c.id)},row=archiveMetadata.get(c.id+':'+c.hash);
      const raw=rawFacing(c,d);
      if(raw?.state==='error'&&!facingAnalyzer.get(identity,d))facingAnalyzer.entries.set(facingAnalyzer.key(identity,d),raw);
      if(!raw||['queued','error'].includes(raw.state))facingAnalyzer.request(identity,d,i,c.id===selectedCamera&&i===(selectedVehicle??c.detections.findIndex(d=>d.label==='car'))?1000:(c.id===selectedCamera?100:0)+(d.on_road===true?30:d.on_road===false?0:10));
      else if(raw.version===VehicleFacing.VERSION&&['estimated','unknown'].includes(raw.state)&&row&&SpatialHeading.facing(row,d,i)?.version!==VehicleFacing.VERSION)persistFacing(identity,i,raw);
    }
    persistPaired(c);schedulePairedHistory(c).catch(cacheFailure);
  }
    persistHeadingsForVisible();
  facingAnalyzer.wake();localFacingQueue.wake();
}
function persistHeadingsForVisible(){
  for(const original of visibleSourceCameras()){
    const c=displayedCamera(original),key=c.id+':'+c.hash,row=archiveMetadata.get(key);if(!row)continue;
    const byFingerprint=Object.fromEntries((c.detections||[]).filter(d=>d.heading&&d.orientation&&!['queued','processing','error'].includes(d.orientation.state)).map(d=>[VehicleFacing.fingerprint(d),d.heading]));
    if(!Object.keys(byFingerprint).length)continue;const headings={version:SpatialHeading.VERSION,byFingerprint},value=signature(headings);
    if(signature(row.headings)===value||headingWrites.get(key)===value)continue;headingWrites.set(key,value);
    archiveReady.then(()=>photoArchive.saveHeadings(key,headings)).catch(cacheFailure).finally(()=>{if(headingWrites.get(key)===value)headingWrites.delete(key);});
  }
}
function baseDisplayedCamera(c) {
  if(c._archive)return c;
  if(!$('source-view').checked)return {...c,_renderSource:publishedPhotoSources.get(c.hash)||null};
  const local=localAnalysis(c),source=sourceEntry(c);
  if(source?.url&&(source.hash===c.hash||local?.status==='ready')){
    const states={moving:0,stationary:0,unknown:local?.detections?.filter(d=>vehicleLabels.includes(d.label)).length||0};
    const view=source.hash===c.hash?{...c,_renderSource:source}:{...c,hash:source.hash,detections:local.detections,counts:local.counts,observed:local.observed,browser_analysis:true,_analysisAvailable:true,_renderSource:source,analysis:{states,road_model:null,motion:{status:'insufficient_cadence',gap_seconds:null}}};
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
    needing.length ? `Browser YOLO: ${completed}/${needing.length} fresh frames detected${browserDetector.backend==='webgl'?' · GPU / WebGL':browserDetector.backend==='webgpu'?' · GPU / WebGPU':browserDetector.backend==='wasm'?' · CPU / WASM':''}${Number.isFinite(browserDetector.lastInferenceMs)?' · '+browserDetector.lastInferenceMs+' ms':''}${browserDetector.state === 'loading' ? ' · downloading model on first use' : completed < needing.length ? ' · processing images in the background' : ' · current-source boxes ready'}.` : 'YOLO boxes match the displayed analyzed frames. Fresh-image detection loads when needed.';
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
  if(localHistoryOnly){loadSavedVisibleSources();return;}
  if(data?.storage_mode==='browser-only')loadSavedVisibleSources();
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
  return viewCameras().filter(c => (!$('region').value || c.region === $('region').value) && (!$('route-highway').value || String(c.highway) === $('route-highway').value) &&
    `${c.name} ${c.highway} ${c.caption}`.toLowerCase().includes(search));
}
function age(c) {
  if(!c.observed)return 'Awaiting local analysis';
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
  const missingPhoto=!source?.url&&(!c.frame||c._archive);
  const mismatched=missingPhoto; 
  if(!missingPhoto)image.src=source?.url||`${c.frame}?v=${c.hash.slice(0,12)}`;
  image.onerror = () => { image.alt = 'Cached image unavailable'; };
  image.onload = function(){
    const target=this.closest('.picture');if(!target)return;
    if(this.naturalWidth&&this.naturalHeight){imageSizes.set(c.hash,[this.naturalWidth,this.naturalHeight]);target.style.aspectRatio=`${this.naturalWidth} / ${this.naturalHeight}`;if(!mismatched)renderVectorLayer(target,c,detail,this.naturalWidth,this.naturalHeight,selectedVehicle);}
  };
  const knownSize=imageSizes.get(c.hash)||(source?.width&&source?.height?[source.width,source.height]:null);if(knownSize)picture.style.aspectRatio=`${knownSize[0]} / ${knownSize[1]}`;
  picture.append(image,node('span',`HWY ${c.highway}`,'tag'),node('span',c._archive?'SAVED PHOTO':c._waitingForSource?'UPDATING · PREVIOUS FRAME':source?.url?(c.browser_analysis?'CURRENT SOURCE · LOCAL YOLO':'ANALYSIS MATCHED'):'ANALYZED FRAME','frame-kind'));
  if(missingPhoto){picture.dataset.loading='true';image.alt=c._archive?'Loading saved photo…':'Loading photo directly from DriveBC…';return picture;}

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
  if (!mismatched) {renderHeadingLayer(picture,c,detail);renderVectorLayer(picture,c,detail,...(imageSizes.get(c.hash)||[640,400]),selectedVehicle);renderFacingLayer(picture,c,detail,selectedVehicle);}
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
    row.append(node('span', `#${i + 1} ${d.label} · ${Math.round(d.confidence * 100)}%`), node('span', `${VehicleFacing.description(d.orientation)}${d.heading?.direction?' · '+(d.heading.measured_motion?'Measured heading ':'Estimated facing direction ')+d.heading.direction:''}${d.orientation?.road_direction?.state==='estimated'?' · road direction '+d.orientation.road_direction.direction:''} · ${motionDescription(d)} · ${road}`));
    return row;
  });
  renderFacingInspector(c);renderHeadingControls(c);
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
    const facing=vehicles(c).filter(d=>d.orientation?.state==='estimated'),states=motionCounts([c]);
    details.append(node('div',`Facing: ${facing.filter(d=>d.orientation.facing==='toward_camera').length} toward · ${facing.filter(d=>d.orientation.facing==='away_from_camera').length} away · ${facing.filter(d=>d.orientation.facing?.startsWith('image_')).length} side${states.moving||states.stationary?` · motion ${states.moving} moving / ${states.stationary} stationary`:''}`,'facing-tag'));
    const sources=node('div',undefined,'camera-sources');for(const [text,url]of[['DriveBC camera',c.source],['Original image',c.image]])if(url)sources.append(link(text+' ↗',url));details.append(sources);
    details.append(node('div',sourceLabel(original),'source-status'));
    const inspect=node('button','Inspect','inspect-button');inspect.onclick=()=>{selectCamera(c.id);$('detail-pane').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'center'});};inspect.title='Hover or focus a card to replay its saved frames';details.append(inspect,node('span',undefined,'update-badge'));
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
  window.renderRoutes?.();
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
  $('status').textContent = `${isLive()?(data.storage_mode==='browser-only'?'Browser-only analysis · camera catalog ':'Published analysis ')+new Date(data.generated).toLocaleString():'Local history at '+new Date(timelineAt).toLocaleString()}${isLive()&&data.errors?.length ? ' · ' + data.errors.length + ' fetch errors' : ''}`;
}
function apply(payload, history) {
  const localLatest=new Map(Monitoring.latestAt([...archiveMetadata.values()],Date.now()).map(r=>[String(r.cameraId),r]));
  if(payload.storage_mode==='browser-only')payload={...payload,cameras:payload.cameras.map(c=>{
    const saved=localLatest.get(String(c.id));
    return saved?{...c,hash:saved.hash,observed:new Date(saved.observedAt).toISOString(),detections:SpatialHeading.frameDetections(saved),counts:saved.browserResult?.counts||saved.counts||{},analysis:saved.analysis||{},_analysisAvailable:Array.isArray(saved.detections)||!!saved.browserResult}: {...c,hash:'',observed:null,detections:[],counts:{},analysis:{},_analysisAvailable:false};
  })};
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
    if (results[0].status === 'fulfilled') {await archiveReady;localHistoryOnly=false;apply(results[0].value, results[1].status === 'fulfilled' ? results[1].value : null);}
    else { analysisError = true; if (!data) throw results[0].reason; }
    historyError = results[1].status === 'rejected';
    const sources = fromSource ? await loadVisibleSources(true) : [];
    const failed = sources.filter(s => s.error).length;
    $('refresh-result').textContent = (fromSource ? (sources.length ? `${sources.length - failed}/${sources.length} visible camera images checked directly with DriveBC.${failed ? ' ' + failed + ' source requests failed; last good frames retained.' : ''}` : 'No visible cameras match the current filters.') : 'Published analysis checked.') + (analysisError ? ' Analysis refresh failed; previous results retained.' : ' Analysis published ' + new Date(data.generated).toLocaleString() + '.') + (historyError ? ' Pattern history unavailable; prior history retained.' : '');
  } catch (e) {
    $('refresh-result').textContent = 'Refresh failed. Try again shortly.' + (data ? ' Keeping the last loaded analysis.' : '');
    if(!data&&await restoreLocalArchive()){ $('refresh-result').textContent='Showing photos and analysis saved in this browser. Network sources are unavailable.';return;}
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
if (new URLSearchParams(location.search).get('serverEvents')==='1') {
  const stream = new EventSource('api/events'); stream.onmessage = async e => { const p = JSON.parse(e.data); if(p.cameras){if(isLive())apply(p);else data=p;} }; stream.onerror = () => stream.close();
}

// Monitoring preferences and persistent local history.
$('power-budget').addEventListener('input',()=>setPower($('power-budget').value));
function setGPUBatch(value){
  gpuBatchSize=gpuBatchSizes[Math.max(0,Math.min(3,Number(value)||0))];
  $('gpu-batch').value=gpuBatchSizes.indexOf(gpuBatchSize);
  const label=gpuBatchSize===1?'Off · one car':'Up to '+gpuBatchSize+' cars';
  $('gpu-batch-label').value=label;$('gpu-batch').setAttribute('aria-valuetext',label);
  $('gpu-batch-description').textContent=gpuBatchSize===1?'Process one car at a time with lower GPU memory use. Both verification views stay enabled.':'Process up to '+gpuBatchSize+' cars together on the GPU. Higher batches can improve throughput but use more GPU memory and model-cache storage. Small queues run immediately; unsupported batches use individual inference. Photo storage is separate.';
  try{localStorage.setItem('locaintel.gpuBatch',String(gpuBatchSize));}catch(_){}
  facingAnalyzer.setBudget({batchSize:gpuBatchSize});queueViewPaint();
}
$('gpu-batch').addEventListener('input',()=>setGPUBatch($('gpu-batch').value));
setGPUBatch(gpuBatchSizes.indexOf(gpuBatchSize));
let seekTimer;
$('timeline-position').addEventListener('input',()=>{stopPlayback();const time=Number($('timeline-position').value);timelineAt=time;updateBackgroundBudget();$('timeline-title').textContent='Reviewing '+new Date(time).toLocaleString();clearTimeout(seekTimer);seekTimer=setTimeout(()=>seekHistory(time),120);});
$('timeline-position').addEventListener('change',()=>{clearTimeout(seekTimer);seekHistory(Number($('timeline-position').value));});
$('timeline-date').addEventListener('change',()=>{const time=new Date($('timeline-date').value).getTime();if(Number.isFinite(time))seekHistory(time);});
$('go-live').onclick=goLive;$('timeline-prev').onclick=()=>stepHistory(-1);$('timeline-next').onclick=()=>stepHistory(1);$('timeline-play').onclick=startPlayback;
async function configureStorage(){try{archiveError='';$('cache-status').classList.remove('cache-error');await archiveReady;await photoArchive.configure({budgetBytes:Number($('cache-limit').value)*Monitoring.MB,retentionMs:Number($('cache-retention').value)*3600000});await updateArchiveUI();}catch(error){cacheFailure(error);}}
$('cache-limit').addEventListener('change',configureStorage);$('cache-retention').addEventListener('change',configureStorage);
$('keep-browser-data').onclick=async()=>{try{const granted=await navigator.storage?.persist?.();$('retention-status').textContent=granted?'Browser persistence granted. Photos still follow your size and age limits.':'Browser persistence was not granted. Saved data remains subject to browser eviction.';}catch(error){$('retention-status').textContent=error.message;}};
$('clear-history').onclick=async()=>{if(!confirm('Delete saved camera photos from this browser? This cannot be undone.'))return;try{await photoArchive.clear();for(const source of historyPhotoSources.values())URL.revokeObjectURL(source.url);historyPhotoSources.clear();goLive({refreshSources:false});await updateArchiveUI();}catch(error){cacheFailure(error);}};
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopPlayback();updateBackgroundBudget();if(!document.hidden&&isLive()){scheduleSources();scheduleDetection();}});
window.addEventListener('monitor-frame-commit',releaseOldPhotoURLs);
archiveReady=photoArchive.initialize().then(async()=>{const rows=await photoArchive.metadata();for(const row of rows){archiveMetadata.set(row.key,row);scheduleTemporal(row);}for(const [id,reference]of await photoArchive.references())directionReferences.set(id,reference);for(const row of rows.filter(r=>r.analysisMode==='browser'&&r.detectorVersion===browserDetector.modelVersion&&Array.isArray(r.detections)).sort((a,b)=>b.observedAt-a.observedAt).slice(0,128))browserDetector.entries.set(row.hash,{status:'ready',hash:row.hash,version:row.detectorVersion,detections:row.detections,counts:row.counts||{},observed:new Date(row.observedAt).toISOString()});archiveRevision++;await updateArchiveUI();}).catch(cacheFailure);
fetch('models/paired-facing-v1.json?v=local-1').then(r=>{if(!r.ok)throw Error('Paired model unavailable');return r.json();}).then(m=>{pairedModel=PairedFacing.validateModel(m);queueViewPaint();}).catch(e=>{pairedModelError=e.message;queueViewPaint();});
setPower(powerProfile.level);refresh(true);

// Temporal work runs serially with the same device budget as inference.
const temporalPending=new Set(),hoverCards=new WeakSet();let temporalRunning=false;
function scheduleTemporal(row){
  if(!SpatialHeading.frameDetections(row).length||!isLive())return;
  temporalPending.add(row.cameraId);setTimeout(runTemporal,0);
}
async function runTemporal(){
  if(temporalRunning||!isLive()||document.hidden||!temporalPending.size)return;
  temporalRunning=true;const id=temporalPending.values().next().value;temporalPending.delete(id);
  try{
    const rows=[...archiveMetadata.values()].filter(r=>String(r.cameraId)===String(id)&&SpatialHeading.frameDetections(r).length).sort((a,b)=>a.observedAt-b.observedAt).slice(-PairedFacing.MAX_FRAMES),latest=rows.at(-1),frames=[];
    for(const row of rows){
      let features=row.features;
      if(!PairedFacing.validFeatures(row)){const blob=await photoArchive.photo(row.hash);if(!blob)continue;features={...await TemporalTraffic.features(blob,SpatialHeading.frameDetections(row)),version:1,hash:row.hash,detectionSignature:SpatialHeading.detectionSignature(SpatialHeading.frameDetections(row))};await photoArchive.saveFeatures(row.key,features);}
      frames.push({...row,...features});
    }
    const motionFrames=frames.slice(-3);
    if(latest&&motionFrames.at(-1)?.hash===latest.hash&&(signature(latest.temporal?.log?.frameHashes)!==signature(motionFrames.map(r=>r.hash))||latest.temporal?.log?.detectionSignature!==SpatialHeading.detectionSignature(SpatialHeading.frameDetections(latest)))){
      const result=TemporalTraffic.analyze(motionFrames);result.log.detectionSignature=SpatialHeading.detectionSignature(motionFrames.at(-1).detections);result.scene={background:motionFrames.at(-1).background,pixelSignature:motionFrames.at(-1).pixelSignature,width:motionFrames.at(-1).width,height:motionFrames.at(-1).height};await photoArchive.saveTemporal(latest.key,result);queueViewPaint();
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
  if(motionDownloadURL)URL.revokeObjectURL(motionDownloadURL);motionDownloadURL=URL.createObjectURL(new Blob([JSON.stringify(rows.map(r=>({cameraId:r.cameraId,hash:r.hash,...r.temporal,facing:r.facing||null,pairedFacing:r.pairedFacing||null,features:r.features||null,road:r.roadScene||null,headings:r.headings||null,directionReferences:directionReferences.get(r.cameraId)||null})),null,2)],{type:'application/json'}));const download=node('a','Download camera motion log (JSON)');download.href=motionDownloadURL;download.download=`camera-${selectedCamera}-motion.json`;pane.replaceChildren(summary,explain,list,download);
}

try{$('facing-enabled').checked=localStorage.getItem('locaintel.facing')!=='off';}catch(_){}
$('retry-facing').onclick=async()=>{
  facingAnalyzer.retryFailed();
  for(const row of archiveMetadata.values())for(const [i,d]of SpatialHeading.frameDetections(row).entries()){const result=savedFacing(row,d);if(result?.state==="error")await photoArchive.saveOrientation(row.key,i,{...result,retry_attempts:0,next_retry_at:0,retry_exhausted:false});}
  localFacingQueue.wake();scheduleFacing();queueViewPaint();
};
$('facing-enabled').addEventListener('change',()=>{try{localStorage.setItem('locaintel.facing',$('facing-enabled').checked?'on':'off');}catch(_){}facingAnalyzer.setBudget({delayMs:powerProfile.inferenceDelay});roadAnalyzer.setBudget({delayMs:powerProfile.inferenceDelay});scheduleFacing();localFacingQueue.wake();queueViewPaint();});
function renderFacingStatus(snapshots){
  const ds=snapshots.flatMap(vehicles),totals={toward_camera:0,away_from_camera:0,side:0,unresolved:0},tiers={high:0,medium:0,low:0};let completed=0;
  for(const d of ds){if(d.orientation&&!['queued','processing'].includes(d.orientation.state))completed++;if(d.orientation?.state!=='estimated'){totals.unresolved++;continue;}if(d.orientation.tier)tiers[d.orientation.tier]++;if(d.orientation.facing==='toward_camera'||d.orientation.facing==='away_from_camera')totals[d.orientation.facing]++;else totals.side++;}
  SmoothUI.list($('facing-totals'),Object.entries(totals).map(([key,n])=>{const cell=node('div');cell.dataset.key=key;cell.append(node('strong',String(n)),node('span',{toward_camera:'FACING CAMERA',away_from_camera:'FACING AWAY',side:'SIDE VIEW',unresolved:'WAITING / SKIPPED'}[key]));return cell;}));
  $('facing-status').textContent=!isLive()?'Saved facing estimates at the timeline cursor; no new crop inference.':!$('facing-enabled').checked?'New facing inference paused; saved estimates remain visible.':facingAnalyzer.phase==='loading'?'Loading the cached vehicle-view model (GPU where available); YOLO boxes stay visible.':facingAnalyzer.phase==='error'?'Facing model unavailable. Refresh retries failed crops; vehicle boxes and cached history remain available.':`${totals.toward_camera+totals.away_from_camera+totals.side}/${ds.length} vehicles have facing (${tiers.high} high · ${tiers.medium} medium · ${tiers.low} low confidence) · ${facingAnalyzer.queue.length} queued${facingAnalyzer.backend==='webgl'?' · GPU / WebGL':facingAnalyzer.backend==='webgpu'?' · GPU / WebGPU':facingAnalyzer.backend==='wasm'?' · CPU fallback / WASM':''}${facingAnalyzer.backend==='wasm'&&facingAnalyzer.fallback?' · '+facingAnalyzer.fallback:''}${Number.isFinite(facingAnalyzer.lastInferenceMs)?' · '+(facingAnalyzer.lastReused?'exact crop reused':facingAnalyzer.lastInferenceMs+' ms / crop'):''}${facingAnalyzer.running?' · analyzing one crop':''}. Runs on viewed feeds; vehicles under 8×6 pixels, motorcycles and bicycles are skipped.`;
  notifyPane('facing','Vehicle facing',[totals,completed,facingAnalyzer.phase,facingAnalyzer.backend,facingAnalyzer.path,facingAnalyzer.lastInferenceMs,isLive(),$('facing-enabled').checked],$('facing-updated'),$('facing-pane'));
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
  if(o?.paired)details.append(node('p',o.paired.state==='confirmed'?`${o.paired.count} distinct cached screenshots agree · ${o.paired.frameHashes.map(h=>h.slice(0,8)).join(' / ')}`:`Awaiting confirmed second screenshot · ${o.reason?.replaceAll('_',' ')}`,'widget-note'));
  const lightText={headlight_pair:'Paired bright white/yellow pixels: possible headlights.',taillight_pair:'Paired red pixels: possible taillights.',ambiguous:'Both lamp colors present; lamp evidence is ambiguous.',none:'No reliable paired-lamp cue.'}[o?.lights?.type]||'Lamp evidence pending.';
  details.append(node('p',d.road_axis?'Local road axis available · '+(o?.road_direction?.state==='estimated'?'road-aligned estimated direction: '+o.road_direction.direction:o?.road_direction?.reason==='facing_road_axis_disagree'?'axis and facing disagree; direction unresolved':'waiting for consistent facing to choose its direction'):d.on_road===false?'Road model does not support this car footprint.':'Road-axis evidence is unavailable or ambiguous here.','widget-note'),node('p',lightText,'widget-note'),node('p',o?.model?`${o.tier||'?'} confidence · ${o.group||''} probability ${Number.isFinite(o.confidence)?o.confidence.toFixed(2):'?'}${o.truncated?' · cut off at image edge':''} · ${o.crop?.width||'?'}×${o.crop?.height||'?'} source pixels`:'Vehicles at least 8×6 source pixels are evaluated by the DriveBC-adapted facing model.','widget-note'),node('p','This is an appearance estimate. Movement and stationary evidence stay separate.','widget-note'));
  if(o?.model)details.append(node('p',`${o.model}${o.backend==='webgl'?' · GPU / WebGL':o.backend==='webgpu'?' · GPU / WebGPU':o.backend==='wasm'?' · CPU fallback / WASM':''}${o.path==='mirror-averaged'?' · crop + mirror averaged':''}${Number.isFinite(o.inference_ms)?' · '+o.inference_ms+' ms':''}${o.backend==='wasm'&&o.fallback?' · '+o.fallback:''}`,'widget-note'));
  pane.replaceChildren(canvas,details);
  (async()=>{const blob=source?.blob||await photoArchive.photo(c.hash);if(!blob||pane.dataset.cropKey!==key)return;const image=await createImageBitmap(blob);try{if(pane.dataset.cropKey!==key)return;const [x,y,xx,yy]=d.box;canvas.height=Math.max(1,Math.round(224*(yy-y)*image.height/((xx-x)*image.width)));const ctx=canvas.getContext('2d');ctx.drawImage(image,x*image.width,y*image.height,(xx-x)*image.width,(yy-y)*image.height,0,0,224,canvas.height);for(const [px,py]of o?.lights?.points||[]){ctx.beginPath();ctx.arc((px-x)/(xx-x)*224,(py-y)/(yy-y)*canvas.height,5,0,Math.PI*2);ctx.strokeStyle=o.lights.type==='taillight_pair'?'#ff8b8b':'#fff3b0';ctx.lineWidth=2;ctx.stroke();}}finally{image.close();}})().catch(()=>{if(pane.dataset.cropKey===key)canvas.setAttribute('aria-label','Saved crop unavailable');});
}

function renderRoadStatus(snapshots){
  const ready=snapshots.filter(c=>c.analysis?.road_model),directions=snapshots.flatMap(vehicles).filter(d=>d.orientation?.road_direction?.state==='estimated'),axes=snapshots.flatMap(vehicles).filter(d=>d.road_axis);
  $('road-status').textContent=`${ready.length}/${snapshots.length} filtered frames have road predictions · ${axes.length} local road axes · ${directions.length} road-aligned facing estimates. ${Number.isFinite(roadAnalyzer.lastInferenceMs)?roadAnalyzer.lastInferenceMs+' ms / scene · ':''}${roadAnalyzer.running?'Predicting one scene… ':''}${roadAnalyzer.backend==='webgl'?'GPU / WebGL':roadAnalyzer.backend==='webgpu'?'GPU / WebGPU':roadAnalyzer.backend==='wasm'?'CPU / WASM':roadAnalyzer.phase==='loading'?'Loading compact road model…':''}`;
  notifyPane('road','Road surface and direction',[ready.map(c=>[c.id,c.hash,c.analysis.road_backend]),axes.length,directions.length,roadAnalyzer.phase],$('road-updated'),$('road-pane'));
}

function headingScene(c){const row=archiveMetadata.get(c.id+':'+c.hash),size=imageSizes.get(c.hash);return {hash:c.hash,width:size?.[0]||row?.width||640,height:size?.[1]||row?.height||400,background:row?.temporal?.scene?.background};}
function drawSpatialArrow(svg,start,end,color,label,dashed=false){
  const add=(tag,attrs)=>{const n=document.createElementNS(svg.namespaceURI,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);svg.append(n);return n;};
  const dx=end[0]-start[0],dy=end[1]-start[1],length=Math.hypot(dx,dy);if(length<1)return;
  const ux=dx/length,uy=dy/length,head=Math.min(14,length*.3),base=[end[0]-ux*head,end[1]-uy*head];
  const line=add('line',{x1:start[0],y1:start[1],x2:end[0],y2:end[1],stroke:color,'stroke-width':3,'vector-effect':'non-scaling-stroke',...(dashed?{'stroke-dasharray':'5 4'}:{})});
  const title=document.createElementNS(svg.namespaceURI,'title');title.textContent=label;line.append(title);
  add('polygon',{points:[end,[base[0]-uy*head*.5,base[1]+ux*head*.5],[base[0]+uy*head*.5,base[1]-ux*head*.5]].map(p=>p.join(',')).join(' '),fill:color});
}
function renderHeadingLayer(picture,c,detail){
  picture.querySelector('.spatial-vector-layer')?.remove();if(!detail||!$('heading-field').checked)return;
  const scene=headingScene(c),{width,height}=scene,reference=directionReferences.get(String(c.id)),svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.classList.add('spatial-vector-layer');svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('preserveAspectRatio','none');svg.setAttribute('role','img');svg.setAttribute('aria-label','Vehicle direction evidence and user reference vectors');
  for(const [i,v]of (reference?.vectors||[]).entries())drawSpatialArrow(svg,v.start.map((x,j)=>x*(j?height:width)),v.end.map((x,j)=>x*(j?height:width)),SpatialHeading.stableView(reference,scene)?'#c3a1ff':'#7b7d88',`Reference ${i+1}: user annotation; not measured movement`,true);
  for(const [i,d]of (c.detections||[]).entries()){
    if(!vehicleLabels.includes(d.label))continue;const h=d.heading,center=[(d.box[0]+d.box[2])/2*width,d.box[3]*height];
    if(h?.vector){const unit=[h.vector[0]*width,h.vector[1]*height],n=Math.hypot(...unit),length=Math.min(width,height)*.09,start=center.map((v,j)=>v-unit[j]/n*length*.4),end=center.map((v,j)=>Math.max(2,Math.min((j?height:width)-2,v+unit[j]/n*length*.6)));drawSpatialArrow(svg,start,end,h.measured_motion?'#c5ff72':'#88e2f2',`Vehicle #${i+1}: ${h.measured_motion?'measured motion':'estimated facing'} ${h.direction}; ${h.sources.join(', ')}`,!h.measured_motion);}
    else{const dot=document.createElementNS(svg.namespaceURI,'circle');for(const [k,v]of Object.entries({cx:center[0],cy:center[1],r:3,fill:'#efbd7b'}))dot.setAttribute(k,v);const title=document.createElementNS(svg.namespaceURI,'title');title.textContent=`Vehicle #${i+1}: ${d.orientation?.state==='queued'||d.orientation?.state==='processing'?'facing inference pending':h?.reason||'direction unresolved'}`;dot.append(title);svg.append(dot);}
  }
  picture.classList.toggle('drawing-reference',drawReference);picture.append(svg);
}
function renderHeadingControls(c){
  const reference=directionReferences.get(String(c.id)),scene=headingScene(c),vectors=reference?.vectors||[],ds=vehicles(c),known=ds.filter(d=>d.heading?.direction),measured=known.filter(d=>d.heading.measured_motion);
  $('heading-status').textContent=`${known.length}/${ds.length} vehicle directions · ${measured.length} measured · ${known.length-measured.length} appearance estimates · ${vectors.length}/64 reference arrows. ${vectors.length&&!SpatialHeading.stableView(reference,scene)?'Reference view is unverified or changed; annotations cannot influence classification. ':''}${drawReference?'Drag from tail to arrowhead on the camera photo. ':''}Cyan is estimated facing, green is measured motion, purple is your reference geometry. Arrow length is not speed.`;
  $('draw-reference').setAttribute('aria-pressed',String(drawReference));$('draw-reference').textContent=drawReference?'Finish drawing':'Draw direction references';$('draw-reference').disabled=!isLive()||!c._renderSource?.blob;
  $('undo-reference').disabled=!isLive()||!vectors.length;$('clear-references').disabled=!isLive()||!vectors.length;
  const items=vectors.map((v,i)=>{const li=node('li');li.dataset.key=String(i);li.textContent=`${i+1}: (${v.start.map(n=>n.toFixed(2)).join(', ')}) → (${v.end.map(n=>n.toFixed(2)).join(', ')})`;return li;});SmoothUI.list($('reference-list'),items);
}
async function changeReferences(c,vectors){
  const normalized=SpatialHeading.sanitize(vectors),scene=headingScene(c),prior=directionReferences.get(String(c.id));
  if(normalized.length&&prior?.vectors?.length&&!SpatialHeading.stableView(prior,scene))throw Error('The camera view changed or is unverified. Clear its old references before adding new arrows.');
  let background=scene.background;
  if(!background&&c._renderSource?.blob)background=(await TemporalTraffic.features(c._renderSource.blob,c.detections||[])).background;
  const reference={vectors:normalized,frameHash:c.hash,width:scene.width,height:scene.height,background,revision:(prior?.revision||0)+1,updatedAt:Date.now()};
  await photoArchive.saveReferences(String(c.id),reference);directionReferences.set(String(c.id),reference);queueViewPaint();renderDetail();persistHeadingsForVisible();
}
$('heading-field').addEventListener('change',()=>{if(!$('heading-field').checked)drawReference=false;renderDetail();});
$('draw-reference').onclick=()=>{drawReference=!drawReference;if(drawReference)$('heading-field').checked=true;renderDetail();};
for(const [id,update]of [['undo-reference',v=>v.slice(0,-1)],['clear-references',()=>[]]])$(id).onclick=async()=>{try{const original=viewCameras().find(c=>c.id===selectedCamera);if(original)await changeReferences(displayedCamera(original),update(directionReferences.get(String(selectedCamera))?.vectors||[]));}catch(error){$('heading-status').textContent=error.message;}};
$('detail-pane').addEventListener('pointerdown',e=>{
  const picture=e.target.closest('#detail-picture');if(!drawReference||!picture||!isLive())return;
  const original=viewCameras().find(c=>c.id===selectedCamera);if(!original)return;const c=displayedCamera(original),rect=picture.getBoundingClientRect();
  e.preventDefault();picture.setPointerCapture(e.pointerId);referenceDrag={camera:c,picture,pointerId:e.pointerId,rect,start:[(e.clientX-rect.left)/rect.width,(e.clientY-rect.top)/rect.height]};
});
$('detail-pane').addEventListener('pointerup',async e=>{
  if(!referenceDrag||referenceDrag.pointerId!==e.pointerId)return;const drag=referenceDrag;referenceDrag=null;e.preventDefault();
  try{const original=viewCameras().find(c=>c.id===selectedCamera);if(!original||original.id!==drag.camera.id||displayedCamera(original).hash!==drag.camera.hash)throw Error('The displayed photo changed. Draw this arrow again on the current photo.');
    const end=[Math.max(0,Math.min(1,(e.clientX-drag.rect.left)/drag.rect.width)),Math.max(0,Math.min(1,(e.clientY-drag.rect.top)/drag.rect.height))];
    await changeReferences(drag.camera,[...(directionReferences.get(String(selectedCamera))?.vectors||[]),{start:drag.start,end}]);
  }catch(error){$('heading-status').textContent=error.message;}
});
$('detail-pane').addEventListener('pointercancel',()=>{referenceDrag=null;});
$('reference-coordinates').onsubmit=async e=>{e.preventDefault();try{const original=viewCameras().find(c=>c.id===selectedCamera);if(!original||!isLive())return;const v={start:[$('reference-x1'),$('reference-y1')].map(n=>Number(n.value)),end:[$('reference-x2'),$('reference-y2')].map(n=>Number(n.value))};await changeReferences(displayedCamera(original),[...(directionReferences.get(String(selectedCamera))?.vectors||[]),v]);}catch(error){$('heading-status').textContent=error.message;}};
const savedSourceLoads=new Map();
async function loadSavedVisibleSources(){
  for(const c of visibleSourceCameras())if(!publishedPhotoSources.has(c.hash)&&!savedSourceLoads.has(c.hash)){
    const work=photoArchive.photo(c.hash).then(blob=>{if(blob){publishedPhotoSources.set(c.hash,{hash:c.hash,blob,url:URL.createObjectURL(blob),cameraId:c.id,checked:c.observed});queueViewPaint();}}).catch(cacheFailure).finally(()=>savedSourceLoads.delete(c.hash));savedSourceLoads.set(c.hash,work);
  }
  await Promise.all(savedSourceLoads.values());
}
async function restoreLocalArchive(){
  await archiveReady;const rows=Monitoring.latestAt([...archiveMetadata.values()],Date.now());if(!rows.length)return false;
  localHistoryOnly=true;$('source-view').checked=false;
  const cameras=rows.map(row=>({...row.camera,hash:row.hash,observed:new Date(row.observedAt).toISOString(),detections:SpatialHeading.frameDetections(row),counts:row.browserResult?.counts||row.counts||{},analysis:row.analysis||{},frame:'frames/'+row.cameraId+'.jpg'}));
  apply({cameras,generated:new Date(Math.max(...rows.map(r=>r.observedAt))).toISOString(),catalog_size:cameras.length,model:'Saved browser analysis',errors:[]},Monitoring.patternsAt([...archiveMetadata.values()],Date.now()));await loadSavedVisibleSources();return true;
}
if('serviceWorker'in navigator){
  // When an updated worker takes over a tab that was running older scripts, reload once.
  const hadController=!!navigator.serviceWorker.controller;let reloaded=false;
  navigator.serviceWorker.addEventListener('controllerchange',()=>{if(hadController&&!reloaded){reloaded=true;location.reload();}});
  navigator.serviceWorker.register('sw.js').then(r=>r.update()).catch(error=>console.warn('Offline shell unavailable',error));
}

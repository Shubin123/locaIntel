/* Local-only scale verification: 1,000 native photos, real GPU model, IndexedDB.
 * Prepare with python -m pipeline.prepare_facing_verification. No images leave
 * localhost; no screenshots/profiles/traces/photos are uploaded or published. */
const {chromium}=require('playwright');
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const option=(name,fallback)=>process.argv.find(v=>v.startsWith('--'+name+'='))?.slice(name.length+3)||fallback;
const directory=path.resolve(option('data','.cache/facing-verification')),root=path.resolve(__dirname,'../web');
const manifest=JSON.parse(fs.readFileSync(path.join(directory,'manifest.json'))),count=1000;
assert.equal(manifest.count,count);assert.equal(manifest.photos.length,count);assert.equal(new Set(manifest.photos.map(p=>p.hash)).size,count);
const photosByHash=new Map(manifest.photos.map(p=>[p.hash,p]));
const camera=p=>({id:'verify-'+String(p.index).padStart(4,'0'),name:'Local CFV scale fixture '+p.index,caption:'Synthetic cache identity; CFV photo, not a verified camera sequence',region:'Local verification',highway:'',lat:49+p.index*.0001,lon:-123+p.index*.0001,image:'https://www.drivebc.ca/images/238.jpg',source:'https://www.drivebc.ca/cameras/238'});
let catalog={storage_mode:'browser-only',generated:new Date().toISOString(),catalog_size:0,cameras:[],errors:[],model:'Local verification fixture',schema_version:2};
const mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.wasm':'application/wasm','.json':'application/json','.jpg':'image/jpeg'},requests=[];
const server=http.createServer((req,res)=>{
 const url=new URL(req.url,'http://localhost');requests.push({path:url.pathname,method:req.method});
 if(req.method!=='GET'){res.writeHead(405);res.end();return;}
 if(url.pathname==='/data/latest.json'){res.setHeader('content-type','application/json');res.end(JSON.stringify(catalog));return;}
 if(url.pathname==='/data/patterns.json'){res.setHeader('content-type','application/json');res.end('{"cameras":{},"hourly":[]}');return;}
 if(url.pathname==='/orientation-worker.js'){
  const source=fs.readFileSync(path.join(root,'orientation-worker.js'),'utf8'),end=source.indexOf('\n');
  const hook="const __verificationRun=ModelRuntime.Session.prototype.run;ModelRuntime.Session.prototype.run=function(feeds,...args){self.postMessage({verification_stage:{file:this.variant,rows:Object.values(feeds)[0]?.dims?.[0],backend:this.backend}});return __verificationRun.call(this,feeds,...args);};\n";
  res.setHeader('content-type','text/javascript');res.end(source.slice(0,end+1)+hook+source.slice(end+1));return;
 }
 let file;
 if(url.pathname.startsWith('/__verification/images/')){const hash=url.pathname.slice('/__verification/images/'.length).replace(/\.jpg$/,'');if(!/^[a-f0-9]{64}$/.test(hash)||!photosByHash.has(hash)){res.writeHead(404);res.end();return;}file=path.resolve(directory,photosByHash.get(hash).file);if(!file.startsWith(path.join(directory,'images')+path.sep)){res.writeHead(404);res.end();return;}}
 else file=path.resolve(root,'.'+(url.pathname==='/'?'/index.html':url.pathname));
 if((!file.startsWith(root+path.sep)&&!file.startsWith(directory+path.sep))||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);res.end();return;}
 res.setHeader('content-type',mime[path.extname(file)]||'application/octet-stream');fs.createReadStream(file).pipe(res);
});
const quantile=(values,p)=>{const sorted=values.slice().sort((a,b)=>a-b);return sorted.length?sorted[Math.min(sorted.length-1,Math.floor(p*(sorted.length-1)))]:null;};
function accuracy(records){
 const groups=[0,0,1,2,2,2,1,0],groupOf=p=>{const scores=[p[0]+p[1]+p[7],p[2]+p[6],p[3]+p[4]+p[5]];return scores.indexOf(Math.max(...scores));};
 let right=0,accepted=0,acceptedRight=0,acceptedGroupRight=0,groupRight=0,angleError=0;const reasons={},paths={};
 for(let i=0;i<count;i++){const r=records[manifest.photos[i].hash],truth=manifest.photos[i],predicted=r.view_scores.indexOf(Math.max(...r.view_scores));right+=predicted===truth.label;groupRight+=groupOf(r.view_scores)===groups[truth.label];angleError+=Math.abs(((r.angle_degrees-truth.angle+540)%360)-180);if(r.state==='estimated'){accepted++;acceptedRight+=predicted===truth.label;acceptedGroupRight+=groupOf(r.view_scores)===groups[truth.label];}reasons[r.reason]=(reasons[r.reason]||0)+1;paths[r.path]=(paths[r.path]||0)+1;}
 return {images:count,eight_view_correct:right,eight_view_accuracy:right/count,production_group_accuracy:groupRight/count,mean_angle_error_degrees:angleError/count,accepted,coverage:accepted/count,accepted_eight_view_accuracy:accepted?acceptedRight/accepted:null,wrong_accepted:accepted-acceptedRight,accepted_production_group_accuracy:accepted?acceptedGroupRight/accepted:null,wrong_accepted_groups:accepted-acceptedGroupRight,reasons,paths};
}
(async()=>{let browser,sampler;const errors=[],external=[],heap=[];
 try{
  await new Promise(r=>server.listen(0,'127.0.0.1',r));const base=`http://127.0.0.1:${server.address().port}`;
  browser=await chromium.launch({executablePath:process.env.GPU_BENCH_CHROME||undefined,headless:true,args:['--enable-unsafe-webgpu',...(process.platform==='linux'?['--use-angle=vulkan','--enable-features=Vulkan','--disable-vulkan-surface']:[])]});
  const context=await browser.newContext(),page=await context.newPage();page.setDefaultTimeout(60000);
  page.on('pageerror',e=>{errors.push(e.message);console.error('Browser error:',e.message);});page.on('console',m=>{if(m.text().startsWith('verification:'))console.log(m.text());});
  await context.route('**/*',route=>{const u=new URL(route.request().url());if(u.hostname!=='127.0.0.1'){external.push({url:u.href,method:route.request().method()});return route.abort();}return route.continue();});
  await page.addInitScript(()=>{localStorage.setItem('locaintel.facing','off');localStorage.setItem('locaintel.power','1');document.addEventListener('DOMContentLoaded',()=>{timelineAt=0;updateBackgroundBudget();});});
  const cdp=await context.newCDPSession(page);await cdp.send('Performance.enable');sampler=setInterval(async()=>{try{const m=await cdp.send('Performance.getMetrics');heap.push(m.metrics.find(v=>v.name==='JSHeapUsedSize')?.value||0);}catch(_){}},500);
  await page.goto(base,{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>typeof data!=='undefined'&&data&&typeof localFacingQueue!=='undefined');
  const initial=await page.evaluate(async()=>{await archiveReady;clearTimeout(autoRefreshTimer);timelineAt=0;localHistoryOnly=true;$('source-view').checked=false;$('facing-enabled').checked=false;updateBackgroundBudget();roadAnalyzer.isPaused=()=>true;browserDetector.isPaused=()=>true;let a=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});if(!a)a=await navigator.gpu?.requestAdapter();if(!a||/swiftshader|llvmpipe|software/i.test(JSON.stringify(a.info)))throw Error('Hardware WebGPU required');return {adapter:{vendor:a.info.vendor,architecture:a.info.architecture,description:a.info.description,shader_f16:a.features.has('shader-f16')}};});
  console.log('verification: importing 1000 unique native JPEGs into local IndexedDB');
  const imported=await page.evaluate(async photos=>{
   const start=performance.now(),baseTime=Date.now()-photos.length*1000;
   for(let startIndex=0;startIndex<photos.length;startIndex+=4){await Promise.all(photos.slice(startIndex,startIndex+4).map(async p=>{
    const blob=await(await fetch('/__verification/images/'+p.hash+'.jpg')).blob(),hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',await blob.arrayBuffer()))].map(v=>v.toString(16).padStart(2,'0')).join('');if(hash!==p.hash||blob.size!==p.bytes)throw Error('Native photo hash/size mismatch');
    const camera={id:'verify-'+String(p.index).padStart(4,'0'),name:'Local CFV scale fixture '+p.index,caption:'Synthetic cache identity, not a verified camera sequence',region:'Local verification',highway:'',lat:49+p.index*.0001,lon:-123+p.index*.0001,image:'https://www.drivebc.ca/images/238.jpg',source:'https://www.drivebc.ca/cameras/238'},d={label:'car',box:p.box,confidence:1,motion:{state:'unknown',reason:'CFV_ground_truth_box'}};
    await photoArchive.saveFrame(camera,{hash,blob,width:p.width,height:p.height},{observedAt:baseTime+p.index*1000,analysisMode:'browser',detectorVersion:browserDetector.modelVersion,detections:[d],counts:{car:1},analysis:{}});
   }));if((startIndex+4)%100===0)console.log('verification: cached '+Math.min(photos.length,startIndex+4)+'/'+photos.length);}
   await photoArchive.chain;return {elapsed_ms:performance.now()-start,stats:await photoArchive.stats()};
  },manifest.photos);
  assert.equal(imported.stats.photos,count);assert.equal(imported.stats.frames,count);assert.equal(imported.stats.usedBytes,manifest.photos_bytes);
  // A reusable runner uses the actual production cache queue and crop preparer.
  await page.evaluate(()=>{window.runLocalVerification=async({batchSize,reference=false,targets=null,injectFailures=[]})=>{
   const originalRows=[...archiveMetadata.values()],targetSet=targets?new Set(targets):null,photos=originalRows.filter(r=>!targetSet||targetSet.has(r.hash)),expected=photos.length,results={},durations=[],stages=[],failures=[],attempted=new Set(),injected=new Set(injectFailures),dispatch=new Map();
   let analyzer,queue,doneResolve,doneReject,warming=true,maxQueue=0,maxEntries=0,maxVisual=0,maxActivePhotos=0,liveFrames=0,liveBytes=0,maxFrames=0,maxBytes=0,maxSingleBitmap=0;
   let started=performance.now();const makeBitmap=window.createImageBitmap.bind(window);
   window.createImageBitmap=async(...args)=>{const bitmap=await makeBitmap(...args),bytes=bitmap.width*bitmap.height*4,close=bitmap.close.bind(bitmap);let closed=false;liveFrames++;liveBytes+=bytes;maxFrames=Math.max(maxFrames,liveFrames);maxBytes=Math.max(maxBytes,liveBytes);maxSingleBitmap=Math.max(maxSingleBitmap,bytes);bitmap.close=()=>{if(!closed){closed=true;liveFrames--;liveBytes-=bytes;}close();};return bitmap;};
   if(reference){for(const row of photos){const d=row.detections[0];await photoArchive.saveOrientation(row.key,0,{version:VehicleFacing.VERSION,state:'error',fingerprint:VehicleFacing.fingerprint(d),retry_attempts:0,next_retry_at:0,retry_exhausted:false,reason:'Local scalar verification reset'});}await photoArchive.chain;}
   const prepared=VehicleFacing.createPreparer(),prepare=async(s,d)=>{if(injected.has(s.hash)&&!attempted.has(s.hash)){attempted.add(s.hash);throw Error('Injected local verification retry');}return prepared(s,d);};
   const finished=new Promise((resolve,reject)=>{doneResolve=resolve;doneReject=reject;}),timeout=setTimeout(()=>doneReject(Error('1000-image verification timed out')),600000);
   const check=()=>{maxQueue=Math.max(maxQueue,analyzer.queue.length);maxEntries=Math.max(maxEntries,analyzer.entries.size);maxVisual=Math.max(maxVisual,analyzer.visualEntries.size);maxActivePhotos=Math.max(maxActivePhotos,queue.activeHashes.size);if(Object.keys(results).length===expected&&!analyzer.running&&!analyzer.queue.length)doneResolve();};
   analyzer=new VehicleFacing.CropOrientation({batchSize,delayMs:0,prepareCrop:prepare,isWanted:hash=>queue.wants(hash),isPaused:()=>warming,
    workerFactory:()=>{const worker=new Worker('orientation-worker.js?v=local-scale'),send=worker.postMessage.bind(worker);worker.postMessage=(message,transfer)=>{if(message.id)dispatch.set(message.id,{at:performance.now(),cars:message.jobs?.length||1});send(message,transfer);};worker.addEventListener('message',({data})=>{if(data.verification_stage)stages.push(data.verification_stage);const job=dispatch.get(data.id);if(job){durations.push({elapsed_ms:performance.now()-job.at,cars:job.cars,error:data.error||data.batchError||null});dispatch.delete(data.id);}});return worker;},
    onModelUpdate:modelCacheUpdate,onChange:()=>{check();queue.wake();},
    onResult:(source,index,result)=>{
     if(result.state==='error'){failures.push({hash:source.hash,attempt:result.retry_attempts,reason:result.reason});if(!injected.has(source.hash)||result.retry_exhausted)doneReject(Error('Unexpected or exhausted inference failure: '+result.reason));}
     else{if(!result.model_ran)doneReject(Error('Facing model did not evaluate a valid source crop'));if(result.backend!=='webgpu'&&result.backend!=='cache')doneReject(Error('GPU fallback in scale verification'));results[source.hash]=result;if(Object.keys(results).length%100===0||Object.keys(results).length===expected)console.log('verification: GPU batch '+batchSize+' completed '+Object.keys(results).length+'/'+expected);}
     photoArchive.saveOrientation(source.cameraId+':'+source.hash,index,result).catch(doneReject);
    }
   });
   queue=new LocalFacingQueue({rows:()=>[...archiveMetadata.values()].filter(r=>!targetSet||targetSet.has(r.hash)),photo:hash=>photoArchive.photo(hash),analyzer,isPaused:()=>warming,delayMs:50,onError:doneReject});
   const interval=setInterval(check,100);
   let warmupMs=0;
   try{
    const warmStart=performance.now();for(let i=0;i<3;i++){const item={input:new Float32Array(3*224*224).fill(.5),lights:{type:'none'},crop:{width:224,height:224}},reply=await analyzer.infer(batchSize===1?item:[item],batchSize),list=Array.isArray(reply)?reply:[reply];if(list.some(r=>r.backend!=='webgpu'||r.path!=='full-resolution'))throw Error('Warmup did not initialize both GPU stages');}
    warmupMs=performance.now()-warmStart;durations.length=0;stages.length=0;started=performance.now();warming=false;
    queue.wake();await finished;await photoArchive.chain;return {records:results,elapsed_ms:performance.now()-started,model_warmup_ms:warmupMs,dispatches:durations,gpu_stages:stages,failures,bounds:{max_queue:maxQueue,max_result_entries:maxEntries,max_visual_cache:maxVisual,max_active_photos:maxActivePhotos,peak_decoded_frames:maxFrames,peak_decoded_bytes:maxBytes,max_single_bitmap_bytes:maxSingleBitmap,final_result_entries:analyzer.entries.size,final_visual_cache:analyzer.visualEntries.size}};}
   finally{clearInterval(interval);clearTimeout(timeout);queue.dispose();analyzer.worker?.terminate();prepared.dispose();window.createImageBitmap=makeBitmap;}
  };});
  const runnerSource=await page.evaluate(()=>runLocalVerification.toString());
  console.log('verification: real GPU batched pass begins');const batched=await page.evaluate(()=>runLocalVerification({batchSize:8}));
  assert.equal(Object.keys(batched.records).length,count);assert.equal(batched.failures.length,0);assert(batched.bounds.max_queue<=8);assert(batched.bounds.max_active_photos<=8);assert(batched.bounds.final_result_entries<=512);assert(batched.bounds.final_visual_cache<=256);assert(batched.bounds.peak_decoded_frames<=5);assert(batched.bounds.peak_decoded_bytes<=32*1024*1024+batched.bounds.max_single_bitmap_bytes);
  console.log('verification: scalar GPU reference over the same 1000 native photos');const scalar=await page.evaluate(()=>runLocalVerification({batchSize:1,reference:true}));
  assert.equal(Object.keys(scalar.records).length,count);assert.equal(scalar.failures.length,0);
  let changed=0,changedAcceptance=0,maxScore=0,maxMirrorScore=0,maxAngle=0,maxMirrorAngle=0;
  for(const p of manifest.photos){const a=batched.records[p.hash],b=scalar.records[p.hash],view=r=>r.view_scores.indexOf(Math.max(...r.view_scores));changed+=view(a)!==view(b);changedAcceptance+=a.state!==b.state;maxScore=Math.max(maxScore,...a.view_scores.map((s,i)=>Math.abs(s-b.view_scores[i])));maxAngle=Math.max(maxAngle,Math.abs(((a.angle_degrees-b.angle_degrees+540)%360)-180));maxMirrorScore=Math.max(maxMirrorScore,...a.mirror_view_scores.map((s,i)=>Math.abs(s-b.mirror_view_scores[i])));maxMirrorAngle=Math.max(maxMirrorAngle,Math.abs(((a.mirror_angle_degrees-b.mirror_angle_degrees+540)%360)-180));}
  const fidelity={compared:count,changed_view_decisions:changed,changed_acceptance:changedAcceptance,max_score_error:maxScore,max_mirror_score_error:maxMirrorScore,max_angle_error_degrees:maxAngle,max_mirror_angle_error_degrees:maxMirrorAngle};
  assert.equal(changed,0);assert.equal(changedAcceptance,0);assert(maxScore<1e-4&&maxMirrorScore<1e-4);assert(maxAngle<.1&&maxMirrorAngle<.1);
  // Hydrate the actual dashboard with 1000 synthetic cache IDs, not 1000 photos
  // pretending to be a verified traffic-camera chronology.
  catalog={...catalog,catalog_size:count,cameras:manifest.photos.map(camera)};
  const ui=await page.evaluate(async()=>{timelineAt=null;localHistoryOnly=true;$('source-view').checked=false;clearTimeout(autoRefreshTimer);const start=performance.now();await refresh(false);await loadSavedVisibleSources();render();return {initial_render_ms:performance.now()-start,catalog:data.cameras.length,retained_metadata:archiveMetadata.size,feed_cards:document.querySelectorAll('#feeds .card').length,photo_urls:publishedPhotoSources.size};});
  assert.equal(ui.catalog,count);assert.equal(ui.retained_metadata,count);assert.equal(ui.feed_cards,24);assert(ui.photo_urls<=128);
  await page.evaluate(async()=>{await navigator.serviceWorker.ready;await photoArchive.chain;});await page.waitForFunction(()=>!!navigator.serviceWorker.controller);
  const beforeOfflineRequests=requests.filter(r=>r.path.startsWith('/__verification/images/')).length;
  await context.setOffline(true);await page.reload({waitUntil:'domcontentloaded'});await page.waitForFunction(()=>typeof data!=='undefined'&&data?.cameras?.length===1000&&archiveMetadata.size===1000);await page.evaluate(async()=>{await archiveReady;clearTimeout(autoRefreshTimer);timelineAt=0;updateBackgroundBudget();});
  const cacheCheck=await page.evaluate(async()=>{const rows=await photoArchive.metadata();let requests=0;const analyzer=new VehicleFacing.CropOrientation({batchSize:8,workerFactory:()=>{requests++;throw Error('Completed cache should not infer');}}),queue=new LocalFacingQueue({rows:()=>rows,photo:hash=>photoArchive.photo(hash),analyzer});try{await queue.tick();return {stats:await photoArchive.stats(),completed:rows.filter(r=>Object.values(r.facing?.byFingerprint||{}).some(o=>o.model_ran)).length,inference_workers_created:requests};}finally{queue.dispose();}});
  assert.equal(cacheCheck.stats.photos,count);assert.equal(cacheCheck.completed,count);assert.equal(cacheCheck.inference_workers_created,0);
  // Reinstall the runner function after navigation without fetching any photos.
  await page.evaluate(source=>{window.runLocalVerification=eval('('+source+')');},runnerSource);
  const targetHashes=manifest.photos.slice(0,10).map(p=>p.hash);
  await page.evaluate(async hashes=>{for(const hash of hashes){const row=[...archiveMetadata.values()].find(r=>r.hash===hash),d=row.detections[0];await photoArchive.saveOrientation(row.key,0,{version:VehicleFacing.VERSION,state:'error',fingerprint:VehicleFacing.fingerprint(d),retry_attempts:0,next_retry_at:0,retry_exhausted:false,reason:'Verification of missed local facing'});}},targetHashes);
  console.log('verification: retrying 10 missed photos offline, including 3 transient failures');const retry=await page.evaluate(args=>runLocalVerification(args),{batchSize:8,targets:targetHashes,injectFailures:targetHashes.slice(0,3)});
  assert.equal(Object.keys(retry.records).length,10);assert.equal(retry.failures.length,3);
  const pruning=await page.evaluate(async budget=>{const before=await photoArchive.stats();await photoArchive.configure({budgetBytes:budget});const rows=await photoArchive.metadata(),stats=await photoArchive.stats();let bytes=0;for(const row of rows){const photo=await photoArchive.photo(row.hash);if(!photo||!Object.values(row.facing?.byFingerprint||{}).some(o=>o.model_ran))throw Error('Pruning broke photo/evidence association');bytes+=photo.size;}return {before:before.photos,retained:rows.length,stats,retained_photo_bytes:bytes};},Math.floor(manifest.photos_bytes/2));
  assert(pruning.retained>0&&pruning.retained<count);assert.equal(pruning.retained,pruning.stats.photos);assert(pruning.stats.usedBytes<=Math.floor(manifest.photos_bytes/2));assert.equal(pruning.retained_photo_bytes,pruning.stats.usedBytes);
  assert.equal(requests.filter(r=>r.path.startsWith('/__verification/images/')).length,beforeOfflineRequests);assert(!requests.some(r=>r.method!=='GET'));assert.deepEqual(errors,[]);
  const timing=run=>({elapsed_ms:run.elapsed_ms,images_per_second:count/(run.elapsed_ms/1000),model_warmup_ms:run.model_warmup_ms,warmup_requests_excluded:3,worker_dispatches:run.dispatches.length,worker_request_batch_histogram:run.dispatches.reduce((r,x)=>(r[x.cars]=(r[x.cars]||0)+1,r),{}),gpu_stage_invocations:run.gpu_stages.length,gpu_stage_row_histogram:run.gpu_stages.reduce((r,s)=>(r[s.rows]=(r[s.rows]||0)+1,r),{}),gpu_stage_input_rows:run.gpu_stages.reduce((sum,s)=>sum+s.rows,0),median_dispatch_ms:quantile(run.dispatches.map(x=>x.elapsed_ms),.5),p95_dispatch_ms:quantile(run.dispatches.map(x=>x.elapsed_ms),.95),...run.bounds});
  const report={version:1,passed:true,created:new Date().toISOString(),images:count,logical_original_mirror_views_per_pass:count*2,source:{dataset:manifest.dataset,revision:manifest.revision,archive_sha256:manifest.archive_sha256,manifest_sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(directory,'manifest.json'))).digest('hex'),photos_bytes:manifest.photos_bytes,identities:14},adapter:initial.adapter,browser:browser.version(),implementation_sha256:Object.fromEntries(['scripts/verify-facing-1000.cjs','web/orientation-worker.js','web/orientation.js','web/facing-cache.js'].map(file=>[file,crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')])),import:imported,batched:{...timing(batched),accuracy:accuracy(batched.records)},scalar:{...timing(scalar),accuracy:accuracy(scalar.records)},fidelity,offline_cache:cacheCheck,offline_retry:{images:10,transient_failures:retry.failures.length,completed:Object.keys(retry.records).length,elapsed_ms:retry.elapsed_ms},dashboard:ui,pruning,main_renderer_js_heap_peak_bytes:Math.max(0,...heap),network:{photo_gets:beforeOfflineRequests,photo_gets_after_offline:0,uploads:0,blocked_external_requests:external.length},limitations:['CFV supplied ground-truth boxes; YOLO recall is not measured.','Both timings include the same local queue, photo reads, crop preparation and per-image evidence writes. Reference evidence reset and three synthetic full-path warmups are excluded, reported separately. Worker dispatches and Session.run stage invocations differ.','GPU stage calls use a test-only worker observer; production inputs, outputs, thresholds and inference body are unchanged.','All-angle official test photos; test identities were previously evaluated upstream. No policy was tuned on this run.','Synthetic cache IDs and observation times test storage/UI scale, not same-camera temporal accuracy.','JS heap and bitmap bounds do not measure full native browser or GPU VRAM allocation.','One local hardware/browser run is not a universal big-data guarantee.']};
  fs.writeFileSync(path.join(directory,'report.json'),JSON.stringify(report,null,2)+'\n');fs.writeFileSync(path.join(directory,'predictions.json'),JSON.stringify({batched:batched.records,scalar:scalar.records})+'\n');console.log(JSON.stringify(report,null,2));
 }finally{clearInterval(sampler);await browser?.close();if(server.listening)await new Promise(r=>server.close(r));}
})().catch(error=>{console.error(error);process.exitCode=1;});

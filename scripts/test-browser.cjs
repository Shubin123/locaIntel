/* Browser-only storage and full facing coverage, using a static fixture host. */
const {chromium}=require('playwright'),http=require('node:http'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../web'),mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.wasm':'application/wasm','.json':'application/json'};
let png,payload;const requests=[],network=[];const deployed=process.env.BROWSER_BASE_URL,expectedBackend=process.env.BROWSER_BACKEND||'webgpu';
const server=http.createServer((req,res)=>{
 const url=new URL(req.url,'http://localhost');requests.push(url.pathname);
 if(url.pathname==='/data/latest.json'){res.setHeader('content-type','application/json');res.end(JSON.stringify(payload));return;}
 if(url.pathname==='/data/patterns.json'){res.setHeader('content-type','application/json');res.end('{"cameras":{},"hourly":[]}');return;}
 if(url.pathname==='/frames/test.png'){res.setHeader('content-type','image/png');res.end(png);return;}
 const file=path.resolve(root,'.'+(url.pathname==='/'?'/index.html':url.pathname));if(!file.startsWith(root+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);res.end();return;}
 res.setHeader('content-type',mime[path.extname(file)]||'application/octet-stream');fs.createReadStream(file).pipe(res);
});
(async()=>{let browser;
 try{
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  browser=await chromium.launch({executablePath:process.env.GPU_BENCH_CHROME||undefined,headless:true,args:expectedBackend==='wasm'?['--disable-webgpu','--disable-webgl']:['--enable-unsafe-webgpu',...(process.platform==='linux'?['--use-angle=vulkan','--enable-features=Vulkan','--disable-vulkan-surface']:[])]});
  const context=await browser.newContext({permissions:['geolocation'],geolocation:{latitude:49.002,longitude:-123,accuracy:5}}),page=await context.newPage({viewport:{width:1280,height:900}}),errors=[];
  page.on('request',r=>network.push({url:r.url(),method:r.method()}));
  page.on('pageerror',e=>{errors.push(e.message);console.error('Browser error:',e.message);});
  png=Buffer.from(await page.evaluate(()=>{const c=document.createElement('canvas');c.width=640;c.height=400;const g=c.getContext('2d');g.fillStyle='#777';g.fillRect(0,0,640,400);for(const [x,y,w,h,color]of [[70,100,100,70,'#b51e1e'],[230,130,110,65,'#eeeeee'],[420,160,100,70,'#223344']]){g.fillStyle=color;g.fillRect(x,y,w,h);}return c.toDataURL('image/png').split(',')[1];}),'base64');
  const hash=crypto.createHash('sha256').update(png).digest('hex'),detections=[[.11,.25,.26,.425],[.36,.325,.53,.49],[.66,.4,.81,.575],[.2,.2,.21,.21],[0,.6,.15,.8]].map(box=>({label:'car',box,confidence:.9,motion:{state:'unknown'},on_road:null}));
  payload={generated:new Date().toISOString(),catalog_size:1,model:'Browser regression fixture',schema_version:2,errors:[],cameras:[{id:'238',name:'Browser fixture',caption:'Local retention test',region:'Test',highway:'1',lat:49,lon:-123,hash,frame:'frames/test.png',image:'https://www.drivebc.ca/images/238.jpg',observed:new Date().toISOString(),detections,counts:{car:5},analysis:{}}]};
  if(deployed){const r=await context.request.get(new URL('data/latest.json?e2e='+Date.now(),deployed).href);assert(r.ok(),'deployed catalog available');const live=await r.json();assert.equal(live.storage_mode,'browser-only');assert(live.cameras.length>0);assert(live.cameras.every(c=>!('hash'in c)&&!('frame'in c)&&!('detections'in c)),'deployed metadata contains no photos or per-photo evidence');}
  // Window-level fixture fetch remains deterministic after SW takes control;
  // It avoids depending on service-worker network interception behavior.
  await context.addInitScript(({fixture})=>{
    const networkFetch=window.fetch.bind(window),reviewingCache=localStorage.getItem('locaintel.e2e.catalog')==='1';
    const catalog={storage_mode:'browser-only',generated:fixture.generated,catalog_size:1,errors:[],cameras:fixture.cameras.map(({id,name,caption,region,highway,lat,lon,image})=>({id,name,caption,region,highway,lat,lon,image}))};
    // Force catalog metadata to arrive before the local archive is ready.
    if(reviewingCache){const open=IDBFactory.prototype.open;IDBFactory.prototype.open=function(...args){const request=open.apply(this,args);Object.defineProperty(request,'onsuccess',{set(callback){request.addEventListener('success',event=>setTimeout(()=>callback.call(request,event),120));}});return request;};}
    window.fetch=(input,init)=>{const url=new URL(typeof input==='string'?input:input.url,location.href);if(url.pathname.endsWith('/data/cameras.json'))return Promise.resolve(new Response(JSON.stringify(catalog.cameras),{headers:{'content-type':'application/json'}}));if(url.pathname.endsWith('/data/latest.json'))return Promise.resolve(new Response(JSON.stringify(reviewingCache?catalog:fixture),{headers:{'content-type':'application/json'}}));return networkFetch(input,init);};
  },{fixture:payload});
  await context.route('**/frames/test.png*',r=>r.fulfill({contentType:'image/png',body:png}));
  await context.route('https://www.drivebc.ca/images/238.jpg*',r=>r.fulfill({contentType:'image/png',headers:{'access-control-allow-origin':'*'},body:png}));
  await context.route('**/tile.openstreetmap.org/**',r=>r.abort());
  const origin=deployed||`http://127.0.0.1:${server.address().port}/`;await page.goto(origin);
  await page.waitForFunction(()=>{if(!data)return false;const c=displayedCamera(data.cameras[0]);return c.detections.length===5&&c.detections.every(d=>d.orientation?.model_ran)&&!facingAnalyzer.running;},null,{timeout:180000});
  let coverage=await page.evaluate(()=>{const c=displayedCamera(data.cameras[0]);return {cars:c.detections.length,ran:c.detections.filter(d=>d.orientation.model_ran).length,abstained:c.detections.filter(d=>d.orientation.quality_abstention).length,backend:facingAnalyzer.backend};});
  assert.equal(coverage.ran,5);assert.equal(coverage.abstained,1,'only the sub-8×6-pixel box abstains');assert.equal(coverage.backend,expectedBackend);console.log('Facing coverage:',coverage);
  const single=await page.evaluate(()=>displayedCamera(data.cameras[0]).detections.map(d=>({state:d.orientation.state,tier:d.orientation.tier,version:d.orientation.version})));
  assert.equal(single.filter(o=>o.state==='estimated'&&['high','medium','low'].includes(o.tier)&&o.version==='drivebc-facing-v3.1').length,4,'every car at least 8×6 px, including the clipped one, gets a tiered single-photo estimate');
  // Archived results from the retired CFV cascade still go through two-screenshot
  // confirmation. This fixture checks that plumbing, not model accuracy.
  await page.evaluate(async()=>{
    pairCapture.pause();facingAnalyzer.isPaused=()=>true;localFacingQueue.isPaused=()=>true;localFacingQueue.pause();await photoArchive.chain;
    const camera=data.cameras[0],current=archiveMetadata.get(camera.id+':'+camera.hash),blob=await photoArchive.photo(camera.hash),bitmap=await createImageBitmap(blob),canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;
    const g=canvas.getContext('2d');g.drawImage(bitmap,0,0);bitmap.close();g.fillStyle='#818181';g.fillRect(10,10,15,15);
    const previousBlob=await new Promise(r=>canvas.toBlob(r,'image/png')),previousHash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await previousBlob.arrayBuffer())),v=>v.toString(16).padStart(2,'0')).join('');
    await photoArchive.saveFrame(camera,{hash:previousHash,blob:previousBlob,width:640,height:400},{observedAt:current.observedAt-5000,analysisMode:'published',detections:camera.detections,counts:camera.counts,analysis:{}});
    const manifest={model:'CFV fixture',views:['front','front_left','left_side','rear_left','rear','rear_right','right_side','front_right'],angle_convention:'CFV relative view'},logits=Array.from({length:8},(_,i)=>i===0?8:-5);
    for(const hash of [previousHash,camera.hash])for(let i=0;i<3;i++)await photoArchive.saveOrientation(camera.id+':'+hash,i,{...VehicleFacing.classify(logits,0,manifest,{type:'none'},{logits,angle:0}),fingerprint:VehicleFacing.fingerprint(camera.detections[i]),model_ran:true,backend:'webgpu',fixture:true});
    scheduleTemporal(current);queueViewPaint();
  });
  try{await page.waitForFunction(()=>displayedCamera(data.cameras[0]).detections.slice(0,3).every(d=>d.orientation?.paired?.state==='confirmed'),null,{timeout:30000});}
  catch(error){console.error('Pair diagnostics:',await page.evaluate(()=>({cars:displayedCamera(data.cameras[0]).detections.map(d=>({reason:d.orientation?.reason,paired:d.orientation?.paired?.state})),frames:[...archiveMetadata.values()].map(r=>({hash:r.hash,features:PairedFacing.validFeatures(r),facing:Object.values(r.facing?.byFingerprint||{}).map(o=>({state:o.state,reason:o.reason}))}))})));throw error;}
  assert.equal(await page.evaluate(()=>displayedCamera(data.cameras[0]).detections.filter(d=>d.orientation?.paired?.state==='confirmed').length),3);
  await page.waitForFunction(()=>archiveMetadata.get(data.cameras[0].id+':'+data.cameras[0].hash)?.pairedFacing?.byFingerprint?.[VehicleFacing.fingerprint(data.cameras[0].detections[0])]?.paired?.state==='confirmed');
  await page.evaluate(async()=>{const c=data.cameras[0];await photoArchive.saveTemporal(c.id+':'+c.hash,{detections:[{label:'car',box:[.8,.8,.9,.9],motion:{state:'moving',vector:[.01,0],gap_seconds:1}}],log:{frameHashes:[c.hash],states:{moving:1,stationary:0,unknown:0},status:'test_stale'}});});
  assert.equal(await page.evaluate(()=>displayedCamera(data.cameras[0]).detections.length),5);
  await page.locator('.heading-controls summary').click();
  for(const y of ['0.55']){await page.locator('#reference-y1').fill(y);await page.locator('#reference-y2').fill(y);await page.locator('#reference-coordinates button').click();await page.waitForFunction(n=>directionReferences.get('238')?.vectors.length===n,y==='0.55'?1:2);}
  await page.locator('#draw-reference').click();await page.locator('#detail-picture').scrollIntoViewIfNeeded();const bounds=await page.locator('#detail-picture').boundingBox();
  await page.mouse.move(bounds.x+bounds.width*.2,bounds.y+bounds.height*.6);await page.mouse.down();await page.mouse.move(bounds.x+bounds.width*.7,bounds.y+bounds.height*.6,{steps:8});await page.mouse.up();
  await page.waitForFunction(()=>directionReferences.get('238')?.vectors.length===2);await page.locator('#draw-reference').click();
  await page.waitForFunction(()=>!!navigator.serviceWorker.controller);await page.evaluate(async()=>{await navigator.serviceWorker.ready;await photoArchive.chain;localStorage.setItem('locaintel.e2e.catalog','1');});
  await page.reload();await page.waitForFunction(()=>directionReferences.get('238')?.vectors.length===2&&data&&displayedCamera(data.cameras[0]).detections.length===5&&displayedCamera(data.cameras[0]).detections.every(d=>d.orientation?.model_ran),null,{timeout:60000});
  assert.equal(await page.evaluate(async()=>{await photoArchive.chain;return Object.keys((await photoArchive.metadata()).find(r=>r.hash===data.cameras[0].hash&&r.cameraId==='238').facing.byFingerprint).length;}),5);
  // Catalog-only publication preserves locally retained analysis and photos.
  await page.evaluate(()=>{const c=data.cameras[0];apply({storage_mode:'browser-only',generated:new Date().toISOString(),catalog_size:1,errors:[],cameras:[{id:c.id,name:c.name,caption:c.caption,region:c.region,highway:c.highway,lat:c.lat,lon:c.lon,image:c.image}]},null);});
  assert.equal(await page.evaluate(()=>displayedCamera(data.cameras[0]).detections.filter(d=>d.orientation?.model_ran).length),5);
  const modelCache=await page.evaluate(async()=>{const cache=await caches.open('locaintel-models-v1');return (await cache.keys()).map(r=>r.url);});assert(modelCache.some(u=>u.endsWith('.onnx')),'facing graphs retained locally');assert(!modelCache.some(u=>/\.(png|jpg|jpeg)$/.test(u)));
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await context.setOffline(true);await new Promise(resolve=>server.close(resolve));await page.reload();
  await page.waitForFunction(()=>data&&displayedCamera(data.cameras[0]).detections.filter(d=>d.orientation?.model_ran).length===5&&directionReferences.get('238')?.vectors.length===2,null,{timeout:60000});
  await page.waitForFunction(()=>displayedCamera(data.cameras[0]).detections.filter(d=>d.orientation?.paired?.state==='confirmed').length===3,null,{timeout:30000});
  assert.equal(await page.evaluate(async()=>(await photoArchive.stats()).photos),2);assert.equal(await page.evaluate(()=>displayedCamera(data.cameras[0]).detections.filter(d=>d.orientation?.paired?.state==='confirmed').length),3);assert(!requests.some(p=>p.startsWith('/api/')));assert.deepEqual(errors,[]);
  // Missed, nonvisible archived photos retry locally even after the host stops.
  await page.evaluate(async()=>{
    localHistoryOnly=true;$('source-view').checked=false;clearTimeout(autoRefreshTimer);pairCapture.pause();
    const c=data.cameras[0],blob=await photoArchive.photo(c.hash),prepare=facingAnalyzer.prepareCrop;let failed=false;
    facingAnalyzer.prepareCrop=async(source,d)=>{if(source.cameraId==='missed-local'&&!failed){failed=true;throw Error('E2E injected transient crop failure');}return prepare(source,d);};
    await photoArchive.saveFrame({id:'missed-local',name:'Missed local cache'}, {hash:c.hash,blob,width:640,height:400},{observedAt:Date.now(),analysisMode:'browser',detections:[c.detections[0]],counts:{car:1}});
    localFacingQueue.wake();
  });
  await page.waitForFunction(()=>[...archiveMetadata.values()].find(r=>r.cameraId==='missed-local')?.facing?.byDetection?.[0]?.state==='error',null,{timeout:15000});
  await page.waitForFunction(()=>[...archiveMetadata.values()].find(r=>r.cameraId==='missed-local')?.facing?.byDetection?.[0]?.model_ran===true,null,{timeout:180000});
  assert(await page.locator('#model-cache-updates').textContent().then(t=>t.includes('Verified local cache')),'offline retry reloads verified model weights from browser cache');
  // GPS drive log: explicit start/stop, timestamped fixes, exports/import, own-car labels.
  await page.locator('#gps-pane').scrollIntoViewIfNeeded();assert(await page.locator('#gps-start').isDisabled(),'logging needs the own-device confirmation');
  assert.equal(await page.evaluate(()=>gpsLog.logger.active),false,'nothing logs before Start');
  await page.locator('#gps-own').check();await page.locator('#gps-start').click();
  await page.waitForFunction(()=>gpsLog.logger.fixes>=1,null,{timeout:15000});
  for(const latitude of [49.0015,49.001,49.0005]){await new Promise(r=>setTimeout(r,3100));await context.setGeolocation({latitude,longitude:-123,accuracy:5});}
  await page.waitForFunction(()=>gpsLog.logger.fixes>=4,null,{timeout:15000});
  const wakeLock=await page.evaluate(()=>gpsLog.logger.state().wakeLock);assert(/^(held|error: )/.test(wakeLock),'wake lock requested: '+wakeLock);
  await page.locator('#gps-stop').click();await page.waitForFunction(()=>!gpsLog.logger.active);
  const drive=await page.evaluate(async()=>{const [s]=await gpsLog.store.sessions(),fixes=await gpsLog.store.fixes(s.id);return {session:s,fixes};});
  assert.equal(drive.session.fixes,drive.fixes.length);assert.notEqual(drive.session.end,null);assert(drive.fixes.every((f,i)=>!i||f.t-drive.fixes[i-1].t>=3000),'fixes every few seconds, never faster');
  const gpsNetwork=network.length;await new Promise(r=>setTimeout(r,3500));assert.equal(await page.evaluate(()=>gpsLog.store.fixes().then(f=>f.length)),drive.fixes.length,'no fixes after Stop');
  // A saved frame captured mid-drive, at a second camera on the same photo (no extra photo bytes).
  const capturedAt=Math.ceil((drive.fixes[1].t+1)/1000)*1000;
  await page.evaluate(async t=>{const c=data.cameras[0],blob=await photoArchive.photo(c.hash);await photoArchive.saveFrame({id:'gps-cam',name:'GPS fixture',lat:49,lon:-123},{hash:c.hash,blob,width:640,height:400,modified:new Date(t).toUTCString()},{observedAt:t+5000,analysisMode:'published',detections:c.detections,counts:c.counts,analysis:{}});},capturedAt);
  await page.waitForFunction(()=>document.querySelectorAll('#gps-frames .gps-choice').length===5,null,{timeout:15000});
  await page.locator('#gps-frames .gps-choice').first().click();await page.waitForFunction(()=>gpsLog.store.labels().then(l=>l.length===1));
  assert.equal(await page.locator('#gps-frames .gps-choice').first().getAttribute('aria-pressed'),'true');
  const saved=async(button)=>{const [file]=await Promise.all([page.waitForEvent('download'),button.click()]);return fs.readFileSync(await file.path(),'utf8');};
  const evaluation=JSON.parse(await saved(page.locator('#gps-export')));
  assert.equal(evaluation.format,'locaintel-own-vehicle-v1');assert.equal(evaluation.labels.length,1);assert.equal(evaluation.labels[0].sighting,'gps-cam:'+payload.cameras[0].hash+':0');
  assert.equal(evaluation.labels[0].capturedAt,new Date(capturedAt).toISOString());assert(evaluation.passes.some(p=>p.cameraId==='238'&&p.replay.list.endsWith('/api/webcams/238/replayTheDay/')),'camera passes carry DriveBC replay links');
  const gpx=await saved(page.locator('#gps-sessions button',{hasText:'GPX'}).first()),csv=await saved(page.locator('#gps-sessions button',{hasText:'CSV'}).first());
  assert.equal((gpx.match(/<trkpt /g)||[]).length,drive.fixes.length);assert.equal(csv.trim().split('\n').length,drive.fixes.length+1);
  await page.locator('#gps-import').setInputFiles({name:'phone.csv',mimeType:'text/csv',buffer:Buffer.from(csv)});await page.waitForFunction(()=>/Imported/.test(document.getElementById('gps-status').textContent));
  assert.equal(await page.evaluate(()=>gpsLog.store.fixes().then(f=>f.length)),drive.fixes.length,'re-importing an export adds nothing');
  assert(!network.slice(gpsNetwork).some(r=>r.method!=='GET')&&!network.some(r=>/49\.00(15|1|05)/.test(r.url)),'GPS fixes never leave the browser');
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'GPS widget fits a phone screen');
  await page.reload();await page.waitForFunction(()=>window.gpsLog);
  assert.deepEqual(await page.evaluate(async()=>[(await gpsLog.store.sessions()).length,(await gpsLog.store.labels()).length]),[1,1],'GPS drive and label survive an offline reload');
  // A second offline drive, exported straight after a fix: the file includes it even before the list repaints.
  await page.locator('#gps-own').check();await page.locator('#gps-start').click();
  await page.waitForFunction(()=>gpsLog.logger.fixes>=1&&document.querySelector('#gps-sessions .gps-session')?.dataset.key===gpsLog.logger.session.id,null,{timeout:15000});
  await new Promise(r=>setTimeout(r,3100));await context.setGeolocation({latitude:49.003,longitude:-123,accuracy:5});await page.waitForFunction(()=>gpsLog.logger.fixes>=2,null,{timeout:15000});
  const fresh=await saved(page.locator('#gps-sessions .gps-session button',{hasText:'CSV'}).first());
  assert.equal(fresh.trim().split('\n').length-1,await page.evaluate(()=>gpsLog.store.fixes(gpsLog.logger.session.id).then(f=>f.length)),'export includes the newest fix');
  await page.locator('#gps-stop').click();await page.waitForFunction(()=>!gpsLog.logger.active);
  console.log('GPS log:',{fixes:drive.fixes.length,wakeLock,labels:evaluation.labels.length,passes:evaluation.passes.length});
  assert(!network.some(r=>/^https?:/.test(r.url)&&r.method!=='GET'),'no photo/evidence uploads');
  const shellPhotos=await page.evaluate(async()=>{const keys=await caches.keys(),urls=[];for(const k of keys.filter(k=>k.startsWith('locaintel-shell-')))urls.push(...(await(await caches.open(k)).keys()).map(r=>r.url));return urls.filter(u=>/\/frames\//.test(u));});assert.deepEqual(shellPhotos,[],'photos stored only in IndexedDB, never shell cache');
  console.log('Exact YOLO boxes, 5/5 model runs, 4 tiered estimates + 1 tiny abstention, legacy two-screenshot confirmation, reference editing/reload, mobile layout, paired offline reload, automatic local missed-photo retry, GPS drive log/labels/exports and no uploads with host stopped: passed.');
 }finally{await browser?.close();if(server.listening)await new Promise(r=>server.close(r));}
})().catch(error=>{console.error(error);process.exitCode=1;});

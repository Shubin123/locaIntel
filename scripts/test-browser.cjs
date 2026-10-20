/* Browser-only storage and full facing coverage, using a static fixture host. */
const {chromium}=require('playwright'),http=require('node:http'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../web'),mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.wasm':'application/wasm','.json':'application/json'};
let png,payload;const requests=[];
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
  browser=await chromium.launch({executablePath:process.env.GPU_BENCH_CHROME||undefined,headless:true,args:['--enable-unsafe-webgpu',...(process.platform==='linux'?['--use-angle=vulkan','--enable-features=Vulkan','--disable-vulkan-surface']:[])]});
  const context=await browser.newContext(),page=await context.newPage({viewport:{width:1280,height:900}}),errors=[];
  page.on('pageerror',e=>{errors.push(e.message);console.error('Browser error:',e.message);});
  png=Buffer.from(await page.evaluate(()=>{const c=document.createElement('canvas');c.width=640;c.height=400;const g=c.getContext('2d');g.fillStyle='#777';g.fillRect(0,0,640,400);for(const [x,y,w,h,color]of [[70,100,100,70,'#b51e1e'],[230,130,110,65,'#eeeeee'],[420,160,100,70,'#223344']]){g.fillStyle=color;g.fillRect(x,y,w,h);}return c.toDataURL('image/png').split(',')[1];}),'base64');
  const hash=crypto.createHash('sha256').update(png).digest('hex'),detections=[[.11,.25,.26,.425],[.36,.325,.53,.49],[.66,.4,.81,.575],[.2,.2,.21,.21],[0,.6,.15,.8]].map(box=>({label:'car',box,confidence:.9,motion:{state:'unknown'},on_road:null}));
  payload={generated:new Date().toISOString(),catalog_size:1,model:'Browser regression fixture',schema_version:2,errors:[],cameras:[{id:'238',name:'Browser fixture',caption:'Local retention test',region:'Test',highway:'1',lat:49,lon:-123,hash,frame:'frames/test.png',image:'https://www.drivebc.ca/images/238.jpg',observed:new Date().toISOString(),detections,counts:{car:5},analysis:{}}]};
  await context.route('https://www.drivebc.ca/images/238.jpg*',r=>r.fulfill({contentType:'image/png',headers:{'access-control-allow-origin':'*'},body:png}));
  await context.route('**/tile.openstreetmap.org/**',r=>r.abort());
  const origin=`http://127.0.0.1:${server.address().port}`;await page.goto(origin);
  await page.waitForFunction(()=>{if(!data)return false;const c=displayedCamera(data.cameras[0]);return c.detections.length===5&&c.detections.every(d=>d.orientation?.model_ran)&&!facingAnalyzer.running;},null,{timeout:90000});
  let coverage=await page.evaluate(()=>{const c=displayedCamera(data.cameras[0]);return {cars:c.detections.length,ran:c.detections.filter(d=>d.orientation.model_ran).length,abstained:c.detections.filter(d=>d.orientation.quality_abstention).length,backend:facingAnalyzer.backend};});
  assert.equal(coverage.ran,5);assert.equal(coverage.abstained,2);assert.equal(coverage.backend,'webgpu');console.log('Facing coverage:',coverage);
  assert.equal(await page.evaluate(()=>displayedCamera(data.cameras[0]).detections.filter(d=>d.orientation.state==='estimated').length),0,'a single screenshot never confirms facing');
  // Controlled confidence fixture tests temporal plumbing, not model accuracy.
  // Per-frame GPU coverage above uses the real production model.
  await page.evaluate(async()=>{
    pairCapture.pause();await photoArchive.chain;
    const camera=data.cameras[0],current=archiveMetadata.get(camera.id+':'+camera.hash),blob=await photoArchive.photo(camera.hash),bitmap=await createImageBitmap(blob),canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;
    const g=canvas.getContext('2d');g.drawImage(bitmap,0,0);bitmap.close();g.fillStyle='#818181';g.fillRect(10,10,15,15);
    const previousBlob=await new Promise(r=>canvas.toBlob(r,'image/png')),previousHash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await previousBlob.arrayBuffer())),v=>v.toString(16).padStart(2,'0')).join('');
    await photoArchive.saveFrame(camera,{hash:previousHash,blob:previousBlob,width:640,height:400},{observedAt:current.observedAt-5000,analysisMode:'published',detections:camera.detections,counts:camera.counts,analysis:{}});
    const manifest=await (await fetch('models/vehicle-facing-v1.json')).json(),logits=Array.from({length:8},(_,i)=>i===0?8:-5);
    for(const hash of [previousHash,camera.hash])for(let i=0;i<3;i++)await photoArchive.saveOrientation(camera.id+':'+hash,i,{...VehicleFacing.classify(logits,0,manifest,{type:'none'},{logits,angle:0}),fingerprint:VehicleFacing.fingerprint(camera.detections[i]),model_ran:true,backend:'webgpu',fixture:true});
    scheduleTemporal(current);queueViewPaint();
  });
  await page.waitForFunction(()=>displayedCamera(data.cameras[0]).detections.slice(0,3).every(d=>d.orientation?.paired?.state==='confirmed'),null,{timeout:30000});
  assert.equal(await page.evaluate(()=>displayedCamera(data.cameras[0]).detections.filter(d=>d.orientation?.paired?.state==='confirmed').length),3);
  await page.waitForFunction(()=>archiveMetadata.get(data.cameras[0].id+':'+data.cameras[0].hash)?.pairedFacing?.byFingerprint?.[VehicleFacing.fingerprint(data.cameras[0].detections[0])]?.paired?.state==='confirmed');
  await page.evaluate(async()=>{const c=data.cameras[0];await photoArchive.saveTemporal(c.id+':'+c.hash,{detections:[{label:'car',box:[.8,.8,.9,.9],motion:{state:'moving',vector:[.01,0],gap_seconds:1}}],log:{frameHashes:[c.hash],states:{moving:1,stationary:0,unknown:0},status:'test_stale'}});});
  assert.equal(await page.evaluate(()=>displayedCamera(data.cameras[0]).detections.length),5);
  await page.locator('.heading-controls summary').click();
  for(const y of ['0.55']){await page.locator('#reference-y1').fill(y);await page.locator('#reference-y2').fill(y);await page.locator('#reference-coordinates button').click();await page.waitForFunction(n=>directionReferences.get('238')?.vectors.length===n,y==='0.55'?1:2);}
  await page.locator('#draw-reference').click();await page.locator('#detail-picture').scrollIntoViewIfNeeded();const bounds=await page.locator('#detail-picture').boundingBox();
  await page.mouse.move(bounds.x+bounds.width*.2,bounds.y+bounds.height*.6);await page.mouse.down();await page.mouse.move(bounds.x+bounds.width*.7,bounds.y+bounds.height*.6,{steps:8});await page.mouse.up();
  await page.waitForFunction(()=>directionReferences.get('238')?.vectors.length===2);await page.locator('#draw-reference').click();
  await page.waitForFunction(()=>!!navigator.serviceWorker.controller);await page.evaluate(async()=>{await navigator.serviceWorker.ready;await photoArchive.chain;});
  await page.reload();await page.waitForFunction(()=>directionReferences.get('238')?.vectors.length===2&&data&&displayedCamera(data.cameras[0]).detections.every(d=>d.orientation?.model_ran),null,{timeout:60000});
  assert.equal(await page.evaluate(async()=>{await photoArchive.chain;return Object.keys((await photoArchive.metadata())[0].facing.byFingerprint).length;}),5);
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  await context.setOffline(true);await new Promise(resolve=>server.close(resolve));await page.reload();
  await page.waitForFunction(()=>data&&displayedCamera(data.cameras[0]).detections.filter(d=>d.orientation?.model_ran).length===5&&directionReferences.get('238')?.vectors.length===2,null,{timeout:60000});
  assert.equal(await page.evaluate(async()=>(await photoArchive.stats()).photos),2);assert.equal(await page.evaluate(()=>displayedCamera(data.cameras[0]).detections.filter(d=>d.orientation?.paired?.state==='confirmed').length),3);assert(!requests.some(p=>p.startsWith('/api/')));assert.deepEqual(errors,[]);
  console.log('Exact YOLO boxes, 5/5 model runs, quality abstentions, two-screenshot confirmation, reference editing/reload, mobile layout, and paired offline reload with host stopped: passed.');
 }finally{await browser?.close();if(server.listening)await new Promise(r=>server.close(r));}
})().catch(error=>{console.error(error);process.exitCode=1;});

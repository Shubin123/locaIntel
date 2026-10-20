/* Real-browser WebGPU throughput and batch-fidelity check.
 * GPU_BENCH_CHROME=/path/to/chrome npm run benchmark:gpu
 * Optional: --crops=/path/to/CHW-float32-crops.f32 (N x 3 x 224 x 224).
 */
const {chromium}=require('playwright');
const http=require('node:http'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../web'),stride=3*224*224;
const cropsPath=process.argv.find(x=>x.startsWith('--crops='))?.slice(8);
const cropBytes=cropsPath?fs.readFileSync(cropsPath):null;
const baselinePath=process.argv.find(x=>x.startsWith('--baseline-worker='))?.slice(18);
const baselineWorker=baselinePath?fs.readFileSync(baselinePath):null;
if(cropBytes&&(!cropBytes.length||cropBytes.length%(stride*4)))throw Error('Expected packed float32 CHW crops at 224×224');
const mime={'.md':'text/plain','.js':'text/javascript','.mjs':'text/javascript','.html':'text/html','.wasm':'application/wasm','.json':'application/json'};
const server=http.createServer((req,res)=>{
 if(req.url==='/__baseline_worker'&&baselineWorker){res.setHeader('content-type','text/javascript');res.end(baselineWorker);return;}
 if(req.url==='/__benchmark_inputs'&&cropBytes){res.end(cropBytes);return;}
 const file=path.resolve(root,'.'+new URL(req.url,'http://localhost').pathname);
 if(!file.startsWith(root+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);res.end();return;}
 res.setHeader('content-type',mime[path.extname(file)]||'application/octet-stream');fs.createReadStream(file).pipe(res);
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
 try{
  browser=await chromium.launch({executablePath:process.env.GPU_BENCH_CHROME||undefined,headless:!process.argv.includes('--headed'),args:['--enable-unsafe-webgpu',...(process.platform==='linux'?['--use-angle=vulkan','--enable-features=Vulkan','--disable-vulkan-surface']:[])]});
  const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/models/README.md`);
  const info=await page.evaluate(async()=>{const a=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});if(!a)throw Error('No WebGPU adapter: use a supported GPU/browser');return {vendor:a.info.vendor,architecture:a.info.architecture,description:a.info.description,shader_f16:a.features.has('shader-f16')};});
  assert(!/swiftshader|llvmpipe|software/i.test(JSON.stringify(info)),'Software GPU detected');
  console.log(JSON.stringify({adapter:info,input:cropsPath||'deterministic synthetic crops',warmup:3,samples:15}));
  const references=[];
  for(const mode of [...(baselineWorker?[{size:1,baseline:true}]:[]),...[1,2,4,8].map(size=>({size,baseline:false}))]){
   const {size,baseline}=mode;
   const report=await page.evaluate(async({size,baseline,hasCrops,references})=>{
    const stride=3*224*224,raw=hasCrops?new Float32Array(await (await fetch('/__benchmark_inputs')).arrayBuffer()):null;
    const count=raw?raw.length/stride:8;
    const inputs=Array.from({length:count},(_,i)=>raw?raw.slice(i*stride,(i+1)*stride):new Float32Array(stride).fill(.5+i*.02));
    const worker=new Worker(baseline?'/__baseline_worker':'/orientation-worker.js');let serial=0,waiter;
    worker.onmessage=({data})=>{if(data.id){clearTimeout(waiter.timer);data.error?waiter.reject(Error(data.error)):waiter.resolve(data);}};
    worker.onerror=e=>waiter?.reject(Error(e.message));
    const run=async(indices)=>{
     const jobs=indices.map(i=>({input:inputs[i].slice(),lights:{type:'none'},crop:{width:224,height:224}}));
     const start=performance.now();const data=await new Promise((resolve,reject)=>{waiter={resolve,reject,timer:setTimeout(()=>reject(Error('GPU benchmark timed out')),120000)};worker.postMessage(size===1?{id:++serial,...jobs[0]}:{id:++serial,jobs,batchSize:size},jobs.map(j=>j.input.buffer));});
     const results=data.results||[data.result];if(data.batchError||results.some(r=>r.backend!=='webgpu'))throw Error(data.batchError||'Inference fell back from WebGPU');
     return {ms:performance.now()-start,results};
    };
    try{
     const indices=Array.from({length:Math.min(size,count)},(_,i)=>i),cold=await run(indices);
     for(let i=0;i<3;i++)await run(indices);
     const times=[];let last;for(let i=0;i<15;i++){last=await run(indices);times.push(last.ms);}times.sort((a,b)=>a-b);
     const predictions=[];
     for(let i=0;i<count;i+=size){const r=await run(Array.from({length:Math.min(size,count-i)},(_,j)=>i+j));predictions.push(...r.results.map(x=>({view:x.view,scores:x.view_scores,angle:x.angle_degrees,path:x.path})));}
     let maxScoreError=0,maxAngleError=0,matchingViews=0;
     if(references.length)predictions.forEach((p,i)=>{const r=references[i];if(p.view===r.view)matchingViews++;maxScoreError=Math.max(maxScoreError,...p.scores.map((s,j)=>Math.abs(s-r.scores[j])));maxAngleError=Math.max(maxAngleError,Math.abs(((p.angle-r.angle+540)%360)-180));});
     return {mode:baseline?'before changes':'current',size,crops_per_call:indices.length,cold_ms:cold.ms,median_batch_ms:times[7],p95_batch_ms:times[14],median_car_ms:times[7]/indices.length,cars_per_second:1000*indices.length/times[7],capture:last.results.every(r=>r.gpu_capture),paths:last.results.map(r=>r.path),fidelity:references.length?{crops:count,matchingViews,maxScoreError,maxAngleError}:null,predictions};
    }finally{worker.terminate();}
   },{size,baseline,hasCrops:!!cropBytes,references});
   if(!references.length)references.push(...report.predictions);
   else {assert.equal(report.fidelity.matchingViews,references.length);assert(report.fidelity.maxScoreError<1e-4);assert(report.fidelity.maxAngleError<.1);}
   const {predictions,...result}=report;console.log(JSON.stringify(result));
  }
 }finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});

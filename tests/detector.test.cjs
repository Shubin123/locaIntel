const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const ctx=vm.createContext({setTimeout,clearTimeout,Date,Float32Array});
vm.runInContext(fs.readFileSync('web/detector.js','utf8'),ctx);
const {letterbox,decode}=ctx.YoloBoxes;
function output(rows){const n=rows.length,data=new Float32Array(84*n);for(let i=0;i<n;i++){const [x,y,w,h,cls,score]=rows[i];for(const [ch,v] of [[0,x],[1,y],[2,w],[3,h],[cls+4,score]])data[ch*n+i]=v;}return{data,dims:[1,84,n]};}

test('letterbox inversion places boxes on the original image',()=>{
 const geometry=letterbox(1280,720);assert.equal(geometry.padY,140);assert.equal(geometry.scale,.5);
 const o=output([[320,320,320,180,2,.9]]),rows=decode(o.data,o.dims,geometry);
 assert.equal(rows.length,1);assert.equal(rows[0].label,'car');
 assert.deepEqual(Array.from(rows[0].box),[.25,.25,.75,.75]);assert.equal(rows[0].motion.state,'unknown');assert.equal(rows[0].on_road,null);
});
test('YOLO decoding filters classes/confidence, clips boxes and performs class-specific NMS',()=>{
 const o=output([[320,320,200,200,2,.9],[321,321,200,200,2,.8],[320,320,200,200,7,.85],[320,320,100,100,2,.1],[100,100,100,100,16,.99],[-10,320,100,100,0,.7]]);
 const rows=decode(o.data,o.dims,letterbox(640,640));assert.equal(rows.length,3);assert.equal(rows[0].label,'car');assert(rows.some(r=>r.label==='truck'));assert.equal(rows.find(r=>r.label==='person').box[0],0);
});
test('invalid tensor dimensions are rejected',()=>assert.throws(()=>decode(new Float32Array(12),[1,6,2],letterbox(640,640)),/Unsupported/));
test('detector serializes work, deduplicates hashes and exposes source-specific results',async()=>{
 let active=0,peak=0,calls=0;
 class Worker{postMessage(m){calls++;active++;peak=Math.max(peak,active);setTimeout(()=>{active--;this.onmessage({data:{id:m.id,detections:[{label:'car',box:[.1,.1,.2,.2],confidence:.8}]}})},2);}terminate(){}}
 const detector=new ctx.BrowserDetector({workerFactory:()=>new Worker(),prepare:async()=>({input:new Float32Array(1),geometry:{}})});
 const a=detector.request({hash:'a'}),duplicate=detector.request({hash:'a'}),b=detector.request({hash:'b'});
 const result=await a;await Promise.all([duplicate,b]);assert.equal(calls,2);assert.equal(peak,1);assert.equal(result.hash,'a');assert.equal(result.counts.car,1);assert.equal(detector.get('b').status,'ready');
 await detector.request({hash:'a'});assert.equal(calls,2);
});
test('failed detection is visible and can be retried instead of reusing old boxes',async()=>{
 let fail=true;
 class Worker{postMessage(m){queueMicrotask(()=>this.onmessage({data:fail?{id:m.id,error:'model unavailable'}:{id:m.id,detections:[]}}));}terminate(){}}
 const detector=new ctx.BrowserDetector({workerFactory:()=>new Worker(),prepare:async()=>({input:new Float32Array(1),geometry:{}})});
 assert.equal((await detector.request({hash:'new'})).status,'error');fail=false;detector.retryFailed();assert.equal((await detector.request({hash:'new'})).status,'ready');assert.equal(detector.get('new').detections.length,0);
});
test('power budget inserts idle time and paused work resumes without losing queued frames',async()=>{
 let paused=true;const starts=[];
 class Worker{postMessage(m){starts.push(Date.now());queueMicrotask(()=>this.onmessage({data:{id:m.id,detections:[]}}));}terminate(){}}
 const detector=new ctx.BrowserDetector({workerFactory:()=>new Worker(),prepare:async()=>({input:new Float32Array(1),geometry:{}}),isPaused:()=>paused,delayMs:30});
 const a=detector.request({hash:'a'}),b=detector.request({hash:'b'});await new Promise(r=>setTimeout(r,10));assert.equal(starts.length,0);
 paused=false;detector.wake();await Promise.all([a,b]);assert.equal(starts.length,2);assert(starts[1]-starts[0]>=25);
 detector.setBudget({delayMs:0});assert.equal(detector.delayMs,0);
});

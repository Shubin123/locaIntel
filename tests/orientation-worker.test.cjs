const test=require('node:test'),assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm');
function worker({ambiguous=false,fastFailure=false,batched=false,batchGraphFailure=false,batchRunFailure=false,mixed=false}={}){
 const sessions=[],runs=[],messages=[];let disposed=0;
 const logits=k=>Array.from({length:8},(_,i)=>i===k?8:-5),tensor=data=>({data,dispose(){disposed++;}});
 const manifest={model:'original CFV',views:['front','front_left','left_side','rear_left','rear','rear_right','right_side','front_right'],fast:{file:'fast.onnx',model:'trained fast',policy:{temperature:1,threshold:.9,mirror_max_error:25}},gpu:{file:'fp16.onnx',batch:batched?2:1}};
 for(const cars of [2,4,8]){manifest.gpuBatches??={};manifest.gpuBatches[cars]={fast:{file:'fast-cars'+cars+'.onnx'},gpu:{file:'half-cars'+cars+'.onnx'},gpu32:{file:'full-cars'+cars+'.onnx'}};}
 class Session{constructor(cpu,gpu){this.cpu=cpu;this.gpu=Array.isArray(gpu)?gpu[0]:gpu;this.backend='webgpu';this.inputNames=['image'];this.variant=this.gpu?.file||cpu.file;sessions.push(this);}async initialize(){if(batchGraphFailure&&this.variant.includes('-cars'))throw Error('GPU batch unsupported');return this;}async release(){this.released=true;}async run(feeds){const quick=this.cpu.file.includes('fast.onnx')||this.variant.includes('fast-cars');runs.push({quick,dims:feeds.image.dims,input:feeds.image.data});if(quick&&fastFailure)throw Error('fast graph failed');if(batchRunFailure&&this.variant.includes('-cars'))throw Error('GPU batch exhausted memory');const rows=feeds.image.dims[0],scores=[],angles=[],vectors=[];for(let i=0;i<rows;i++){const marker=feeds.image.data[Math.floor(i/2)*2*150528];scores.push(...(quick?((ambiguous||mixed&&marker>.3)?Array(8).fill(0):logits(0)):logits(marker>.3?0:4)));angles.push(marker>.3?0:.5);vectors.push(0,1);}return quick?{view_logits:tensor(scores),angle_vector:tensor(vectors)}:{view_logits:tensor(scores),angle_fraction:tensor(angles)};}}
 const context=vm.createContext({Float32Array,Math,Number,Set,Map,Array,Error,performance:{now:()=>10},ModelRuntime:{Session},ort:{Tensor:class{constructor(type,data,dims){Object.assign(this,{type,data,dims});}dispose(){disposed++;}}},fetch:async()=>({ok:true,json:async()=>manifest})});context.self={postMessage:m=>messages.push(m)};context.importScripts=(...files)=>{for(const file of files)if(file.startsWith('orientation.js'))vm.runInContext(fs.readFileSync('web/orientation.js','utf8'),context);};vm.runInContext(fs.readFileSync('web/orientation-worker.js','utf8'),context);
 return {send:async(lights={type:'none'})=>{const input=new Float32Array(150528);input[0]=.25;await context.self.onmessage({data:{id:messages.length+1,input,lights,crop:{width:50,height:40}}});return messages.at(-1);},sendBatch:async(n,batchSize=n)=>{const jobs=Array.from({length:n},(_,i)=>{const input=new Float32Array(150528);input[0]=i%2?.75:.25;return {input,lights:{type:'none'},crop:{width:50+i,height:40}};});await context.self.onmessage({data:{id:99,jobs,batchSize}});return messages.at(-1);},sessions,runs,messages,disposed:()=>disposed};
}
test('confident facing batches original and mirror without loading the large model',async()=>{const w=worker(),reply=await w.send();assert.equal(reply.result.path,'fast-verified');assert.equal(reply.result.backend,'webgpu');assert.equal(reply.result.facing,'toward_camera');assert.equal(w.sessions.length,1);assert.equal(w.runs.length,1);assert.deepEqual([...w.runs[0].dims],[2,3,224,224]);assert.equal(w.runs[0].input[150528+223],.25);assert.equal(w.disposed(),3);});
test('ambiguous fast results get the full-resolution model and retain mirror verification',async()=>{const w=worker({ambiguous:true}),reply=await w.send();assert.equal(reply.result.path,'full-resolution');assert.equal(reply.result.facing,'away_from_camera');assert.equal(reply.result.mirror_consistent,true);assert.equal(w.sessions.length,2);assert.equal(w.runs.length,3);await w.send();assert.equal(w.sessions.length,2);});
test('contradictory lamp evidence cannot bypass the full-resolution check',async()=>{const w=worker(),reply=await w.send({type:'taillight_pair'});assert.equal(reply.result.path,'full-resolution');assert.equal(reply.result.facing,'away_from_camera');assert.equal(reply.result.reason,'model_and_light_pair_agree');});
test('a failed fast graph still permits the full-resolution model',async()=>{const w=worker({fastFailure:true}),reply=await w.send();assert.equal(reply.result.path,'full-resolution');assert.match(reply.result.fast_error,/fast graph failed/);assert.equal(reply.result.state,'estimated');});

test('full-resolution GPU checker batches both verification views',async()=>{const w=worker({ambiguous:true,batched:true}),reply=await w.send();assert.equal(reply.result.path,'full-resolution');assert.equal(reply.result.mirror_consistent,true);assert.equal(w.runs.length,2);assert.equal(w.runs[1].dims[0],2);});

test('four distinct cars share the compact GPU stage and map through bounded checker microbatches',async()=>{
 const w=worker({ambiguous:true}),reply=await w.sendBatch(4);
 assert.equal(reply.batchError,null);assert.equal(reply.results.length,4);
 assert.deepEqual(w.runs.map(r=>r.dims[0]),[8,4,4]);assert.equal(w.sessions.length,2);
 assert.deepEqual(reply.results.map(r=>r.facing),['away_from_camera','toward_camera','away_from_camera','toward_camera']);
 reply.results.forEach((r,i)=>{assert.equal(r.crop.width,50+i);assert.equal(r.gpu_batch_cars,2);assert.equal(r.batch_count,4);assert.equal(r.mirror_consistent,true);});
});
test('mixed fast/full routes compact difficult crops into the checker and retain result ordering',async()=>{
 const w=worker({mixed:true}),reply=await w.sendBatch(4);
 assert.deepEqual(reply.results.map(r=>r.path),['fast-verified','full-resolution','fast-verified','full-resolution']);
 assert.equal(w.runs.length,2);assert.equal(w.runs[1].input[0],.75);assert.equal(w.runs[1].input[2*150528],.75);
});
test('partial batches run immediately and retain the selected graph for subsequent calls',async()=>{
 const w=worker({ambiguous:true});let reply=await w.sendBatch(3,4);assert.equal(reply.results.length,3);assert.equal(w.runs[0].dims[0],8);
 const count=w.sessions.length;reply=await w.sendBatch(1,4);assert.equal(reply.results.length,1);assert.equal(w.sessions.length,count);assert.equal(w.runs.at(-1).dims[0],4);assert.equal(w.runs.at(-1).input[2*150528],0);
});
test('GPU batch creation and execution errors recover through individual verified crops',async()=>{
 for(const options of [{batchGraphFailure:true},{batchRunFailure:true}]){
  const w=worker({...options,ambiguous:true,batched:true}),reply=await w.sendBatch(2);
  assert.match(reply.batchError,/GPU batch/);assert.equal(reply.results.length,2);assert(reply.results.every(r=>r.gpu_batch_cars===1&&r.mirror_consistent));
 }
});
test('changing compact batch size releases its old graph and retains the bounded checker',async()=>{
 const w=worker({ambiguous:true});await w.sendBatch(2);const old=w.sessions.filter(s=>s.variant.includes('-cars'));await w.sendBatch(4);assert(old.filter(s=>s.variant.includes('fast')).every(s=>s.released));assert(old.filter(s=>!s.variant.includes('fast')).every(s=>!s.released));
});

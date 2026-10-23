const test=require('node:test'),assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm');
function worker({failure=false,scores=null}={}){
 const sessions=[],runs=[],messages=[];let disposed=0;
 const logits=k=>Array.from({length:8},(_,i)=>i===k?8:-5),tensor=data=>({data,dispose(){disposed++;}});
 const manifest={model:'DriveBC adapted',file:'drivebc-facing-v3.onnx',sha256:'x',policy:{high:.8,medium:.6}};
 class Session{constructor(spec){this.spec=spec;this.backend='webgpu';this.inputNames=['image'];sessions.push(this);}async initialize(){return this;}async run(feeds){runs.push({dims:feeds.image.dims,input:feeds.image.data});if(failure)throw Error('graph failed');const pairs=feeds.image.dims[0]/2,one=scores||[...logits(0),...logits(0)];return {view_logits:tensor(Array.from({length:pairs},()=>one).flat()),angle_vector:tensor(Array(pairs*4).fill(0))};}}
 const context=vm.createContext({Float32Array,Math,Number,Set,Map,Array,Error,performance:{now:()=>10},ModelRuntime:{Session},ort:{Tensor:class{constructor(type,data,dims){Object.assign(this,{type,data,dims});}dispose(){disposed++;}}},fetch:async()=>({ok:true,json:async()=>manifest})});context.self={postMessage:m=>messages.push(m)};context.importScripts=(...files)=>{for(const file of files)if(file.startsWith('orientation.js'))vm.runInContext(fs.readFileSync('web/orientation.js','utf8'),context);};vm.runInContext(fs.readFileSync('web/orientation-worker.js','utf8'),context);
 return {sendJobs:async jobs=>{await context.self.onmessage({data:{id:99,jobs,batchSize:4}});return messages.at(-1);},send:async(lights={type:'none'},truncated=false)=>{const input=new Float32Array(150528);input[0]=.25;await context.self.onmessage({data:{id:messages.length+1,input,lights,truncated,crop:{width:50,height:40}}});return messages.at(-1);},sessions,runs,messages,disposed:()=>disposed};
}
test('one batched dispatch carries the crop and its mirror; the model loads once',async()=>{const w=worker(),reply=await w.send();assert.equal(reply.result.path,'mirror-averaged');assert.equal(reply.result.backend,'webgpu');assert.equal(reply.result.facing,'toward_camera');assert.equal(reply.result.tier,'high');assert.deepEqual([...w.runs[0].dims],[2,3,224,224]);assert.equal(w.runs[0].input[150528+223],.25);assert.equal(w.disposed(),3);await w.send();assert.equal(w.sessions.length,1);});
test('uncertain crops still return an estimate with a low tier; contradicting lamps and clipped crops lower it',async()=>{assert.equal((await worker({scores:Array(16).fill(0)}).send()).result.tier,'low');const w=worker();assert.equal((await w.send({type:'taillight_pair'})).result.tier,'medium');assert.equal((await w.send({type:'none'},true)).result.tier,'medium');});
test('a failed graph reports an error for that crop',async()=>{const reply=await worker({failure:true}).send();assert.match(reply.error,/graph failed/);});
test('a batch of cars runs as one dispatch of crop/mirror row pairs and keeps result order',async()=>{
 const w=worker({scores:[...Array.from({length:8},(_,i)=>i===0?8:-5),...Array.from({length:8},(_,i)=>i===0?8:-5)]});
 const jobs=[0,1,2].map(i=>{const input=new Float32Array(150528);input[0]=(i+1)/10;return {input,lights:{type:'none'},crop:{width:40+i,height:30}};});
 // The fake graph returns one car's logits; give it one pair per car.
 w.runsScores=null;const reply=await w.sendJobs(jobs);
 assert.equal(w.runs.length,1);assert.deepEqual([...w.runs[0].dims],[6,3,224,224]);assert.ok(Math.abs(w.runs[0].input[2*150528]-.2)<1e-6);
 assert.equal(reply.results.length,3);reply.results.forEach((r,i)=>{assert.equal(r.crop.width,40+i);assert.equal(r.batch_count,3);assert.equal(r.model_ran,true);});
});

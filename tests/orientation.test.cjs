const test=require('node:test'),assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm');
const scope=extra=>{const c=vm.createContext({setTimeout,clearTimeout,queueMicrotask,Date,Float32Array,...extra});vm.runInContext(fs.readFileSync('web/orientation.js','utf8'),c);return c.VehicleFacing;};
const m={model:'test',policy:{high:.8,medium:.6},views:['front','front_left','left_side','rear_left','rear','rear_right','right_side','front_right'],angle_convention:'CFV relative view'},f=scope(),logits=index=>Array.from({length:8},(_,i)=>i===index?8:-5);
test('trained front/rear/side classes decode to facing without claiming motion',()=>{for(const [index,facing]of [[0,'toward_camera'],[4,'away_from_camera'],[2,'image_right']]){const r=f.classify(logits(index),index/8,m,{type:'none'},{logits:logits((8-index)%8),angle:((8-index)%8)/8});assert.equal(r.state,'estimated');assert.equal(r.facing,facing);assert.equal(r.measured_motion,false);assert.equal(r.mirror_consistent,true);}});
test('angle disagreement, mirror disagreement and conflicting lamps reject an estimate',()=>{assert.equal(f.classify(logits(0),.5,m).reason,'angle_view_disagreement');assert.equal(f.classify(logits(0),0,m,{type:'taillight_pair'}).reason,'lamp_model_disagreement');assert.equal(f.classify(logits(1),.125,m,{type:'none'},{logits:logits(3),angle:.375}).reason,'mirror_disagreement');assert.equal(f.classify(Array(8).fill(0),0,m).state,'unknown');assert.throws(()=>f.decode([NaN],0),/Invalid/);});
test('paired lamps distinguish red tail cues and white head cues but reject red body panels',()=>{const pixels=color=>{const data=new Uint8ClampedArray(100*60*4);for(const [x,y,w,h]of [[20,35,4,3],[72,35,4,3]])for(let yy=y;yy<y+h;yy++)for(let xx=x;xx<x+w;xx++)for(let c=0;c<3;c++)data[(yy*100+xx)*4+c]=color[c];return data;};assert.equal(f.lampPairs(pixels([255,240,210]),100,60).type,'headlight_pair');assert.equal(f.lampPairs(pixels([240,30,20]),100,60).type,'taillight_pair');const solid=new Uint8ClampedArray(100*60*4);for(let i=0;i<solid.length;i+=4)solid[i]=240;assert.equal(f.lampPairs(solid,100,60).type,'none');});
test('only sub-8×6-pixel crops abstain; clipped crops are flagged for a lower tier',async()=>{
 let closed=0;const context2d={fillRect(){},drawImage(){},getImageData:()=>({data:new Uint8ClampedArray(224*224*4)})};
 const g=scope({createImageBitmap:async()=>({width:640,height:400,close(){closed++;}}),document:{createElement:()=>({getContext:()=>context2d})}});
 const prediction={state:'estimated',facing:'toward_camera',image_direction:'down',facing_glyph:'↓',model:'CFV',mirror_consistent:true};
 const tiny=await g.prepare({blob:{}},{label:'car',box:[.1,.1,.105,.105]});assert.equal(tiny.skip,undefined);assert.equal(tiny.input.length,150528);assert.equal(tiny.quality,'crop_too_small');
 const result=g.qualityGate(prediction,tiny);assert.equal(result.state,'unknown');assert.equal(result.reason,'crop_too_small');assert.equal(result.model_ran,true);assert.equal(result.facing_glyph,undefined);
 const clipped=await g.prepare({blob:{}},{label:'truck',box:[0,.2,.3,.5]});assert.equal(clipped.quality,null);assert.equal(clipped.truncated,true);assert.equal(g.qualityGate(prediction,clipped).state,'estimated');
 assert.equal(closed,2);
});
test('crop inference is serial, deduplicated, paced, and pauses queued work',async()=>{let paused=true,active=0,peak=0,calls=0;const starts=[],records=[];class Worker{postMessage(msg){calls++;starts.push(Date.now());active++;peak=Math.max(peak,active);setTimeout(()=>{active--;this.onmessage({data:{id:msg.id,result:{state:'estimated',facing:'toward_camera',version:f.VERSION,measured_motion:false}}});},2);}terminate(){}}
const q=new f.CropOrientation({workerFactory:()=>new Worker(),prepareCrop:async()=>({input:new Float32Array(1)}),delayMs:30,isPaused:()=>paused,onResult:(source,index,r)=>records.push({source,index,r})}),source={cameraId:'1',hash:'a'},d={label:'car',box:[.1,.1,.2,.2]};q.request(source,d,0);q.request(source,d,0);q.request({...source,hash:'b'},d,1);await new Promise(r=>setTimeout(r,10));assert.equal(calls,0);paused=false;q.wake();while(records.length<2)await new Promise(r=>setTimeout(r,10));assert.equal(calls,2);assert.equal(peak,1);assert(starts[1]-starts[0]>=25);assert.equal(records[1].index,1);assert.equal(records[0].r.fingerprint,f.fingerprint(d));});
test('bright mirror or windshield reflections in the upper crop are not headlight pairs',()=>{const data=new Uint8ClampedArray(100*60*4);for(const x of [20,72])for(let y=16;y<20;y++)for(let xx=x;xx<x+4;xx++)for(let c=0;c<3;c++)data[(y*100+xx)*4+c]=240;assert.equal(f.lampPairs(data,100,60).type,'none');});
test('identical crop pixels across new source hashes reuse facing but refresh lamp coordinates',async()=>{let calls=0;const records=[];class Worker{postMessage(m){calls++;queueMicrotask(()=>this.onmessage({data:{id:m.id,result:{state:'estimated',facing:'toward_camera',version:f.VERSION,lights:{points:[[.2,.3]]}}}}));}terminate(){}}const q=new f.CropOrientation({workerFactory:()=>new Worker(),delayMs:0,prepareCrop:async source=>({input:new Float32Array(1),visualKey:'same-exact-pixels',lights:{points:[[source.hash==='a'?.2:.4,.3]]}}),onResult:(s,i,r)=>records.push(r)}),d={label:'car',box:[.1,.1,.2,.2]};q.request({cameraId:'1',hash:'a'},d,0);while(records.length<1)await new Promise(r=>setTimeout(r,2));q.request({cameraId:'1',hash:'b'},d,0);while(records.length<2)await new Promise(r=>setTimeout(r,2));assert.equal(calls,1);assert.equal(records[1].reused_exact_crop,true);assert.equal(records[1].lights.points[0][0],.4);});
test('fast facing requires calibrated confidence, exact mirrored view, and stable angle vectors',()=>{
 const manifest={...m,fast:{policy:{temperature:1,threshold:.9,mirror_max_error:25}}};
 const pair=(a,b)=>[...logits(a),...logits(b)];
 const vector=a=>[Math.sin(a*Math.PI/180),Math.cos(a*Math.PI/180)];
 const good=f.fastClassify(pair(1,7),[...vector(45),...vector(315)],manifest);
 assert.equal(good.state,'estimated');assert.equal(good.facing,'toward_camera');assert.equal(good.image_direction,'down-right');assert.equal(good.measured_motion,false);
 for(const [scores,vectors]of [[pair(1,1),[...vector(45),...vector(45)]],[Array(16).fill(0),[...vector(45),...vector(315)]],[pair(1,7),[0,0,0,0]],[pair(1,7),[...vector(45),...vector(280)]]]){
  const result=f.fastClassify(scores,vectors,manifest);assert.equal(result.state,'unknown');assert.equal(result.image_direction,undefined);
 }
 assert.equal(f.fastClassify(pair(0,0),[...vector(0),...vector(0)],manifest,{type:'taillight_pair'}).state,'unknown');
 assert.throws(()=>f.fastClassify([NaN],[],manifest),/Invalid/);
});

test('GPU queue batches priority crops, drains partial batches, and retains exact frame identities',{timeout:2000},async()=>{
 let paused=true;const calls=[],records=[];
 class Worker{postMessage(m){calls.push(m);queueMicrotask(()=>this.onmessage({data:{id:m.id,results:m.jobs.map(p=>({state:'estimated',marker:p.input[0]}))}}));}terminate(){}}
 const q=new f.CropOrientation({batchSize:4,workerFactory:()=>new Worker(),prepareCrop:async s=>({input:Float32Array.of(Number(s.hash))}),isPaused:()=>paused,delayMs:0,onResult:(s,i,r)=>records.push([s.hash,i,r])});
 const d={label:'car',box:[.1,.1,.3,.3]};for(let i=0;i<7;i++)q.request({cameraId:'1',hash:String(i)},d,i,i===5?100:0);
 q.request({cameraId:'1',hash:'5'},d,5,100);paused=false;q.wake();
 while(records.length<7)await new Promise(r=>setTimeout(r,2));
 assert.deepEqual(calls.map(c=>c.jobs.length),[4,3]);assert.equal(calls[0].jobs[0].input[0],5);assert(calls.every(c=>c.batchSize===4));
 assert.equal(new Set(records.map(r=>r[0])).size,7);records.forEach(([hash,index,r])=>{assert.equal(r.marker,Number(hash));assert.equal(index,Number(hash));assert.equal(r.fingerprint,f.fingerprint(d));});
});
test('pausing during crop preparation requeues the crop before GPU dispatch',{timeout:2000},async()=>{
 let paused=false,calls=0,prepared=0;const records=[];
 class Worker{postMessage(m){calls++;queueMicrotask(()=>this.onmessage({data:{id:m.id,results:m.jobs.map(()=>({state:'unknown'}))}}));}terminate(){}}
 const q=new f.CropOrientation({batchSize:2,workerFactory:()=>new Worker(),prepareCrop:async()=>{if(++prepared===1)paused=true;return {input:Float32Array.of(1)};},isPaused:()=>paused,delayMs:0,onResult:r=>records.push(r)});
 q.request({cameraId:'1',hash:'a'},{label:'car',box:[.1,.1,.3,.3]},0);await new Promise(r=>setTimeout(r,5));
 assert.equal(calls,0);assert.equal(q.queue.length,1);assert.equal(q.entries.values().next().value.state,'queued');
 paused=false;q.wake();while(!records.length)await new Promise(r=>setTimeout(r,2));assert.equal(calls,1);
});
test('changing the slider during preparation preserves the in-flight batch shape',{timeout:2000},async()=>{
 let paused=true,preparations=0;const calls=[],records=[];
 class Worker{postMessage(m){calls.push(m);queueMicrotask(()=>this.onmessage({data:m.jobs?{id:m.id,results:m.jobs.map(()=>({state:'unknown'}))}:{id:m.id,result:{state:'unknown'}}}));}terminate(){}}
 const q=new f.CropOrientation({batchSize:4,isPaused:()=>paused,delayMs:0,workerFactory:()=>new Worker(),prepareCrop:async()=>{if(++preparations===1)q.setBudget({batchSize:1});return {input:Float32Array.of(1)};},onResult:r=>records.push(r)});
 for(let i=0;i<5;i++)q.request({cameraId:'1',hash:String(i)},{label:'car',box:[.1,.1,.3,.3]},i);
 paused=false;q.wake();while(records.length<5)await new Promise(r=>setTimeout(r,2));
 assert.equal(calls[0].jobs.length,4);assert.equal(calls[0].batchSize,4);assert(calls[1].input);
});

test('failed GPU batching keeps subsequent work individual until the batch setting changes',{timeout:2000},async()=>{
 let paused=true;const calls=[],records=[];
 class Worker{postMessage(m){calls.push(m);queueMicrotask(()=>this.onmessage({data:m.jobs?{id:m.id,batchError:'GPU batch exhausted memory',results:m.jobs.map(()=>({state:'unknown'}))}:{id:m.id,result:{state:'unknown'}}}));}terminate(){}}
 const q=new f.CropOrientation({batchSize:2,isPaused:()=>paused,delayMs:0,workerFactory:()=>new Worker(),prepareCrop:async()=>({input:Float32Array.of(1)}),onResult:r=>records.push(r)}),d={label:'car',box:[.1,.1,.3,.3]};
 for(let i=0;i<5;i++)q.request({cameraId:'1',hash:String(i)},d,i);
 paused=false;q.wake();while(records.length<5)await new Promise(r=>setTimeout(r,2));
 assert.equal(calls[0].jobs.length,2);assert(calls.slice(1).every(c=>c.input));assert.equal(q.batchError,'GPU batch exhausted memory');
 q.setBudget({batchSize:4});assert.equal(q.batchError,null);
});
test('every supported crop gets a facing with a confidence tier; mirror views are averaged',()=>{
 const pair=(a,b)=>[...logits(a),...logits(b)];
 const front=f.estimate(pair(0,0),m);assert.equal(front.state,'estimated');assert.equal(front.facing,'toward_camera');assert.equal(front.tier,'high');assert.equal(front.travel_offset,180);assert.equal(front.facing_glyph,'↓');assert.equal(front.measured_motion,false);
 const rear=f.estimate(pair(4,4),m);assert.equal(rear.facing,'away_from_camera');assert.equal(rear.travel_offset,0);
 // A left-side view and the mirror of a right-side view agree: nose points image-right.
 const side=f.estimate(pair(2,6),m);assert.equal(side.group,'side');assert.equal(side.facing,'image_right');assert.equal(side.travel_offset,90);assert.equal(side.facing_glyph,'→');
 const unsure=f.estimate(Array(16).fill(0),m);assert.equal(unsure.state,'estimated');assert.equal(unsure.tier,'low');
 assert.equal(f.estimate(pair(0,0),m,{type:'taillight_pair'}).tier,'medium');assert.equal(f.estimate(pair(0,0),m,{type:'none'},{truncated:true}).tier,'medium');
 assert.throws(()=>f.estimate([NaN],m),/Invalid/);
});

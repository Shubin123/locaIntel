const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
function load(temporal=true){const c=vm.createContext({});for(const name of ['orientation',...(temporal?['temporal']:[]),'paired-facing'])vm.runInContext(fs.readFileSync(`web/${name}.js`,'utf8'),c);return c;}
const c=load(),P=c.PairedFacing,views=['front','front_left','left_side','rear_left','rear','rear_right','right_side','front_right'];
const model=JSON.parse(fs.readFileSync('web/models/paired-facing-v1.json','utf8'));
const clone=x=>JSON.parse(JSON.stringify(x));
const detection=(x=.2)=>({label:'car',box:[x,.4,x+.08,.5],confidence:.9,appearance:Array(432).fill(.4),motion:{state:'unknown'}});
function base(d,index=0,probability=null){
  const logits=probability?probability.map(Math.log):Array.from({length:8},(_,i)=>i===index?7:-7),mirrored=Array.from({length:8},(_,i)=>logits[(8-i)%8]);
  return {...c.VehicleFacing.classify(logits,index/8,{model:'test production model',views},{type:'none'},{logits:mirrored,angle:((8-index)%8)/8}),model_ran:true,path:'full-resolution',fingerprint:c.VehicleFacing.fingerprint(d)};
}
function row(hash,time,x=.2){
  const d=detection(x),fingerprint=c.VehicleFacing.fingerprint(d),signature=JSON.stringify([fingerprint]);
  return {key:'238:'+hash,cameraId:'238',hash,observedAt:time,detections:[d],features:{version:1,hash,detectionSignature:signature,width:640,height:400,pixelSignature:'pixels-'+hash,background:Array(100).fill(.2),detections:[clone(d)]},facing:{byFingerprint:{[fingerprint]:base(d)}}};
}
const combine=(current,history,selectedModel=model)=>P.combine(P.baseFor(current,current.detections[0]),current.detections[0],P.context(current,history),selectedModel);
function setBase(r,value){r.facing.byFingerprint[c.VehicleFacing.fingerprint(r.detections[0])]=value;}
test('one screenshot and its mirrored model view cannot confirm facing',()=>{
  const current=row('current',10000),result=combine(current,[]);assert.equal(result.state,'unknown');assert.equal(result.reason,'need_second_screenshot');assert.equal(result.paired.count,1);assert.equal(result.mirror_consistent,true);assert.equal(result.motion,undefined);
});
test('two distinct same-camera observations confirm unchanged latest output without inventing movement',()=>{
  const previous=row('previous',5000),current=row('current',10000,.21),result=combine(current,[previous]);
  assert.equal(result.state,'estimated');assert.equal(result.facing,'toward_camera');assert.equal(result.paired.state,'confirmed');assert.equal(result.paired.count,2);assert.equal(result.paired.measured_motion,false);assert.equal(result.measured_motion,false);assert.equal(current.detections[0].motion.state,'unknown');assert.equal(result.angle_degrees,P.baseFor(current,current.detections[0]).angle_degrees);
});
test('association requires the same normalized camera, strictly earlier time and at most 15 seconds',()=>{
  const current=row('current',20000),previous=row('previous',5000);previous.cameraId=238;assert.equal(combine(current,[previous]).paired.state,'confirmed');
  for(const change of [r=>r.cameraId='different',r=>r.observedAt=4999,r=>r.observedAt=20000,r=>r.observedAt=20001,r=>r.observedAt=NaN,r=>r.hash='current']){const wrong=clone(previous);change(wrong);assert.equal(combine(current,[wrong]).state,'unknown');}
  assert.equal(P.rowsFor(null,[]).length,0);assert.equal(P.rowsFor(current,null).length,0);
});
test('nonadjacent duplicate pixels and repeated photo hashes never increase evidence count',()=>{
  const current=row('current',20000),a=row('a',19000),b=row('b',18000),d=row('d',17000);b.features.pixelSignature=a.features.pixelSignature;d.features.pixelSignature=current.features.pixelSignature;
  const result=combine(current,[a,b,d,a]);assert.equal(result.paired.state,'confirmed');assert.equal(result.paired.count,2);
  assert.equal(combine(current,[d]).reason,'duplicate_screenshot_pixels');
});
test('at most five cached observations contribute, ordered before the current screenshot',()=>{
  const current=row('current',20000),history=Array.from({length:8},(_,i)=>row('old'+i,19000-i*1000)),result=combine(current,history);
  assert.equal(result.paired.count,5);assert.equal(result.paired.frames.at(-1).hash,'current');assert(result.paired.frames.every(f=>f.observedAt<=current.observedAt));
});
test('ambiguous vehicles and changed backgrounds cannot establish paired identity',()=>{
  const current=row('current',10000),previous=row('previous',5000),other=detection(.205);previous.detections.push(other);previous.features.detections.push(clone(other));previous.features.detectionSignature=JSON.stringify(previous.detections.map(c.VehicleFacing.fingerprint));
  assert.equal(combine(current,[previous]).state,'unknown');
  const changed=row('changed',5000);changed.features.background.fill(.5);assert.equal(combine(current,[changed]).reason,'camera_view_changed');changed.features.background.fill(.2);changed.features.width=641;assert.equal(combine(current,[changed]).state,'unknown');
});
test('cached feature corruption fails closed without rendering exceptions',()=>{
  const mutations=[r=>r.features.detectionSignature='stale',r=>r.features.detections=[null],r=>r.detections=[null],r=>r.detections={},r=>r.features.detections[0].box=[.1,.1,.2,.2],r=>r.features.detections[0].appearance[0]=NaN,r=>r.features.detections[0].appearance[0]=2,r=>r.features.detections[0].appearance=[],r=>r.features.width=Infinity,r=>r.features.height=0,r=>r.features.background[0]=NaN,r=>r.features.background[0]=2,r=>r.features.background=[],r=>r.features.pixelSignature='',r=>r.features.hash='wrong'];
  for(const change of mutations){const current=row('current',10000);change(current);assert.equal(P.validFeatures(current),false);assert.doesNotThrow(()=>P.context(current,[row('prior',5000)]));assert.equal(P.context(current,[]).candidates.length,0);}
  assert.equal(P.validFeatures(null),false);
  const current=row('current',10000);current.features.detections=[null];assert.doesNotThrow(()=>P.combine(base(detection()),detection(),P.context(current,[]),model));
});
test('missing temporal dependency and missing learned model remain unresolved',()=>{
  const current=row('current',10000),previous=row('previous',5000),withoutTemporal=load(false);
  assert.equal(withoutTemporal.PairedFacing.context(current,[previous]).reason,'screenshot_association_unavailable');assert.equal(combine(current,[previous],null).reason,'paired_facing_model_loading');
});
test('unknown crops, quality abstentions and legacy results can never be promoted by consensus',()=>{
  for(const mutate of [r=>r.state='unknown',r=>r.quality_abstention=true,r=>r.version='cfv-facing-cascade-v3',r=>delete r.mirror_view_scores,r=>r.model_ran=false]){
    const current=row('current',10000),previous=row('previous',5000),record=base(current.detections[0]);mutate(record);setBase(current,record);
    const result=P.combine(record,current.detections[0],P.context(current,[previous]),model);assert.equal(result.state,'unknown');assert.notEqual(result.paired.state,'confirmed');
  }
});
test('all matched cached observations must be verified and agree; conflicts remain unresolved',()=>{
  const current=row('current',10000),a=row('a',9000),b=row('b',8000);setBase(b,base(b.detections[0],4));
  assert.equal(combine(current,[a,b]).paired.state,'conflicting');
  const unknown=base(b.detections[0]);unknown.state='unknown';setBase(b,unknown);assert.equal(combine(current,[a,b]).reason,'previous_facing_pending');assert.equal(combine(current,[a,b]).state,'unknown');
});
test('original and mirror evidence is rechecked instead of trusting persisted acceptance flags',()=>{
  const corruptions=[r=>r.view_scores.fill(0),r=>r.view_scores[0]=NaN,r=>r.mirror_view_scores.fill(-100),r=>r.mirror_view_scores.fill(0),r=>r.mirror_view_scores=[1,0],r=>r.mirror_view_scores=[0,0,0,0,1,0,0,0],r=>r.angle_degrees=361,r=>r.mirror_angle_degrees=-1,r=>r.mirror_angle_degrees=120,r=>r.score=.2,r=>r.margin=.1,r=>r.facing='away_from_camera',r=>r.fingerprint='wrong box',r=>r.lights={type:'taillight_pair'},r=>r.paired={state:'confirmed'}];
  for(const mutate of corruptions){const current=row('current',10000),previous=row('previous',5000),record=base(current.detections[0]);mutate(record);setBase(current,record);assert.notEqual(combine(current,[previous])?.state,'estimated');}
  const previous=row('previous',5000),current=row('current',10000);previous.facing.byFingerprint[c.VehicleFacing.fingerprint(previous.detections[0])].fingerprint='wrong';assert.equal(combine(current,[previous]).state,'unknown');previous.facing.byDetection={0:null};assert.doesNotThrow(()=>P.baseFor(previous,previous.detections[0]));
});
test('fast-path evidence must retain its stronger mirror and probability gates',()=>{
  const current=row('current',10000),previous=row('previous',5000),record=base(current.detections[0],0,[.8,.04,.02,.02,.02,.02,.02,.06]);assert.equal(P.eligible(record),true);record.path='fast-verified';assert.equal(P.eligible(record),false);setBase(current,record);assert.equal(combine(current,[previous]).state,'unknown');
});
test('a learned head disagreement rejects rather than overriding the latest accepted class',()=>{
  const disagree=clone(model);disagree.bias[4]=1000;
  const current=row('current',10000),result=combine(current,[row('previous',5000)],disagree);
  assert.equal(result.state,'unknown');assert.equal(result.paired.state,'conflicting');assert.equal(P.baseFor(current,current.detections[0]).facing,'toward_camera');
});
test('model shape, finite numbers and temporal/base version contracts are validated',()=>{
  assert.equal(P.validateModel(model),model);
  for(const change of [m=>m.version='other',m=>m.base_version='old',m=>m.minimum_frames=1,m=>m.max_frames=6,m=>m.max_gap_ms=16000,m=>m.mean={},m=>m.std[0]=0,m=>m.weights[0]=null,m=>m.bias[0]=NaN,m=>m.threshold='0',m=>m.threshold=-1]){const invalid=clone(model);change(invalid);assert.throws(()=>P.validateModel(invalid),/Invalid paired facing model/);assert.equal(combine(row('current',10000),[row('previous',5000)],invalid).reason,'paired_facing_model_invalid');}
});

const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const ctx=vm.createContext({});vm.runInContext(fs.readFileSync('web/motion.js','utf8'),ctx);vm.runInContext(fs.readFileSync('web/heading.js','utf8'),ctx);const h=ctx.SpatialHeading;
const car={label:'car',box:[.4,.4,.5,.5],orientation:{state:'estimated',image_direction:'right',mirror_consistent:true}},scene={width:640,height:400,hash:'a'};
const reference=(vectors=[{start:[.2,.5],end:[.8,.5]}])=>({frameHash:'a',width:640,height:400,revision:1,vectors});
test('spatial fusion uses local user vectors only with mirror-verified trained facing',()=>{
 const r=h.classify(car,{...scene,reference:reference()});assert.equal(r.state,'estimated');assert.equal(r.direction,'right');assert.equal(r.measured_motion,false);assert(r.sources.includes('user_vectors'));
 const unknown=h.classify({...car,orientation:{state:'unknown'}},{...scene,reference:reference()});assert.equal(unknown.state,'unknown');assert.equal(unknown.vector,undefined);
});
test('crossing and opposing local references abstain instead of outvoting a vehicle',()=>{
 const r=h.classify(car,{...scene,reference:reference([{start:[.2,.5],end:[.8,.5]},{start:[.45,.2],end:[.45,.8]}])});assert.equal(r.state,'unknown');assert.equal(r.reason,'conflicting_reference_vectors');
 const opposite=h.classify(car,{...scene,reference:reference([{start:[.8,.5],end:[.2,.5]}])});assert.equal(opposite.reason,'reference_facing_disagreement');
});
test('moving-camera, unknown-background and aspect changes prevent annotation transfer',()=>{
 const ref={...reference(),background:Array(100).fill(.2)};
 assert.equal(h.stableView(ref,{...scene,hash:'b',background:Array(100).fill(.2)}),true);
 for(const altered of [{hash:'b'},{hash:'b',background:Array(100).fill(.6)},{hash:'b',width:400,background:Array(100).fill(.2)}])assert.equal(h.stableView(ref,{...scene,...altered}),false);
});
test('measured temporal displacement stays separate from facing and sparse guesses',()=>{
 const moving={...car,motion:{state:'moving',vector:[-.02,0],gap_seconds:1}};const r=h.classify(moving,scene);assert.equal(r.state,'measured');assert.equal(r.direction,'left');assert.equal(r.facing_conflict,true);
 assert.equal(h.classify({...moving,motion:{...moving.motion,gap_seconds:90}},scene).measured_motion,false);
});
test('temporal overlays never replace, reorder or drop current YOLO boxes',()=>{
 const a={label:'car',box:[.1,.1,.2,.2]},b={label:'car',box:[.3,.3,.4,.4]},old={label:'car',box:[.11,.1,.21,.2],motion:{state:'moving'}};
 const ds=h.mergeTemporal([a,b],{detections:[old,{...b,motion:{state:'stationary'}}]});assert.equal(ds.length,2);assert.equal(ds[0],a);assert.equal(ds[1].motion.state,'stationary');assert.deepEqual(ds[1].box,b.box);
});
test('persisted orientation lookup follows exact fingerprints across detection reordering',()=>{
 const a={label:'car',box:[.1,.1,.2,.2]},b={label:'car',box:[.3,.3,.4,.4]},r={fingerprint:h.fingerprint(a),state:'estimated'};
 const row={facing:{byDetection:{0:r},byFingerprint:{[r.fingerprint]:r}}};assert.equal(h.facing(row,a,1),r);assert.equal(h.facing(row,b,0),null);
});
test('invalid coordinates and excessive spatial references are rejected',()=>{
 assert.throws(()=>h.sanitize([{start:[NaN,0],end:[1,1]}]));assert.throws(()=>h.sanitize([{start:[0,0],end:[0,0]}]));assert.throws(()=>h.sanitize(Array(65).fill({start:[0,0],end:[1,1]})));
});

test('historical headings cannot reuse a stale estimate after paired evidence becomes unresolved',()=>{
 const d={...car,orientation:{state:'unknown',paired:{state:'pending'}}},fp=h.fingerprint(d),row={headings:{version:h.VERSION,byFingerprint:{[fp]:{state:'estimated',direction:'right'}}}};
 const c=h.attach({_archive:true,hash:'a',detections:[d]},row,null);assert.equal(c.detections[0].heading.state,'unknown');assert.equal(c.detections[0].heading.direction,undefined);
});

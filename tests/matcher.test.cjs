const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const ctx=vm.createContext({});for(const f of ['temporal','routes','matcher'])vm.runInContext(fs.readFileSync(`web/${f}.js`,'utf8'),ctx);
const M=ctx.VehicleMatcher,R=ctx.VehicleRoutes,plain=v=>JSON.parse(JSON.stringify(v));
const T0=Date.UTC(2026,9,26,21,0,0);
// Southbound at ~80 km/h on lon -123: a fix every 3 s, 66.7 m apart, passing the camera at i≈16.7.
const drive=(session='drive-a',n=34,lat0=49.01)=>Array.from({length:n},(_,i)=>({session,t:T0+3000*i,lat:lat0-.0006*i,lon:-123,accuracy:5,altitude:null,altitudeAccuracy:null,speed:22,heading:180}));
const north={id:'100',name:'Site - N',lat:49,lon:-123,orientation:'N',source:'https://www.drivebc.ca/cameras/100'},south={...north,id:'101',name:'Site - S',orientation:'S'};
const red=[.5,0,0,0,0,0,0,0,.1,.1,.2,.1,.4],white=[0,0,0,0,0,0,0,0,0,0,.2,.8,1];
const car=(label,colour,facing,tier='high')=>({label,colour,facing,tier});
const row=(camera,captured,cars=[car('car',red,'toward_camera')],extra={})=>({key:camera.id+':h'+captured,cameraId:camera.id,hash:'h'+captured,observedAt:captured+20000,sourceModified:new Date(captured).toUTCString(),camera,
  detections:[...cars.map(c=>({label:c.label,box:[.4,.5,.5,.6]})),{label:'person',box:[.1,.1,.2,.3]}],
  facing:{byDetection:Object.fromEntries(cars.map((c,i)=>[i,c.facing?{state:'estimated',facing:c.facing,tier:c.tier}:null]))},routeFeatures:{version:R.VERSION,byDetection:Object.fromEntries(cars.map((c,i)=>[i,c.colour]))},...extra});

test('positions interpolate between close fixes, never across long gaps',()=>{
  const fixes=drive();
  const p=M.positionAt(fixes,T0+4500);assert.ok(Math.abs(p.lat-(49.01-.0009))<1e-9);assert.equal(Math.round(p.course),180);assert.ok(Math.abs(p.speed-22.2)<.1);assert.equal(p.session,'drive-a');
  const gap=[fixes[0],{...fixes[1],t:T0+60000}];assert.equal(M.positionAt(gap,T0+30000),null);
  assert.equal(M.positionAt(gap,T0+4000).lat,49.01,'within 5 s of the edge fix');assert.equal(M.positionAt(fixes,T0-6000),null);
  const stopped=[fixes[0],{...fixes[0],t:T0+3000,heading:null}];assert.equal(M.positionAt(stopped,T0+1500).course,180,'stationary pair keeps the reported heading');
});

test('expected facing follows travel direction relative to the camera look direction',()=>{
  assert.equal(M.expectedFacing(0,180),'toward_camera');assert.equal(M.expectedFacing(0,10),'away_from_camera');assert.equal(M.expectedFacing(0,90),'image_right');
  assert.equal(M.expectedFacing(90,0),'image_left');assert.equal(M.expectedFacing(null,0),null);assert.equal(M.expectedFacing(0,null),null);
});

test('the camera index finds exactly the cameras within its radius',()=>{
  const cams=[];for(let i=0;i<40;i++)for(let j=0;j<40;j++)cams.push({id:i+'-'+j,lat:48.9+i*.004,lon:-123.1+j*.006});
  const index=M.cameraIndex([...cams,{id:'no-location',lat:null,lon:1}],500);assert.equal(index.cameras.length,1600);
  let seed=3;const rand=()=>(seed=(seed*16807)%2147483647)/2147483647;
  for(let k=0;k<200;k++){
    const p={lat:48.89+rand()*.18,lon:-123.11+rand()*.26},brute=cams.filter(c=>R.distance(c,p)<=500).map(c=>c.id).sort();
    assert.deepEqual(plain(index.near(p).map(n=>n.camera.id).sort()),brute);
  }
  assert.deepEqual(plain(M.cameraIndex([]).near({lat:49,lon:-123})),[]);
});

test('only fixes near a camera are kept, with one boundary fix each side',()=>{
  const index=M.cameraIndex([north]),fixes=drive('d',60,49.02),filter=new M.RelevanceFilter(index);
  const kept=fixes.flatMap(f=>filter.push(f)),inside=fixes.filter(f=>R.distance(north,f)<=500);
  assert.equal(kept.length,inside.length+2);assert.ok(R.distance(north,kept[0])>500&&R.distance(north,kept.at(-1))>500,'boundary fixes sit just outside');
  assert.ok(kept.every(f=>R.distance(north,f)<500+67));assert.deepEqual(plain(kept.map(f=>f.t)),plain(kept.map(f=>f.t).sort((a,b)=>a-b)));
  assert.deepEqual(plain(M.compact([...drive('e',60,49.02)].reverse(),index).map(f=>f.t)),plain(kept.map(f=>f.t)),'compaction sorts and filters the same way');
  // A fix minutes before entering is not a boundary: it could be anywhere, such as home.
  const late=new M.RelevanceFilter(index),home={session:'d',t:T0-600000,lat:49.2,lon:-123};
  assert.deepEqual(plain(late.push(home)),[]);assert.deepEqual(plain(late.push({...home,t:T0,lat:49.001})).length,1);
  assert.deepEqual(plain(M.compact(drive('far',20,50),index)),[]);
});

test('camera passes list each in-view visit with closest approach and the 24-hour replay window',()=>{
  const fixes=drive(),loop=[...fixes,...drive('drive-a',34).map(f=>({...f,t:f.t+600000}))];
  const passes=M.passes([north,south,{...north,id:'far',lat:50}],fixes);
  assert.deepEqual(plain(passes.map(p=>[p.cameraId,p.fixes,p.expectedFacing])),[['100',5,'toward_camera'],['101',4,'away_from_camera']]);
  const p=passes[0];assert.equal(p.start,T0+39000);assert.equal(p.end,T0+51000);assert.equal(p.closest.t,T0+51000);assert.equal(Math.round(p.closest.distance),22);assert.equal(Math.round(p.closest.course),180);
  assert.deepEqual(plain(p.replay),{page:'https://www.drivebc.ca/cameras/100',list:'https://www.drivebc.ca/api/webcams/100/replayTheDay/',availableUntil:T0+39000+86400000});
  assert.equal(M.passes([north],loop).length,2,'a second pass ten minutes later is a separate visit');
  assert.deepEqual(plain(M.passes([north],[])),[]);
  const coarse=M.passes([north],fixes.map(f=>({...f,accuracy:80})));assert.deepEqual(plain(coarse),[]);
});

test('frames match by capture time; fetch-time-only frames are uncertain; the rest are near misses',()=>{
  const fixes=drive(),passes=M.passes([north,south],fixes);
  const rows=[row(north,T0+46000),row(south,T0+46000),row(north,T0+60000),row(south,T0+60000),row(north,T0+46000+600000),
    {...row(north,T0+45000),key:'fetch-only',sourceModified:null,observedAt:T0+47000},{...row(north,T0+20000),key:'fetch-early',sourceModified:null,observedAt:T0-200000}];
  const [atNorth,atSouth]=M.matchFrames(passes,rows,fixes);
  assert.deepEqual(plain(atNorth.map(f=>[f.key,f.kind])),[['100:h'+(T0+46000),'in_view'],['fetch-only','uncertain'],['100:h'+(T0+60000),'missed']]);
  const seen=atNorth[0];assert.equal(seen.exact,true);assert.equal(seen.expectedFacing,'toward_camera');assert.equal(Math.round(seen.distance),89);assert.equal(seen.offAxis,0);assert.equal(seen.offset,-5000);
  assert.equal(atNorth[1].exact,false);assert.equal(atNorth[1].expectedFacing,'toward_camera');
  assert.deepEqual(plain(atSouth.map(f=>[f.key,f.kind])),[['101:h'+(T0+46000),'missed'],['101:h'+(T0+60000),'in_view']]);
  assert.equal(atSouth[1].expectedFacing,'away_from_camera');assert.equal(Math.round(atSouth[1].distance),222);
  const unknownLook=M.passes([{...north,orientation:'',caption:'Site'}],fixes);assert.equal(unknownLook[0].expectedFacing,null);
  assert.equal(M.matchFrames(unknownLook,[row(north,T0+60000)],fixes)[0][0].inView,null,'no look direction: radius only');
});

test('verification scores body class, colour and facing, and needs a clear margin to confirm',()=>{
  const frame={key:'f',kind:'in_view',expectedFacing:'toward_camera'},vehicle={id:'v',bodyType:'sedan',colour:'red'};
  const r=row(north,T0,[car('car',white,'toward_camera'),car('car',red,'toward_camera'),car('truck',red,'toward_camera')]);
  const v=M.verify(frame,r,{vehicle});
  assert.deepEqual(plain(v.detections.map(d=>d.index)),[1,2,0],'people are never candidates');
  assert.deepEqual(plain(v.best.cues),{body:1,colour:.8,facing:1});assert.ok(Math.abs(v.best.score-.9)<1e-9);assert.equal(v.status,'confirmed');
  assert.equal(M.verify({...frame,kind:'uncertain'},r,{vehicle}).status,'possible','without a capture time a match is never confirmed');
  const twins=M.verify(frame,row(north,T0,[car('car',red,'toward_camera'),car('car',red,'toward_camera')]),{vehicle});
  assert.equal(twins.margin,0);assert.equal(twins.status,'possible','two lookalikes in one frame');
  const wrongWay=M.verify(frame,row(north,T0,[car('car',red,'away_from_camera')]),{vehicle});assert.equal(wrongWay.best.cues.facing,0);
  const lowTier=M.verify(frame,row(north,T0,[car('car',red,'away_from_camera','low')]),{vehicle});assert.equal(lowTier.best.cues.facing,undefined,'low-tier facing is ignored');
  const facingOnly=M.verify(frame,row(north,T0,[car('car',red,'toward_camera'),car('car',red,'away_from_camera')]),{vehicle:{id:'blank'}});
  assert.equal(facingOnly.confidence,1);assert.equal(facingOnly.status,'possible','facing alone never confirms');
  assert.equal(M.verify({...frame,expectedFacing:null},r,{vehicle:{id:'blank'}}).status,'unverified','an empty profile cannot verify on its own');
  assert.equal(M.verify(frame,{detections:[{label:'person',box:[0,0,.1,.1]}]},{vehicle}).status,'empty');
  assert.equal(M.verify(frame,{key:'x'},{vehicle}).status,'pending','detection has not run yet');
  // References replace the named colour; a label is never used to score its own frame.
  const refs=M.references({id:'v',references:[]},[{key:'v|other',vehicle:'v',frame:'other',detection:0,descriptor:white},{key:'v|f',vehicle:'v',frame:'f',detection:1,descriptor:red},{key:'w|f',vehicle:'w',frame:'f',detection:0,descriptor:red}]);
  assert.equal(refs.length,2);const byRef=M.verify(frame,r,{vehicle:{id:'v'},references:refs});
  assert.equal(byRef.best.index,0,'only the white reference from another frame counts');assert.equal(byRef.best.cues.colour,1);
  const extra=M.verify(frame,r,{vehicle,cues:{plate:d=>d.index===0?1:0},options:{weights:{plate:5}}});assert.equal(extra.best.index,0,'extra cues plug in');
  // A red just under 360° lands in the pink bin; it still counts toward a red profile, and wraps.
  const pinkish=[0,0,0,0,0,0,0,.5,.1,.1,.2,.1,.4],orange=[0,.5,0,0,0,0,0,0,.1,.1,.2,.1,.4],green=[0,0,0,.5,0,0,0,0,.1,.1,.2,.1,.4];
  const named=colour=>M.verify(frame,row(north,T0,[car('car',colour,null)]),{vehicle:{id:'v',colour:'red'}}).best.cues.colour;
  assert.deepEqual([named(red),named(pinkish),named(orange),named(green),named(white)],[.8,.5,.5,0,0]);
});

test('the full track: sightings in time order, labels override, decoy rate and leave-one-out agreement',()=>{
  const fixes=drive(),vehicle={id:'v1',name:'Test',bodyType:'sedan',colour:'red'};
  const seen=row(north,T0+46000),later=row(south,T0+60000,[car('car',white,'away_from_camera')]);
  const strangers=[row(north,T0+3600000),row(north,T0+7200000,[car('car',white,'toward_camera')]),row(north,T0-3600000,[])];
  const labels=[{key:'v1|'+seen.key,vehicle:'v1',frame:seen.key,detection:0,descriptor:red},{key:'v1|'+later.key,vehicle:'v1',frame:later.key,detection:null},{key:'other|'+seen.key,vehicle:'other',frame:seen.key,detection:null}];
  const result=M.track({vehicle,fixes,cameras:[north,south],rows:[seen,later,...strangers],labels});
  assert.deepEqual(plain(result.sightings.map(s=>[s.key,s.status,s.verification.status,s.detection])),[[later.key,'not_visible','possible',null],[seen.key,'labelled','confirmed',0]]);
  assert.deepEqual(plain(result.passes.map(p=>[p.cameraId,p.status,p.frames.length])),[['100','labelled',1],['101','not_visible',1]]);
  assert.deepEqual(plain(result.counts),{not_visible:1,labelled:1});assert.equal(result.references,1);
  assert.deepEqual(plain(result.decoy),{frames:2,confirmed:1,possible:1,rate:.5},'the red stranger an hour later looks just like the vehicle');
  assert.deepEqual(plain(result.agreement),{labelled:1,agree:1,hidden:1,falseConfirmed:0});
  const none=M.track({vehicle,fixes:[],cameras:[north],rows:[seen]});assert.deepEqual(plain(none.sightings),[]);assert.equal(none.decoy.rate,null);
});

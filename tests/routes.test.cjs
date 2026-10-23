const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
const context=vm.createContext({});vm.runInContext(fs.readFileSync('web/temporal.js','utf8'),context);vm.runInContext(fs.readFileSync('web/routes.js','utf8'),context);const R=context.VehicleRoutes;
const red=[.5,0,0,0,0,0,0,0,.1,.1,.2,.1,.4],white=[0,0,0,0,0,0,0,0,0,0,.2,.8,1],blue=[0,0,0,0,0,.5,0,0,.1,.2,.1,.1,.5];
// Three cameras ~5 km apart heading east along one road.
const cams=[{id:'a',lat:49,lon:-122.6,orientation:'E'},{id:'b',lat:49,lon:-122.53,orientation:'E'},{id:'c',lat:49,lon:-122.46,orientation:'W'}];
const row=(cam,t,dets)=>({key:cam.id+':'+t,hash:cam.id+t,cameraId:cam.id,observedAt:t,camera:cam,detections:dets.map(d=>({label:d.label||'car',box:[.4,.4,.5,.5]})),
  facing:{byDetection:Object.fromEntries(dets.map((d,i)=>[i,{state:'estimated',facing:d.facing}]))},routeFeatures:{byDetection:Object.fromEntries(dets.map((d,i)=>[i,d.colour]))}});

test('camera look direction comes from catalog orientation or caption',()=>{
  assert.equal(R.lookBearing({orientation:'NE'}),45);assert.equal(R.lookBearing({caption:'Highway 1 at Main St, looking west.'}),270);
  assert.equal(R.lookBearing({caption:'Hwy 5 looking south-east'}),135);assert.equal(R.lookBearing({caption:'No direction'}),null);
});
test('facing converts to compass travel heading relative to the camera view',()=>{
  const cam={orientation:'N'};
  assert.equal(R.heading(cam,{state:'estimated',facing:'away_from_camera'}),0);assert.equal(R.heading(cam,{state:'estimated',facing:'toward_camera'}),180);
  assert.equal(R.heading(cam,{state:'estimated',facing:'image_right'}),90);assert.equal(R.heading(cam,{state:'unknown'}),null);
});
test('a distinctive vehicle is linked across cameras at plausible highway speed',()=>{
  // 5.1 km per hop; 4 minutes ≈ 77 km/h.
  const rows=[row(cams[0],0,[{facing:'away_from_camera',colour:red},{facing:'toward_camera',colour:white}]),
    row(cams[1],240000,[{facing:'away_from_camera',colour:red}]),row(cams[2],480000,[{facing:'toward_camera',colour:red}])];
  const found=R.routes(R.sightings(rows));
  assert.equal(found.length,1);assert.equal(found[0].hops.length,3);assert.equal(found[0].heading,'E');assert.equal(found[0].colour,'red');
  assert.ok(found[0].speed*3.6>60&&found[0].speed*3.6<95);
});
test('implausible speed, wrong direction, colour mismatch and ambiguous candidates abstain',()=>{
  const s=rows=>R.routes(R.sightings(rows)).length;
  assert.equal(s([row(cams[0],0,[{facing:'away_from_camera',colour:red}]),row(cams[1],20000,[{facing:'away_from_camera',colour:red}])]),0);
  assert.equal(s([row(cams[0],0,[{facing:'toward_camera',colour:red}]),row(cams[1],240000,[{facing:'away_from_camera',colour:red}])]),0);
  assert.equal(s([row(cams[0],0,[{facing:'away_from_camera',colour:red}]),row(cams[1],240000,[{facing:'away_from_camera',colour:blue}])]),0);
  assert.equal(s([row(cams[0],0,[{facing:'away_from_camera',colour:white}]),row(cams[1],240000,[{facing:'away_from_camera',colour:white},{facing:'away_from_camera',colour:white}])]),0);
});
test('flows summarise compass headings per camera and colours are named',()=>{
  const f=R.flows(R.sightings([row(cams[0],0,[{facing:'away_from_camera',colour:red},{facing:'away_from_camera',colour:white},{facing:'toward_camera',colour:white}])]));
  assert.equal(f[0].headed,3);assert.equal(f[0].dominant.direction,'E');assert.equal(f[0].secondary.direction,'W');
  assert.equal(R.colourName(white),'white');assert.equal(R.colourName(red),'red');
  const px=new Uint8ClampedArray([250,250,250,255,10,10,10,255]);assert.equal(R.describePixels(px).length,13);
});
test('low-confidence facing gives no heading, so it cannot steer a route',()=>{
  const r=row(cams[0],0,[{facing:'away_from_camera',colour:red}]);r.facing.byDetection[0].tier='low';
  assert.equal(R.sightings([r])[0].heading,null);r.facing.byDetection[0].tier='medium';assert.equal(R.sightings([r])[0].heading,90);
});
test('decoy clock shifts estimate chance links and flow series bins headings through time',()=>{
  const rows=[row(cams[0],0,[{facing:'away_from_camera',colour:red}]),row(cams[1],240000,[{facing:'away_from_camera',colour:red}]),row(cams[2],480000,[{facing:'toward_camera',colour:red}])];
  const list=R.sightings(rows);assert.equal(R.routes(list).length,1);const c=R.chance(list);assert.ok(c>=0&&c<=1,'chance is the mean decoy route count');
  const series=R.flowSeries(list,0,600000,3);assert.equal(series.length,3);assert.equal(series.reduce((n,b)=>n+b.counts.E,0),3);assert.equal(series[0].vehicles,1);
});
test('same-camera tracks chain a vehicle across consecutive frames; mirrored decoys and changed views do not',()=>{
  const cam={id:'q',lat:49,lon:-123,orientation:'N'},patch=Array.from({length:432},(_,i)=>(i%36<18?.9:.1)),other=Array.from({length:432},(_,i)=>i%2?.2:.6);
  const frame=(n,x,hash=String(n),background=Array(50).fill(.3))=>({key:'q:'+hash,hash,cameraId:'q',observedAt:n*120000,camera:cam,
    facing:{byFingerprint:{[JSON.stringify(['car',[x,.5,x+.1,.6]])]:{state:'estimated',facing:'away_from_camera',tier:'high',fingerprint:JSON.stringify(['car',[x,.5,x+.1,.6]])}}},
    features:{version:1,hash,pixelSignature:hash,background,detections:[{label:'car',box:[x,.5,x+.1,.6],appearance:patch},{label:'car',box:[.8,.1,.9,.2],appearance:other}]}});
  const rows=[frame(0,.2),frame(1,.25),frame(2,.3)],found=R.tracks(rows);
  const moving=found.find(t=>t.state==='moving');assert.equal(moving.hops.length,3);assert.equal(moving.imageDirection,'→');assert.equal(moving.heading,'N');assert.equal(moving.seconds,240);
  assert.equal(found.find(t=>t.state==='stationary').hops.length,3,'the parked car persists too');
  assert.equal(R.tracks(rows,{decoy:true}).filter(t=>t.state==='moving').length,0,'an asymmetric vehicle does not match its mirror image');
  assert.equal(R.tracks([frame(0,.2),frame(1,.25,'1',Array(50).fill(.9))]).length,0,'a changed camera view breaks tracks');
  assert.equal(R.tracks([frame(0,.2),frame(40,.25)]).length,0,'gaps over 30 minutes break tracks');
});
test('same-site handoffs link one vehicle between views captured seconds apart, and journeys join them with tracks',()=>{
  // Two views from one pole: looking north and looking south.
  const north={id:'n',lat:49.2,lon:-123.1,orientation:'N'},south={id:'s',lat:49.2,lon:-123.1,orientation:'S'};
  const view=(cam,t,facing,colour,box=[.4,.6,.5,.7])=>({key:cam.id+':'+t,hash:cam.id+t,cameraId:cam.id,observedAt:t,camera:cam,detections:[{label:'car',box}],
    facing:{byDetection:{0:{state:'estimated',facing,tier:'high'}}},routeFeatures:{byDetection:{0:colour}}});
  // Southbound red car: toward the north-looking camera, then away from the south-looking one 4 s later.
  const rows=[view(north,1000,'toward_camera',red),view(south,5000,'away_from_camera',red),view(south,5000,'away_from_camera',blue,[.1,.6,.2,.7])];
  const list=R.sightings(rows),h=R.handoffs(list);
  assert.equal(h.length,1);assert.equal(h[0].a.cameraId,'n');assert.equal(h[0].b.cameraId,'s');assert.equal(h[0].seconds,4);
  assert.equal(R.handoffs(R.sightings([view(north,1000,'toward_camera',red),view(south,5000,'toward_camera',red)])).length,0,'opposite headings never hand off');
  assert.equal(R.handoffs(R.sightings([view(north,1000,'toward_camera',red),view(south,61000,'away_from_camera',red)])).length,0,'captures a minute apart are too far apart');
  const journeys=R.journeys(list,[],h);assert.equal(journeys.length,1);assert.deepEqual([...journeys[0].cameras],['n','s']);assert.equal(journeys[0].heading,'S');
  assert.ok(R.handoffChance(list)<=1);
});
test('a route never loops back to a camera through handoffs',()=>{
  const cams=['s','w','n'].map(id=>({id,lat:49.1,lon:-122.5,orientation:id.toUpperCase()}));
  const mk=(cam,t)=>({id:cam.id+t,key:cam.id+t,cameraId:cam.id,camera:cam,t,label:'car',heading:180,colour:red,colourName:'red'});
  const a=mk(cams[0],0),b=mk(cams[1],4000),c=mk(cams[2],8000),d=mk(cams[0],12000);
  const j=R.journeys([a,b,c,d],[],[{a,b},{a:b,b:c},{a:c,b:d}]);
  assert.equal(j.length,1);assert.equal(JSON.stringify(j[0].hops.map(h=>h.cameraId)),'["s","w","n"]','the return to camera s is cut');
});

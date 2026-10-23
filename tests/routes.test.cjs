const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
const context=vm.createContext({});vm.runInContext(fs.readFileSync('web/routes.js','utf8'),context);const R=context.VehicleRoutes;
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

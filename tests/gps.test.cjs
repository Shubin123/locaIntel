const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {IDBFactory,IDBKeyRange}=require('fake-indexeddb');
function scope(){const ctx=vm.createContext({indexedDB:new IDBFactory(),IDBKeyRange});for(const f of ['temporal','routes','gps'])vm.runInContext(fs.readFileSync(`web/${f}.js`,'utf8'),ctx);return ctx.GpsLog;}
const plain=v=>JSON.parse(JSON.stringify(v));
const T0=Date.UTC(2026,9,26,21,0,0);
// Southbound at ~80 km/h on lon -123: a fix every 3 s, 66.7 m apart, passing the camera at i≈16.7.
const drive=(session='drive-a',n=34)=>Array.from({length:n},(_,i)=>({session,t:T0+3000*i,lat:49.01-.0006*i,lon:-123,accuracy:5,altitude:null,altitudeAccuracy:null,speed:22,heading:180}));
const north={id:'100',name:'Site - N',lat:49,lon:-123,orientation:'N',source:'https://www.drivebc.ca/cameras/100'},south={...north,id:'101',name:'Site - S',orientation:'S'};
const row=(camera,captured,extra={})=>({key:camera.id+':h'+captured,cameraId:camera.id,hash:'h'+captured,observedAt:captured+20000,sourceModified:new Date(captured).toUTCString(),camera,detections:[{label:'car',box:[.4,.5,.5,.6]},{label:'person',box:[.1,.1,.2,.3]}],...extra});

function geolocation(){
  const g={watchers:new Map(),n:0,current:[],cleared:[],
    watchPosition(ok,error,options){g.options=options;g.watchers.set(++g.n,{ok,error});return g.n;},
    clearWatch(id){g.cleared.push(id);g.watchers.delete(id);},
    getCurrentPosition(ok,error,options){g.current.push(options);},
    emit(p){for(const w of g.watchers.values())w.ok(p);},fail(e){return Promise.all([...g.watchers.values()].map(w=>w.error(e)));}};
  return g;
}
const position=(t,lat=49,lon=-123,coords={})=>({timestamp:t,coords:{latitude:lat,longitude:lon,accuracy:5,altitude:null,altitudeAccuracy:null,speed:20,heading:NaN,...coords}});
function wakeLock({fail=false}={}){
  const w={requests:0,sentinels:[],async request(type){w.requests++;w.type=type;await Promise.resolve();if(fail)throw Error('NotAllowedError');const listeners=[],s={released:false,addEventListener:(_,f)=>listeners.push(f),async release(){if(s.released)return;s.released=true;listeners.forEach(f=>f());}};w.sentinels.push(s);return s;}};
  return w;
}
function page(){const d={visibilityState:'visible',listeners:new Set(),addEventListener:(_,f)=>d.listeners.add(f),removeEventListener:(_,f)=>d.listeners.delete(f),show(state){d.visibilityState=state;d.listeners.forEach(f=>f());}};return d;}
function logger(G,options={}){
  let now=T0;const timers=new Map();let serial=0;
  const store=new G.GpsStore(),geo=geolocation(),lock=options.wakeLock||wakeLock(),doc=page();
  const log=new G.GpsLogger({store,geolocation:geo,wakeLock:lock,document:doc,intervalMs:3000,now:()=>now,setTimer:(fn,ms)=>{timers.set(++serial,{fn,ms});return serial;},clearTimer:id=>timers.delete(id),...options});
  return {log,store,geo,lock,doc,timers,tick:ms=>{now+=ms;for(const t of timers.values())t.fn();},get now(){return now;}};
}

test('nothing is logged before an explicit start; start asks for high-accuracy fresh fixes and a screen lock',async()=>{
  const G=scope(),{log,geo,lock,store}=logger(G);
  assert.equal(geo.watchers.size,0);assert.equal(log.position(position(T0)),null);assert.equal(lock.requests,0);
  const session=await log.start();
  assert.match(session.id,/^drive-20261026T210000Z$/);assert.equal(plain(geo.options).enableHighAccuracy,true);assert.equal(geo.options.maximumAge,0);
  assert.equal(lock.type,'screen');assert.equal(log.state().wakeLock,'held');
  assert.equal((await store.sessions())[0].end,null);
  assert.equal(await log.start(),session,'a second start reuses the running session');assert.equal(geo.watchers.size,1);
});

test('fixes are stored every few seconds with timestamps; stop clears the watch, releases the lock and ends the session',async()=>{
  const G=scope(),{log,geo,lock,store,tick}=logger(G);await log.start();
  for(const dt of [0,1000,2000,3000,3500,6100])geo.emit(position(T0+dt,49+dt*1e-7,-123,dt?{heading:180}:{}));
  geo.emit({timestamp:T0+9000,coords:{latitude:NaN,longitude:-123}});
  geo.emit(position(T0+2000));// a stale cached fix arriving late
  const fixes=await store.fixes(log.session.id);
  assert.deepEqual(fixes.map(f=>f.t-T0),[0,3000,6100]);assert.equal(fixes[0].heading,null,'NaN heading while stationary is stored as unknown');assert.equal(fixes[1].heading,180);
  tick(5000);const id=log.session.id,ended=await log.stop();
  assert.equal(geo.watchers.size,0);assert.equal(lock.sentinels[0].released,true);assert.equal(log.state().wakeLock,'idle');
  assert.equal(ended.end,T0+5000);assert.equal(ended.fixes,3);assert.equal(ended.last,T0+6100);
  geo.emit(position(T0+20000));assert.equal((await store.fixes(id)).length,3,'no fixes after stop');
  assert.equal(await log.stop(),null);
});

test('the wake lock is requested again when the page returns, never twice at once, and a refusal does not stop logging',async()=>{
  const G=scope(),{log,lock,doc,geo,store}=logger(G);await log.start();
  doc.show('hidden');await lock.sentinels[0].release();assert.equal(log.state().wakeLock,'released');
  doc.show('visible');doc.show('visible');await log.locking;
  assert.equal(lock.requests,2);assert.equal(log.state().wakeLock,'held');
  await log.stop();doc.show('visible');assert.equal(lock.requests,2,'no lock after stop');assert(lock.sentinels.every(s=>s.released));
  const refused=logger(scope(),{wakeLock:wakeLock({fail:true})});await refused.log.start();
  assert.match(refused.log.state().wakeLock,/^error: NotAllowedError/);refused.geo.emit(position(T0));
  assert.equal((await refused.store.fixes()).length,1);
  const none=logger(scope(),{wakeLock:undefined});await none.log.start();assert.equal(none.log.state().wakeLock,'unsupported');
});

test('a quiet watch is topped up with a direct fix request; permission denial ends the session',async()=>{
  const G=scope(),{log,geo,tick,store}=logger(G);await log.start();
  tick(3000);assert.equal(geo.current.length,0);tick(3000);assert.equal(geo.current.length,1);assert.equal(geo.current[0].enableHighAccuracy,true);
  geo.fail({code:3,message:'timeout'});assert.equal(log.active,true);assert.match(log.state().error,/Waiting/);
  const id=log.session.id;await geo.fail({code:1,message:'denied'});
  assert.equal(log.active,false);assert.equal(log.state().error,'Location permission denied');assert.notEqual((await store.sessions()).find(s=>s.id===id).end,null);
});

test('sessions persist, interrupted ones close at their last fix, imports deduplicate and deleting a drive removes its fixes and labels',async()=>{
  const G=scope(),store=new G.GpsStore();
  await store.begin({id:'open',start:T0,end:null,last:null,fixes:0});for(const f of drive('open',3))await store.add(f);await store.add(drive('open',1)[0]);
  const again=new G.GpsStore();assert.deepEqual(plain(await again.recover('other')),['open']);
  const [s]=await again.sessions();assert.equal(s.fixes,3);assert.equal(s.end,T0+6000);assert.equal(s.interrupted,true);
  assert.deepEqual(plain(await again.importFixes(drive('phone',5),{name:'phone.csv'})),['phone']);await again.importFixes(drive('phone',5));
  const phone=(await again.sessions()).find(s=>s.id==='phone');assert.equal(phone.fixes,5);assert.equal(phone.imported,true);assert.equal(phone.name,'phone.csv');
  await again.saveLabel({key:'a',session:'phone'});await again.saveLabel({key:'b',session:'open'});
  await again.remove('phone');
  assert.deepEqual(plain((await again.sessions()).map(s=>s.id)),['open']);assert.equal((await again.fixes()).length,3);assert.deepEqual(plain((await again.labels()).map(l=>l.key)),['b']);
});

test('GPX and CSV exports carry every timestamped fix and import back unchanged',()=>{
  const G=scope(),fixes=[...drive('drive-a',3),{...drive('drive-a',1)[0],t:T0+120000,altitude:12.5,heading:null}];
  const gpx=G.gpx(fixes,{name:'Drive <1> & "two"'});
  assert.match(gpx,/<gpx version="1.1" creator="LocaIntel" xmlns="http:\/\/www.topografix.com\/GPX\/1\/1"/);assert.match(gpx,/<name>Drive &lt;1&gt; &amp; &quot;two&quot;<\/name>/);
  assert.equal((gpx.match(/<trkseg>/g)||[]).length,2,'a 2-minute gap starts a new segment');assert.match(gpx,/<trkpt lat="49.01" lon="-123"><time>2026-10-26T21:00:00.000Z<\/time><extensions><li:accuracy>5<\/li:accuracy><li:speed>22<\/li:speed><li:heading>180<\/li:heading><\/extensions><\/trkpt>/);
  assert.match(gpx,/<ele>12.5<\/ele><time>2026-10-26T21:02:00.000Z<\/time>/);
  assert.deepEqual(plain(G.parseTrack(gpx,{session:'drive-a'})),plain(fixes));
  const csv=G.csv(fixes);assert.equal(csv.split('\n')[0],'session,time,unix_ms,lat,lon,accuracy_m,altitude_m,altitude_accuracy_m,speed_mps,heading_deg');
  assert.equal(csv.split('\n')[1],`drive-a,2026-10-26T21:00:00.000Z,${T0},49.01,-123,5,,,22,180`);
  assert.deepEqual(plain(G.parseTrack(csv,{session:'ignored'})),plain(fixes),'CSV keeps its own session ids');
  const risky=G.csv([{...fixes[0],session:'=HYPERLINK("x")'}]);assert.match(risky,/\n"'=HYPERLINK\(""x""\)",/);assert.equal(G.parseCSV(risky,'s')[0].session,'=HYPERLINK("x")');
});

test('third-party GPX 1.0 and CSV tracks import; untimed points are skipped',()=>{
  const G=scope();
  const gpx=`﻿<?xml version="1.0"?><gpx version="1.0"><trk><trkseg><trkpt lat='49.1' lon='-122.9'><ele>3</ele><time>2026-10-26T21:00:00Z</time><speed>12.5</speed><course>270</course></trkpt><trkpt lat="49.2" lon="-122.9"/><trkpt lat="91" lon="0"><time>2026-10-26T21:00:05Z</time></trkpt></trkseg></trk></gpx>`;
  assert.deepEqual(plain(G.parseTrack(gpx,{name:'watch.gpx'})),[{session:'import-watch.gpx',t:T0,lat:49.1,lon:-122.9,accuracy:null,altitude:3,altitudeAccuracy:null,speed:12.5,heading:270}]);
  const csv='Timestamp,Latitude,Longitude,Speed\r\n2026-10-26T21:00:00Z,49.1,-122.9,10\r\n,49,-123,\r\n"2026-10-26T21:00:03Z","49.2","-122.8",""\r\n';
  assert.deepEqual(plain(G.parseTrack(csv,{name:'x.csv'})).map(f=>[f.t-T0,f.lat,f.speed]),[[0,49.1,10],[3000,49.2,null]]);
  assert.throws(()=>G.parseTrack('a,b\n1,2',{name:'bad.csv'}),/lat, lon and time/);assert.throws(()=>G.parseTrack('<gpx></gpx>',{name:'empty.gpx'}),/No timestamped positions/);
});

test('positions interpolate between close fixes, never across long gaps or coarse fixes',()=>{
  const G=scope(),fixes=drive();
  const p=G.positionAt(fixes,T0+4500);assert.ok(Math.abs(p.lat-(49.01-.0009))<1e-9);assert.equal(Math.round(p.course),180);assert.ok(Math.abs(p.speed-22.2)<.1);assert.equal(p.session,'drive-a');
  const gap=[fixes[0],{...fixes[1],t:T0+60000}];assert.equal(G.positionAt(gap,T0+30000),null);
  assert.equal(G.positionAt(gap,T0+4000).lat,49.01,'within 5 s of the edge fix');assert.equal(G.positionAt(fixes,T0-6000),null);
  const stopped=[fixes[0],{...fixes[0],t:T0+3000,heading:null}];assert.equal(G.positionAt(stopped,T0+1500).course,180,'stationary pair keeps the reported heading');
  const coarse=G.frames([row(north,T0+46000)],fixes.map(f=>({...f,accuracy:80})));assert.deepEqual(plain(coarse),[]);
});

test('expected facing follows travel direction relative to the camera look direction',()=>{
  const G=scope();
  assert.equal(G.expectedFacing(0,180),'toward_camera');assert.equal(G.expectedFacing(0,10),'away_from_camera');assert.equal(G.expectedFacing(0,90),'image_right');
  assert.equal(G.expectedFacing(90,0),'image_left');assert.equal(G.expectedFacing(null,0),null);assert.equal(G.expectedFacing(0,null),null);
});

test('saved frames match only when the capture time puts the car in front of the camera',()=>{
  const G=scope(),fixes=drive(),far={...north,id:'200',lat:49.05};
  const rows=[row(north,T0+46000),row(south,T0+46000),row(north,T0+60000),row(south,T0+60000),row(far,T0+46000),row(north,T0+46000+600000),
    {...row(north,T0+45000),key:'fetch-only',sourceModified:null,observedAt:T0+46000},{...row(north,T0+101000),key:'implausible',observedAt:T0+40000},{...row({...north,lat:null},T0+46000),key:'no-location'}];
  const found=G.frames(rows,fixes);
  assert.deepEqual(plain(found.map(m=>m.key).sort()),['100:h'+(T0+46000),'101:h'+(T0+60000)]);
  const approaching=found.find(m=>m.cameraId==='100');assert.equal(approaching.expectedFacing,'toward_camera');assert.equal(Math.round(approaching.distance),89);assert.equal(approaching.offAxis,0);
  const leaving=found.find(m=>m.cameraId==='101');assert.equal(leaving.expectedFacing,'away_from_camera');assert.equal(Math.round(leaving.distance),222);
  const unknownLook=G.frames([row({...north,orientation:'',caption:'Site'},T0+60000)],fixes);assert.equal(unknownLook.length,1,'no look direction: radius only');assert.equal(unknownLook[0].inView,null);assert.equal(unknownLook[0].expectedFacing,null);
  assert.equal(G.frames(rows,fixes,{radius:80}).length,0);
});

test('camera passes list each in-view visit with closest approach and the 24-hour replay window',()=>{
  const G=scope(),fixes=drive(),loop=[...fixes,...drive('drive-a',34).map(f=>({...f,t:f.t+600000}))];
  const passes=G.passes([north,south,{...north,id:'far',lat:50}],fixes);
  assert.deepEqual(plain(passes.map(p=>[p.cameraId,p.fixes,p.expectedFacing])),[['100',5,'toward_camera'],['101',4,'away_from_camera']]);
  const p=passes[0];assert.equal(p.start,T0+39000);assert.equal(p.end,T0+51000);assert.equal(p.closest.t,T0+51000);assert.equal(Math.round(p.closest.distance),22);assert.equal(Math.round(p.closest.course),180);
  assert.deepEqual(plain(p.replay),{page:'https://www.drivebc.ca/cameras/100',list:'https://www.drivebc.ca/api/webcams/100/replayTheDay/',availableUntil:T0+39000+86400000});
  assert.equal(G.passes([north],loop).length,2,'a second pass ten minutes later is a separate visit');
  assert.deepEqual(plain(G.passes([north],[])),[]);
});

test('own-car labels join tracker sightings and the evaluation export separates labeled, unlabeled and replay-only passes',()=>{
  const G=scope(),fixes=drive(),frame={...row(north,T0+46000),facing:{byDetection:{0:{facing:'toward_camera'}}}},other=row(south,T0+60000);
  const [m]=G.frames([frame],fixes),label=G.labelRecord(m,frame,0,T0+3600000);
  assert.equal(label.sighting,frame.key+':0');assert.deepEqual(plain(label.box),[.4,.5,.5,.6]);assert.equal(label.modelFacing,'toward_camera');assert.equal(label.expectedFacing,'toward_camera');assert.equal(label.session,'drive-a');
  assert.throws(()=>G.labelRecord(m,frame,1),/Not a vehicle/);
  const hidden=G.labelRecord(m,frame,null,T0);assert.equal(hidden.detection,null);assert.equal(hidden.sighting,null);
  const set=plain(G.evaluationSet({sessions:[{id:'drive-a',start:T0,end:T0+100000,fixes:34}],labels:[label],frames:G.frames([frame,other],fixes),passes:G.passes([north,south],fixes),generated:T0}));
  assert.equal(set.format,'locaintel-own-vehicle-v1');assert.equal(set.labels.length,1);assert.equal(set.labels[0].capturedAt,'2026-10-26T21:00:46.000Z');
  assert.deepEqual(set.unlabeled.map(u=>u.key),[other.key]);assert.equal(set.passes.length,2);assert.equal(set.passes[0].replay.availableUntil,'2026-10-27T21:00:39.000Z');
  assert.equal(set.passes[0].closest.time,'2026-10-26T21:00:51.000Z');
});

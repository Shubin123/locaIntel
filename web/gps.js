/* Own-vehicle GPS log for a tracking evaluation set. Logging runs only between an
   explicit Start and Stop on this device; fixes stay in IndexedDB until exported.
   Fixes matched to saved camera frames (by upstream capture time) say which frames
   should contain your car, so you can label it. */
(function(root){
  const {lookBearing,distance,bearing,captured}=root.VehicleRoutes;
  const DAY=86400000,VEHICLES=['car','truck','bus','motorcycle'];
  const wrap=a=>((a%360)+360)%360,diff=(a,b)=>Math.abs(((a-b+540)%360)-180);
  const finite=v=>v==null||v===''?null:Number.isFinite(Number(v))?Number(v):null;
  const iso=t=>new Date(t).toISOString();
  const request=r=>new Promise((resolve,reject)=>{r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
  const complete=tx=>new Promise((resolve,reject)=>{tx.oncomplete=resolve;tx.onabort=tx.onerror=()=>reject(tx.error||Error('GPS storage transaction failed'));});

  function validFix(fix){return fix&&Number.isFinite(fix.t)&&Number.isFinite(fix.lat)&&Number.isFinite(fix.lon)&&Math.abs(fix.lat)<=90&&Math.abs(fix.lon)<=180?fix:null;}
  // A GeolocationPosition as a plain record. Heading is NaN while stationary.
  function fromPosition(position,session){
    const c=position?.coords||{};
    return validFix({session,t:Math.round(finite(position?.timestamp)),lat:finite(c.latitude),lon:finite(c.longitude),accuracy:finite(c.accuracy),altitude:finite(c.altitude),altitudeAccuracy:finite(c.altitudeAccuracy),speed:finite(c.speed),heading:finite(c.heading)});
  }

  class GpsStore{
    constructor({indexedDB=root.indexedDB,IDBKeyRange=root.IDBKeyRange,name='locaintel-gps-v1'}={}){Object.assign(this,{indexedDB,IDBKeyRange,name});this.chain=Promise.resolve();this.dbPromise=null;}
    open(){
      if(this.dbPromise)return this.dbPromise;
      this.dbPromise=new Promise((resolve,reject)=>{
        if(!this.indexedDB){reject(Error('Browser storage is unavailable'));return;}
        const r=this.indexedDB.open(this.name,1);
        r.onupgradeneeded=()=>{const db=r.result;db.createObjectStore('sessions',{keyPath:'id'});db.createObjectStore('fixes',{keyPath:['session','t']});db.createObjectStore('labels',{keyPath:'key'});};
        r.onerror=()=>reject(r.error);r.onblocked=()=>reject(Error('Close another app tab to update GPS storage'));
        r.onsuccess=()=>{const db=r.result;db.onversionchange=()=>db.close();resolve(db);};
      });return this.dbPromise;
    }
    enqueue(fn){const result=this.chain.then(fn);this.chain=result.catch(()=>{});return result;}
    range(id){return this.IDBKeyRange.bound([id,-Infinity],[id,Infinity]);}
    begin(session){return this.enqueue(async()=>{const db=await this.open(),tx=db.transaction('sessions','readwrite'),done=complete(tx);tx.objectStore('sessions').put(session);await done;return session;});}
    add(fix){return this.enqueue(async()=>{
      const db=await this.open(),tx=db.transaction(['sessions','fixes'],'readwrite'),done=complete(tx),fixes=tx.objectStore('fixes'),sessions=tx.objectStore('sessions');
      const [existing,session]=await Promise.all([request(fixes.getKey([fix.session,fix.t])),request(sessions.get(fix.session))]);
      fixes.put(fix);if(session&&existing===undefined){session.fixes=(session.fixes||0)+1;session.last=Math.max(session.last||0,fix.t);sessions.put(session);}
      await done;return existing===undefined;
    });}
    // Imported tracks keep their session ids, so importing the same file twice adds nothing.
    importFixes(list,{name}={}){return this.enqueue(async()=>{
      const db=await this.open(),tx=db.transaction(['sessions','fixes'],'readwrite'),done=complete(tx),fixes=tx.objectStore('fixes'),sessions=tx.objectStore('sessions'),ids=[...new Set(list.map(f=>f.session))];
      for(const fix of list)fixes.put(fix);
      for(const id of ids){
        const all=await request(fixes.getAll(this.range(id))),previous=await request(sessions.get(id));
        const end=previous&&previous.end==null?null:Math.max(previous?.end??-Infinity,all.at(-1).t);
        sessions.put({...previous,id,name:previous?.name||name||id,imported:previous?previous.imported:true,start:Math.min(previous?.start??Infinity,all[0].t),end,last:all.at(-1).t,fixes:all.length});
      }
      await done;return ids;
    });}
    finish(id,end){return this.enqueue(async()=>{const db=await this.open(),tx=db.transaction('sessions','readwrite'),done=complete(tx),store=tx.objectStore('sessions'),s=await request(store.get(id));if(s){s.end=end;store.put(s);}await done;return s;});}
    // A page closed mid-drive leaves an open session; it ends at its last fix.
    recover(activeId){return this.enqueue(async()=>{
      const db=await this.open(),tx=db.transaction('sessions','readwrite'),done=complete(tx),store=tx.objectStore('sessions'),closed=[];
      for(const s of await request(store.getAll()))if(s.end==null&&s.id!==activeId){s.end=s.last||s.start;s.interrupted=true;store.put(s);closed.push(s.id);}
      await done;return closed;
    });}
    async sessions(){await this.chain;const db=await this.open();return (await request(db.transaction('sessions','readonly').objectStore('sessions').getAll())).sort((a,b)=>a.start-b.start);}
    async fixes(id){await this.chain;const db=await this.open(),store=db.transaction('fixes','readonly').objectStore('fixes');return (await request(id==null?store.getAll():store.getAll(this.range(id)))).sort((a,b)=>a.t-b.t);}
    remove(id){return this.enqueue(async()=>{
      const db=await this.open(),tx=db.transaction(['sessions','fixes','labels'],'readwrite'),done=complete(tx),labels=tx.objectStore('labels');
      tx.objectStore('sessions').delete(id);tx.objectStore('fixes').delete(this.range(id));
      for(const label of await request(labels.getAll()))if(label.session===id)labels.delete(label.key);
      await done;
    });}
    saveLabel(label){return this.enqueue(async()=>{const db=await this.open(),tx=db.transaction('labels','readwrite'),done=complete(tx);tx.objectStore('labels').put(label);await done;return label;});}
    removeLabel(key){return this.enqueue(async()=>{const db=await this.open(),tx=db.transaction('labels','readwrite'),done=complete(tx);tx.objectStore('labels').delete(key);await done;});}
    async labels(){await this.chain;const db=await this.open();return request(db.transaction('labels','readonly').objectStore('labels').getAll());}
  }

  // Phones pause background tabs, so a screen wake lock keeps the page (and its
  // watchPosition) running. The lock drops whenever the page is hidden and is
  // requested again when it returns.
  class GpsLogger{
    constructor({store,geolocation=root.navigator?.geolocation,wakeLock=root.navigator?.wakeLock,document=root.document,intervalMs=3000,now=()=>Date.now(),setTimer=(fn,ms)=>setInterval(fn,ms),clearTimer=id=>clearInterval(id),onChange=()=>{}}={}){
      Object.assign(this,{store,geolocation,wakeLock,document,intervalMs,now,setTimer,clearTimer,onChange});
      this.session=null;this.watch=null;this.timer=null;this.sentinel=null;this.locking=null;this.last=null;this.lastSeen=0;this.fixes=0;this.error='';this.wake=wakeLock?'idle':'unsupported';
      this.visibility=()=>{if(this.session&&this.document?.visibilityState==='visible')this.lock();};
    }
    get active(){return !!this.session;}
    state(){return {active:this.active,session:this.session,fixes:this.fixes,last:this.last,wakeLock:this.wake,error:this.error};}
    changed(){try{this.onChange(this.state());}catch(_){}}
    async start(){
      if(this.session)return this.session;
      if(!this.geolocation)throw Error('This browser has no location access');
      const start=this.now(),session={id:'drive-'+iso(start).replace(/[-:]/g,'').replace(/\.\d+/,''),start,end:null,last:null,fixes:0};
      Object.assign(this,{session,last:null,lastSeen:start,fixes:0,error:''});
      // Store writes are queued in order, so fixes saved later land after the session record.
      const begun=this.store.begin(session),options={enableHighAccuracy:true,maximumAge:0,timeout:30000};
      // Ask for location (and the screen lock) before any await, inside the tap that called start,
      // so the browser treats it as the user's request and can show its permission prompt.
      this.watch=this.geolocation.watchPosition(p=>this.position(p),e=>this.failed(e),options);
      // watchPosition can go quiet while stationary; ask directly so a queue still gets fixes.
      this.timer=this.setTimer(()=>{if(this.session&&this.now()-this.lastSeen>this.intervalMs*1.5)this.geolocation.getCurrentPosition(p=>this.position(p),e=>this.failed(e),{...options,maximumAge:this.intervalMs});},this.intervalMs);
      this.document?.addEventListener?.('visibilitychange',this.visibility);
      const locked=this.lock();
      try{await begun;}catch(error){if(this.session===session)this.halt();throw error;}
      await locked;this.changed();return session;
    }
    lock(){
      if(!this.wakeLock||this.sentinel||this.document?.visibilityState==='hidden')return Promise.resolve();
      // One request at a time: a second concurrent request would leak an unreleased lock.
      return this.locking||=(async()=>{
        try{
          const sentinel=await this.wakeLock.request('screen');
          if(!this.session){await sentinel.release?.();return;}
          this.sentinel=sentinel;this.wake='held';
          sentinel.addEventListener?.('release',()=>{if(this.sentinel===sentinel){this.sentinel=null;this.wake=this.session?'released':'idle';this.changed();}});
        }catch(error){this.wake='error: '+(error?.message||error);}
        finally{this.locking=null;}
        this.changed();
      })();
    }
    position(position){
      const session=this.session;if(!session)return null;
      const fix=fromPosition(position,session.id);if(!fix)return null;
      this.lastSeen=this.now();
      if(this.last&&fix.t-this.last.t<this.intervalMs)return null;
      this.last=fix;this.fixes++;this.error='';
      this.store.add(fix).catch(error=>{this.error='Fix not saved: '+error.message;this.changed();});
      this.changed();return fix;
    }
    failed(error){
      // PERMISSION_DENIED ends the session; timeouts and lost signal keep watching.
      this.error=error?.code===1?'Location permission denied':error?.code===3?'Waiting for a GPS fix…':'Location unavailable: '+(error?.message||'unknown error');
      // A drive refused before its first fix is not kept as an empty session.
      if(error?.code===1)return this.stop({discard:!this.fixes});this.changed();
    }
    halt(){
      this.session=null;this.geolocation.clearWatch(this.watch);this.watch=null;this.clearTimer(this.timer);this.timer=null;
      this.document?.removeEventListener?.('visibilitychange',this.visibility);
      const sentinel=this.sentinel;this.sentinel=null;this.wake=this.wakeLock?'idle':'unsupported';
      return Promise.resolve(sentinel?.release()).catch(()=>{});
    }
    async stop({discard=false}={}){
      const session=this.session;if(!session)return null;
      await this.halt();
      const ended=discard?await this.store.remove(session.id).then(()=>null):await this.store.finish(session.id,this.now());this.changed();return ended;
    }
  }

  // Interpolated position at time t. Gaps over maxGap are not bridged: a stop at
  // the edge of a tunnel is not evidence of where the car was in between.
  const defaults={radius:250,fov:70,near:40,maxGap:30000,edge:5000,maxAccuracy:50,minMove:3};
  const usable=(fixes,o)=>fixes.filter(f=>f.accuracy==null||f.accuracy<=o.maxAccuracy).sort((a,b)=>a.t-b.t);
  function course(a,b,o){return a&&b&&distance(a,b)>=o.minMove?bearing(a,b):null;}
  function positionAt(list,t,options={}){
    const o={...defaults,...options};let lo=0,hi=list.length;
    while(lo<hi){const m=lo+hi>>1;if(list[m].t<t)lo=m+1;else hi=m;}
    const a=list[lo-1],b=list[lo];
    if(a&&b&&b.t-a.t<=o.maxGap){
      const k=(t-a.t)/(b.t-a.t),moved=distance(a,b),c=course(a,b,o);
      return {t,lat:a.lat+(b.lat-a.lat)*k,lon:a.lon+(b.lon-a.lon)*k,accuracy:Math.max(a.accuracy??0,b.accuracy??0)||null,course:c??a.heading??b.heading??null,speed:moved/Math.max(.001,(b.t-a.t)/1000),gap:b.t-a.t,session:a.session};
    }
    const f=[a,b].filter(Boolean).sort((x,y)=>Math.abs(x.t-t)-Math.abs(y.t-t))[0];
    if(!f||Math.abs(f.t-t)>o.edge)return null;
    return {t,lat:f.lat,lon:f.lon,accuracy:f.accuracy,course:f.heading,speed:f.speed,gap:Math.abs(f.t-t),session:f.session};
  }
  // Where the car is relative to the camera's view. Without a look direction we
  // cannot tell whether it is in front of or behind the camera.
  function view(camera,p,o){
    const d=distance(camera,p),look=lookBearing(camera),offAxis=look==null?null:diff(look,bearing(camera,p));
    return {distance:d,look,offAxis,inView:look==null?null:d<=o.near||offAxis<=o.fov};
  }
  // Travelling the way the camera looks shows the car's rear, i.e. away_from_camera.
  function expectedFacing(look,travel){
    if(look==null||travel==null)return null;const r=wrap(travel-look);
    return r<45||r>=315?'away_from_camera':r<135?'image_right':r<225?'toward_camera':'image_left';
  }
  // Saved frames whose upstream capture time puts your car in front of the camera.
  // Fetch times can trail a capture by minutes, so frames without one are skipped.
  function frames(rows,fixes,options={}){
    const o={...defaults,...options},list=usable(fixes,o),out=[];if(!list.length)return out;
    const first=list[0].t-o.edge,last=list.at(-1).t+o.edge;
    for(const row of rows){
      const camera=row.camera||{},t=captured(row);
      if(!Number.isFinite(camera.lat)||!Number.isFinite(camera.lon)||t!==Date.parse(row.sourceModified||'')||t<first||t>last)continue;
      const p=positionAt(list,t,o);if(!p)continue;
      const v=view(camera,p,o);if(v.distance>o.radius||v.inView===false)continue;
      out.push({key:row.key,cameraId:String(row.cameraId),hash:row.hash,camera,t,position:p,...v,expectedFacing:expectedFacing(v.look,p.course)});
    }
    return out.sort((a,b)=>b.t-a.t);
  }
  // Every catalog camera the drive passed in view of, for DriveBC's Replay the Day
  // (about 24 hours of images) when no frame was saved locally.
  function passes(cameras,fixes,options={}){
    const o={...defaults,...options},list=usable(fixes,o),out=[];if(!list.length)return out;
    let south=90,north=-90,west=180,east=-180;for(const f of list){south=Math.min(south,f.lat);north=Math.max(north,f.lat);west=Math.min(west,f.lon);east=Math.max(east,f.lon);}
    const padLat=o.radius/111320,padLon=o.radius/(111320*Math.max(.05,Math.cos((south+north)/2*Math.PI/180)));
    for(const camera of cameras){
      if(!Number.isFinite(camera.lat)||!Number.isFinite(camera.lon)||camera.lat<south-padLat||camera.lat>north+padLat||camera.lon<west-padLon||camera.lon>east+padLon)continue;
      let visit=null;const close=()=>{if(visit)out.push(visit);visit=null;};
      for(let i=0;i<list.length;i++){
        const f=list[i],v=view(camera,f,o);if(v.distance>o.radius||v.inView===false)continue;
        if(visit&&f.t-visit.end>o.maxGap)close();
        if(!visit)visit={cameraId:String(camera.id),camera,start:f.t,end:f.t,fixes:0,closest:null,session:f.session};
        visit.end=f.t;visit.fixes++;
        if(!visit.closest||v.distance<visit.closest.distance){
          const prev=list[i-1]?.t>=f.t-o.maxGap?list[i-1]:null,next=list[i+1]?.t<=f.t+o.maxGap?list[i+1]:null;
          visit.closest={t:f.t,lat:f.lat,lon:f.lon,accuracy:f.accuracy,distance:v.distance,offAxis:v.offAxis,course:course(prev||f,next||f,o)??f.heading};
        }
      }
      close();
    }
    for(const p of out)Object.assign(p,{expectedFacing:expectedFacing(lookBearing(p.camera),p.closest.course),replay:replay(p.camera,p.start)});
    return out.sort((a,b)=>a.start-b.start);
  }
  function replay(camera,start){
    const id=encodeURIComponent(String(camera.id));
    return {page:camera.source||`https://www.drivebc.ca/cameras/${id}`,list:`https://www.drivebc.ca/api/webcams/${id}/replayTheDay/`,availableUntil:start+DAY};
  }
  // One own-car label per frame; detection null means the car is not visible.
  // Its index matches VehicleRoutes.sightings keys (frame key + ':' + index).
  function labelRecord(match,row,index,labeledAt=Date.now()){
    const d=index==null?null:row.detections?.[index];if(index!=null&&!VEHICLES.includes(d?.label))throw Error('Not a vehicle detection');
    const p=match.position;
    return {key:match.key,cameraId:match.cameraId,hash:match.hash,capturedAt:match.t,session:p.session,detection:index==null?null:index,sighting:index==null?null:match.key+':'+index,label:d?.label||null,box:d?.box||null,
      modelFacing:index==null?null:row.facing?.byDetection?.[index]?.facing||null,expectedFacing:match.expectedFacing,distance:match.distance,offAxis:match.offAxis,
      gps:{lat:p.lat,lon:p.lon,accuracy:p.accuracy,course:p.course,speed:p.speed,gap:p.gap},labeledAt};
  }
  function evaluationSet({sessions=[],labels=[],frames:candidates=[],passes:passed=[],generated=Date.now()}){
    const labeled=new Set(labels.map(l=>l.key)),round=v=>v==null?null:Math.round(v*10)/10;
    return {format:'locaintel-own-vehicle-v1',generated:iso(generated),
      description:'Own-vehicle labels from a GPS log matched to DriveBC frames by upstream capture time. detection null = car not visible. sighting joins VehicleRoutes.sightings keys.',
      sessions:sessions.map(s=>({id:s.id,name:s.name||null,start:iso(s.start),end:s.end==null?null:iso(s.end),fixes:s.fixes||0,interrupted:!!s.interrupted})),
      labels:labels.slice().sort((a,b)=>a.capturedAt-b.capturedAt).map(l=>({...l,capturedAt:iso(l.capturedAt),labeledAt:iso(l.labeledAt)})),
      unlabeled:candidates.filter(m=>!labeled.has(m.key)).map(m=>({key:m.key,cameraId:m.cameraId,hash:m.hash,capturedAt:iso(m.t),distance:round(m.distance),expectedFacing:m.expectedFacing})),
      passes:passed.map(p=>({cameraId:p.cameraId,name:p.camera.name||null,session:p.session,start:iso(p.start),end:iso(p.end),closest:{time:iso(p.closest.t),distance:round(p.closest.distance),course:round(p.closest.course),lat:p.closest.lat,lon:p.closest.lon},expectedFacing:p.expectedFacing,replay:{...p.replay,availableUntil:iso(p.replay.availableUntil)}}))};
  }

  const xml=s=>String(s).replace(/[<>&'"]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;',"'":'&apos;','"':'&quot;'}[c]));
  const EXTENSIONS=['accuracy','altitudeAccuracy','speed','heading'];
  // GPX 1.1; a new segment starts at a session change or a gap over splitMs.
  function gpx(fixes,{name='LocaIntel drive',splitMs=60000}={}){
    const sorted=[...fixes].sort((a,b)=>a.t-b.t),segments=[];let previous=null;
    for(const f of sorted){if(!previous||f.t-previous.t>splitMs||f.session!==previous.session)segments.push([]);segments.at(-1).push(f);previous=f;}
    const point=f=>{const ext=EXTENSIONS.filter(k=>f[k]!=null).map(k=>`<li:${k}>${f[k]}</li:${k}>`).join('');return `<trkpt lat="${f.lat}" lon="${f.lon}">${f.altitude!=null?`<ele>${f.altitude}</ele>`:''}<time>${iso(f.t)}</time>${ext?`<extensions>${ext}</extensions>`:''}</trkpt>`;};
    return `<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="LocaIntel" xmlns="http://www.topografix.com/GPX/1/1" xmlns:li="https://github.com/Shubin123/locaIntel/gpx/1">\n<metadata><name>${xml(name)}</name>${sorted.length?`<time>${iso(sorted[0].t)}</time>`:''}</metadata>\n<trk><name>${xml(name)}</name>\n${segments.map(s=>'<trkseg>\n'+s.map(point).join('\n')+'\n</trkseg>').join('\n')}\n</trk>\n</gpx>\n`;
  }
  const CSV_FIELDS=['session','time','unix_ms','lat','lon','accuracy_m','altitude_m','altitude_accuracy_m','speed_mps','heading_deg'];
  // Text cells that a spreadsheet would run as a formula get a leading apostrophe.
  const cell=v=>{if(v==null)return '';let s=String(v);if(typeof v==='string'&&/^[=+\-@\t\r]/.test(s))s="'"+s;return /[",\n\r]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;};
  function csv(fixes){return [CSV_FIELDS.join(','),...[...fixes].sort((a,b)=>a.t-b.t).map(f=>[f.session,iso(f.t),f.t,f.lat,f.lon,f.accuracy,f.altitude,f.altitudeAccuracy,f.speed,f.heading].map(cell).join(','))].join('\n')+'\n';}
  function splitCSV(text){
    const rows=[];let row=[],value='',quoted=false;
    for(let i=0;i<text.length;i++){
      const c=text[i];
      if(quoted){if(c!=='"')value+=c;else if(text[i+1]==='"'){value+='"';i++;}else quoted=false;}
      else if(c==='"')quoted=true;else if(c===','){row.push(value);value='';}
      else if(c==='\n'||c==='\r'){if(c==='\r'&&text[i+1]==='\n')i++;row.push(value);rows.push(row);row=[];value='';}
      else value+=c;
    }
    if(value||row.length){row.push(value);rows.push(row);}
    return rows.filter(r=>r.some(v=>v.trim()));
  }
  function parseCSV(text,session){
    const [header=[],...rows]=splitCSV(String(text).replace(/^﻿/,'')),names=header.map(h=>h.trim().toLowerCase());
    const col=(...keys)=>keys.map(k=>names.indexOf(k)).find(i=>i>=0)??-1;
    const at={session:col('session'),time:col('time','timestamp','date'),ms:col('unix_ms'),lat:col('lat','latitude'),lon:col('lon','lng','longitude'),accuracy:col('accuracy_m','accuracy'),altitude:col('altitude_m','altitude','ele','elevation'),altitudeAccuracy:col('altitude_accuracy_m'),speed:col('speed_mps','speed'),heading:col('heading_deg','heading','course','bearing')};
    if(at.lat<0||at.lon<0||at.time<0&&at.ms<0)throw Error('CSV needs lat, lon and time (or unix_ms) columns');
    const get=(r,k)=>at[k]<0?null:r[at[k]];
    return rows.map(r=>validFix({session:get(r,'session')?.trim().replace(/^'(?=[=+\-@])/,'')||session,t:finite(get(r,'ms'))??Date.parse(get(r,'time')||''),lat:finite(get(r,'lat')),lon:finite(get(r,'lon')),
      accuracy:finite(get(r,'accuracy')),altitude:finite(get(r,'altitude')),altitudeAccuracy:finite(get(r,'altitudeAccuracy')),speed:finite(get(r,'speed')),heading:finite(get(r,'heading'))})).filter(Boolean);
  }
  // Track points with a time; GPX 1.0 speed/course and this app's extensions are kept.
  function parseGPX(text,session){
    const out=[];
    for(const m of String(text).matchAll(/<(?:\w+:)?trkpt\b([^>]*?)(?:\/>|>([\s\S]*?)<\/(?:\w+:)?trkpt>)/g)){
      const attr=n=>finite((new RegExp(`\\b${n}\\s*=\\s*["']([^"']*)["']`).exec(m[1])||[])[1]);
      const tag=n=>(new RegExp(`<(?:\\w+:)?${n}>\\s*([^<]*?)\\s*</(?:\\w+:)?${n}>`).exec(m[2]||'')||[])[1];
      const fix=validFix({session,t:Date.parse(tag('time')||''),lat:attr('lat'),lon:attr('lon'),accuracy:finite(tag('accuracy')),altitude:finite(tag('ele')),altitudeAccuracy:finite(tag('altitudeAccuracy')),speed:finite(tag('speed')),heading:finite(tag('heading'))??finite(tag('course'))});
      if(fix)out.push(fix);
    }
    return out;
  }
  function parseTrack(text,{name='track',session='import-'+name}={}){
    const fixes=/^\s*(<\?xml|<gpx)/i.test(text)?parseGPX(text,session):parseCSV(text,session);
    if(!fixes.length)throw Error('No timestamped positions found in '+name);
    return fixes;
  }
  function trackLength(fixes){let m=0;for(let i=1;i<fixes.length;i++)if(fixes[i].session===fixes[i-1].session)m+=distance(fixes[i-1],fixes[i]);return m;}

  root.GpsLog={DAY,VEHICLES,defaults,fromPosition,GpsStore,GpsLogger,positionAt,expectedFacing,frames,passes,labelRecord,evaluationSet,gpx,csv,parseCSV,parseGPX,parseTrack,trackLength};
})(globalThis);

/* GPS storage and logging for vehicle tracking. Logging runs only between an explicit
   Start and Stop on this device. Fixes pass through VehicleMatcher's relevance
   filter, so only positions near a DriveBC camera are stored, and they stay in
   IndexedDB until exported. Every drive and label belongs to a vehicle profile, so
   any number of vehicles can be followed, each from its own GPS track. */
(function(root){
  const M=root.VehicleMatcher,{distance}=root.VehicleRoutes;
  const DEFAULT_VEHICLE='vehicle-1';
  const finite=v=>v==null||v===''?null:Number.isFinite(Number(v))?Number(v):null;
  const iso=t=>new Date(t).toISOString();
  const request=r=>new Promise((resolve,reject)=>{r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
  const complete=tx=>new Promise((resolve,reject)=>{tx.oncomplete=resolve;tx.onabort=tx.onerror=()=>reject(tx.error||Error('GPS storage transaction failed'));});
  const vehicle=(id=DEFAULT_VEHICLE,fields={})=>({id,name:'My vehicle',bodyType:'',colour:'',references:[],created:Date.now(),...fields});

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
        const r=this.indexedDB.open(this.name,2);
        r.onupgradeneeded=event=>{
          const db=r.result,tx=r.transaction;
          if(event.oldVersion<1){db.createObjectStore('sessions',{keyPath:'id'});db.createObjectStore('fixes',{keyPath:['session','t']});db.createObjectStore('labels',{keyPath:'key'});}
          if(event.oldVersion<2){
            // Version 1 followed one own vehicle: its drives and labels move to the first profile.
            const vehicles=db.createObjectStore('vehicles',{keyPath:'id'}),sessions=tx.objectStore('sessions'),labels=tx.objectStore('labels');
            vehicles.put(vehicle());
            sessions.getAll().onsuccess=e=>{for(const s of e.target.result)if(!s.vehicle)sessions.put({...s,vehicle:DEFAULT_VEHICLE});};
            labels.getAll().onsuccess=e=>{for(const l of e.target.result)if(!l.vehicle){labels.delete(l.key);labels.put({...l,vehicle:DEFAULT_VEHICLE,frame:l.key,key:DEFAULT_VEHICLE+'|'+l.key});}};
          }
        };
        r.onerror=()=>reject(r.error);r.onblocked=()=>reject(Error('Close another app tab to update GPS storage'));
        r.onsuccess=()=>{const db=r.result;db.onversionchange=()=>db.close();resolve(db);};
      });return this.dbPromise;
    }
    enqueue(fn){const result=this.chain.then(fn);this.chain=result.catch(()=>{});return result;}
    range(id){return this.IDBKeyRange.bound([id,-Infinity],[id,Infinity]);}
    async vehicles(){await this.chain;const db=await this.open();return (await request(db.transaction('vehicles','readonly').objectStore('vehicles').getAll())).sort((a,b)=>(a.created||0)-(b.created||0));}
    saveVehicle(v){return this.enqueue(async()=>{const db=await this.open(),tx=db.transaction('vehicles','readwrite'),done=complete(tx);tx.objectStore('vehicles').put(v);await done;return v;});}
    // Deleting a vehicle deletes its drives, their fixes and its labels.
    removeVehicle(id){return this.enqueue(async()=>{
      const db=await this.open(),tx=db.transaction(['vehicles','sessions','fixes','labels'],'readwrite'),done=complete(tx),sessions=tx.objectStore('sessions'),labels=tx.objectStore('labels');
      tx.objectStore('vehicles').delete(id);
      for(const s of await request(sessions.getAll()))if(s.vehicle===id){sessions.delete(s.id);tx.objectStore('fixes').delete(this.range(s.id));}
      for(const l of await request(labels.getAll()))if(l.vehicle===id)labels.delete(l.key);
      await done;
    });}
    begin(session){return this.enqueue(async()=>{const db=await this.open(),tx=db.transaction('sessions','readwrite'),done=complete(tx);tx.objectStore('sessions').put(session);await done;return session;});}
    add(fix){return this.enqueue(async()=>{
      const db=await this.open(),tx=db.transaction(['sessions','fixes'],'readwrite'),done=complete(tx),fixes=tx.objectStore('fixes'),sessions=tx.objectStore('sessions');
      const [existing,session]=await Promise.all([request(fixes.getKey([fix.session,fix.t])),request(sessions.get(fix.session))]);
      fixes.put(fix);if(session&&existing===undefined){session.fixes=(session.fixes||0)+1;session.last=Math.max(session.last||0,fix.t);sessions.put(session);}
      await done;return existing===undefined;
    });}
    // Imported tracks keep their session ids, so importing the same file twice adds nothing.
    importFixes(list,{name,vehicle:owner=DEFAULT_VEHICLE}={}){return this.enqueue(async()=>{
      const db=await this.open(),tx=db.transaction(['sessions','fixes'],'readwrite'),done=complete(tx),fixes=tx.objectStore('fixes'),sessions=tx.objectStore('sessions'),ids=[...new Set(list.map(f=>f.session))];
      for(const fix of list)fixes.put(fix);
      for(const id of ids){
        const all=await request(fixes.getAll(this.range(id))),previous=await request(sessions.get(id));
        const end=previous&&previous.end==null?null:Math.max(previous?.end??-Infinity,all.at(-1).t);
        sessions.put({...previous,id,vehicle:previous?.vehicle||owner,name:previous?.name||name||id,imported:previous?previous.imported:true,start:Math.min(previous?.start??Infinity,all[0].t),end,last:all.at(-1).t,fixes:all.length});
      }
      await done;return ids;
    });}
    // Replace a drive's fixes with the given subset (dropping positions away from cameras).
    trim(id,keep){return this.enqueue(async()=>{
      const db=await this.open(),tx=db.transaction(['sessions','fixes'],'readwrite'),done=complete(tx),fixes=tx.objectStore('fixes'),sessions=tx.objectStore('sessions');
      fixes.delete(this.range(id));for(const f of keep)fixes.put(f);
      const s=await request(sessions.get(id));if(s){s.fixes=keep.length;s.last=keep.at(-1)?.t??s.last;s.trimmed=true;sessions.put(s);}
      await done;return keep.length;
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
  // requested again when it returns. With a filter, only the fixes it keeps are
  // stored; `seen` counts every fix, `fixes` the stored ones.
  class GpsLogger{
    constructor({store,geolocation=root.navigator?.geolocation,wakeLock=root.navigator?.wakeLock,document=root.document,intervalMs=3000,filter=()=>null,now=()=>Date.now(),setTimer=(fn,ms)=>setInterval(fn,ms),clearTimer=id=>clearInterval(id),onChange=()=>{}}={}){
      Object.assign(this,{store,geolocation,wakeLock,document,intervalMs,makeFilter:filter,now,setTimer,clearTimer,onChange});
      this.session=null;this.filter=null;this.watch=null;this.timer=null;this.sentinel=null;this.locking=null;this.last=null;this.lastSeen=0;this.seen=0;this.fixes=0;this.error='';this.wake=wakeLock?'idle':'unsupported';
      this.visibility=()=>{if(this.session&&this.document?.visibilityState==='visible')this.lock();};
    }
    get active(){return !!this.session;}
    state(){return {active:this.active,session:this.session,seen:this.seen,fixes:this.fixes,last:this.last,wakeLock:this.wake,error:this.error};}
    changed(){try{this.onChange(this.state());}catch(_){}}
    async start({vehicle:owner=DEFAULT_VEHICLE}={}){
      if(this.session)return this.session;
      if(!this.geolocation)throw Error('This browser has no location access');
      const start=this.now(),session={id:'drive-'+iso(start).replace(/[-:]/g,'').replace(/\.\d+/,''),vehicle:owner,start,end:null,last:null,fixes:0};
      Object.assign(this,{session,filter:this.makeFilter(),last:null,lastSeen:start,seen:0,fixes:0,error:''});
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
      this.last=fix;this.seen++;this.error='';
      for(const kept of this.filter?this.filter.push(fix):[fix]){this.fixes++;this.store.add(kept).catch(error=>{this.error='Fix not saved: '+error.message;this.changed();});}
      this.changed();return fix;
    }
    failed(error){
      // PERMISSION_DENIED ends the session; timeouts and lost signal keep watching.
      this.error=error?.code===1?'Location permission denied':error?.code===3?'Waiting for a GPS fix…':'Location unavailable: '+(error?.message||'unknown error');
      if(error?.code===1)return this.stop();this.changed();
    }
    halt(){
      this.session=null;this.filter=null;this.geolocation.clearWatch(this.watch);this.watch=null;this.clearTimer(this.timer);this.timer=null;
      this.document?.removeEventListener?.('visibilitychange',this.visibility);
      const sentinel=this.sentinel;this.sentinel=null;this.wake=this.wakeLock?'idle':'unsupported';
      return Promise.resolve(sentinel?.release()).catch(()=>{});
    }
    // A drive that never came near a camera stored nothing, so it is not kept as an empty session.
    async stop({discard=!this.fixes}={}){
      const session=this.session;if(!session)return null;
      await this.halt();
      const ended=discard?await this.store.remove(session.id).then(()=>null):await this.store.finish(session.id,this.now());this.changed();return ended;
    }
  }

  // One label per vehicle and frame; detection null means the vehicle is not visible.
  // Its index matches VehicleRoutes.sightings keys (frame key + ':' + index), and the
  // colour descriptor is kept so the label stays a reference after the photo is pruned.
  function labelRecord(frame,row,index,{vehicle:owner=DEFAULT_VEHICLE,descriptor=null,labeledAt=Date.now()}={}){
    const d=index==null?null:row.detections?.[index];if(index!=null&&!M.VEHICLES.includes(d?.label))throw Error('Not a vehicle detection');
    const p=frame.position||{};
    return {key:owner+'|'+frame.key,vehicle:owner,frame:frame.key,cameraId:frame.cameraId,hash:frame.hash,capturedAt:frame.t,exactTime:!!frame.exact,session:p.session??frame.session??null,detection:index==null?null:index,sighting:index==null?null:frame.key+':'+index,label:d?.label||null,box:d?.box||null,
      descriptor:index==null?null:descriptor,modelFacing:index==null?null:row.facing?.byDetection?.[index]?.facing||null,expectedFacing:frame.expectedFacing??null,distance:frame.distance??null,offAxis:frame.offAxis??null,
      gps:{lat:p.lat??null,lon:p.lon??null,accuracy:p.accuracy??null,course:p.course??null,speed:p.speed??null,gap:p.gap??null},labeledAt};
  }
  // passes keep the v1 shape, so pipeline.gps_replay reads either format.
  function exportSet({vehicle:v,sessions=[],labels=[],result,generated=Date.now()}){
    const round=v=>v==null?null:Math.round(v*10)/10,score=v=>v==null?null:Math.round(v*1000)/1000,own=labels.filter(l=>l.vehicle===v.id),labeled=new Set(own.map(l=>l.frame));
    return {format:'locaintel-vehicle-track-v2',generated:iso(generated),
      description:'One vehicle followed through DriveBC cameras from its GPS track. Only fixes near cameras are stored. detection null = vehicle not visible. sighting joins VehicleRoutes.sightings keys.',
      vehicle:{id:v.id,name:v.name||null,bodyType:v.bodyType||null,colour:v.colour||null,references:(v.references||[]).map(r=>({source:'photo',descriptor:r.descriptor}))},
      relevance:{radiusMetres:M.RELEVANT_RADIUS,rule:'fixes within the radius of a camera, plus one boundary fix each side within 30 s'},
      sessions:sessions.filter(s=>s.vehicle===v.id).map(s=>({id:s.id,name:s.name||null,start:iso(s.start),end:s.end==null?null:iso(s.end),fixes:s.fixes||0,interrupted:!!s.interrupted,trimmed:!!s.trimmed})),
      sightings:(result?.sightings||[]).map(s=>({key:s.key,cameraId:s.cameraId,name:s.camera.name||null,capturedAt:iso(s.t),exactTime:s.exact,status:s.status,confidence:score(s.verification.confidence),margin:score(s.verification.margin),detection:s.detection,box:s.box,distance:round(s.distance),expectedFacing:s.expectedFacing,cues:s.detection==null?null:s.verification.detections.find(d=>d.index===s.detection)?.cues||null})),
      labels:own.slice().sort((a,b)=>a.capturedAt-b.capturedAt).map(l=>({...l,capturedAt:iso(l.capturedAt),labeledAt:iso(l.labeledAt)})),
      unlabeled:(result?.sightings||[]).filter(s=>!labeled.has(s.key)).map(s=>({key:s.key,cameraId:s.cameraId,hash:s.hash,capturedAt:iso(s.t),distance:round(s.distance),expectedFacing:s.expectedFacing})),
      passes:(result?.passes||[]).map(p=>({cameraId:p.cameraId,name:p.camera.name||null,session:p.session,start:iso(p.start),end:iso(p.end),closest:{time:iso(p.closest.t),distance:round(p.closest.distance),course:round(p.closest.course),lat:p.closest.lat,lon:p.closest.lon},expectedFacing:p.expectedFacing,status:p.status,frames:p.frames.length,replay:{...p.replay,availableUntil:iso(p.replay.availableUntil)}})),
      checks:{decoy:result?.decoy||null,agreement:result?.agreement||null}};
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
  // Stored tracks are pieces near cameras, so only consecutive fixes within a minute count toward length.
  function trackLength(fixes){let m=0;for(let i=1;i<fixes.length;i++)if(fixes[i].session===fixes[i-1].session&&fixes[i].t-fixes[i-1].t<=60000)m+=distance(fixes[i-1],fixes[i]);return m;}

  root.GpsLog={DEFAULT_VEHICLE,vehicle,fromPosition,GpsStore,GpsLogger,labelRecord,exportSet,gpx,csv,parseCSV,parseGPX,parseTrack,trackLength};
})(globalThis);

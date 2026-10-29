/* Vehicle tracking: the focal panel and the map's tracking layer. Each vehicle has a
   profile (name, body type, colour, reference photos) and its own drives; the panel
   runs VehicleMatcher over the shared photo archive from app.js. GPS fixes and
   profiles live in their own IndexedDB store and leave this browser only as files
   you export. */
(function(){
  const G=GpsLog,M=VehicleMatcher,R=VehicleRoutes,store=new G.GpsStore();
  let vehicles=[],vehicleId=null,sessions=[],fixes=[],labels=[],catalog=[],index=null,result=null,trimmable=new Map();
  let lastVersion='',paintTimer=null,reloadTimer=null,lastRevision=-1,message='',permission='',cameraCache={revision:-1,list:[]},liveMarker=null,preparing=false;
  const crops=new Map(),wanted=new Set(),describing=new Set(),capture=new Map();
  const clock=t=>new Date(t).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const day=t=>new Date(t).toLocaleDateString([],{month:'short',day:'numeric'});
  const metres=m=>m==null?'':m<1000?Math.round(m)+' m':(m/1000).toFixed(1)+' km';
  const pct=v=>Math.round(v*100)+'%',plural=(n,one,many=one+'s')=>n+' '+(n===1?one:many);
  const ago=ms=>{const s=Math.round(Math.abs(ms)/1000);return (s<90?s+' s':Math.round(s/60)+' min')+(ms<0?' before':' after')+' your closest approach';};
  const hex=buffer=>Array.from(new Uint8Array(buffer),b=>b.toString(16).padStart(2,'0')).join('');
  const FACING={toward_camera:'front expected (driving toward camera)',away_from_camera:'rear expected (driving away)',image_left:'side expected (moving left)',image_right:'side expected (moving right)'};
  const STATUS={labelled:'marked by you',confirmed:'confirmed',possible:'possible',unmatched:'no vehicle matches the profile',unverified:'pick your vehicle: the profile has no body type, colour or references yet',pending:'waiting for detection',empty:'no vehicle detected',not_visible:'marked not visible'};
  const SHORT={labelled:'marked by you',confirmed:'confirmed',possible:'possible',unmatched:'no match',unverified:'needs your pick',pending:'detecting',empty:'no vehicle detected',not_visible:'not visible'};
  const COLOUR={labelled:'#b8ed86',confirmed:'#ffd36b',possible:'#ffd36b',unverified:'#c3a1ff',pending:'#8c9da7'};
  const WAKE={held:'screen kept awake',idle:'',released:'screen lock dropped while hidden; keep this tab in front',unsupported:'this browser cannot keep the screen awake; keep it on with this tab in front'};
  // A page cannot force the permission prompt: once location is blocked, the browser refuses without asking.
  const IOS=/iP(hone|ad|od)/.test(navigator.userAgent)||navigator.platform==='MacIntel'&&navigator.maxTouchPoints>1;
  const BLOCKED=IOS?'Location is blocked, so iOS will not ask. In Settings › Privacy & Security › Location Services, turn Location Services on and set Safari Websites (or Chrome) to While Using the App with Precise Location on. Then in Safari tap aA › Website Settings › Location › Ask, reload, and tap Start.'
    :'Location is blocked for this site, so the browser will not ask. Allow location in the site settings next to the address bar, reload, and tap Start.';
  navigator.permissions?.query?.({name:'geolocation'}).then(p=>{const update=()=>{permission=p.state;status();};p.onchange=update;update();}).catch(()=>{});
  const current=()=>vehicles.find(v=>v.id===vehicleId)||null;
  const own=list=>list.filter(x=>(x.vehicle||G.DEFAULT_VEHICLE)===vehicleId);
  const radius=()=>Number($('gps-radius').value),tolerance=()=>Number($('gps-tolerance').value);
  const button=(text,fn,cls)=>{const b=node('button',text,cls);b.type='button';b.onclick=fn;return b;};
  const remember=()=>{try{localStorage.setItem('locaintel.vehicle',vehicleId);}catch(_){}};

  const logger=new G.GpsLogger({store,filter:()=>new M.RelevanceFilter(index),onChange:state=>{
    if(state.last)noteCapture(state.last);
    status();renderLive();
    // Stopped, or refused permission: show the final drive list now.
    if(!state.active)reload();else if(!reloadTimer)reloadTimer=setTimeout(()=>{reloadTimer=null;reload();},4000);
  }});
  function fail(error){message=error?.message||String(error);status();}
  function download(name,text,type){
    const url=URL.createObjectURL(new Blob([text],{type})),a=Object.assign(document.createElement('a'),{href:url,download:name});
    document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function status(){
    const s=logger.state(),wake=WAKE[s.wakeLock]??`screen lock unavailable (${s.wakeLock.replace(/^error: /,'')}); keep the screen on`,owner=vehicles.find(v=>v.id===s.session?.vehicle);
    const error=/permission denied/.test(s.error)?BLOCKED:s.error,coarse=s.last?.accuracy>1000&&(IOS?'approximate location only: turn on Precise Location':'approximate location only');
    $('gps-status').textContent=s.active?[`Logging ${owner?.name||'vehicle'}`,`${plural(s.seen,'fix','fixes')}, ${s.fixes} kept near cameras`,s.last?`last ${clock(s.last.t)}${s.last.accuracy!=null?' ±'+Math.round(s.last.accuracy)+' m':''}`:'waiting for the first fix',capture.size&&`capturing ${plural(capture.size,'camera')}`,coarse,wake,error].filter(Boolean).join(' · ')
      :error||message||(!index?'Loading the camera catalog…':permission==='denied'?BLOCKED:permission==='prompt'?'Not logging. Start will ask to use your location.':'Not logging.');
    $('gps-status').classList.toggle('weak',!!(s.error||coarse||!s.active&&(message||permission==='denied')||s.active&&s.wakeLock!=='held'));
    $('gps-start').disabled=s.active||!$('gps-own').checked||!index||!current();$('gps-stop').disabled=!s.active;$('gps-own').disabled=s.active;
    // A running drive belongs to the vehicle it started with.
    $('vehicle-select').disabled=s.active;$('vehicle-add').disabled=s.active;$('vehicle-delete').disabled=s.active;
  }
  async function reload(){
    try{
      let [v,s,f,l]=await Promise.all([store.vehicles(),store.sessions(),store.fixes(),store.labels()]);
      if(!v.length)v=[await store.saveVehicle(G.vehicle('vehicle-'+Date.now().toString(36)))];
      vehicles=v;sessions=s;fixes=f;labels=l;
      if(!current())vehicleId=(()=>{try{return localStorage.getItem('locaintel.vehicle');}catch(_){return null;}})();
      if(!current())vehicleId=vehicles[0].id;
      updateTrimmable();lastVersion='';renderVehicles();schedule();status();
    }catch(error){fail(error);}
  }
  // Drives saved before only relevant GPS was kept can be cut down to the same camera zones.
  function updateTrimmable(){
    trimmable=new Map();let extra=0;
    if(index)for(const s of sessions){if(s.id===logger.session?.id)continue;const mine=fixes.filter(f=>f.session===s.id),kept=M.compact(mine,index);if(kept.length<mine.length){trimmable.set(s.id,kept);extra+=mine.length-kept.length;}}
    $('gps-trim').hidden=!trimmable.size;if(!trimmable.size)return;
    $('gps-trim').replaceChildren(node('span',`${plural(trimmable.size,'drive')} saved earlier still ${trimmable.size===1?'holds':'hold'} ${plural(extra,'fix','fixes')} away from any camera. `),button('Keep only camera zones',async()=>{
      if(!confirm(`Delete ${plural(extra,'fix','fixes')} farther than ${M.RELEVANT_RADIUS} m from every camera? Export GPX first if you want the full track.`))return;
      try{for(const [id,kept]of trimmable)await store.trim(id,kept);message=`Kept only the camera zones of ${plural(trimmable.size,'drive')}.`;await reload();}catch(error){fail(error);}
    },'gps-hidden'));
  }

  // Crops come from the browser's photo archive; a pruned photo leaves an empty tile.
  function crop(hash,box,key,width=72,height=54){
    const canvas=node('canvas');canvas.width=width;canvas.height=height;
    (async()=>{try{let bitmap=crops.get(key);if(!bitmap){const blob=await photoArchive.photo(hash);if(!blob)return;const image=await createImageBitmap(blob),[x,y,xx,yy]=box;
      bitmap=await createImageBitmap(image,x*image.width,y*image.height,Math.max(1,(xx-x)*image.width),Math.max(1,(yy-y)*image.height));image.close();crops.set(key,bitmap);if(crops.size>160){const [k,b]=crops.entries().next().value;b.close?.();crops.delete(k);}}
      canvas.getContext('2d').drawImage(bitmap,0,0,width,height);}catch(_){}})();
    return canvas;
  }
  async function thumbnail(blob,box){
    const image=await createImageBitmap(blob);
    try{const [x,y,xx,yy]=box,canvas=node('canvas');canvas.width=96;canvas.height=72;canvas.getContext('2d').drawImage(image,x*image.width,y*image.height,(xx-x)*image.width,(yy-y)*image.height,0,0,96,72);return canvas.toDataURL('image/jpeg',.8);}
    finally{image.close();}
  }

  function renderVehicles(){
    SmoothUI.options($('vehicle-select'),vehicles.map(v=>[v.name||'Unnamed vehicle',v.id]),vehicleId||'');
    const v=current();if(!v)return;
    if(document.activeElement!==$('vehicle-name'))$('vehicle-name').value=v.name||'';
    $('vehicle-body').value=v.bodyType||'';$('vehicle-colour').value=v.colour||'';
    const photos=v.references||[],marked=own(labels).filter(l=>l.detection!=null&&Array.isArray(l.descriptor)),empty=!v.bodyType&&!v.colour&&!photos.length&&!marked.length;
    $('vehicle-summary').textContent=['Profile',M.BODY_TYPES[v.bodyType]?.name,v.colour,plural(photos.length,'photo'),plural(marked.length,'marked sighting')].filter(Boolean).join(' · ');
    if(empty&&!$('vehicle-profile').dataset.opened){$('vehicle-profile').open=true;$('vehicle-profile').dataset.opened='1';}
    const strip=[...photos.map(r=>{
      const fig=node('figure',undefined,'reference');fig.dataset.key='photo:'+r.id;const img=node('img');img.src=r.thumb||'';img.alt=r.located?'Reference photo':'Reference photo (no vehicle found, centre used)';
      fig.append(img,node('figcaption',`${R.colourName(r.descriptor)}${r.located?'':' · centre'}`),button('×',()=>saveProfile({references:(current().references||[]).filter(x=>x.id!==r.id)}),'remove'));fig.lastChild.title='Remove this reference';fig.lastChild.setAttribute('aria-label','Remove reference photo');return fig;
    }),...marked.slice(-12).map(l=>{const fig=node('figure',undefined,'reference marked');fig.dataset.key='label:'+l.key;fig.append(crop(l.hash,l.box,l.hash+':'+l.detection,96,72),node('figcaption',`${R.colourName(l.descriptor)} · marked`));fig.title=`Marked ${new Date(l.capturedAt).toLocaleString()}`;return fig;})];
    SmoothUI.list($('vehicle-references'),strip.length?strip:[Object.assign(node('p','No references yet.','widget-note'),{id:'no-references'})]);
    $('vehicle-note').textContent=empty?'Add a body type, a colour or reference photos so sightings can be verified. Until then, pick the vehicle in each frame yourself; every pick becomes a reference.'
      :'Each vehicle you mark in a frame becomes a reference too. Crops from DriveBC frames match better than phone photos, which have different light and colour.';
  }
  async function saveProfile(patch){
    try{const v={...current(),...patch};await store.saveVehicle(v);vehicles=vehicles.map(x=>x.id===v.id?v:x);lastVersion='';renderVehicles();schedule();}catch(error){fail(error);}
  }
  // A reference photo is located with the same YOLO model, then described like a camera crop.
  async function addPhotos(files){
    const v=current();if(!v||!files.length)return;let added=0;
    for(const file of files){
      try{
        message=`Locating the vehicle in ${file.name}…`;status();
        const blob=file,hash=hex(await crypto.subtle.digest('SHA-256',await blob.arrayBuffer()));wanted.add(hash);
        const found=await browserDetector.request({hash,blob},true);wanted.delete(hash);
        const area=d=>(d.box[2]-d.box[0])*(d.box[3]-d.box[1]),best=(found?.detections||[]).filter(d=>M.VEHICLES.includes(d.label)).sort((a,b)=>area(b)-area(a))[0];
        const box=best?.box||[.1,.1,.9,.9],[descriptor]=await R.describe(blob,[{box}]);if(!descriptor)throw Error('The photo is too small');
        const ref={id:hash.slice(0,16),source:'photo',descriptor,thumb:await thumbnail(blob,box),located:!!best,label:best?.label||null,addedAt:Date.now()};
        await saveProfile({references:[...(current().references||[]).filter(r=>r.id!==ref.id),ref]});added++;
      }catch(error){message=`${file.name}: ${error.message}`;status();}
    }
    if(added){message=`Added ${plural(added,'reference photo')}.`;status();}
  }

  function cameraList(){
    if(cameraCache.revision===archiveRevision&&cameraCache.catalog===catalog)return cameraCache.list;
    // A saved frame keeps its camera's metadata, so frames from cameras outside the catalog still match.
    const byId=new Map(catalog.map(c=>[String(c.id),c]));
    for(const row of archiveMetadata.values()){const c=row.camera;if(c&&!byId.has(String(row.cameraId))&&Number.isFinite(c.lat)&&Number.isFinite(c.lon))byId.set(String(row.cameraId),{...c,id:String(row.cameraId)});}
    cameraCache={revision:archiveRevision,catalog,list:[...byId.values()]};return cameraCache.list;
  }
  function render(){
    const v=current();if(!v)return;
    const mine=own(sessions),ids=new Set(mine.map(s=>s.id)),track=fixes.filter(f=>ids.has(f.session)),marks=own(labels);
    const version=signature([vehicleId,v.name,v.bodyType,v.colour,(v.references||[]).map(r=>r.id),track.length,track.at(-1)?.t,mine.map(s=>[s.id,s.fixes,s.end]),archiveRevision,marks.map(l=>[l.key,l.detection,!!l.descriptor]),radius(),tolerance(),catalog.length,Math.floor(Date.now()/60000)]);
    if(version===lastVersion)return;lastVersion=version;
    result=M.track({vehicle:v,fixes:track,cameras:cameraList(),rows:[...archiveMetadata.values()],labels:marks,options:{radius:radius(),tolerance:tolerance()}});
    renderSummary();renderSightings(track);renderPasses(track);renderSessions(mine);renderLayer(track);prepare();
    notifyPane('gps','Vehicle tracking',[vehicleId,mine.length,result.sightings.map(s=>[s.key,s.status]),result.passes.length],$('gps-updated'),$('gps-pane'));
  }
  function schedule(){clearTimeout(paintTimer);paintTimer=setTimeout(render,250);}
  function renderSummary(){
    const c=result.counts,d=result.decoy,a=result.agreement;
    SmoothUI.list($('gps-summary'),[['PASSES',result.passes.length],['IN VIEW',result.sightings.length],['CONFIRMED',(c.confirmed||0)+(c.labelled||0)],['POSSIBLE',c.possible||0]].map(([label,value])=>{const cell=node('div');cell.dataset.key=label;cell.append(node('strong',String(value)),node('span',label));return cell;}));
    const chance=d.frames?`Chance check: ${d.confirmed} of ${plural(d.frames,'frame')} from these cameras without this vehicle also confirm (${pct(d.rate)}).`:result.passes.length?'Chance check: no frames from these cameras without this vehicle yet.':'';
    const agree=a.labelled?` The automatic pick agrees with ${a.agree} of the ${plural(a.labelled,'frame')} you marked.`:'',hidden=a.hidden?` ${a.falseConfirmed} of ${plural(a.hidden,'frame')} you marked not visible were confirmed automatically.`:'';
    $('gps-checks').textContent=chance+agree+hidden;$('gps-checks').classList.toggle('weak',d.rate>.1||a.falseConfirmed>0);
  }
  function renderSightings(track){
    const list=result.sightings;
    $('gps-frames-note').textContent=!track.length?'No drives for this vehicle yet. Start logging, or import a GPX or CSV track.':list.length?`${plural(list.length,'saved frame')} ${list.length===1?'was':'were'} captured while the vehicle was within ${radius()} m in front of a camera. Pick the vehicle, or Not visible.`:'No saved frame was captured while the vehicle was in view. Turn on capture while you drive, or import Replay the Day frames.';
    SmoothUI.list($('gps-frames'),list.slice(0,40).map(frameItem));
  }
  function frameItem(s){
    const row=archiveMetadata.get(s.key),v=s.verification,label=s.label,item=node('li',undefined,`route-item gps-frame sighting ${s.status}`+(label?' labeled':''));item.dataset.key=s.key;
    const head=node('button',undefined,'route-summary');head.type='button';head.title='Show this camera on the map';
    head.append(node('strong',`${day(s.t)} ${clock(s.t)} · ${s.camera.name||s.cameraId}`),node('span',[STATUS[s.status]+(v.confidence!=null&&!label?' · '+pct(v.confidence):''),s.exact?'capture time':'fetch time only',`${metres(s.distance)} from camera`,s.position?.accuracy!=null&&`GPS ±${Math.round(s.position.accuracy)} m`,FACING[s.expectedFacing]].filter(Boolean).join(' · ')));
    head.onclick=()=>AppMap.frame([[s.camera.lat,s.camera.lon]],{maxZoom:14});
    const choices=node('div',undefined,'route-hops');
    for(const d of v.detections.slice(0,6)){
      const b=node('button',undefined,'route-hop gps-choice'+(!label&&s.detection===d.index?' best':''));b.type='button';b.dataset.key=String(d.index);b.setAttribute('aria-pressed',String(label?.detection===d.index));
      b.title=`Mark vehicle ${d.index+1} as ${current()?.name||'this vehicle'}`+(Object.keys(d.cues).length?` · ${Object.entries(d.cues).map(([k,x])=>k+' '+pct(x)).join(', ')}`:'');
      b.append(crop(s.hash,d.box,s.hash+':'+d.index),node('span',[`${d.index+1} · ${d.label}`,d.colour,d.score!=null&&pct(d.score)].filter(Boolean).join(' · ')));b.onclick=()=>setLabel(s,d.index);choices.append(b);
    }
    if(v.detections.length>6)choices.append(node('span',`+${v.detections.length-6} less likely`,'widget-note'));
    const hidden=button('Not visible',()=>setLabel(s,null),'gps-hidden');hidden.dataset.key='none';hidden.setAttribute('aria-pressed',String(label?.detection===null));
    const inspect=button('Inspect',async()=>{if(row)await seekHistory(row.observedAt);showCamera(s.cameraId);},'gps-hidden');inspect.dataset.key='inspect';inspect.title='Open this frame in the camera panel (pauses live updates)';
    choices.append(hidden,inspect);item.append(head,choices);return item;
  }
  async function setLabel(s,index){
    try{
      const key=vehicleId+'|'+s.key,prior=labels.find(l=>l.key===key),row=archiveMetadata.get(s.key);
      if(prior&&prior.detection===index){await store.removeLabel(key);labels=labels.filter(l=>l.key!==key);}
      else{
        let descriptor=index==null?null:row?.routeFeatures?.byDetection?.[index]||null;
        if(index!=null&&!descriptor){const blob=await photoArchive.photo(row.hash);if(blob)[descriptor]=await R.describe(blob,[row.detections[index]]);}
        const label=G.labelRecord(s,row,index,{vehicle:vehicleId,descriptor});await store.saveLabel(label);labels=[...labels.filter(l=>l.key!==key),label];
      }
      lastVersion='';render();renderVehicles();
    }catch(error){fail(error);}
  }
  function renderPasses(track){
    const list=result.passes.slice().reverse();
    $('gps-passes-note').textContent=!track.length?'':list.length?`${plural(list.length,'camera pass','camera passes')}. DriveBC keeps about 24 hours of images per camera.`:'No camera was within range and facing the route.';
    SmoothUI.list($('gps-passes'),list.slice(0,60).map(p=>{
      const item=node('li',undefined,'route-item gps-pass '+(p.status||'none'));item.dataset.key=p.cameraId+':'+p.start;const live=Date.now()<p.replay.availableUntil;
      item.append(node('strong',`${p.camera.name||p.cameraId} · ${day(p.start)} ${clock(p.start)}–${clock(p.end)}`),
        node('span',[`closest ${metres(p.closest.distance)}`,FACING[p.expectedFacing],p.frames.length?`${plural(p.frames.length,'frame')} in view · ${SHORT[p.status]}`:p.nearest?`no frame in view; nearest ${ago(p.nearest.offset)}`:'no saved frame',live?`replay until ${day(p.replay.availableUntil)} ${clock(p.replay.availableUntil)}`:'replay expired'].filter(Boolean).join(' · ')));
      if(live)item.append(link('Replay the Day on DriveBC ↗',p.replay.page));return item;
    }));
  }
  function renderSessions(mine){
    SmoothUI.list($('gps-sessions'),mine.length?mine.slice().reverse().map(s=>{
      const kept=fixes.filter(f=>f.session===s.id),item=node('li',undefined,'route-item gps-session');item.dataset.key=s.id;
      item.append(node('strong',`${s.name&&s.name!==s.id?s.name:day(s.start)} · ${clock(s.start)}–${s.end==null?'now':clock(s.end)}`),
        node('span',[`${plural(s.fixes||0,'fix','fixes')} near cameras`,metres(G.trackLength(kept))+' of road near cameras',s.imported&&'imported',s.trimmed&&'trimmed to camera zones',s.interrupted&&'page closed while logging'].filter(Boolean).join(' · ')));
      const actions=node('div',undefined,'gps-actions'),file=s.id.replace(/[^\w.-]+/g,'_');
      // Export what is stored now, not the list from the last repaint, so the newest fix is included.
      const saved=async(write,ext,type)=>{try{download(`locaintel-${file}.${ext}`,write(await store.fixes(s.id)),type);}catch(error){fail(error);}};
      actions.append(button('GPX',()=>saved(f=>G.gpx(f,{name:s.name||s.id}),'gpx','application/gpx+xml')),button('CSV',()=>saved(G.csv,'csv','text/csv')),
        button('Delete',async()=>{if(logger.session?.id===s.id||!confirm('Delete this drive and its labels from this browser?'))return;try{await store.remove(s.id);await reload();}catch(error){fail(error);}}));
      actions.lastChild.disabled=logger.session?.id===s.id;item.append(actions);return item;
    }):[Object.assign(node('li','No drives yet. Start logging, or import a GPX or CSV file exported from a phone.','widget-note'),{id:'gps-no-sessions'})]);
  }

  // Map layer: the GPS pieces kept near cameras, a ring per pass coloured by its best
  // sighting, and this device's position while logging (shown, not stored).
  function renderLayer(track){
    const layer=AppMap.layer('tracking');if(!layer)return;layer.clearLayers();liveMarker=null;
    let piece=[];const flush=()=>{if(piece.length>1)L.polyline(piece,{pane:'tracking',color:'#ffd36b',weight:3,opacity:.75}).addTo(layer);piece=[];};
    for(let i=0;i<track.length;i++){const f=track[i],p=track[i-1];if(p&&(f.t-p.t>60000||f.session!==p.session))flush();piece.push([f.lat,f.lon]);}flush();
    for(const p of result.passes){
      const colour=COLOUR[p.status]||'#8c9da7',solid=['labelled','confirmed'].includes(p.status);
      L.circleMarker([p.camera.lat,p.camera.lon],{pane:'tracking',radius:solid?9:8,color:colour,weight:solid?3:2,dashArray:p.frames.length?null:'3 4',fillColor:colour,fillOpacity:solid?.55:0})
        .bindTooltip(`${clock(p.closest.t)} · ${p.camera.name||p.cameraId} · ${p.frames.length?SHORT[p.status]:'passed, no frame in view'}`).on('click',()=>focusPass(p)).addTo(layer);
    }
    renderLive();
  }
  function renderLive(){
    const layer=AppMap.layer('tracking'),s=logger.state();if(!layer)return;
    if(!s.active||!s.last){liveMarker?.remove();liveMarker=null;return;}
    if(!liveMarker)liveMarker=L.circleMarker([s.last.lat,s.last.lon],{pane:'tracking',radius:7,color:'#0b1115',weight:2,fillColor:'#4fc3ff',fillOpacity:1}).bindTooltip('This device now. Positions are stored only near cameras.');
    liveMarker.setLatLng([s.last.lat,s.last.lon]);if(!layer.hasLayer(liveMarker))liveMarker.addTo(layer);
  }
  function focusPass(p){
    openPanel('track');const frame=p.frames[0],target=frame&&$('gps-frames').querySelector(`[data-key="${CSS.escape(frame.key)}"]`)||$('gps-passes').querySelector(`[data-key="${CSS.escape(p.cameraId+':'+p.start)}"]`);
    if(target){target.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'center'});target.classList.add('pane-flash');setTimeout(()=>target.classList.remove('pane-flash'),1000);}
  }

  // A photo the detector has already seen returns its cached result without calling
  // onResult again, so that result is saved to the frame here.
  function detect(source){const done=browserDetector.get(source.hash);if(done?.status==='ready'){recordDetection(source,done);return;}wanted.add(source.hash);browserDetector.request(source);}
  // In-view frames need detections and colour descriptors before they can be verified.
  // Imported replay frames have neither; live frames usually have detections already.
  async function prepare(){
    if(preparing||!result||document.hidden)return;preparing=true;
    try{
      for(const s of result.sightings.slice(0,60)){
        const row=archiveMetadata.get(s.key);if(!row)continue;
        if(!Array.isArray(row.detections)){if(!wanted.has(row.hash)){const blob=await photoArchive.photo(row.hash);if(blob)detect({hash:row.hash,blob,cameraId:String(row.cameraId),width:row.width,height:row.height});}continue;}
        if(row.routeFeatures?.version===R.VERSION||describing.has(row.key)||!row.detections.length)continue;
        describing.add(row.key);
        try{const blob=await photoArchive.photo(row.hash);if(!blob)continue;const features=await R.describe(blob,row.detections);await photoArchive.saveRouteFeatures(row.key,{version:R.VERSION,byDetection:Object.fromEntries(features.map((f,i)=>[i,f]))});}
        finally{describing.delete(row.key);}
      }
    }catch(error){console.warn('Tracking features unavailable',error);}
    finally{preparing=false;}
  }

  // While logging, cameras within the relevance radius are fetched as you arrive, every
  // 30 s, and for 3 minutes after you leave: a capture made while you were in view stays
  // DriveBC's current image until the next one, so a fetch shortly afterwards still has it.
  function noteCapture(fix){
    if(!index||!$('gps-capture').checked||!logger.active)return;
    for(const {camera}of index.near(fix)){const id=String(camera.id),fresh=!capture.has(id);capture.set(id,Date.now()+180000);if(fresh)fetchCamera(id);}
  }
  function fetchCamera(id){const c=data?.cameras.find(c=>String(c.id)===id);if(c&&isLive())cameraSources.load(c,true).then(()=>scheduleDetection()).catch(()=>{});}
  setInterval(()=>{const now=Date.now();for(const [id,until]of capture)if(until<now)capture.delete(id);if($('gps-capture').checked)for(const id of capture.keys())fetchCamera(id);if(logger.active)status();},30000);
  window.trackingCameras=()=>{if(!capture.size||!$('gps-capture').checked)return [];return (data?.cameras||[]).filter(c=>capture.has(String(c.id)));};
  window.trackingWants=hash=>wanted.has(hash);

  for(const [id,options]of [['vehicle-body',[['Unknown',''],...Object.entries(M.BODY_TYPES).map(([k,b])=>[b.name,k])]],['vehicle-colour',[['Unknown',''],...M.COLOURS.map(c=>[c[0].toUpperCase()+c.slice(1),c])]]])SmoothUI.options($(id),options,'');
  $('vehicle-select').addEventListener('change',()=>{vehicleId=$('vehicle-select').value;remember();lastVersion='';renderVehicles();render();status();});
  $('vehicle-name').addEventListener('change',()=>saveProfile({name:$('vehicle-name').value.trim()||'Unnamed vehicle'}));
  $('vehicle-body').addEventListener('change',()=>saveProfile({bodyType:$('vehicle-body').value}));
  $('vehicle-colour').addEventListener('change',()=>saveProfile({colour:$('vehicle-colour').value}));
  $('vehicle-photos').addEventListener('change',async event=>{const files=[...event.target.files];event.target.value='';await addPhotos(files);});
  $('vehicle-add').onclick=async()=>{try{const v=G.vehicle('vehicle-'+Date.now().toString(36),{name:`Vehicle ${vehicles.length+1}`});await store.saveVehicle(v);vehicleId=v.id;remember();await reload();$('vehicle-profile').open=true;$('vehicle-name').focus();$('vehicle-name').select();}catch(error){fail(error);}};
  $('vehicle-delete').onclick=async()=>{const v=current();if(!v||logger.active||!confirm(`Delete ${v.name||'this vehicle'}, its drives and its labels from this browser?`))return;try{await store.removeVehicle(v.id);vehicleId=null;await reload();}catch(error){fail(error);}};
  $('vehicle-zoom').onclick=()=>{
    const mine=new Set(own(sessions).map(s=>s.id)),points=[...fixes.filter(f=>mine.has(f.session)).map(f=>[f.lat,f.lon]),...(result?.passes||[]).map(p=>[p.camera.lat,p.camera.lon])];
    if(!AppMap.enabled('tracking')){$('layer-tracking').checked=true;$('layer-tracking').dispatchEvent(new Event('change'));}
    if(points.length)AppMap.frame(points,{maxZoom:14});else{message='Nothing to show yet: this vehicle has no drives.';status();}
  };
  $('gps-own').addEventListener('change',status);
  $('gps-capture').addEventListener('change',()=>{if(!$('gps-capture').checked)capture.clear();status();});
  // Start requests location first, synchronously in this tap, so the browser can show its prompt.
  // Drives exist only in this browser until exported; ask it not to evict them under storage pressure.
  $('gps-start').onclick=async()=>{if(!$('gps-own').checked||!index||!current())return;message='';const started=logger.start({vehicle:vehicleId});navigator.storage?.persist?.().catch(()=>{});try{await started;await reload();}catch(error){fail(error);}};
  $('gps-stop').onclick=async()=>{try{await logger.stop();}catch(error){fail(error);}await reload();};
  for(const id of ['gps-radius','gps-tolerance'])$(id).addEventListener('change',schedule);
  $('gps-import').addEventListener('change',async event=>{
    const file=event.target.files[0];if(!file)return;
    try{
      if(!index)throw Error('The camera catalog is still loading; try again in a moment');
      const parsed=G.parseTrack(await file.text(),{name:file.name}),kept=M.compact(parsed,index);
      if(!kept.length)throw Error(`No part of ${file.name} passes within ${M.RELEVANT_RADIUS} m of a DriveBC camera, so nothing was kept`);
      await store.importFixes(kept,{name:file.name,vehicle:vehicleId});
      message=`Imported ${file.name}: kept ${kept.length} of ${plural(parsed.length,'fix','fixes')}, those within ${M.RELEVANT_RADIUS} m of a camera.`;await reload();
    }catch(error){fail(error);}finally{event.target.value='';}
  });
  // Frames saved by python -m pipeline.gps_replay: <camera id>/<UTC stamp>.jpg.
  $('replay-import').addEventListener('change',async event=>{
    const files=[...event.target.files];event.target.value='';let saved=0,skipped=0;
    for(const file of files){
      const m=/(?:^|\/)(\d{1,7})\/(\d{14})\.jpe?g$/i.exec(file.webkitRelativePath||file.name),camera=m&&cameraList().find(c=>String(c.id)===m[1]);
      if(!camera){skipped++;continue;}
      const s=m[2],t=Date.UTC(+s.slice(0,4),+s.slice(4,6)-1,+s.slice(6,8),+s.slice(8,10),+s.slice(10,12),+s.slice(12,14));
      try{
        const bytes=await file.arrayBuffer(),hash=hex(await crypto.subtle.digest('SHA-256',bytes)),blob=new Blob([bytes],{type:'image/jpeg'}),bitmap=await createImageBitmap(blob),width=bitmap.width,height=bitmap.height;bitmap.close();
        await archiveReady;await photoArchive.saveFrame(camera,{hash,blob,width,height,modified:new Date(t).toUTCString(),checked:new Date(t).toISOString()},{observedAt:t});
        detect({hash,blob,cameraId:String(camera.id),width,height});saved++;
      }catch(_){skipped++;}
    }
    message=`Imported ${plural(saved,'replay frame')}${skipped?`; skipped ${skipped} that are not Replay the Day images of a known camera`:''}. Detection runs in the background.`;status();
  });
  $('gps-export').onclick=async()=>{
    await reload();lastVersion='';render();const v=current();if(!v)return;
    download(`locaintel-${(v.name||v.id).toLowerCase().replace(/[^\w]+/g,'-')}-${new Date().toISOString().slice(0,19).replace(/[-:]/g,'')}.json`,JSON.stringify(G.exportSet({vehicle:v,sessions,labels,result}),null,1)+'\n','application/json');
  };
  const useCatalog=rows=>{if(!Array.isArray(rows)||!rows.length)return;catalog=rows;index=M.cameraIndex(rows);updateTrimmable();lastVersion='';schedule();status();};
  fetch('data/cameras.json').then(r=>r.ok?r.json():[]).then(useCatalog).catch(()=>{});
  store.recover().then(reload).catch(fail);status();
  // Without the catalog file (offline, no cache), the app's own camera list will do.
  setInterval(()=>{if(!index&&data?.cameras?.length)useCatalog(data.cameras);if(archiveRevision!==lastRevision){lastRevision=archiveRevision;schedule();}prepare();},Math.max(1500,powerProfile.paintMs*10));
  window.gpsLog={store,logger,reload,get result(){return result;}};
})();

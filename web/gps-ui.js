/* GPS drive log widget. Reads the shared local archive from app.js; GPS fixes live
   in their own IndexedDB store and leave this browser only as files you export. */
(function(){
  const G=GpsLog,store=new G.GpsStore();
  let sessions=[],fixes=[],labels=new Map(),catalog=[],lastVersion='',paintTimer=null,reloadTimer=null,lastRevision=-1,message='';
  const crops=new Map();
  const clock=t=>new Date(t).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const day=t=>new Date(t).toLocaleDateString([],{month:'short',day:'numeric'});
  const metres=m=>m<1000?Math.round(m)+' m':(m/1000).toFixed(1)+' km';
  const FACING={toward_camera:'front expected (driving toward camera)',away_from_camera:'rear expected (driving away)',image_left:'side expected (moving left)',image_right:'side expected (moving right)'};
  const WAKE={held:'screen kept awake',idle:'',released:'screen lock dropped while hidden; keep this tab in front',unsupported:'this browser cannot keep the screen awake; keep it on with this tab in front'};
  const radius=()=>Number($('gps-radius').value);
  const logger=new G.GpsLogger({store,onChange:state=>{
    if(state.last&&!fixes.includes(state.last)){fixes.push(state.last);if(fixes.length>1&&fixes.at(-2).t>state.last.t)fixes.sort((a,b)=>a.t-b.t);}
    status();schedule();if(!reloadTimer)reloadTimer=setTimeout(()=>{reloadTimer=null;reload();},4000);
  }});
  function fail(error){message=error?.message||String(error);status();}
  function download(name,text,type){
    const url=URL.createObjectURL(new Blob([text],{type})),a=Object.assign(document.createElement('a'),{href:url,download:name});
    document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function status(){
    const s=logger.state(),wake=WAKE[s.wakeLock]??`screen lock unavailable (${s.wakeLock.replace(/^error: /,'')}); keep the screen on`;
    $('gps-status').textContent=s.active?['Logging',s.fixes+(s.fixes===1?' fix':' fixes'),s.last?`last ${clock(s.last.t)}${s.last.accuracy!=null?' ±'+Math.round(s.last.accuracy)+' m':''}`:'waiting for the first fix',wake,s.error].filter(Boolean).join(' · '):s.error||message||'Not logging.';
    $('gps-status').classList.toggle('weak',!!(s.error||!s.active&&message||s.active&&s.wakeLock!=='held'));
    $('gps-start').disabled=s.active||!$('gps-own').checked;$('gps-stop').disabled=!s.active;$('gps-own').disabled=s.active;
  }
  async function reload(){
    try{const [s,f,l]=await Promise.all([store.sessions(),store.fixes(),store.labels()]);sessions=s;fixes=f;labels=new Map(l.map(x=>[x.key,x]));lastVersion='';schedule();}
    catch(error){fail(error);}
  }
  function crop(row,d,i){
    const canvas=node('canvas');canvas.width=72;canvas.height=54;const key=row.hash+':'+i;
    (async()=>{try{let bitmap=crops.get(key);if(!bitmap){const blob=await photoArchive.photo(row.hash);if(!blob)return;const image=await createImageBitmap(blob),[x,y,xx,yy]=d.box;
      bitmap=await createImageBitmap(image,x*image.width,y*image.height,Math.max(1,(xx-x)*image.width),Math.max(1,(yy-y)*image.height));image.close();crops.set(key,bitmap);if(crops.size>120){const [k,b]=crops.entries().next().value;b.close?.();crops.delete(k);}}
      canvas.getContext('2d').drawImage(bitmap,0,0,72,54);}catch(_){}})();
    return canvas;
  }
  async function setLabel(match,row,index){
    try{
      const current=labels.get(match.key);
      if(current&&current.detection===index){await store.removeLabel(match.key);labels.delete(match.key);}
      else{const label=G.labelRecord(match,row,index);await store.saveLabel(label);labels.set(match.key,label);}
      lastVersion='';render();
    }catch(error){fail(error);}
  }
  function sessionItem(s){
    const own=fixes.filter(f=>f.session===s.id),item=node('li',undefined,'route-item gps-session');item.dataset.key=s.id;
    item.append(node('strong',`${s.name&&s.name!==s.id?s.name:day(s.start)} · ${clock(s.start)}–${s.end==null?'now':clock(s.end)}`),
      node('span',[`${s.fixes||0} fix${s.fixes===1?'':'es'}`,metres(G.trackLength(own)),s.imported&&'imported',s.interrupted&&'page closed while logging'].filter(Boolean).join(' · ')));
    const actions=node('div',undefined,'gps-actions'),file=s.id.replace(/[^\w.-]+/g,'_');
    const button=(text,fn)=>{const b=node('button',text);b.type='button';b.onclick=fn;return b;};
    actions.append(button('GPX',()=>download(`locaintel-${file}.gpx`,G.gpx(own,{name:s.name||s.id}),'application/gpx+xml')),button('CSV',()=>download(`locaintel-${file}.csv`,G.csv(own),'text/csv')),
      button('Delete',async()=>{if(logger.session?.id===s.id||!confirm('Delete this drive and its own-car labels from this browser?'))return;try{await store.remove(s.id);await reload();}catch(error){fail(error);}}));
    actions.lastChild.disabled=logger.session?.id===s.id;item.append(actions);return item;
  }
  function frameItem(match){
    const row=archiveMetadata.get(match.key),label=labels.get(match.key),item=node('li',undefined,'route-item gps-frame'+(label?' labeled':''));item.dataset.key=match.key;
    item.append(node('strong',`${match.camera.name||match.cameraId} · ${clock(match.t)}`),
      node('span',[`car ${metres(match.distance)} from camera`,match.position.accuracy!=null&&`GPS ±${Math.round(match.position.accuracy)} m`,FACING[match.expectedFacing],label&&(label.detection==null?'labeled: not visible':`labeled: vehicle ${label.detection+1}`)].filter(Boolean).join(' · ')));
    const choices=node('div',undefined,'route-hops');
    for(const [i,d]of (row.detections||[]).entries()){
      if(!G.VEHICLES.includes(d.label))continue;
      const b=node('button',undefined,'route-hop gps-choice');b.type='button';b.dataset.key=String(i);b.setAttribute('aria-pressed',String(label?.detection===i));
      const facing=row.facing?.byDetection?.[i]?.facing;b.title=`Mark vehicle ${i+1} as my car`;
      b.append(crop(row,d,i),node('span',`${i+1} · ${d.label}${facing&&facing===match.expectedFacing?' · facing fits':''}`));b.onclick=()=>setLabel(match,row,i);choices.append(b);
    }
    const hidden=node('button','Not visible','gps-hidden');hidden.type='button';hidden.dataset.key='none';hidden.setAttribute('aria-pressed',String(label?.detection===null));hidden.onclick=()=>setLabel(match,row,null);
    const inspect=node('button','Inspect','gps-hidden');inspect.type='button';inspect.dataset.key='inspect';inspect.onclick=()=>{selectCamera(match.cameraId);$('detail-pane').scrollIntoView({behavior:'smooth',block:'center'});};
    choices.append(hidden,inspect);item.append(choices);return item;
  }
  function passItem(p,saved){
    const item=node('li',undefined,'route-item');item.dataset.key=p.cameraId+':'+p.start;const live=Date.now()<p.replay.availableUntil;
    item.append(node('strong',`${p.camera.name||p.cameraId} · ${day(p.start)} ${clock(p.start)}–${clock(p.end)}`),
      node('span',[`closest ${metres(p.closest.distance)}`,FACING[p.expectedFacing],saved?`${saved} saved frame${saved>1?'s':''}`:'no saved frame',live?`replay until ${day(p.replay.availableUntil)} ${clock(p.replay.availableUntil)}`:'replay expired'].filter(Boolean).join(' · ')));
    if(live)item.append(link('Replay the Day on DriveBC ↗',p.replay.page));return item;
  }
  function render(){
    const rows=[...archiveMetadata.values()],r=radius(),cams=catalog.length?catalog:data?.cameras||[];
    const version=signature([r,fixes.length,fixes.at(-1)?.t,sessions.map(s=>[s.id,s.fixes,s.end]),archiveRevision,[...labels.values()].map(l=>[l.key,l.detection]),cams.length,Math.floor(Date.now()/60000)]);
    if(version===lastVersion)return;lastVersion=version;
    SmoothUI.list($('gps-sessions'),sessions.length?sessions.slice().reverse().map(sessionItem):[Object.assign(node('li','No drives yet. Start logging, or import a GPX/CSV file exported from your phone.','widget-note'),{id:'gps-no-sessions'})]);
    const frames=G.frames(rows,fixes,{radius:r}),passes=G.passes(cams,fixes,{radius:r}),done=frames.filter(m=>labels.has(m.key)).length;
    $('gps-frames-note').textContent=!fixes.length?'':frames.length?`${frames.length} saved frame${frames.length>1?'s were':' was'} captured while your car was within ${r} m in front of the camera · ${done} labeled. Pick your car, or Not visible.`:'No saved frame was captured while your car was in view. Sample the corridor while you drive, or use Replay the Day below.';
    SmoothUI.list($('gps-frames'),frames.slice(0,40).map(frameItem));
    $('gps-passes-note').textContent=!fixes.length?'':passes.length?`${passes.length} camera pass${passes.length>1?'es':''}. DriveBC keeps about 24 hours of images per camera.`:'No catalog camera was within range and facing your route.';
    SmoothUI.list($('gps-passes'),passes.slice().reverse().slice(0,60).map(p=>passItem(p,frames.filter(m=>m.cameraId===p.cameraId&&m.t>=p.start-G.defaults.maxGap&&m.t<=p.end+G.defaults.maxGap).length)));
    notifyPane('gps','GPS drive log',[sessions.length,frames.length,labels.size,passes.length],$('gps-updated'),$('gps-pane'));
  }
  function schedule(){clearTimeout(paintTimer);paintTimer=setTimeout(render,250);}

  $('gps-own').addEventListener('change',status);
  $('gps-start').onclick=async()=>{if(!$('gps-own').checked)return;message='';try{await logger.start();await reload();}catch(error){fail(error);}};
  $('gps-stop').onclick=async()=>{try{await logger.stop();}catch(error){fail(error);}await reload();};
  $('gps-radius').addEventListener('change',schedule);
  $('gps-import').addEventListener('change',async event=>{
    const file=event.target.files[0];if(!file)return;
    try{const parsed=G.parseTrack(await file.text(),{name:file.name});await store.importFixes(parsed,{name:file.name});message=`Imported ${parsed.length} fixes from ${file.name}.`;await reload();status();}
    catch(error){fail(error);}finally{event.target.value='';}
  });
  $('gps-export').onclick=()=>{
    const r=radius(),cams=catalog.length?catalog:data?.cameras||[];
    const set=G.evaluationSet({sessions,labels:[...labels.values()],frames:G.frames([...archiveMetadata.values()],fixes,{radius:r}),passes:G.passes(cams,fixes,{radius:r})});
    download(`locaintel-own-vehicle-${new Date().toISOString().slice(0,19).replace(/[-:]/g,'')}.json`,JSON.stringify(set,null,1)+'\n','application/json');
  };
  fetch('data/cameras.json').then(r=>r.ok?r.json():[]).then(rows=>{catalog=rows;lastVersion='';schedule();}).catch(()=>{});
  store.recover().then(reload).catch(fail);status();
  setInterval(()=>{if(archiveRevision!==lastRevision){lastRevision=archiveRevision;schedule();}},Math.max(1500,powerProfile.paintMs*10));
  window.gpsLog={store,logger,reload};
})();

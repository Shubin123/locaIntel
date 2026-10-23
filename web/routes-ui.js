/* Vehicle routes widget. Reads the shared local archive and timeline from app.js. */
(function(){
  const R=VehicleRoutes;let routeMap=null,routeLayer=null,flowLayer=null,cameraLayer=null,selectedRoute=null,lastVersion='',featureRunning=false,paintTimer=null,lastRevision=-1;
  const thumbs=new Map();
  const confidenceLabel=c=>c>=.6?'high':c>=.35?'medium':'low';
  const time=t=>new Date(t).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});

  // Colour descriptors for every saved frame with detections, computed once and stored with the frame.
  async function computeFeatures(){
    if(featureRunning||document.hidden)return;featureRunning=true;
    try{
      const pending=[...archiveMetadata.values()].filter(r=>r.detections?.length&&r.routeFeatures?.version!==R.VERSION).sort((a,b)=>b.observedAt-a.observedAt);
      for(const row of pending.slice(0,6)){
        const blob=await photoArchive.photo(row.hash);if(!blob)continue;
        const features=await R.describe(blob,row.detections);
        await photoArchive.saveRouteFeatures(row.key,{version:R.VERSION,byDetection:Object.fromEntries(features.map((f,i)=>[i,f]))});
      }
    }catch(error){console.warn('Route features unavailable',error);}
    finally{featureRunning=false;}
  }
  // Full DriveBC catalog: routes span cameras that may not be in the published analysis.
  let catalog=[];fetch('data/cameras.json').then(r=>r.ok?r.json():[]).then(rows=>{catalog=rows;lastVersion='';schedule();}).catch(()=>{});
  const inCorridor=c=>!$('route-highway').value||String(c?.highway)===$('route-highway').value;
  function corridorCameras(){return (catalog.length?catalog:data?.cameras||[]).filter(inCorridor);}
  function populateCorridors(){
    const cams=catalog.length?catalog:data?.cameras;if(!cams)return;const counts=new Map();for(const c of cams)if(c.highway)counts.set(String(c.highway),(counts.get(String(c.highway))||0)+1);
    const options=[...counts].filter(([,n])=>n>=3).sort((a,b)=>(parseInt(a[0])||999)-(parseInt(b[0])||999)||a[0].localeCompare(b[0]));
    const current=$('route-highway').value,wanted=signature(options);if($('route-highway').dataset.version===wanted)return;$('route-highway').dataset.version=wanted;
    $('route-highway').replaceChildren(new Option('All highways',''),...options.map(([h,n])=>new Option(`${/^\d/.test(h)?'Highway '+h:h} · ${n} cameras`,h)));$('route-highway').value=options.some(([h])=>h===current)?current:'';
  }
  function initMap(){
    if(routeMap||!window.L)return;
    routeMap=L.map('routes-map',{scrollWheelZoom:false}).setView([49.6,-122.5],7);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(routeMap);
    cameraLayer=L.layerGroup().addTo(routeMap);flowLayer=L.layerGroup().addTo(routeMap);routeLayer=L.layerGroup().addTo(routeMap);
  }
  // Short arrow from a camera toward a compass bearing; length grows with vehicle share.
  function arrow(lat,lon,bearing,km,colour,weight){
    const rad=Math.PI/180,dLat=km/111*Math.cos(bearing*rad),dLon=km/(111*Math.cos(lat*rad))*Math.sin(bearing*rad),end=[lat+dLat,lon+dLon];
    const head=(side)=>{const b=bearing+180+side*28,h=km*.35;return [end[0]+h/111*Math.cos(b*rad),end[1]+h/(111*Math.cos(lat*rad))*Math.sin(b*rad)];};
    return [L.polyline([[lat,lon],end],{color:colour,weight,opacity:.9}),L.polyline([head(-1),end,head(1)],{color:colour,weight,opacity:.9})];
  }
  function thumbnail(hop){
    const canvas=node('canvas');canvas.width=64;canvas.height=48;canvas.title=`${hop.camera.name||hop.cameraId} · ${new Date(hop.t).toLocaleString()}`;
    const key=hop.hash+':'+hop.index;
    (async()=>{try{let bitmap=thumbs.get(key);if(!bitmap){const blob=await photoArchive.photo(hop.hash);if(!blob)return;const image=await createImageBitmap(blob),[x,y,xx,yy]=hop.box;
      bitmap=await createImageBitmap(image,x*image.width,y*image.height,Math.max(1,(xx-x)*image.width),Math.max(1,(yy-y)*image.height));image.close();thumbs.set(key,bitmap);if(thumbs.size>120){const [k,b]=thumbs.entries().next().value;b.close?.();thumbs.delete(k);}}
      canvas.getContext('2d').drawImage(bitmap,0,0,64,48);}catch(_){}})();
    return canvas;
  }
  function render(){
    if(!data)return;populateCorridors();initMap();
    const end=currentTime(),windowMs=Number($('route-window').value),minConfidence=Number($('route-confidence').value),byId=new Map(catalog.map(c=>[String(c.id),c]));
    const rows=[...archiveMetadata.values()].filter(r=>r.observedAt<=end&&r.observedAt>=end-windowMs&&r.detections&&inCorridor(r.camera||byId.get(r.cameraId)));
    const sightings=R.sightings(rows.map(r=>({...r,camera:{...byId.get(r.cameraId),...Object.fromEntries(Object.entries(r.camera||{}).filter(([,v])=>v!=null))}})));
    const found=R.routes(sightings).filter(r=>r.confidence>=minConfidence),flows=R.flows(sightings);
    const version=signature([end-end%30000,windowMs,minConfidence,$('route-highway').value,$('route-flows').checked,selectedRoute,sightings.length,found.map(r=>[r.id,r.hops.length]),flows.map(f=>[f.cameraId,f.bins])]);
    if(version===lastVersion)return;lastVersion=version;
    const headed=sightings.filter(s=>s.heading!=null).length,described=sightings.filter(s=>s.colour).length;
    SmoothUI.list($('route-totals'),[['ROUTES',found.length],['SIGHTINGS',sightings.length],['WITH HEADING',sightings.length?Math.round(headed/sightings.length*100)+'%':'—'],['CAMERAS',new Set(rows.map(r=>r.cameraId)).size]].map(([label,value])=>{const cell=node('div');cell.dataset.key=label;cell.append(node('strong',String(value)),node('span',label));return cell;}));
    if(routeMap){
      cameraLayer.clearLayers();flowLayer.clearLayers();routeLayer.clearLayers();
      const points=[];
      for(const c of corridorCameras()){if(!Number.isFinite(c.lat))continue;points.push([c.lat,c.lon]);L.circleMarker([c.lat,c.lon],{radius:3,color:'#92a2ac',weight:1,fillOpacity:.8}).bindTooltip(c.name).addTo(cameraLayer);}
      if($('route-flows').checked)for(const f of flows){
        for(const [part,colour]of [[f.dominant,'#88e2f2'],[f.secondary,'#5d8fa0']]){if(!part||part.share<.15)continue;for(const layer of arrow(f.lat,f.lon,part.bearing,1.2+part.share*2.5,colour,2+Math.min(5,f.headed*part.share/2)))layer.bindTooltip(`${f.camera.name||f.cameraId}: ${Math.round(part.share*100)}% heading ${part.direction} (${f.headed} vehicles with heading)`).addTo(flowLayer);}
      }
      found.forEach((route,i)=>{
        const age=Math.min(1,(end-route.end)/windowMs),active=selectedRoute===route.id,colour=active?'#ffd36b':`hsl(${95-age*60},70%,${65-age*25}%)`;
        const line=L.polyline(route.hops.map(h=>[h.lat,h.lon]),{color:colour,weight:active?6:3,opacity:active?1:.85-age*.4,dashArray:route.confidence<.35?'6 6':null}).addTo(routeLayer);
        line.bindTooltip(`Route ${i+1}: ${route.colour} ${route.label} heading ${route.heading} · ${time(route.start)}–${time(route.end)}`);line.on('click',()=>{selectedRoute=active?null:route.id;lastVersion='';render();});
        for(const h of route.hops)L.circleMarker([h.lat,h.lon],{radius:active?6:4,color:colour,fillColor:colour,fillOpacity:1,weight:1}).bindTooltip(`${h.camera.name||h.cameraId} · ${time(h.t)}`).addTo(routeLayer);
      });
      const focus=found.find(r=>r.id===selectedRoute);
      if(focus)routeMap.fitBounds(focus.hops.map(h=>[h.lat,h.lon]),{padding:[40,40],maxZoom:13});
      else{
        // Frame the cameras that actually have sightings; fall back to the corridor.
        const seen=[...new Map(sightings.map(s=>[s.cameraId,[s.lat,s.lon]])).values()],target=seen.length?seen:points,key=$('route-highway').value+':'+seen.length;
        if(target.length&&$('routes-map').dataset.framed!==key){$('routes-map').dataset.framed=key;routeMap.fitBounds(target,{padding:[40,40],maxZoom:11});}
      }
    }
    const items=found.slice(0,30).map((route,i)=>{
      const item=node('li',undefined,'route-item'+(selectedRoute===route.id?' active':''));item.dataset.key=route.id;
      const button=node('button',undefined,'route-summary');button.type='button';button.setAttribute('aria-pressed',String(selectedRoute===route.id));
      button.append(node('strong',`${i+1}. ${route.colour[0].toUpperCase()+route.colour.slice(1)} ${route.label} → ${route.heading}`),node('span',`${route.hops.length} cameras · ${(route.distance/1000).toFixed(1)} km · ${time(route.start)}–${time(route.end)}${route.speed?` · ~${Math.round(route.speed*3.6)} km/h`:''} · ${confidenceLabel(route.confidence)} match`));
      button.onclick=()=>{selectedRoute=selectedRoute===route.id?null:route.id;lastVersion='';render();};
      const hops=node('div',undefined,'route-hops');for(const h of route.hops){const hop=node('button',undefined,'route-hop');hop.type='button';hop.title=`Inspect ${h.camera.name||h.cameraId}`;hop.append(thumbnail(h),node('span',`${time(h.t)} · ${h.camera.name||h.cameraId}`));hop.onclick=()=>{selectCamera(h.cameraId);$('detail-pane').scrollIntoView({behavior:'smooth',block:'center'});};hops.append(hop);}
      item.append(button,hops);return item;
    });
    SmoothUI.list($('route-list'),items.length?items:[Object.assign(node('li',sightings.length?'No confident cross-camera links in this window yet.':'No saved sightings for this corridor in the window.','widget-note'),{id:'no-routes'})]);
    $('routes-status').textContent=`${rows.length} saved frames from ${new Set(rows.map(r=>r.cameraId)).size} cameras between ${time(end-windowMs)} and ${time(end)}. ${headed}/${sightings.length} vehicles have a compass heading; ${described}/${sightings.length} have a colour profile. ${!$('route-highway').value?'Choose a corridor so its cameras refresh together. ':''}${isLive()?'Live: new photos extend routes as they arrive.':'Showing routes before the timeline cursor.'}`;
    notifyPane('routes','Vehicle routes',[found.map(r=>r.id),flows.length],$('routes-updated'),$('routes-pane'));
  }
  function schedule(){clearTimeout(paintTimer);paintTimer=setTimeout(render,250);}
  for(const id of ['route-window','route-confidence','route-flows'])$(id).addEventListener('input',schedule);
  $('route-highway').addEventListener('change',()=>{selectedRoute=null;limit=24;window.render?.();schedule();});
  setInterval(()=>{computeFeatures();if(archiveRevision!==lastRevision||!isLive()){lastRevision=archiveRevision;schedule();}},Math.max(1500,powerProfile.paintMs*10));
  window.renderRoutes=schedule;
})();

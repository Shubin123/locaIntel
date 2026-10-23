/* Cross-camera vehicle routes through time. Facing + camera look direction give a
   travel heading; colour, class and plausible travel time link sightings between
   cameras. Links are appearance re-identification candidates, never plate-level. */
(function(root){
  const VERSION='routes-v1',R=6371e3,rad=Math.PI/180;
  const COMPASS={n:0,north:0,ne:45,northeast:45,e:90,east:90,se:135,southeast:135,s:180,south:180,sw:225,southwest:225,w:270,west:270,nw:315,northwest:315};
  const NAMES=['N','NE','E','SE','S','SW','W','NW'];
  const wrap=a=>((a%360)+360)%360,diff=(a,b)=>Math.abs(((a-b+540)%360)-180);
  const compassName=a=>NAMES[Math.round(wrap(a)/45)%8];
  function lookBearing(camera){
    const o=String(camera?.orientation||'').trim().toLowerCase();if(o in COMPASS)return COMPASS[o];
    const m=/looking\s+(north|south|east|west)\s*-?\s*(east|west)?|looking\s+(northeast|northwest|southeast|southwest)/i.exec(camera?.caption||'');
    if(!m)return null;const key=(m[3]||(m[1]+(m[2]||''))).toLowerCase();return key in COMPASS?COMPASS[key]:null;
  }
  // Camera-relative travel direction: 0 = away from camera (same as look), 180 = toward.
  function relativeHeading(o){
    if(o?.state!=='estimated')return null;
    if(Number.isFinite(o.travel_offset))return o.travel_offset;
    return {away_from_camera:0,toward_camera:180,image_right:90,image_left:270}[o.facing]??null;
  }
  function heading(camera,o){const look=lookBearing(camera),rel=relativeHeading(o);return look==null||rel==null?null:wrap(look+rel);}
  function distance(a,b){const p=a.lat*rad,q=b.lat*rad,dp=q-p,dl=(b.lon-a.lon)*rad,h=Math.sin(dp/2)**2+Math.cos(p)*Math.cos(q)*Math.sin(dl/2)**2;return 2*R*Math.asin(Math.min(1,Math.sqrt(h)));}
  function bearing(a,b){const p=a.lat*rad,q=b.lat*rad,dl=(b.lon-a.lon)*rad;return wrap(Math.atan2(Math.sin(dl)*Math.cos(q),Math.cos(p)*Math.sin(q)-Math.sin(p)*Math.cos(q)*Math.cos(dl))/rad);}

  // Colour descriptor: 8 saturation-weighted hues + 4 brightness bins + neutral share, from the box interior.
  function describePixels(rgba){
    const hist=Array(13).fill(0);let n=0;
    for(let i=0;i<rgba.length;i+=4){
      const r=rgba[i]/255,g=rgba[i+1]/255,b=rgba[i+2]/255,max=Math.max(r,g,b),min=Math.min(r,g,b),s=max?(max-min)/max:0;
      let h=0;if(max>min){h=max===r?(g-b)/(max-min):max===g?2+(b-r)/(max-min):4+(r-g)/(max-min);h=wrap(h*60);}
      const colour=s>.28&&max>.18?s:0;hist[Math.floor(h/45)%8]+=colour;hist[8+Math.min(3,Math.floor(max*4))]+=1-colour;hist[12]+=1-colour;n++;
    }
    return n?hist.map(v=>Math.round(v/n*1000)/1000):null;
  }
  function colourName(f){
    if(!f)return 'unknown colour';const hue=f.slice(0,8),chroma=hue.reduce((a,b)=>a+b,0);
    if(chroma>.12){const k=hue.indexOf(Math.max(...hue));return ['red','orange','yellow','green','teal','blue','purple','pink'][k];}
    const v=f.slice(8,12),mean=(v[0]*.12+v[1]*.37+v[2]*.62+v[3]*.88)/Math.max(1e-6,v.reduce((a,b)=>a+b,0));
    return mean>.66?'white':mean<.3?'black':'grey';
  }
  // Common neutral colours are weak evidence; saturated colours are distinctive.
  function similarity(a,b){
    if(!a||!b)return 0;let inter=0,total=0;for(let i=0;i<12;i++){inter+=Math.min(a[i],b[i]);total+=Math.max(a[i],b[i]);}
    return total?inter/total:0;
  }
  function distinctiveness(f){if(!f)return 0;const chroma=f.slice(0,8).reduce((a,b)=>a+b,0);return Math.min(1,.35+chroma*2.5);}
  async function describe(blob,detections){
    const bitmap=await createImageBitmap(blob);
    try{
      const canvas=typeof OffscreenCanvas!=='undefined'?new OffscreenCanvas(12,12):Object.assign(document.createElement('canvas'),{width:12,height:12}),ctx=canvas.getContext('2d',{willReadFrequently:true});
      return detections.map(d=>{
        const [x,y,xx,yy]=d.box,w=(xx-x)*bitmap.width,h=(yy-y)*bitmap.height;if(w<6||h<5)return null;
        ctx.clearRect(0,0,12,12);ctx.drawImage(bitmap,(x+(xx-x)*.18)*bitmap.width,(y+(yy-y)*.15)*bitmap.height,w*.64,h*.6,0,0,12,12);
        return describePixels(ctx.getImageData(0,0,12,12).data);
      });
    }finally{bitmap.close();}
  }

  const size=d=>(d.box[2]-d.box[0])*(d.box[3]-d.box[1]);
  const labelsCompatible=(a,b)=>a===b||(a==='car'&&b==='truck')||(a==='truck'&&b==='car');
  // Upstream capture time when known: fetch times can lag a photo by minutes.
  const captured=row=>{const t=Date.parse(row.sourceModified||'');return Number.isFinite(t)&&t<=row.observedAt+60000?t:row.observedAt;};
  // Flatten saved frames into sightings that carry heading and colour.
  function sightings(rows){
    const out=[];
    for(const row of rows){
      const camera=row.camera||{};if(!Number.isFinite(camera.lat)||!Number.isFinite(camera.lon))continue;
      for(const [i,d]of (row.detections||[]).entries()){
        if(!['car','truck','bus','motorcycle'].includes(d.label))continue;
        // Low-tier facing (~55% held-out accuracy) is too weak to steer a link.
        const o=row.facing?.byDetection?.[i],f=row.routeFeatures?.byDetection?.[i]||null,h=o?.tier==='low'?null:heading(camera,o);
        out.push({key:row.key+':'+i,frame:row.key,hash:row.hash,index:i,cameraId:row.cameraId,camera,lat:camera.lat,lon:camera.lon,t:captured(row),label:d.label,box:d.box,area:size(d),heading:h,facing:o?.state==='estimated'?o.facing:null,confidence:o?.confidence??null,colour:f,colourName:colourName(f)});
      }
    }
    return out.sort((a,b)=>a.t-b.t);
  }
  // DriveBC stills refresh every few minutes, so colour coincidences are common:
  // links stay short (next sampled cameras only), at legal-ish speeds, with a heading at both ends.
  const defaults={minSpeed:35/3.6,maxSpeed:130/3.6,maxGap:30*60000,minDistance:400,maxDistance:25000,headingTolerance:60,minScore:.55,minMargin:.08,neighbours:2,neutralSimilarity:.8,minSimilarity:.5};
  const NEUTRAL=new Set(['white','black','grey']);
  function geometry(cache,a,b){
    const key=a.cameraId+'>'+b.cameraId;let g=cache.get(key);if(!g){g={distance:distance(a,b),course:bearing(a,b)};cache.set(key,g);}return g;
  }
  function plausibility(a,b,opts,cache=new Map()){
    const {distance:d,course}=geometry(cache,a,b),dt=(b.t-a.t)/1000;if(d<opts.minDistance||d>opts.maxDistance||dt<=0||dt*1000>opts.maxGap)return null;
    const speed=d/dt;if(speed<opts.minSpeed*.6||speed>opts.maxSpeed)return null;if(a.heading==null||diff(a.heading,course)>opts.headingTolerance)return null;
    if(b.heading==null||diff(b.heading,course)>opts.headingTolerance+20)return null;
    const speedFit=speed<opts.minSpeed?speed/opts.minSpeed:speed>opts.maxSpeed?opts.maxSpeed/speed:1;
    return {distance:d,seconds:dt,speed,course,fit:speedFit*(1-diff(a.heading,course)/180)*(1-diff(b.heading,course)/240)};
  }
  // Mutual-best, margin-checked links between consecutive sightings at different cameras.
  function link(list,options={}){
    const opts={...defaults,...options},forward=new Map(),cache=new Map();list=[...list].sort((a,b)=>a.t-b.t);
    // The next few sampled cameras ahead of a heading; a vehicle cannot skip past others unseen.
    const cameras=[...new Map(list.map(s=>[s.cameraId,s])).values()],ahead=new Map();
    const nextCameras=(a,h)=>{const key=a.cameraId+':'+Math.round(h/45)%8;if(!ahead.has(key))ahead.set(key,new Set(cameras.filter(c=>c.cameraId!==a.cameraId).map(c=>({c,g:geometry(cache,a,c)})).filter(({g})=>g.distance>=opts.minDistance&&g.distance<=opts.maxDistance&&diff(g.course,h)<=45).sort((x,y)=>x.g.distance-y.g.distance).slice(0,opts.neighbours).map(({c})=>c.cameraId)));return ahead.get(key);};
    for(let i=0;i<list.length;i++){
      const a=list[i];if(a.heading==null||!a.colour)continue;const scored=[],next=nextCameras(a,a.heading),neutral=NEUTRAL.has(colourName(a.colour));
      for(let j=i+1;j<list.length&&list[j].t-a.t<=opts.maxGap;j++){
        const b=list[j];if(b.t<=a.t||!next.has(b.cameraId)||!b.colour||!labelsCompatible(a.label,b.label))continue;
        const p=plausibility(a,b,opts,cache);if(!p)continue;
        const look=similarity(a.colour,b.colour),score=look*(.55+.45*p.fit);
        if(look<(neutral?opts.neutralSimilarity:opts.minSimilarity))continue;scored.push({a,b,...p,similarity:look,score});
      }
      scored.sort((x,y)=>y.score-x.score);
      let best=scored[0];if(!best)continue;
      // A later camera further along the same course is a continuation, not a rival: link the nearest hop.
      const along=(x,y)=>y.b.t>x.b.t&&plausibility(x.b,y.b,{...opts,headingTolerance:opts.headingTolerance+30},cache)&&similarity(x.b.colour,y.b.colour)>=.5;
      for(const c of scored)if(c.b.t<best.b.t&&c.score>=best.score-opts.minMargin*2&&along(c,best))best=c;
      const rivals=scored.filter(c=>c!==best&&!along(best,c));
      best.margin=rivals[0]?best.score-rivals[0].score:best.score;
      best.confidence=Math.max(0,Math.min(1,(best.score-opts.minScore)/(1-opts.minScore)))*.6+Math.min(1,best.margin/.25)*.25+distinctiveness(a.colour)*.15;
      if(neutral)best.confidence=Math.min(best.confidence,.55); // white/black/grey never reach "high"
      if(best.score>=opts.minScore&&best.margin>=opts.minMargin)forward.set(a.key,best);
    }
    const backward=new Map();for(const l of forward.values()){const prior=backward.get(l.b.key);if(!prior||prior.score<l.score)backward.set(l.b.key,l);}
    return [...backward.values()].filter(l=>forward.get(l.a.key)===l);
  }
  function routes(list,options={}){
    const links=link(list,options),next=new Map(links.map(l=>[l.a.key,l])),hasPrior=new Set(links.map(l=>l.b.key)),out=[];
    for(const l of links){
      if(hasPrior.has(l.a.key))continue;
      const hops=[l.a],legs=[];let cur=l;while(cur&&hops.length<12){legs.push(cur);hops.push(cur.b);cur=next.get(cur.b.key);}
      const span=(hops.at(-1).t-hops[0].t)/1000,length=legs.reduce((s,x)=>s+x.distance,0);
      out.push({id:hops[0].key,hops,legs,start:hops[0].t,end:hops.at(-1).t,distance:length,seconds:span,speed:span?length/span:null,
        heading:compassName(bearing(hops[0],hops.at(-1))),colour:hops[0].colourName,label:hops[0].label,
        confidence:legs.reduce((m,x)=>Math.min(m,x.confidence),1)});
    }
    return out.sort((a,b)=>b.end-a.end);
  }
  // Decoy test: shift each camera's clock by ±10–20 min so true vehicles no longer
  // line up; links that still appear are chance. Deterministic for stable UI.
  function chance(list,options={},trials=6){
    if(!list.length)return 0;let seed=7,total=0;const rand=()=>(seed=(seed*16807)%2147483647)/2147483647,ids=[...new Set(list.map(s=>s.cameraId))];
    for(let k=0;k<trials;k++){const shift=new Map(ids.map(id=>[id,(rand()<.5?-1:1)*(10+rand()*10)*60000]));total+=routes(list.map(s=>({...s,t:s.t+shift.get(s.cameraId)})),options).length;}
    return total/trials;
  }
  // Compass headings per time bin for a corridor: direction of travel through time.
  function flowSeries(list,start,end,bins=24){
    const width=Math.max(1,(end-start)/bins),out=Array.from({length:bins},(_,i)=>({start:start+i*width,end:start+(i+1)*width,counts:{N:0,E:0,S:0,W:0},vehicles:0}));
    for(const s of list){if(s.t<start||s.t>end)continue;const bin=out[Math.min(bins-1,Math.floor((s.t-start)/width))];bin.vehicles++;if(s.heading!=null)bin.counts[['N','E','S','W'][Math.round(s.heading/90)%4]]++;}
    return out;
  }
  // Same-camera tracks: the same vehicle across consecutive saved frames, matched by
  // its 12×12 appearance patch, position and size (TemporalTraffic.matches) under a
  // stable camera view. Queues, congestion and slow traffic stay in view for
  // minutes, so these tracks are real even at DriveBC's refresh rate. With
  // decoy:true frame B is mirrored left↔right, which breaks true matches but keeps
  // position statistics; the decoy count estimates chance tracks.
  const meanDifference=(a,b)=>{if(!Array.isArray(a)||!Array.isArray(b)||a.length!==b.length)return 1;let s=0,n=0;for(let i=0;i<a.length;i++)if(Number.isFinite(a[i])&&Number.isFinite(b[i])){s+=Math.abs(a[i]-b[i]);n++;}return n>=a.length*.4?s/n:1;};
  const mirrorPatch=p=>p.length===432?p.map((_,i)=>{const c=i%3,x=Math.floor(i/3)%12,y=Math.floor(i/36);return p[(y*12+11-x)*3+c];}):p;
  const mirrorFrame=f=>({...f,detections:f.detections.map(d=>({...d,box:[1-d.box[2],d.box[1],1-d.box[0],d.box[3]],appearance:Array.isArray(d.appearance)?mirrorPatch(d.appearance):d.appearance}))});
  // Mutual-best matching for frames minutes apart: lighting drifts, so the error
  // limit is looser than temporal motion's, but each match must be unique.
  function trackMatches(a,b,{maxError=.08,maxMove=.12,margin=.015}={}){
    const c=d=>[(d.box[0]+d.box[2])/2,(d.box[1]+d.box[3])/2],area=d=>(d.box[2]-d.box[0])*(d.box[3]-d.box[1]),ok=d=>Array.isArray(d.appearance)&&d.appearance.length>=108&&['car','truck','bus','motorcycle'].includes(d.label);
    const pairs=[];
    for(let j=0;j<b.detections.length;j++){const q=b.detections[j];if(!ok(q))continue;
      for(let i=0;i<a.detections.length;i++){const p=a.detections[i];if(!ok(p)||!labelsCompatible(p.label,q.label)||p.appearance.length!==q.appearance.length)continue;
        const [px,py]=c(p),[qx,qy]=c(q);if(Math.hypot(px-qx,py-qy)>maxMove||Math.abs(Math.log(area(p)/area(q)))>.5)continue;
        pairs.push({i,j,e:meanDifference(p.appearance,q.appearance)});}}
    const best=(key,id)=>pairs.filter(x=>x[key]===id).sort((x,y)=>x.e-y.e),result=new Map();
    for(const x of pairs){if(x.e>maxError)continue;const fromA=best('i',x.i),fromB=best('j',x.j);
      if(fromA[0]!==x||fromB[0]!==x)continue;if((fromA[1]&&fromA[1].e-x.e<margin)||(fromB[1]&&fromB[1].e-x.e<margin))continue;result.set(x.j,x.i);}
    return result;
  }
  function tracks(rows,{maxGap=30*60000,decoy=false,matcher=trackMatches,match={}}={}){
    if(typeof matcher!=='function')return [];
    const ready=rows.filter(r=>r.features?.version===1&&r.features.hash===r.hash&&Array.isArray(r.features.detections)),cams=new Map(),out=[];
    for(const r of ready){if(!cams.has(r.cameraId))cams.set(r.cameraId,[]);cams.get(r.cameraId).push(r);}
    for(const list of cams.values()){
      list.sort((a,b)=>captured(a)-captured(b));const open=new Map();
      for(let i=1;i<list.length;i++){
        const A=list[i-1],B=list[i],gap=captured(B)-captured(A);
        if(gap<=0||gap>maxGap||A.features.pixelSignature===B.features.pixelSignature||meanDifference(A.features.background,B.features.background)>=.035)continue;
        for(const [j,ia]of matcher(A.features,decoy?mirrorFrame(B.features):B.features,match)){
          const hop=(row,k)=>{const d=row.features.detections[k],fp=JSON.stringify([d.label,d.box]),o=row.facing?.byFingerprint?.[fp]||Object.values(row.facing?.byDetection||{}).find(x=>x?.fingerprint===fp);return {key:row.key+':'+k,frame:row.key,hash:row.hash,index:k,cameraId:row.cameraId,camera:row.camera||{},t:captured(row),label:d.label,box:d.box,orientation:o?.state==='estimated'?o:null};};
          let track=open.get(A.key+':'+ia);if(!track){track={hops:[hop(A,ia)]};out.push(track);}
          track.hops.push(hop(B,j));open.set(B.key+':'+j,track);
        }
      }
    }
    return out.map(track=>{
      const h=track.hops,c=x=>[(x.box[0]+x.box[2])/2,(x.box[1]+x.box[3])/2],first=c(h[0]),last=c(h.at(-1)),moved=Math.hypot(last[0]-first[0],last[1]-first[1]);
      const o=h.map(x=>x.orientation).reverse().find(x=>x&&x.tier!=='low'),compass=o?heading(h[0].camera,o):null;
      return {id:h[0].key,cameraId:h[0].cameraId,camera:h[0].camera,hops:h,start:h[0].t,end:h.at(-1).t,seconds:(h.at(-1).t-h[0].t)/1000,label:h[0].label,
        moved,state:moved>=.02?'moving':'stationary',imageDirection:moved>=.02?['→','↘','↓','↙','←','↖','↑','↗'][Math.round(((Math.atan2(last[1]-first[1],last[0]-first[0])*180/Math.PI+360)%360)/45)%8]:null,
        heading:compass==null?null:compassName(compass)};
    }).sort((a,b)=>b.hops.length-a.hops.length||b.seconds-a.seconds);
  }
  // Per-camera compass flow from facing alone, robust even when no route links.
  function flows(list){
    const cams=new Map();
    for(const s of list){
      let c=cams.get(s.cameraId);if(!c){c={cameraId:s.cameraId,camera:s.camera,lat:s.lat,lon:s.lon,look:lookBearing(s.camera),vehicles:0,headed:0,bins:Array(8).fill(0)};cams.set(s.cameraId,c);}
      c.vehicles++;if(s.heading!=null){c.headed++;c.bins[Math.round(s.heading/45)%8]++;}
    }
    for(const c of cams.values()){const order=c.bins.map((n,i)=>[n,i]).sort((a,b)=>b[0]-a[0]);c.dominant=order[0][0]?{direction:NAMES[order[0][1]],bearing:order[0][1]*45,share:order[0][0]/c.headed}:null;c.secondary=order[1][0]?{direction:NAMES[order[1][1]],bearing:order[1][1]*45,share:order[1][0]/c.headed}:null;}
    return [...cams.values()];
  }
  root.VehicleRoutes={VERSION,lookBearing,relativeHeading,heading,distance,bearing,describePixels,describe,colourName,similarity,sightings,link,routes,chance,trackMatches,tracks,flowSeries,flows,compassName};
})(globalThis);

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
  // Flatten saved frames into sightings that carry heading and colour.
  function sightings(rows){
    const out=[];
    for(const row of rows){
      const camera=row.camera||{};if(!Number.isFinite(camera.lat)||!Number.isFinite(camera.lon))continue;
      for(const [i,d]of (row.detections||[]).entries()){
        if(!['car','truck','bus','motorcycle'].includes(d.label))continue;
        const o=row.facing?.byDetection?.[i],f=row.routeFeatures?.byDetection?.[i]||null,h=heading(camera,o);
        out.push({key:row.key+':'+i,frame:row.key,hash:row.hash,index:i,cameraId:row.cameraId,camera,lat:camera.lat,lon:camera.lon,t:row.observedAt,label:d.label,box:d.box,area:size(d),heading:h,facing:o?.state==='estimated'?o.facing:null,confidence:o?.confidence??null,colour:f,colourName:colourName(f)});
      }
    }
    return out.sort((a,b)=>a.t-b.t);
  }
  const defaults={minSpeed:20/3.6,maxSpeed:140/3.6,maxGap:45*60000,minDistance:400,maxDistance:60000,headingTolerance:60,minScore:.55,minMargin:.08};
  function geometry(cache,a,b){
    const key=a.cameraId+'>'+b.cameraId;let g=cache.get(key);if(!g){g={distance:distance(a,b),course:bearing(a,b)};cache.set(key,g);}return g;
  }
  function plausibility(a,b,opts,cache=new Map()){
    const {distance:d,course}=geometry(cache,a,b),dt=(b.t-a.t)/1000;if(d<opts.minDistance||d>opts.maxDistance||dt<=0||dt*1000>opts.maxGap)return null;
    const speed=d/dt;if(speed<opts.minSpeed*.6||speed>opts.maxSpeed*1.25)return null;if(a.heading==null||diff(a.heading,course)>opts.headingTolerance)return null;
    if(b.heading!=null&&diff(b.heading,course)>opts.headingTolerance+20)return null;
    const speedFit=speed<opts.minSpeed?speed/opts.minSpeed:speed>opts.maxSpeed?opts.maxSpeed/speed:1;
    return {distance:d,seconds:dt,speed,course,fit:speedFit*(1-diff(a.heading,course)/180)*(b.heading==null?.85:1-diff(b.heading,course)/240)};
  }
  // Mutual-best, margin-checked links between consecutive sightings at different cameras.
  function link(list,options={}){
    const opts={...defaults,...options},forward=new Map(),cache=new Map();list=[...list].sort((a,b)=>a.t-b.t);
    for(let i=0;i<list.length;i++){
      const a=list[i];if(a.heading==null||!a.colour)continue;const scored=[];
      for(let j=i+1;j<list.length&&list[j].t-a.t<=opts.maxGap;j++){
        const b=list[j];if(b.t<=a.t||b.cameraId===a.cameraId||!b.colour||!labelsCompatible(a.label,b.label))continue;
        const p=plausibility(a,b,opts,cache);if(!p)continue;
        const look=similarity(a.colour,b.colour),score=look*(.55+.45*p.fit);
        if(look<.5)continue;scored.push({a,b,...p,similarity:look,score});
      }
      scored.sort((x,y)=>y.score-x.score);
      let best=scored[0];if(!best)continue;
      // A later camera further along the same course is a continuation, not a rival: link the nearest hop.
      const along=(x,y)=>y.b.t>x.b.t&&plausibility(x.b,y.b,{...opts,headingTolerance:opts.headingTolerance+30},cache)&&similarity(x.b.colour,y.b.colour)>=.5;
      for(const c of scored)if(c.b.t<best.b.t&&c.score>=best.score-opts.minMargin*2&&along(c,best))best=c;
      const rivals=scored.filter(c=>c!==best&&!along(best,c));
      best.margin=rivals[0]?best.score-rivals[0].score:best.score;
      best.confidence=Math.max(0,Math.min(1,(best.score-opts.minScore)/(1-opts.minScore)))*.6+Math.min(1,best.margin/.25)*.25+distinctiveness(a.colour)*.15;
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
  root.VehicleRoutes={VERSION,lookBearing,relativeHeading,heading,distance,bearing,describePixels,describe,colourName,similarity,sightings,link,routes,flows,compassName};
})(globalThis);

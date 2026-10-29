/* Follows one vehicle through DriveBC cameras from its GPS track.
   1. Relevance: only fixes within RELEVANT_RADIUS of a camera are ever stored.
   2. Camera matching: a pass is a visit within the radius and in front of a camera.
   3. Time matching: saved frames captured while the vehicle was in view.
   4. Verification: every detected vehicle in those frames is scored against the
      vehicle's profile (body class, colour, facing), and a decoy run on frames
      from the same cameras without the vehicle shows how often a stranger scores
      as well. Any vehicle with a GPS track can be followed; nothing here looks for
      a vehicle that has no track. */
(function(root){
  const R=root.VehicleRoutes,{lookBearing,distance,bearing,captured,similarity,colourName,NEUTRAL}=R;
  const VERSION='matcher-v1',DAY=86400000,RELEVANT_RADIUS=500,VEHICLES=['car','truck','bus','motorcycle'];
  const wrap=a=>((a%360)+360)%360,diff=(a,b)=>Math.abs(((a-b+540)%360)-180);
  const defaults={radius:250,fov:70,near:40,maxGap:30000,edge:5000,maxAccuracy:50,minMove:3,tolerance:150000,decoyGap:600000,maxDecoys:300,confirm:.6,possible:.35};
  // YOLO's COCO classes are coarse: SUVs are mostly "car", pickups often "truck".
  // These weights are hand-set compatibilities, not a trained body-type classifier.
  const BODY_TYPES={
    sedan:{name:'Car (sedan, hatchback, wagon)',classes:{car:1,truck:.15}},
    suv:{name:'SUV, crossover or minivan',classes:{car:.85,truck:.5}},
    pickup:{name:'Pickup truck',classes:{truck:1,car:.45}},
    van:{name:'Van',classes:{truck:.8,car:.6,bus:.2}},
    truck:{name:'Large truck',classes:{truck:1,bus:.3}},
    bus:{name:'Bus',classes:{bus:1,truck:.4}},
    motorcycle:{name:'Motorcycle',classes:{motorcycle:1}}
  };
  // colourName's hue order, which wraps from pink back to red.
  const HUES=['red','orange','yellow','green','blue','indigo','purple','pink'],COLOURS=['white','grey','black',...HUES];

  // Interpolated position at time t. Gaps over maxGap are not bridged: a stop at
  // the edge of a tunnel is not evidence of where the vehicle was in between.
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
  // Where the vehicle is relative to the camera's view. Without a look direction we
  // cannot tell whether it is in front of or behind the camera.
  function view(camera,p,o){
    const d=distance(camera,p),look=lookBearing(camera),offAxis=look==null?null:diff(look,bearing(camera,p));
    return {distance:d,look,offAxis,inView:look==null?null:d<=o.near||offAxis<=o.fov};
  }
  // Travelling the way the camera looks shows the vehicle's rear, i.e. away_from_camera.
  function expectedFacing(look,travel){
    if(look==null||travel==null)return null;const r=wrap(travel-look);
    return r<45||r>=315?'away_from_camera':r<135?'image_right':r<225?'toward_camera':'image_left';
  }

  // Cameras in a grid one radius wide, so a 3×3 block of cells holds every camera within the radius.
  function cameraIndex(cameras,radius=RELEVANT_RADIUS){
    const list=[...new Map(cameras.filter(c=>Number.isFinite(c?.lat)&&Number.isFinite(c?.lon)).map(c=>[String(c.id),c])).values()];
    const top=Math.min(80,Math.max(0,...list.map(c=>Math.abs(c.lat)))),dLat=radius/111320,dLon=radius/(111320*Math.cos(top*Math.PI/180)),cells=new Map();
    const cell=p=>[Math.floor(p.lat/dLat),Math.floor(p.lon/dLon)];
    for(const c of list){const k=cell(c).join(':');if(!cells.has(k))cells.set(k,[]);cells.get(k).push(c);}
    function near(p){
      const [i,j]=cell(p),out=[];
      for(let a=i-1;a<=i+1;a++)for(let b=j-1;b<=j+1;b++)for(const camera of cells.get(a+':'+b)||[]){const d=distance(camera,p);if(d<=radius)out.push({camera,distance:d});}
      return out.sort((x,y)=>x.distance-y.distance);
    }
    return {radius,cameras:list,near};
  }
  // Keeps the fixes that can place a vehicle at a camera: those within the radius,
  // plus the fix just before entering and just after leaving, so positions
  // interpolate across the edge. A boundary fix more than maxGap away is useless for
  // interpolation and could be anywhere, so it is dropped too.
  class RelevanceFilter{
    constructor(index,{maxGap=defaults.maxGap}={}){Object.assign(this,{index,maxGap});this.inside=false;this.previous=null;}
    push(fix){
      const inside=this.index.near(fix).length>0,prev=this.previous,close=!!prev&&fix.t-prev.fix.t<=this.maxGap,out=[];
      if(inside&&!this.inside&&close&&!prev.kept)out.push(prev.fix);
      const keep=inside||this.inside&&close;if(keep)out.push(fix);
      this.inside=inside;this.previous={fix,kept:keep};return out;
    }
  }
  function compact(fixes,index,options){
    const filters=new Map(),out=[];
    for(const f of [...fixes].sort((a,b)=>a.t-b.t)){let filter=filters.get(f.session);if(!filter)filters.set(f.session,filter=new RelevanceFilter(index,options));out.push(...filter.push(f));}
    return out;
  }

  // Camera matching: every camera the track passed in view of, one entry per visit.
  function passes(cameras,fixes,options={}){
    const o={...defaults,...options},list=usable(fixes,o),out=[];if(!list.length)return out;
    let south=90,north=-90,west=180,east=-180;for(const f of list){south=Math.min(south,f.lat);north=Math.max(north,f.lat);west=Math.min(west,f.lon);east=Math.max(east,f.lon);}
    const padLat=o.radius/111320,padLon=o.radius/(111320*Math.max(.05,Math.cos((south+north)/2*Math.PI/180)));
    for(const camera of cameras){
      if(!Number.isFinite(camera.lat)||!Number.isFinite(camera.lon)||camera.lat<south-padLat||camera.lat>north+padLat||camera.lon<west-padLon||camera.lon>east+padLon)continue;
      let visit=null;const close=()=>{if(visit)out.push(visit);visit=null;};
      for(let i=0;i<list.length;i++){
        const f=list[i],v=view(camera,f,o);if(v.distance>o.radius||v.inView===false)continue;
        if(visit&&(f.t-visit.end>o.maxGap||f.session!==visit.session))close();
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
  // DriveBC's Replay the Day keeps about 24 hours of images per camera.
  function replay(camera,start){
    const id=encodeURIComponent(String(camera.id));
    return {page:camera.source||`https://www.drivebc.ca/cameras/${id}`,list:`https://www.drivebc.ca/api/webcams/${id}/replayTheDay/`,availableUntil:start+DAY};
  }

  // Upstream capture time (DriveBC's Expires header or a replay stamp) when the frame
  // has one; otherwise only the time it was fetched, which can trail capture by minutes.
  function frameTime(row){const t=captured(row);return {t,exact:Number.isFinite(t)&&t===Date.parse(row.sourceModified||'')};}
  // Time matching. A frame with a capture time is in view when the interpolated
  // position at that moment is. One with only a fetch time was captured up to
  // `tolerance` earlier, so it is uncertain when that window overlaps the pass.
  // Other frames within the tolerance are near misses, kept to say how close they were.
  function matchFrames(passList,rows,fixes,options={}){
    const o={...defaults,...options},list=usable(fixes,o),byCamera=new Map();
    for(const row of rows){const id=String(row.cameraId);if(!byCamera.has(id))byCamera.set(id,[]);byCamera.get(id).push(row);}
    return passList.map(pass=>{
      const out=[];
      for(const row of byCamera.get(pass.cameraId)||[]){
        const {t,exact}=frameTime(row);if(!Number.isFinite(t))continue;
        const from=exact?t:t-o.tolerance;if(t<pass.start-o.tolerance||from>pass.end+o.tolerance)continue;
        const base={key:row.key,cameraId:pass.cameraId,hash:row.hash,camera:pass.camera,t,exact,offset:t-pass.closest.t,session:pass.session};
        if(exact){
          const p=positionAt(list,t,o),v=p&&view(pass.camera,p,o);
          if(p&&v.distance<=o.radius&&v.inView!==false)out.push({...base,kind:'in_view',position:p,...v,expectedFacing:expectedFacing(v.look,p.course)});
          else out.push({...base,kind:'missed'});
        }else if(Math.min(t,pass.end)>=Math.max(from,pass.start)){
          const c=pass.closest;
          out.push({...base,kind:'uncertain',position:{t:c.t,lat:c.lat,lon:c.lon,accuracy:c.accuracy,course:c.course,session:pass.session},distance:c.distance,offAxis:c.offAxis,look:lookBearing(pass.camera),inView:true,expectedFacing:pass.expectedFacing});
        }else out.push({...base,kind:'missed'});
      }
      return out.sort((a,b)=>a.t-b.t);
    });
  }

  // References: the profile's photos plus every frame where this vehicle was marked.
  function references(vehicle,labels=[]){
    return [...(vehicle?.references||[]).filter(r=>Array.isArray(r.descriptor)).map(r=>({source:'photo',id:r.id,descriptor:r.descriptor})),
      ...labels.filter(l=>l.vehicle===vehicle?.id&&l.detection!=null&&Array.isArray(l.descriptor)).map(l=>({source:'label',id:l.key,frame:l.frame,descriptor:l.descriptor}))];
  }
  // Each cue scores one detection 0–1 against the profile, or null without evidence.
  // Extra cues (an embedding model, say) can be passed to verify() alongside these.
  const side=f=>f==='image_left'||f==='image_right';
  const CUES={
    body:(d,c)=>BODY_TYPES[c.vehicle?.bodyType]?BODY_TYPES[c.vehicle.bodyType].classes[d.label]??0:null,
    // A frame is never scored against a label made on that same frame.
    colour:(d,c)=>{
      if(!d.descriptor)return null;const refs=c.references.filter(r=>r.frame!==c.frame.key);
      if(refs.length)return Math.max(...refs.map(r=>similarity(d.descriptor,r.descriptor)));
      const want=c.vehicle?.colour;if(!want)return null;const got=colourName(d.descriptor);
      // Hue bins are 45° wide from 0°, so a red near 360° reads as pink: neighbouring hues get partial credit.
      const i=HUES.indexOf(got),j=HUES.indexOf(want),near=i>=0&&j>=0&&Math.min(Math.abs(i-j),HUES.length-Math.abs(i-j))===1;
      return got===want?.8:near?.5:NEUTRAL.has(got)&&NEUTRAL.has(want)?.3:0;
    },
    // Low-tier facing (~55% held-out accuracy) is too weak to count.
    facing:(d,c)=>{
      const want=c.frame.expectedFacing,o=d.facing;if(!want||o?.state!=='estimated'||o.tier==='low')return null;
      return o.facing===want?1:side(o.facing)&&side(want)?.2:side(o.facing)!==side(want)?.35:0;
    }
  };
  const WEIGHTS={body:.25,colour:.5,facing:.25};
  function verify(frame,row,{vehicle,references:refs=[],options={},cues={}}={}){
    const o={...defaults,...options},all={...CUES,...cues},weights={...WEIGHTS,...o.weights};
    if(!row||!Array.isArray(row.detections))return {detections:[],best:null,margin:null,confidence:null,status:'pending'};
    const features=row.routeFeatures?.byDetection||{},scored=[];
    for(const [index,d]of row.detections.entries()){
      if(!VEHICLES.includes(d.label))continue;
      const detection={index,label:d.label,box:d.box,descriptor:features[index]||null,facing:row.facing?.byDetection?.[index]||null},parts={};let sum=0,total=0;
      for(const [name,cue]of Object.entries(all)){const v=cue(detection,{vehicle,references:refs,frame});if(v==null)continue;const w=weights[name]??.25;parts[name]=Math.round(v*1000)/1000;sum+=v*w;total+=w;}
      scored.push({index,label:d.label,box:d.box,colour:detection.descriptor?colourName(detection.descriptor):null,score:total?sum/total:null,cues:parts});
    }
    scored.sort((a,b)=>(b.score??-1)-(a.score??-1)||a.index-b.index);
    const [best,second]=scored;
    if(!best)return {detections:scored,best:null,margin:null,confidence:null,status:'empty'};
    if(best.score==null)return {detections:scored,best,margin:null,confidence:null,status:'unverified'};
    // A lookalike in the same frame lowers confidence; white, grey and black are common, so they count for less.
    const margin=best.score-(second?.score??0),tone=vehicle?.colour||(refs[0]&&colourName(refs[0].descriptor));
    const confidence=best.score*(.55+.45*Math.min(1,margin/.25))*(NEUTRAL.has(tone)?.85:1);
    // Without a capture time the vehicle may not be in the frame at all; facing is
    // shared by everyone driving the same way and almost everything is a "car". So a
    // confirmation needs a capture time and an appearance cue beyond class and facing.
    const appearance=Object.keys(best.cues).some(k=>k!=='facing'&&k!=='body');
    const status=confidence>=o.confirm&&frame.kind==='in_view'&&appearance?'confirmed':confidence>=o.possible?'possible':'unmatched';
    return {detections:scored,best,margin,confidence,status};
  }

  // Decoy: the same verification on frames of the passed cameras captured when the
  // vehicle was nowhere near (over decoyGap from every visit), scored as if it were
  // in view. Confirmations there are strangers that look like the profile.
  function decoy({vehicle,passes:passList,rows,references:refs,options={},cues}){
    const o={...defaults,...options},visits=new Map(),out={frames:0,confirmed:0,possible:0,rate:null};
    for(const p of passList){if(!visits.has(p.cameraId))visits.set(p.cameraId,[]);visits.get(p.cameraId).push(p);}
    for(const row of [...rows].sort((a,b)=>b.observedAt-a.observedAt)){
      const list=visits.get(String(row.cameraId));if(!list||!row.detections?.some(d=>VEHICLES.includes(d.label)))continue;
      const {t}=frameTime(row);if(list.some(p=>t>=p.start-o.decoyGap&&t<=p.end+o.decoyGap))continue;
      const pass=list.reduce((a,b)=>Math.abs(b.closest.t-t)<Math.abs(a.closest.t-t)?b:a),v=verify({key:row.key,kind:'in_view',expectedFacing:pass.expectedFacing},row,{vehicle,references:refs,options:o,cues});
      if(v.confidence==null)continue;out.frames++;if(v.status==='confirmed')out.confirmed++;else if(v.status==='possible')out.possible++;
      if(out.frames>=o.maxDecoys)break;
    }
    out.rate=out.frames?out.confirmed/out.frames:null;return out;
  }

  // The whole pipeline for one vehicle: passes, the frames of each pass with their
  // verification, a time-ordered list of sightings, the decoy rate, and how often the
  // automatic pick agrees with the frames you labelled (leave-one-out).
  function track({vehicle,fixes=[],cameras=[],rows=[],labels=[],options={},cues}={}){
    const o={...defaults,...options},own=labels.filter(l=>l.vehicle===vehicle?.id),refs=references(vehicle,own),byFrame=new Map(own.map(l=>[l.frame,l])),rowByKey=new Map(rows.map(r=>[r.key,r]));
    const passList=passes(cameras,fixes,o),matched=matchFrames(passList,rows,fixes,o);
    // A frame inside two nearby visits belongs to the one it was closest to.
    const owner=new Map();for(const list of matched)for(const f of list)if(f.kind!=='missed'){const prior=owner.get(f.key);if(!prior||Math.abs(f.offset)<Math.abs(prior.offset))owner.set(f.key,f);}
    const sightings=[],agreement={labelled:0,agree:0,hidden:0,falseConfirmed:0};
    const out=passList.map((pass,i)=>{
      const frames=[],missed=[];
      for(const f of matched[i]){
        if(f.kind==='missed'){missed.push(f);continue;}if(owner.get(f.key)!==f)continue;
        const v=verify(f,rowByKey.get(f.key),{vehicle,references:refs,options:o,cues}),label=byFrame.get(f.key)||null;
        if(label?.detection!=null){agreement.labelled++;if(v.best?.index===label.detection&&['confirmed','possible'].includes(v.status))agreement.agree++;}
        else if(label){agreement.hidden++;if(v.status==='confirmed')agreement.falseConfirmed++;}
        const status=label?label.detection==null?'not_visible':'labelled':v.status,pick=label?label.detection:['confirmed','possible'].includes(v.status)?v.best.index:null;
        const s={...f,pass,verification:v,label,status,detection:pick,box:pick==null?null:rowByKey.get(f.key)?.detections?.[pick]?.box||null};
        frames.push(s);sightings.push(s);
      }
      const nearest=missed.sort((a,b)=>Math.abs(a.offset)-Math.abs(b.offset))[0]||null;
      const rank=['labelled','confirmed','possible','unverified','pending','unmatched','empty','not_visible'],best=frames.map(f=>f.status).sort((a,b)=>rank.indexOf(a)-rank.indexOf(b))[0]||null;
      return {...pass,frames,nearest,status:best};
    });
    const counts={};for(const s of sightings)counts[s.status]=(counts[s.status]||0)+1;
    return {version:VERSION,passes:out,sightings:sightings.sort((a,b)=>b.t-a.t),counts,references:refs.length,decoy:decoy({vehicle,passes:passList,rows,references:refs,options:o,cues}),agreement};
  }

  root.VehicleMatcher={VERSION,DAY,RELEVANT_RADIUS,VEHICLES,BODY_TYPES,COLOURS,defaults,positionAt,view,expectedFacing,cameraIndex,RelevanceFilter,compact,passes,replay,frameTime,matchFrames,references,CUES,verify,decoy,track};
})(globalThis);
